"""Core data models: academic data tables, timetable configuration, placements.

Everything the solver and the NLP layer need to know about the college lives
here, so both layers share one source of truth.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pandas as pd

import re as _re


def is_batch_key(key: str) -> bool:
    """True for a lab-batch key such as 'CSE-5-A-B1' (not for a section called 'CSE-5-B')."""
    return bool(_re.search(r"-B\d+$", key))


ALL_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

SUBJECT_COLS = ["name", "aliases", "group", "teacher", "sessions_per_week", "type", "duration",
                "batches", "shared_with", "elective_block", "room_pref"]
TEACHER_COLS = ["name", "unavailable_days", "max_per_day", "max_consecutive"]
ROOM_COLS = ["name", "capacity", "type", "building"]
GROUP_COLS = ["name", "department", "semester", "section", "size"]


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass
class Config:
    """Timetable grid configuration (days, periods, lunch)."""

    days: list = field(default_factory=lambda: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
    start_time: str = "09:00"
    slot_minutes: int = 60
    num_slots: int = 9
    lunch_slots: list = field(default_factory=lambda: [4])  # 13:00-14:00 by default
    travel_gap: bool = False                  # keep a free period when moving between buildings
    day_slots: dict = field(default_factory=dict)  # {"Saturday": [0, 4]} -> only slots 0..3 are open that day

    def slot_open(self, day: str, slot: int) -> bool:
        rng = self.day_slots.get(day)
        return rng is None or rng[0] <= slot < rng[1]

    def open_teaching_slots(self, day: str) -> list:
        return [s for s in self.teaching_slots() if self.slot_open(day, s)]

    # ---- time helpers -------------------------------------------------
    def start_min(self) -> int:
        h, m = self.start_time.split(":")
        return int(h) * 60 + int(m)

    def slot_start(self, i: int) -> int:
        return self.start_min() + i * self.slot_minutes

    def slot_end(self, i: int) -> int:
        return self.slot_start(i) + self.slot_minutes

    @staticmethod
    def fmt(minutes: int) -> str:
        minutes = max(0, min(minutes, 24 * 60))
        return f"{minutes // 60:02d}:{minutes % 60:02d}"

    def slot_label(self, i: int) -> str:
        return f"{self.fmt(self.slot_start(i))}-{self.fmt(self.slot_end(i))}"

    def teaching_slots(self) -> list:
        return [i for i in range(self.num_slots) if i not in self.lunch_slots]

    def resolve_period(self, n: int):
        """0-based teaching-period ordinal (or -1 for last) -> slot index."""
        ts = self.teaching_slots()
        if not ts:
            return None
        if n == -1:
            return ts[-1]
        return ts[n] if 0 <= n < len(ts) else None

    def lunch_window(self):
        if not self.lunch_slots:
            return None
        return self.slot_start(min(self.lunch_slots)), self.slot_end(max(self.lunch_slots))

    def copy(self) -> "Config":
        return copy.deepcopy(self)


@dataclass
class Placement:
    """One scheduled session (a block of consecutive periods)."""

    group: str
    subject: str
    teacher: str
    room: str
    day: str
    start: int
    duration: int
    kind: str  # "theory" | "lab"
    groups: tuple = ()      # every student group attending (shared classes, lab batches + their parent)
    occupies: tuple = ()    # student-group keys that are busy during this placement
    size: int = 0           # students in the room (0 = look it up from the group)
    building: str = ""
    label: str = ""         # what to print as the group, e.g. "CSE-7-A+CSE-7-B" or "CSE-7-A-B1"
    elective: str = ""      # elective block name, if any

    @property
    def all_groups(self) -> tuple:
        return self.groups or (self.group,)

    @property
    def keys(self) -> tuple:
        return self.occupies or (self.group,)

    @property
    def end(self) -> int:  # exclusive slot index
        return self.start + self.duration

    def minutes(self, cfg: Config):
        return cfg.slot_start(self.start), cfg.slot_end(self.end - 1)

    def slots(self):
        return range(self.start, self.end)


@dataclass
class Part:
    """One subject inside an elective block (all parts of a block run at the same time)."""

    subject: str
    teacher: str
    kind: str = "theory"
    room_pref: str = ""


@dataclass
class Session:
    """A teaching session that must be placed (one weekly occurrence)."""

    sid: int
    group: str
    subject: str
    teacher: str
    duration: int
    kind: str
    index: int
    groups: tuple = ()          # all attending groups (shared classes / batch + parent)
    occupies: tuple = ()        # student-group keys that cannot do anything else meanwhile
    size: int = 0
    parts: list | None = None   # elective block: list[Part]
    room_pref: str = ""
    label: str = ""
    elective: str = ""

    @property
    def all_groups(self) -> tuple:
        return self.groups or (self.group,)

    def teachers(self) -> list:
        return [p.teacher for p in self.parts] if self.parts else [self.teacher]

    def subjects(self) -> list:
        return [p.subject for p in self.parts] if self.parts else [self.subject]


# --------------------------------------------------------------------------
# Data store
# --------------------------------------------------------------------------
def parse_days_text(text, valid_days=ALL_DAYS) -> set:
    out = set()
    for tok in re.split(r"[,;/|]+", str(text or "")):
        tok = tok.strip().lower()
        if len(tok) < 3:
            continue
        for d in valid_days:
            if d.lower().startswith(tok[:3]):
                out.add(d)
    return out


class DataStore:
    """Holds the four academic tables plus the grid configuration."""

    def __init__(self, subjects=None, teachers=None, rooms=None, groups=None, config=None):
        self.subjects = subjects if subjects is not None else pd.DataFrame(columns=SUBJECT_COLS)
        self.teachers = teachers if teachers is not None else pd.DataFrame(columns=TEACHER_COLS)
        self.rooms = rooms if rooms is not None else pd.DataFrame(columns=ROOM_COLS)
        self.groups = groups if groups is not None else pd.DataFrame(columns=GROUP_COLS)
        self.config = config if config is not None else Config()
        self.clean()

    # ---- normalisation -------------------------------------------------
    def clean(self):
        """Make types sane after the user edits tables in the UI."""

        def _txt(df, cols):
            for c in cols:
                df[c] = df[c].fillna("").astype(str).str.strip()

        def _int(df, col, default):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default).astype(int)

        for df, cols in ((self.subjects, SUBJECT_COLS), (self.teachers, TEACHER_COLS),
                         (self.rooms, ROOM_COLS), (self.groups, GROUP_COLS)):
            for c in cols:
                if c not in df.columns:
                    df[c] = ""
        _txt(self.subjects, ["name", "aliases", "group", "teacher", "type", "shared_with", "elective_block", "room_pref"])
        _int(self.subjects, "sessions_per_week", 1)
        _int(self.subjects, "duration", 1)
        _int(self.subjects, "batches", 0)
        self.subjects["type"] = self.subjects["type"].str.lower().where(
            self.subjects["type"].str.lower().isin(["theory", "lab"]), "theory")
        _txt(self.teachers, ["name", "unavailable_days"])
        _int(self.teachers, "max_per_day", 0)
        _int(self.teachers, "max_consecutive", 0)
        _txt(self.rooms, ["name", "type", "building"])
        _int(self.rooms, "capacity", 0)
        self.rooms["type"] = self.rooms["type"].str.lower().where(
            self.rooms["type"].str.lower().isin(["theory", "lab"]), "theory")
        _txt(self.groups, ["name", "department", "section"])
        _int(self.groups, "semester", 0)
        _int(self.groups, "size", 0)
        for name in ("subjects", "teachers", "rooms", "groups"):
            df = getattr(self, name)
            setattr(self, name, df[df["name"] != ""].reset_index(drop=True))

    # ---- queries -------------------------------------------------------
    def group_names(self):
        return list(self.groups["name"])

    def group_size(self, group: str) -> int:
        row = self.groups[self.groups["name"] == group]
        return int(row["size"].iloc[0]) if len(row) else 0

    def resolve_groups(self, department=None, semester=None, section=None):
        g = self.groups
        if department:
            g = g[g["department"].str.lower() == department.lower()]
        if semester:
            g = g[g["semester"] == int(semester)]
        if section:
            g = g[g["section"].str.lower() == section.lower()]
        return list(g["name"])

    def room_building(self, room: str) -> str:
        row = self.rooms[self.rooms["name"] == room]
        return str(row["building"].iloc[0]) if len(row) else ""

    def teacher_unavailable(self, teacher: str) -> set:
        row = self.teachers[self.teachers["name"] == teacher]
        if not len(row):
            return set()
        return parse_days_text(row["unavailable_days"].iloc[0])

    # ---- persistence ---------------------------------------------------
    def to_dict(self):
        return {
            "subjects": self.subjects.to_dict("records"),
            "teachers": self.teachers.to_dict("records"),
            "rooms": self.rooms.to_dict("records"),
            "groups": self.groups.to_dict("records"),
            "config": asdict(self.config),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            pd.DataFrame(d["subjects"], columns=SUBJECT_COLS),
            pd.DataFrame(d["teachers"], columns=TEACHER_COLS),
            pd.DataFrame(d["rooms"], columns=ROOM_COLS),
            pd.DataFrame(d["groups"], columns=GROUP_COLS),
            Config(**d["config"]),
        )

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def signature(self) -> str:
        return hashlib.md5(json.dumps(self.to_dict(), sort_keys=True, default=str).encode()).hexdigest()


# --------------------------------------------------------------------------
# Sample data
# --------------------------------------------------------------------------
def sample_store() -> DataStore:
    def subj(name, aliases, group, teacher, n, typ="theory", dur=1):
        return dict(name=name, aliases=aliases, group=group, teacher=teacher,
                    sessions_per_week=n, type=typ, duration=dur)

    subjects = []
    for sec in ("A", "B"):
        g = f"CSE-5-{sec}"
        subjects += [
            subj("DBMS", "database management system; database management systems; database", g, "Dr. Rao", 4),
            subj("Operating Systems", "os; operating system", g, "Prof. Mehta", 4),
            subj("Computer Networks", "cn; networks; networking", g, "Dr. Iyer", 3),
            subj("Java Programming", "java; java class", g, "Mr. Kumar", 3),
            subj("Software Engineering", "software eng; softeng", g, "Ms. Sharma", 3),
            subj("Machine Learning", "ml", g, "Dr. Nair", 3),
            subj("DBMS Lab", "dbms practical; database lab", g, "Dr. Rao", 1, "lab", 2),
            subj("Java Lab", "java practical", g, "Mr. Kumar", 1, "lab", 2),
        ]
    g = "CSE-3-A"
    subjects += [
        subj("Data Structures", "ds; dsa", g, "Prof. Gupta", 4),
        subj("Discrete Mathematics", "discrete maths; dm", g, "Ms. Das", 3),
        subj("Digital Logic", "dld; digital logic design", g, "Mr. Singh", 3),
        subj("Data Structures Lab", "ds lab; dsa lab", g, "Prof. Gupta", 1, "lab", 2),
    ]
    teachers = [
        dict(name="Dr. Rao", unavailable_days=""),
        dict(name="Prof. Mehta", unavailable_days=""),
        dict(name="Dr. Iyer", unavailable_days=""),
        dict(name="Mr. Kumar", unavailable_days=""),
        dict(name="Ms. Sharma", unavailable_days=""),
        dict(name="Dr. Nair", unavailable_days=""),
        dict(name="Prof. Gupta", unavailable_days=""),
        dict(name="Ms. Das", unavailable_days=""),
        dict(name="Mr. Singh", unavailable_days=""),
    ]
    rooms = [
        dict(name="Room 101", capacity=70, type="theory"),
        dict(name="Room 102", capacity=65, type="theory"),
        dict(name="Room 103", capacity=40, type="theory"),
        dict(name="Lab 1", capacity=65, type="lab"),
        dict(name="Lab 2", capacity=65, type="lab"),
    ]
    groups = [
        dict(name="CSE-5-A", department="CSE", semester=5, section="A", size=60),
        dict(name="CSE-5-B", department="CSE", semester=5, section="B", size=58),
        dict(name="CSE-3-A", department="CSE", semester=3, section="A", size=55),
    ]
    return DataStore(pd.DataFrame(subjects), pd.DataFrame(teachers),
                     pd.DataFrame(rooms), pd.DataFrame(groups), Config())


def advanced_sample_store() -> DataStore:
    """Phase 3 demo data: shared class, elective block, lab batches, two buildings,
    teacher limits, room preferences and a short Saturday."""
    def subj(name, aliases, group, teacher, n, typ="theory", dur=1, **kw):
        return dict(name=name, aliases=aliases, group=group, teacher=teacher, sessions_per_week=n,
                    type=typ, duration=dur, batches=kw.get("batches", 0), shared_with=kw.get("shared_with", ""),
                    elective_block=kw.get("elective_block", ""), room_pref=kw.get("room_pref", ""))
    A, B = "CSE-5-A", "CSE-5-B"
    subjects = []
    for g in (A, B):
        subjects += [
            subj("DBMS", "database management system; database", g, "Dr. Rao", 4),
            subj("Operating Systems", "os; operating system", g, "Prof. Mehta", 3),
            subj("Computer Networks", "cn; networks", g, "Dr. Iyer", 3),
            subj("Java Programming", "java; java class", g, "Mr. Kumar", 3),
            subj("DBMS Lab", "dbms practical; database lab", g, "Dr. Rao", 1, "lab", 2),
            subj("Java Lab", "java practical", g, "Mr. Kumar", 1, "lab", 2, batches=2, room_pref="Lab 2"),
        ]
    subjects += [
        # one class attended by both sections together
        subj("Machine Learning", "ml", A, "Dr. Nair", 3, shared_with=B),
        # students of both sections choose one of these; they run at the same time
        subj("Artificial Intelligence", "ai", A, "Ms. Sharma", 2, shared_with=B, elective_block="Elective 1"),
        subj("Cloud Computing", "cloud", A, "Prof. Gupta", 2, shared_with=B, elective_block="Elective 1"),
    ]
    teachers = [dict(name=n, unavailable_days="", max_per_day=0, max_consecutive=0) for n in
                ["Dr. Rao", "Prof. Mehta", "Dr. Iyer", "Mr. Kumar", "Ms. Sharma", "Dr. Nair", "Prof. Gupta"]]
    teachers[0].update(max_per_day=3, max_consecutive=2)
    teachers[4].update(unavailable_days="Saturday")
    rooms = [
        dict(name="Room 101", capacity=130, type="theory", building="Main"),
        dict(name="Room 102", capacity=70, type="theory", building="Main"),
        dict(name="Room 201", capacity=70, type="theory", building="Annex"),
        dict(name="Lab 1", capacity=65, type="lab", building="Main"),
        dict(name="Lab 2", capacity=65, type="lab", building="Main"),
        dict(name="Lab 3", capacity=65, type="lab", building="Annex"),
    ]
    groups = [dict(name=A, department="CSE", semester=5, section="A", size=60),
              dict(name=B, department="CSE", semester=5, section="B", size=58)]
    cfg = Config(days=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
                 day_slots={"Saturday": [0, 4]}, travel_gap=False)
    return DataStore(pd.DataFrame(subjects), pd.DataFrame(teachers), pd.DataFrame(rooms), pd.DataFrame(groups), cfg)
