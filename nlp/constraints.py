"""Structured scheduling constraints produced by the NLP layer (and derived from academic data).

A constraint is split into two parts so the same object can be compiled into
the CP-SAT model *and* checked independently on a finished timetable:

* **scope**  - which sessions it talks about (groups, subject, teacher, room, kind)
* **where**  - the time region it talks about (days, time window, periods)

Types
-----
forbid           sessions in scope must NOT touch the region   ("keep Friday free")
confine          sessions in scope must lie INSIDE the region  ("classes only after 10 AM")
require          at least one session in scope lies inside it  ("DBMS on Monday morning")
room_pref        sessions in scope must be in `room`           ("Java Lab in Lab 2")
max_labs_per_day at most `value` lab sessions per student group per day
max_consecutive  at most `value` occupied periods in a row (per group, or per teacher if given)
max_daily_load   at most `value` occupied periods per day (per group, or per teacher if given)
minimize_gaps    (soft) fewer idle periods between a group's classes
travel_gap       keep a free period when a group / teacher changes buildings

`origin == "data"` marks constraints derived from the academic tables
(teacher limits, room preferences) or the settings (travel buffer).
"""
from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Optional

from models import Config, Placement, is_batch_key

_ids = itertools.count(1)


def new_id() -> str:
    return f"c{next(_ids)}"


def fmt12(minutes: int) -> str:
    h, m = divmod(int(minutes), 60)
    suffix = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    return f"{h12}:{m:02d} {suffix}" if m else f"{h12} {suffix}"


def key_in_groups(key: str, groups) -> bool:
    """Is a student-group key (a section or one of its lab batches 'SEC-B1') inside the given groups?"""
    if groups is None:
        return True
    return key in groups or (is_batch_key(key) and key.rsplit("-B", 1)[0] in groups)


@dataclass
class Constraint:
    type: str
    hard: bool
    text: str = ""
    id: str = ""
    groups: Optional[list] = None      # None -> every group in the timetable
    subject: Optional[str] = None
    teacher: Optional[str] = None
    room: Optional[str] = None
    kind: Optional[str] = None         # "lab" | "theory"
    days: Optional[list] = None
    window: Optional[tuple] = None     # (start_min, end_min)
    window_label: Optional[str] = None  # e.g. "afternoon" (display only)
    start_at: Optional[int] = None     # exact start time in minutes
    periods: Optional[list] = None     # 0-based teaching-period ordinals, -1 = last
    value: int = 1                     # for max_* constraints
    weight: int = 10                   # penalty weight when soft
    origin: str = ""                   # "move", "data", ...

    def __post_init__(self):
        if not self.id:
            self.id = new_id()

    # ---- predicates -----------------------------------------------------
    def in_scope(self, p: Placement, ignore_room: bool = False) -> bool:
        if self.groups is not None and not (set(self.groups) & set(p.all_groups)):
            return False
        if self.subject and p.subject != self.subject:
            return False
        if self.teacher and p.teacher != self.teacher:
            return False
        if self.room and not ignore_room and p.room != self.room:
            return False
        if self.kind and p.kind != self.kind:
            return False
        return True

    def has_where(self) -> bool:
        return bool(self.days or self.window or self.periods or self.start_at is not None)

    def in_where(self, p: Placement, cfg: Config, overlap: bool) -> bool:
        """Does the placement fall in the time region?  `overlap=True` means
        touching the region is enough (used by forbid); otherwise it must lie inside."""
        if self.days and p.day not in self.days:
            return False
        if self.window:
            s, e = p.minutes(cfg)
            w0, w1 = self.window
            ok = (s < w1 and e > w0) if overlap else (s >= w0 and e <= w1)
            if not ok:
                return False
        if self.start_at is not None:
            if cfg.slot_start(p.start) != self.start_at:
                return False
        if self.periods:
            wanted = {cfg.resolve_period(n) for n in self.periods}
            wanted.discard(None)
            covered = set(p.slots())
            if overlap:
                if not (wanted & covered):
                    return False
            elif p.start not in wanted:
                return False
        return True

    def is_forbidden(self, p: Placement, cfg: Config) -> bool:
        """Placement-level violation test for forbid / confine / room_pref constraints."""
        if self.type == "room_pref":
            return self.in_scope(p, ignore_room=True) and p.room != self.room
        if not self.in_scope(p):
            return False
        if self.type == "forbid":
            return self.in_where(p, cfg, overlap=True)
        if self.type == "confine":
            return not self.in_where(p, cfg, overlap=False)
        return False

    def matches_require(self, p: Placement, cfg: Config) -> bool:
        return self.in_scope(p) and self.in_where(p, cfg, overlap=False)

    # ---- display --------------------------------------------------------
    def _where_text(self) -> str:
        parts = []
        if self.days:
            parts.append("on " + "/".join(self.days))
        if self.window_label:
            parts.append(self.window_label)
        elif self.window:
            w0, w1 = self.window
            if w0 <= 0:
                parts.append(f"before {fmt12(w1)}")
            elif w1 >= 24 * 60:
                parts.append(f"after {fmt12(w0)}")
            else:
                parts.append(f"between {fmt12(w0)} and {fmt12(w1)}")
        if self.start_at is not None:
            parts.append(f"starting at {fmt12(self.start_at)}")
        if self.periods:
            names = {0: "first", 1: "second", 2: "third", 3: "fourth", 4: "fifth", -1: "last"}
            parts.append("in the " + "/".join(names.get(n, f"#{n + 1}") for n in self.periods) + " period")
        return " ".join(parts)

    def _target_text(self) -> str:
        bits = []
        if self.subject:
            bits.append(self.subject)
        if self.teacher:
            bits.append(f"{self.teacher}'s classes")
        if self.kind == "lab" and not self.subject:
            bits.append("lab sessions")
        if self.kind == "theory" and not self.subject:
            bits.append("theory classes")
        if self.room:
            bits.append(f"classes in {self.room}")
        return " / ".join(bits) if bits else "classes"

    def describe(self) -> str:
        level = "HARD" if self.hard else "SOFT"
        scope = f" [{', '.join(self.groups)}]" if self.groups else ""
        who = f"{self.teacher}: " if self.teacher else ""
        if self.type == "max_labs_per_day":
            body = f"At most {self.value} lab session(s) per day"
        elif self.type == "max_consecutive":
            body = f"{who}at most {self.value} period(s) in a row"
        elif self.type == "max_daily_load":
            body = f"{who}at most {self.value} period(s) per day"
        elif self.type == "minimize_gaps":
            body = "Keep a group's classes together (few idle periods between them)"
        elif self.type == "travel_gap":
            body = "Leave a free period when changing buildings"
        elif self.type == "room_pref":
            body = f"{self.subject or 'classes'} in {self.room}"
        elif self.type == "forbid":
            body = f"No {self._target_text()} {self._where_text()}".strip()
        elif self.type == "confine":
            body = f"{self._target_text().capitalize()} only {self._where_text()}".strip()
        else:
            body = f"{self._target_text()} {self._where_text()}".strip()
            body = "At least one " + body if self.subject or self.teacher else body
        tag = " (from academic data / settings)" if self.origin == "data" else ""
        return f"{level}: {body}{scope}{tag}"

    def to_dict(self):
        d = asdict(self)
        d["description"] = self.describe()
        return d


# --------------------------------------------------------------------------
# Independent verification on a finished timetable
# --------------------------------------------------------------------------
def _occupancy(placements, cfg, c: Constraint, by_teacher: bool):
    """{(key, day): set(slots)} for the groups (or the teacher) a constraint is about."""
    busy = defaultdict(set)
    for p in placements:
        if by_teacher:
            if p.teacher == c.teacher:
                busy[(p.teacher, p.day)].update(p.slots())
        else:
            for k in p.keys:
                if key_in_groups(k, c.groups):
                    busy[(k, p.day)].update(p.slots())
    return busy


def _longest_run(slots) -> int:
    run = best = 0
    prev = None
    for sl in sorted(slots):
        run = run + 1 if prev is not None and sl == prev + 1 else 1
        best, prev = max(best, run), sl
    return best


def count_gaps(placements, cfg: Config) -> list:
    """[(key, day, idle_periods)] for every group/day that has idle teaching periods between classes."""
    busy = defaultdict(set)
    for p in placements:
        for k in p.keys:
            busy[(k, p.day)].update(p.slots())
    teaching = cfg.teaching_slots()
    out = []
    for (k, day), slots in busy.items():
        lo, hi = min(slots), max(slots)
        idle = [s for s in teaching if lo < s < hi and s not in slots and cfg.slot_open(day, s)]
        if idle:
            out.append((k, day, len(idle)))
    return out


def check_constraint(c: Constraint, placements: list, cfg: Config):
    """Returns (satisfied: bool, detail: str)."""
    if c.type in ("forbid", "confine", "room_pref"):
        bad = [p for p in placements if c.is_forbidden(p, cfg)]
        if bad:
            p = bad[0]
            return False, f"{len(bad)} session(s) violate it, e.g. {p.subject} ({p.label or p.group}) on {p.day} {cfg.slot_label(p.start)} in {p.room}"
        return True, ""
    if c.type == "require":
        if any(c.matches_require(p, cfg) for p in placements):
            return True, ""
        return False, "no session satisfies it"
    if c.type == "max_labs_per_day":
        counts = defaultdict(int)
        for p in placements:
            if p.kind == "lab":
                for k in p.keys:
                    if key_in_groups(k, c.groups):
                        counts[(k, p.day)] += 1
        over = [(k, v) for k, v in counts.items() if v > c.value]
        if over:
            (g, d), v = over[0]
            return False, f"{g} has {v} lab sessions on {d}"
        return True, ""
    if c.type in ("max_consecutive", "max_daily_load"):
        busy = _occupancy(placements, cfg, c, by_teacher=bool(c.teacher))
        for (key, day), slots in busy.items():
            val = _longest_run(slots) if c.type == "max_consecutive" else len(slots)
            if val > c.value:
                what = "periods in a row" if c.type == "max_consecutive" else "periods"
                return False, f"{key} has {val} {what} on {day}"
        return True, ""
    if c.type == "minimize_gaps":
        gaps = count_gaps(placements, cfg)
        total = sum(g[2] for g in gaps)
        return (total == 0), ("" if total == 0 else f"{total} idle period(s) between classes, e.g. {gaps[0][0]} on {gaps[0][1]}")
    if c.type == "travel_gap":
        occ = defaultdict(set)                      # (owner, day, slot) -> buildings
        for p in placements:
            for sl in p.slots():
                for owner in list(p.keys) + [p.teacher]:
                    occ[(owner, p.day, sl)].add(p.building)
        for (owner, day, sl), b1 in occ.items():
            b1 = {b for b in b1 if b}
            b2 = {b for b in occ.get((owner, day, sl + 1), set()) if b}
            if b1 and b2 and len(b1 | b2) > 1 and not (len(b1) == 1 and b1 == b2):
                return False, f"{owner} changes building between periods {sl + 1} and {sl + 2} on {day}"
        return True, ""
    return True, ""
