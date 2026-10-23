"""Manual edits: move a session by hand, with a live list of everything that would break.

`find_conflicts` checks a complete timetable and returns the problems as data, so the UI can
highlight the offending cells in red *before* the edit is applied.
"""
from __future__ import annotations

import copy
from collections import defaultdict

from models import Config, DataStore
from nlp.constraints import check_constraint
from solver.scheduler import build_sessions, derived_constraints


def find_conflicts(data: DataStore, cfg: Config, placements: list, constraints: list = (), groups=None,
                   num_subjects=None) -> tuple:
    """Returns (messages, bad) where bad is the set of indexes into `placements` involved in a problem."""
    msgs, bad = [], set()
    busy = defaultdict(list)
    for i, p in enumerate(placements):
        ident = (p.elective or p.subject, p.group, p.day, p.start)
        for sl in p.slots():
            if sl >= cfg.num_slots or sl < 0:
                msgs.append(f"{p.subject} ({p.label or p.group}) runs outside the grid on {p.day}")
                bad.add(i)
                continue
            if sl in cfg.lunch_slots:
                msgs.append(f"{p.subject} ({p.label or p.group}) overlaps lunch on {p.day}")
                bad.add(i)
            elif not cfg.slot_open(p.day, sl):
                msgs.append(f"{p.subject} ({p.label or p.group}) is placed after closing time on {p.day}")
                bad.add(i)
            for k in p.keys:
                busy[("group", k, p.day, sl)].append((i, ident))
            busy[("teacher", p.teacher, p.day, sl)].append((i, (i,)))
            busy[("room", p.room, p.day, sl)].append((i, (i,)))
        if p.day in data.teacher_unavailable(p.teacher):
            msgs.append(f"{p.teacher} is unavailable on {p.day} ({p.subject})")
            bad.add(i)
        row = data.rooms[data.rooms["name"] == p.room]
        if len(row):
            r = row.iloc[0]
            size = p.size or data.group_size(p.group)
            if r["type"] != ("lab" if p.kind == "lab" else "theory"):
                msgs.append(f"{p.subject}: {p.room} is a {r['type']} room but the session is a {p.kind}")
                bad.add(i)
            elif r["capacity"] and size and r["capacity"] < size:
                msgs.append(f"{p.room} (capacity {r['capacity']}) is too small for {p.label or p.group} ({size})")
                bad.add(i)
    seen = set()
    for (what, name, day, sl), items in busy.items():
        if len({ident for _, ident in items}) > 1:
            idx = sorted({i for i, _ in items})
            bad.update(idx)
            text = f"{what} clash: {name} on {day} at {cfg.slot_label(sl)} " \
                   f"({' vs '.join(sorted({placements[i].subject for i in idx}))})"
            if text not in seen:
                seen.add(text)
                msgs.append(text)
    if groups:
        sessions = build_sessions(data, groups, num_subjects)
        hard = [c for c in list(constraints) + derived_constraints(data, cfg, sessions) if c.hard]
    else:
        hard = [c for c in constraints if c.hard]
    for c in hard:
        ok, detail = check_constraint(c, list(placements), cfg)
        if not ok:
            msgs.append(f"Breaks rule: {c.describe()} - {detail}")
            for i, p in enumerate(placements):
                if c.type in ("forbid", "confine", "room_pref") and c.is_forbidden(p, cfg):
                    bad.add(i)
    return msgs, bad


def move_session(placements: list, index: int, day: str, start: int, room: str | None = None) -> list:
    """Return a NEW placement list with one session moved (all parts of an elective block move together)."""
    src = placements[index]
    new = []
    for i, p in enumerate(placements):
        together = src.elective and p.elective == src.elective and (p.day, p.start) == (src.day, src.start) \
            and p.all_groups == src.all_groups
        if i == index or together:
            q = copy.copy(p)
            q.day, q.start = day, start
            if room and i == index:
                q.room = room
                q.building = ""
            new.append(q)
        else:
            new.append(p)
    return new
