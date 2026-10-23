"""Constraint-programming timetable solver (Google OR-Tools CP-SAT).

Model
-----
One Boolean variable per feasible (session, day, start period, rooms).
Every session is placed exactly once; teachers, rooms and student groups can
never be double-booked; labs occupy consecutive periods that never cross lunch
or the closing time of a short day.

Session kinds
-------------
* ordinary session      one group, one teacher, one room
* shared session        several groups attend ONE class together (``shared_with``)
* lab batches           ``batches`` = N splits a section into sub-groups ``SEC-B1..BN``;
                        each batch gets its own lab sessions, while the whole section's
                        theory classes block every batch
* elective block        several subjects with the same ``elective_block`` run at the same
                        time (different teachers and rooms); students pick one

Hard NL / data constraints are guarded by *assumption literals*.  If the model is
infeasible, CP-SAT returns an unsat core - the subset of constraints that cannot hold
together - which we shrink and turn into a human explanation.  Soft constraints and
built-in quality rules become weighted penalties; hard rules are never traded away.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product

from ortools.sat.python import cp_model

from models import Config, DataStore, Part, Placement, Session
from nlp.constraints import Constraint, check_constraint, key_in_groups

# soft weights for built-in quality rules
W_SAME_SUBJECT_SAME_DAY = 4
W_BALANCE = 2
W_LATE_SLOT = 1
W_STABILITY = 3
W_GAP = 1
W_ROOM_PREF = 5


@dataclass
class SolveResult:
    status: str                      # OPTIMAL | FEASIBLE | INFEASIBLE | UNKNOWN | INVALID
    placements: list = field(default_factory=list)
    objective: float | None = None
    soft_report: list = field(default_factory=list)   # [{constraint, satisfied, detail}]
    conflict: list = field(default_factory=list)      # constraints in the unsat core
    issues: list = field(default_factory=list)        # structural problems found before/after solving
    suggestions: list = field(default_factory=list)
    seconds: float = 0.0
    sessions: list = field(default_factory=list)
    derived: list = field(default_factory=list)       # constraints derived from tables / settings

    @property
    def ok(self) -> bool:
        return self.status in ("OPTIMAL", "FEASIBLE")


# --------------------------------------------------------------------------
# Building the list of sessions
# --------------------------------------------------------------------------
def _split(text) -> list:
    return [x.strip() for x in str(text or "").replace(";", ",").split(",") if x.strip()]


def limit_subjects(rows, n: int):
    """Keep the first n theory subjects plus the labs that belong to them
    (a lab belongs to a theory subject when its name starts with that subject's name or first word)."""
    theory = rows[rows["type"] == "theory"].head(n)
    keys = {name.lower() for name in theory["name"]} | {name.split()[0].lower() for name in theory["name"]}
    all_theory = {name.lower() for name in rows[rows["type"] == "theory"]["name"]}
    all_first = {name.split()[0].lower() for name in rows[rows["type"] == "theory"]["name"]}

    def keep_lab(name):
        stem = name.lower().replace("lab", "").strip()
        first = stem.split()[0] if stem else ""
        if stem in keys or first in keys:
            return True
        return stem not in all_theory and first not in all_first   # orphan lab: keep
    labs = rows[(rows["type"] == "lab") & rows["name"].map(keep_lab)]
    return rows[rows.index.isin(theory.index) | rows.index.isin(labs.index)]


def build_sessions(data: DataStore, groups: list, num_subjects: int | None = None) -> list:
    active = set(groups)
    rows_by_group = {}
    for g in groups:
        rows = data.subjects[data.subjects["group"] == g]
        if num_subjects:
            rows = limit_subjects(rows, int(num_subjects))
        rows_by_group[g] = rows

    consumed = set()            # (group, subject) rows that are served by another group's shared session
    batch_keys = defaultdict(set)
    for g, rows in rows_by_group.items():
        for _, r in rows.iterrows():
            if int(r["batches"]) >= 2 and not r["elective_block"]:
                batch_keys[g].update(f"{g}-B{i}" for i in range(1, int(r["batches"]) + 1))
            elif r["shared_with"]:
                for o in _split(r["shared_with"]):
                    if o in active:
                        consumed.add((o, r["name"]))

    def occupy(members):
        keys = []
        for m in members:
            keys.append(m)
            keys += sorted(batch_keys[m])
        return tuple(keys)

    sessions, sid = [], 0
    for g in groups:
        rows = rows_by_group[g]
        size_g = data.group_size(g)
        # --- elective blocks -------------------------------------------------
        blocks = [b for b in dict.fromkeys(rows["elective_block"]) if b]
        for block in blocks:
            brows = [r for _, r in rows[rows["elective_block"] == block].iterrows() if (g, r["name"]) not in consumed]
            if not brows:
                continue
            first = brows[0]
            members = [g] + [o for o in _split(first["shared_with"]) if o in active]
            parts = [Part(r["name"], r["teacher"], r["type"], r["room_pref"]) for r in brows]
            for k in range(int(first["sessions_per_week"])):
                sessions.append(Session(sid, g, block, brows[0]["teacher"], max(1, int(first["duration"])),
                                        parts[0].kind, k, groups=tuple(members), occupies=occupy(members),
                                        size=math.ceil(sum(data.group_size(m) for m in members) / len(parts)), parts=parts,
                                        label="+".join(members), elective=block))
                sid += 1
        # --- ordinary / shared / batched ------------------------------------------
        for _, r in rows.iterrows():
            if r["elective_block"] or (g, r["name"]) in consumed:
                continue
            dur, n = max(1, int(r["duration"])), int(r["sessions_per_week"])
            nb = int(r["batches"])
            if nb >= 2:
                for i in range(1, nb + 1):
                    batch = f"{g}-B{i}"
                    for k in range(n):
                        sessions.append(Session(sid, batch, r["name"], r["teacher"], dur, r["type"], k,
                                                groups=(batch, g), occupies=(batch,),
                                                size=math.ceil(size_g / nb), room_pref=r["room_pref"], label=batch))
                        sid += 1
                continue
            members = [g] + [o for o in _split(r["shared_with"]) if o in active]
            for k in range(n):
                sessions.append(Session(sid, g, r["name"], r["teacher"], dur, r["type"], k, groups=tuple(members),
                                        occupies=occupy(members), size=sum(data.group_size(m) for m in members),
                                        room_pref=r["room_pref"], label="+".join(members)))
                sid += 1
    return sessions


def derived_constraints(data: DataStore, cfg: Config, sessions: list) -> list:
    """Constraints that come from the tables and settings rather than from a sentence."""
    out = []
    teachers = {t for s in sessions for t in s.teachers()}
    for _, t in data.teachers.iterrows():
        if t["name"] not in teachers:
            continue
        if int(t["max_per_day"]) > 0:
            out.append(Constraint("max_daily_load", True, text="teacher data", teacher=t["name"],
                                  value=int(t["max_per_day"]), origin="data"))
        if int(t["max_consecutive"]) > 0:
            out.append(Constraint("max_consecutive", True, text="teacher data", teacher=t["name"],
                                  value=int(t["max_consecutive"]), origin="data"))
    active = {g for s in sessions for g in s.all_groups}
    subjects = {n for s in sessions for n in s.subjects()}
    rooms = set(data.rooms["name"])
    for _, r in data.subjects.iterrows():
        if r["room_pref"] and r["room_pref"] in rooms and r["name"] in subjects and r["group"] in active:
            out.append(Constraint("room_pref", False, text="room preference", subject=r["name"], room=r["room_pref"],
                                  groups=[r["group"]], weight=W_ROOM_PREF, origin="data"))
    buildings = {b for b in data.rooms["building"] if b}
    if cfg.travel_gap and len(buildings) > 1:
        out.append(Constraint("travel_gap", True, text="travel buffer setting", origin="data"))
    return out


# --------------------------------------------------------------------------
# Domains and structural diagnostics
# --------------------------------------------------------------------------
def _rooms_for(data: DataStore, kind: str, size: int) -> list:
    rooms = data.rooms[data.rooms["type"] == ("lab" if kind == "lab" else "theory")]
    return [r["name"] for _, r in rooms.iterrows() if r["capacity"] <= 0 or size <= 0 or r["capacity"] >= size]


def _part_kinds(s: Session) -> list:
    return [p.kind for p in s.parts] if s.parts else [s.kind]


def _room_combos(data: DataStore, s: Session) -> list:
    lists = [_rooms_for(data, k, s.size) for k in _part_kinds(s)]
    if any(not l for l in lists):
        return []
    combos = [c for c in product(*lists) if len(set(c)) == len(c)]
    return combos


def _domain(data: DataStore, cfg: Config, s: Session) -> list:
    """All feasible (day, start, rooms) triples before user constraints."""
    teachers = s.teachers()
    if len(set(teachers)) != len(teachers):
        return []                                    # two parts of an elective share a teacher
    unavailable = set().union(*(data.teacher_unavailable(t) for t in teachers))
    combos = _room_combos(data, s)
    out = []
    for day in cfg.days:
        if day in unavailable:
            continue
        for start in range(0, cfg.num_slots - s.duration + 1):
            slots = range(start, start + s.duration)
            if any(sl in cfg.lunch_slots or not cfg.slot_open(day, sl) for sl in slots):
                continue
            for rooms in combos:
                out.append((day, start, rooms))
    return out


def precheck(data: DataStore, cfg: Config, sessions: list) -> list:
    """Cheap structural diagnostics that explain impossible *base* data."""
    issues = []
    cap = sum(len(cfg.open_teaching_slots(d)) for d in cfg.days)
    by_key, by_teacher, by_kind = defaultdict(int), defaultdict(int), defaultdict(int)
    for s in sessions:
        for k in (s.occupies or (s.group,)):
            by_key[k] += s.duration
        for t in s.teachers():
            by_teacher[t] += s.duration
        for kind in _part_kinds(s):
            by_kind[kind] += s.duration
        label = s.elective or s.subject
        if s.parts and len(set(s.teachers())) != len(s.teachers()):
            issues.append(f"Elective block {s.elective} ({s.group}) has two subjects with the same teacher; "
                          f"they run at the same time, so the teachers must differ.")
            continue
        for kind in set(_part_kinds(s)):
            if not _rooms_for(data, kind, s.size):
                issues.append(f"No {kind} room is large enough for {label} ({s.label or s.group}, {s.size} students). "
                              f"Add a bigger {kind} room or reduce the group size.")
        if s.parts and len(_room_combos(data, s)) == 0 and all(_rooms_for(data, k, s.size) for k in _part_kinds(s)):
            issues.append(f"Elective block {s.elective} needs {len(s.parts)} different rooms at once but there are not enough.")
        elif not _domain(data, cfg, s) and not any("No " in i and label in i for i in issues):
            issues.append(f"{label} ({s.label or s.group}) has no valid slot: a teacher is unavailable, or the "
                          f"{s.duration}-period block cannot fit around lunch / the open hours.")
    seen = set()
    for k, n in by_key.items():
        if n > cap:
            issues.append(f"{k} needs {n} periods per week but only {cap} teaching periods are open.")
    for t, n in by_teacher.items():
        avail = sum(len(cfg.open_teaching_slots(d)) for d in cfg.days if d not in data.teacher_unavailable(t))
        if n > avail:
            issues.append(f"{t} is assigned {n} periods per week but is only available for {avail}.")
    for kind in ("lab", "theory"):
        nrooms = len(data.rooms[data.rooms["type"] == kind])
        if by_kind[kind] > nrooms * cap:
            issues.append(f"{by_kind[kind]} {kind} periods are needed per week but {nrooms} {kind} room(s) "
                          f"provide only {nrooms * cap}.")
    return [i for i in issues if not (i in seen or seen.add(i))]


def scope_warnings(sessions: list, constraints: list) -> list:
    """Constraints that talk about things that are not in the current timetable."""
    subjects = {n for s in sessions for n in s.subjects()}
    teachers = {t for s in sessions for t in s.teachers()}
    groups = {g for s in sessions for g in s.all_groups}
    out = []
    for c in constraints:
        if c.origin == "data":
            continue
        if c.subject and c.subject not in subjects:
            out.append((c, f"'{c.subject}' is not part of the current timetable"))
        elif c.teacher and c.teacher not in teachers:
            out.append((c, f"{c.teacher} has no classes in the current timetable"))
        elif c.groups is not None and not (set(c.groups) & groups):
            out.append((c, f"the group(s) {', '.join(c.groups)} are not in the current timetable"))
    return out


# --------------------------------------------------------------------------
# The CP-SAT model
# --------------------------------------------------------------------------
def _solve_once(data, cfg, sessions, constraints, previous, time_limit, seed, workers):
    model = cp_model.CpModel()
    recs = []                                   # (var, session, [Placement per part])
    by_session = defaultdict(list)
    cover = defaultdict(list)
    dayvars = defaultdict(list)                 # (sid, day) -> vars
    travel = any(c.type == "travel_gap" for c in constraints)

    for s in sessions:
        for day, start, rooms in _domain(data, cfg, s):
            v = model.new_bool_var(f"x{s.sid}_{day}_{start}")
            pls = []
            for i, room in enumerate(rooms):
                part = s.parts[i] if s.parts else None
                pls.append(Placement(
                    group=s.group, subject=part.subject if part else s.subject,
                    teacher=part.teacher if part else s.teacher, room=room, day=day, start=start,
                    duration=s.duration, kind=part.kind if part else s.kind, groups=s.all_groups,
                    occupies=s.occupies or (s.group,), size=s.size, building=data.room_building(room),
                    label=s.label or s.group, elective=s.elective))
            recs.append((v, s, pls))
            by_session[s.sid].append((v, pls))
            dayvars[(s.sid, day)].append(v)
            owners = set(s.occupies or (s.group,))
            teachers = {p.teacher for p in pls}
            room_names = {p.room for p in pls}
            for sl in range(start, start + s.duration):
                for k in owners:
                    cover[("g", k, day, sl)].append(v)
                for t in teachers:
                    cover[("t", t, day, sl)].append(v)
                for r in room_names:
                    cover[("r", r, day, sl)].append(v)
                if travel:
                    for b in {p.building for p in pls if p.building}:
                        for k in owners:
                            cover[("gb", k, b, day, sl)].append(v)
                    for p in pls:
                        if p.building:
                            cover[("tb", p.teacher, p.building, day, sl)].append(v)

    for s in sessions:
        model.add_exactly_one([v for v, _ in by_session[s.sid]])
    for lst in cover.values():
        if len(lst) > 1:
            model.add_at_most_one(lst)

    # symmetry breaking: identical weekly sessions of one subject are ordered in time
    group_subject = defaultdict(list)
    for s in sessions:
        group_subject[(s.group, s.subject)].append(s)
    day_idx = {d: i for i, d in enumerate(cfg.days)}
    for sess_list in group_subject.values():
        pos = [sum(v * (day_idx[pls[0].day] * 100 + pls[0].start) for v, pls in by_session[s.sid]) for s in sess_list]
        for a, b in zip(pos, pos[1:]):
            model.add(a + 1 <= b)

    penalties = []                              # (weight, expression)
    assumption = {}                             # literal index -> (literal, constraint)
    all_keys = sorted({k for s in sessions for k in (s.occupies or (s.group,))})

    def guard(c):
        lit = model.new_bool_var(f"a_{c.id}")
        assumption[lit.index] = (lit, c)
        return lit

    def owners_for(c):
        if c.teacher:
            return [("t", c.teacher)]
        return [("g", k) for k in all_keys if key_in_groups(k, c.groups)]

    # ---- constraints -------------------------------------------------------------
    gap_weight = W_GAP
    for c in constraints:
        if c.type in ("forbid", "confine", "room_pref"):
            bad = [v for v, _, pls in recs if any(c.is_forbidden(p, cfg) for p in pls)]
            if c.hard:
                lit = guard(c)
                for v in bad:
                    model.add_implication(lit, v.negated())
            else:
                for v in bad:
                    penalties.append((c.weight, v))
        elif c.type == "require":
            good = [v for v, _, pls in recs if any(c.matches_require(p, cfg) for p in pls)]
            if c.hard:
                lit = guard(c)
                model.add(sum(good) >= 1).only_enforce_if(lit)
            else:
                sat = model.new_bool_var(f"sat_{c.id}")
                if good:
                    model.add(sum(good) >= 1).only_enforce_if(sat)
                else:
                    model.add(sat == 0)
                penalties.append((c.weight, sat.negated()))
        elif c.type == "max_labs_per_day":
            lit = guard(c) if c.hard else None
            for key in all_keys:
                if not key_in_groups(key, c.groups):
                    continue
                labs = [s for s in sessions if s.kind == "lab" and key in (s.occupies or (s.group,))]
                if len(labs) <= c.value:
                    continue
                for day in cfg.days:
                    expr = sum(sum(dayvars[(s.sid, day)]) for s in labs)
                    if c.hard:
                        model.add(expr <= c.value).only_enforce_if(lit)
                    else:
                        excess = model.new_int_var(0, len(labs), f"exc_{c.id}_{key}_{day}")
                        model.add(excess >= expr - c.value)
                        penalties.append((c.weight, excess))
        elif c.type in ("max_consecutive", "max_daily_load"):
            lit = guard(c) if c.hard else None
            for kind, name in owners_for(c):
                for day in cfg.days:
                    if c.type == "max_daily_load":
                        windows = [list(range(cfg.num_slots))]
                        limit = c.value
                    else:
                        windows = [list(range(st, st + c.value + 1)) for st in range(0, cfg.num_slots - c.value)
                                   if not any(sl in cfg.lunch_slots for sl in range(st, st + c.value + 1))]
                        limit = c.value
                    for win in windows:
                        terms = [cover.get((kind, name, day, sl), []) for sl in win]
                        if not any(terms):
                            continue
                        expr = sum(sum(t) for t in terms)
                        if c.hard:
                            model.add(expr <= limit).only_enforce_if(lit)
                        else:
                            exc = model.new_int_var(0, cfg.num_slots, f"ex_{c.id}_{name}_{day}_{win[0]}")
                            model.add(exc >= expr - limit)
                            penalties.append((c.weight, exc))
        elif c.type == "travel_gap":
            lit = guard(c)
            buildings = sorted({b for b in data.rooms["building"] if b})
            owners = [("gb", k) for k in all_keys] + [("tb", t) for t in sorted({t for s in sessions for t in s.teachers()})]
            for kind, name in owners:
                for day in cfg.days:
                    for sl in range(cfg.num_slots - 1):
                        for b1, b2 in product(buildings, buildings):
                            if b1 == b2:
                                continue
                            a, b = cover.get((kind, name, b1, day, sl), []), cover.get((kind, name, b2, day, sl + 1), [])
                            if a and b:
                                model.add(sum(a) + sum(b) <= 1).only_enforce_if(lit)
        elif c.type == "minimize_gaps":
            gap_weight += c.weight

    # ---- idle periods between a group's classes (soft) -----------------------------
    if gap_weight > 0:
        teaching = [sl for sl in range(cfg.num_slots) if sl not in cfg.lunch_slots]
        for key in all_keys:
            for day in cfg.days:
                seq = [sl for sl in teaching if cfg.slot_open(day, sl)]
                if len(seq) < 3 or not any(cover.get(("g", key, day, sl)) for sl in seq):
                    continue
                occ = []
                for sl in seq:
                    o = model.new_bool_var(f"o_{key}_{day}_{sl}")
                    lst = cover.get(("g", key, day, sl), [])
                    model.add(o == (sum(lst) if lst else 0))
                    occ.append(o)
                seen = [model.new_bool_var(f"s_{key}_{day}_{i}") for i in range(len(seq))]
                later = [model.new_bool_var(f"l_{key}_{day}_{i}") for i in range(len(seq))]
                for i in range(len(seq)):
                    model.add(seen[i] >= occ[i])
                    model.add(later[i] >= occ[i])
                    if i > 0:
                        model.add(seen[i] >= seen[i - 1])
                        model.add(seen[i] <= occ[i] + seen[i - 1])
                    else:
                        model.add(seen[i] <= occ[i])
                    j = len(seq) - 1 - i
                    if j < len(seq) - 1:
                        model.add(later[j] >= later[j + 1])
                        model.add(later[j] <= occ[j] + later[j + 1])
                    else:
                        model.add(later[j] <= occ[j])
                for i in range(1, len(seq) - 1):
                    gap = model.new_bool_var(f"gap_{key}_{day}_{i}")
                    model.add(gap >= seen[i - 1] + later[i + 1] - occ[i] - 1)
                    penalties.append((gap_weight, gap))

    # ---- other built-in soft quality rules ----------------------------------------------
    for (g, subj), sess_list in group_subject.items():
        if len(sess_list) < 2:
            continue
        for day in cfg.days:
            cnt = sum(sum(dayvars[(s.sid, day)]) for s in sess_list)
            exc = model.new_int_var(0, len(sess_list), f"same_{g}_{subj}_{day}")
            model.add(exc >= cnt - 1)
            penalties.append((W_SAME_SUBJECT_SAME_DAY, exc))
    real_groups = [g for g in data.group_names() if any(g in (s.occupies or (s.group,)) for s in sessions)]
    for g in real_groups:
        maxload = model.new_int_var(0, cfg.num_slots, f"maxload_{g}")
        for day in cfg.days:
            load = sum(s.duration * v for v, s, pls in recs if g in (s.occupies or (s.group,)) and pls[0].day == day)
            model.add(load <= maxload)
        penalties.append((W_BALANCE, maxload))
    last = cfg.teaching_slots()[-1] if cfg.teaching_slots() else 0
    for v, _, pls in recs:
        if pls[0].end - 1 >= last:
            penalties.append((W_LATE_SLOT, v))

    # ---- stability: keep unrelated sessions where they were -------------------------------
    if previous:
        prev = defaultdict(set)
        for p in previous:
            prev[(p.group, p.subject)].add((p.day, p.start))
        for v, s, pls in recs:
            if any((p.group, p.subject) in prev and (p.day, p.start) not in prev[(p.group, p.subject)] for p in pls):
                penalties.append((W_STABILITY, v))

    model.minimize(sum(w * e for w, e in penalties) if penalties else 0)
    lits = [lit for k, (lit, _) in assumption.items()]
    if lits:
        model.add_assumptions(lits)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    solver.parameters.random_seed = seed
    status = solver.solve(model)
    name = solver.status_name(status)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        placed = [p for v, _, pls in recs if solver.boolean_value(v) for p in pls]
        return name, placed, solver.objective_value, []
    core = []
    if status == cp_model.INFEASIBLE and lits:
        idx = set(solver.sufficient_assumptions_for_infeasibility())
        core = [c for lit, c in assumption.values() if lit.index in idx]
    return name, [], None, core


def shrink_core(data, cfg, sessions, core, budget_s=12.0):
    """Deletion-based minimisation: drop every constraint that is not needed to stay infeasible."""
    t0 = time.time()
    core = list(core)
    if len(core) > 12:
        return core
    for c in list(core):
        if time.time() - t0 > budget_s or len(core) <= 1:
            break
        trial = [x for x in core if x is not c]
        st, _, _, _ = _solve_once(data, cfg, sessions, trial, None, 3, 0, 8)
        if st == "INFEASIBLE":
            core = trial
    return core


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------
def validate(data: DataStore, cfg: Config, sessions: list, placements: list, constraints: list) -> list:
    """Independent re-check of every hard rule.  Returns a list of violations (empty = valid)."""
    errs = []
    need = defaultdict(int)
    for s in sessions:
        for subj in s.subjects():
            need[(s.group, subj)] += 1
    have = defaultdict(int)
    for p in placements:
        have[(p.group, p.subject)] += 1
    for k, n in need.items():
        if have[k] != n:
            errs.append(f"{k[1]} ({k[0]}): expected {n} sessions, found {have[k]}")
    busy = defaultdict(set)                       # resource slot -> identities using it
    for idx, p in enumerate(placements):
        ident = (p.elective or p.subject, p.group, p.day, p.start)
        for sl in p.slots():
            if sl in cfg.lunch_slots:
                errs.append(f"{p.subject} ({p.label}) overlaps lunch on {p.day}")
            if not cfg.slot_open(p.day, sl):
                errs.append(f"{p.subject} ({p.label}) is placed after closing time on {p.day}")
            for k in p.keys:
                busy[("g", k, p.day, sl)].add(ident)
            busy[("t", p.teacher, p.day, sl)].add((idx,))
            busy[("r", p.room, p.day, sl)].add((idx,))
        if p.day in data.teacher_unavailable(p.teacher):
            errs.append(f"{p.teacher} is unavailable on {p.day} but teaches {p.subject}")
        room = data.rooms[data.rooms["name"] == p.room]
        if len(room):
            r = room.iloc[0]
            size = p.size or data.group_size(p.group)
            if r["capacity"] and size and r["capacity"] < size:
                errs.append(f"{p.room} (capacity {r['capacity']}) too small for {p.label} ({size} students)")
            if r["type"] != ("lab" if p.kind == "lab" else "theory"):
                errs.append(f"{p.subject} placed in wrong room type {p.room}")
    for key, idents in busy.items():
        if len(idents) > 1:
            who = {"g": "group", "t": "teacher", "r": "room"}[key[0]]
            errs.append(f"{who} clash: {key[1]} on {key[2]} slot {key[3]}")
    for c in constraints:
        if c.hard:
            ok, detail = check_constraint(c, placements, cfg)
            if not ok:
                errs.append(f"Hard constraint violated: {c.describe()} - {detail}")
    return errs


def soft_report(constraints: list, placements: list, cfg: Config) -> list:
    out = []
    for c in constraints:
        ok, detail = check_constraint(c, placements, cfg)
        out.append({"constraint": c, "satisfied": ok, "detail": detail})
    return out


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def solve(data: DataStore, cfg: Config, groups: list, constraints: list, num_subjects: int | None = None,
          previous: list | None = None, time_limit: float = 10.0, seed: int = 0, explain: bool = True) -> SolveResult:
    t0 = time.time()
    sessions = build_sessions(data, groups, num_subjects)
    res = SolveResult("UNKNOWN", sessions=sessions)
    if not sessions:
        res.status = "INVALID"
        res.issues.append("There are no subjects for the selected group(s). Add subjects in 'Manage Academic Data'.")
        return res

    res.derived = derived_constraints(data, cfg, sessions)
    # a room preference typed by the user overrides the one in the subject table
    user_rooms = {(c.subject, g) for c in constraints if c.type == "room_pref" for g in (c.groups or data.group_names())}
    res.derived = [d for d in res.derived if not (d.type == "room_pref" and any(
        (d.subject, g) in user_rooms for g in d.groups))]
    cons = list(constraints) + res.derived
    issues = precheck(data, cfg, sessions)
    blockers = [i for i in issues if "no valid slot" in i or i.startswith("No ") or "Elective block" in i]
    if blockers:
        res.status, res.issues = "INFEASIBLE", issues
        res.suggestions.append("Fix the data problems listed above, then try again.")
        res.seconds = time.time() - t0
        return res

    status, placed, obj, core = _solve_once(data, cfg, sessions, cons, previous, time_limit, seed, 8)
    if status == "INFEASIBLE" and any(c.hard for c in cons):
        # re-run single-threaded to obtain a reliable unsat core
        status, placed, obj, core = _solve_once(data, cfg, sessions, cons, previous, time_limit, seed, 1)
    res.status, res.placements, res.objective = status, placed, obj

    if res.ok:
        errs = validate(data, cfg, sessions, placed, cons)
        if errs:                      # safety net: never hand back an invalid timetable
            res.status, res.placements, res.issues = "INVALID", [], errs
        else:
            res.soft_report = soft_report([c for c in cons if not c.hard], placed, cfg)
    elif status == "INFEASIBLE":
        res.conflict = shrink_core(data, cfg, sessions, core) if core else core
        res.issues = issues
        if explain:
            _suggest(res, data, cfg, sessions, cons, previous, seed)
    else:
        res.issues = issues + ["The solver ran out of time before finding a timetable. "
                               "Try fewer constraints or increase the time limit."]
    res.seconds = time.time() - t0
    return res


def _suggest(res, data, cfg, sessions, constraints, previous, seed):
    hard = [c for c in constraints if c.hard]
    culprits = res.conflict or []
    if not culprits and not hard:
        res.suggestions.append("The academic data itself leaves no valid timetable (see the issues above). "
                               "Add rooms, reduce weekly sessions, or add teaching periods.")
        return
    if not culprits:
        culprits = hard
    fixes = []
    for c in culprits[:5]:
        rest = [x for x in constraints if x.id != c.id]
        st, _, _, _ = _solve_once(data, cfg, sessions, rest, None, 4, seed, 8)
        if st in ("OPTIMAL", "FEASIBLE"):
            fixes.append(c)
    for c in fixes:
        if c.origin == "data":
            res.suggestions.append(f"Relaxing '{c.describe()}' makes a valid timetable possible "
                                   f"(change it under Manage Academic Data or Constraints & Settings).")
        else:
            res.suggestions.append(f"Removing or relaxing '{c.describe()}' makes a valid timetable possible "
                                   f"(say 'prefer ...' to turn it into a soft preference).")
    if not fixes and culprits:
        res.suggestions.append("Dropping any single one of them is not enough - they use up more of the week than the "
                               "sessions, teachers and rooms can spare. Relax at least two, or reduce weekly sessions.")
