"""Phase 3 tests: shared classes, electives, lab batches, teacher limits, travel gaps, short days, NL parsing."""
import pytest

from engine import TimetableAssistant, filter_placements
from models import advanced_sample_store
from nlp.constraints import check_constraint, count_gaps, Constraint
from nlp.parser import Parser, ParseContext
from solver.scheduler import build_sessions, solve, validate, derived_constraints

GROUPS = ["CSE-5-A", "CSE-5-B"]


@pytest.fixture(scope="module")
def store():
    return advanced_sample_store()


@pytest.fixture(scope="module")
def base(store):
    res = solve(store, store.config, GROUPS, [], time_limit=8)
    assert res.ok, res.issues
    return res


def test_session_structure(store):
    ss = build_sessions(store, GROUPS)
    ml = [s for s in ss if s.subject == "Machine Learning"]
    assert len(ml) == 3 and set(ml[0].groups) == set(GROUPS)           # one shared class, not two
    el = [s for s in ss if s.elective]
    assert len(el) == 2 and len(el[0].parts) == 2
    jl = [s for s in ss if s.subject == "Java Lab"]
    assert {s.group for s in jl} == {"CSE-5-A-B1", "CSE-5-A-B2", "CSE-5-B-B1", "CSE-5-B-B2"}


def test_solution_is_valid(store, base):
    cons = base.derived
    assert validate(store, store.config, base.sessions, base.placements, cons) == []


def test_shared_class_counted_once_and_visible_to_both(base):
    ml = [p for p in base.placements if p.subject == "Machine Learning"]
    assert len(ml) == 3
    for g in GROUPS:
        assert len(filter_placements(base.placements, {"groups": [g], "subject": "Machine Learning"})) == 3


def test_elective_parts_run_together_in_different_rooms(base):
    by_start = {}
    for p in base.placements:
        if p.elective:
            by_start.setdefault((p.day, p.start), []).append(p)
    assert len(by_start) == 2
    for parts in by_start.values():
        assert len(parts) == 2 and len({p.room for p in parts}) == 2 and len({p.teacher for p in parts}) == 2


def test_lab_batches_do_not_clash_with_theory(base):
    for p in base.placements:
        if p.group.endswith(("-B1", "-B2")):
            sec = p.group.rsplit("-B", 1)[0]
            for q in base.placements:
                if q is not p and q.day == p.day and sec in q.keys and q.group == sec:
                    assert not (set(p.slots()) & set(q.slots()))


def test_teacher_limits_from_data(store, base):
    rao = [p for p in base.placements if p.teacher == "Dr. Rao"]
    for day in store.config.days:
        slots = sorted(s for p in rao if p.day == day for s in p.slots())
        assert len(slots) <= 3
        runs = [1 + sum(1 for i in range(k) if slots[i + 1] == slots[i] + 1) for k in range(len(slots))] if slots else [0]
        c = Constraint("max_consecutive", True, teacher="Dr. Rao", value=2)
        assert check_constraint(c, rao, store.config)[0]


def test_short_saturday_respected(store, base):
    assert all(p.end <= 4 for p in base.placements if p.day == "Saturday")


def test_unavailable_teacher_not_on_saturday(base):
    assert not [p for p in base.placements if p.teacher == "Ms. Sharma" and p.day == "Saturday"]


def test_travel_gap_hard(store):
    cfg = store.config.copy()
    cfg.travel_gap = True
    res = solve(store, cfg, GROUPS, [], time_limit=10)
    assert res.ok, (res.issues, [c.describe() for c in res.conflict])
    tg = Constraint("travel_gap", True)
    assert check_constraint(tg, res.placements, cfg)[0]


def test_gap_minimisation_gives_compact_days(store):
    # soft quality depends on search time: with a generous limit the strong preference gets (nearly) no idle periods
    gaps = Constraint("minimize_gaps", False, weight=30)
    tight = solve(store, store.config, GROUPS, [gaps], time_limit=30)
    assert tight.ok
    assert sum(x[2] for x in count_gaps(tight.placements, store.config)) <= 3


def test_room_preference_from_user_overrides_data(store):
    c = Constraint("room_pref", False, subject="Java Lab", room="Lab 1", groups=GROUPS, weight=20)
    res = solve(store, store.config, GROUPS, [c], time_limit=8)
    rooms = {p.room for p in res.placements if p.subject == "Java Lab"}
    assert rooms == {"Lab 1"}


def test_impossible_teacher_limit_is_explained(store):
    s2 = advanced_sample_store()
    s2.teachers.loc[s2.teachers["name"] == "Dr. Rao", "max_per_day"] = 1     # 12 periods, 6 days
    res = solve(s2, s2.config, GROUPS, [], time_limit=8)
    assert res.status == "INFEASIBLE"
    assert any(c.origin == "data" and c.teacher == "Dr. Rao" for c in res.conflict)
    assert any("academic data" in s.lower() or "Manage" in s for s in res.suggestions)


def test_elective_with_same_teacher_is_reported():
    s = advanced_sample_store()
    s.subjects.loc[s.subjects["name"] == "Cloud Computing", "teacher"] = "Ms. Sharma"
    res = solve(s, s.config, GROUPS, [], time_limit=5)
    assert res.status == "INFEASIBLE" and any("same teacher" in i for i in res.issues)


# ---------------------------- NL parsing ---------------------------------------
@pytest.fixture(scope="module")
def parser(store):
    return Parser(store)


def parse(parser, store, text):
    ctx = ParseContext(store, store.config, GROUPS, None, None)
    return parser.parse(text, ctx).items[0]


def test_nl_avoid_gaps(parser, store):
    it = parse(parser, store, "Try to avoid gaps between classes.")
    assert it.constraints[0].type == "minimize_gaps" and not it.constraints[0].hard


def test_nl_daily_load(parser, store):
    it = parse(parser, store, "Dr. Rao must teach at most 2 periods a day.")
    c = it.constraints[0]
    assert (c.type, c.teacher, c.value, c.hard) == ("max_daily_load", "Dr. Rao", 2, True)


def test_nl_room_pref(parser, store):
    it = parse(parser, store, "Java Lab should be in Lab 1")
    c = it.constraints[0]
    assert (c.type, c.subject, c.room) == ("room_pref", "Java Lab", "Lab 1")


def test_nl_half_day_and_travel(parser, store):
    it = parse(parser, store, "Make Saturday a half day.")
    assert it.config_changes == [{"day_slots": ("Saturday", 4)}]
    it = parse(parser, store, "Saturday ends at 11 AM.")
    assert it.config_changes == [{"day_slots": ("Saturday", 2)}]
    it = parse(parser, store, "Leave a free period when changing buildings.")
    assert it.config_changes == [{"travel_gap": True}]


def test_engine_short_day_and_travel_roundtrip():
    a = TimetableAssistant(advanced_sample_store(), time_limit=6)
    assert a.handle("Create a timetable for CSE 5th semester.").status == "ok"
    r = a.handle("Make Friday a half day.")
    assert r.status == "ok"
    assert all(p.end <= 4 for p in a.timetable if p.day == "Friday")
    r = a.handle("Make Friday a full day.")
    assert r.status == "ok" and "Friday" not in a.cfg.day_slots
    r = a.handle("Allow travel time when changing buildings.")
    assert r.status == "ok" and a.cfg.travel_gap
    assert a.undo().status == "ok" and not a.cfg.travel_gap
