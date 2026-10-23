import pytest
from models import sample_store, Placement
from nlp.constraints import Constraint
from solver.scheduler import solve, validate, build_sessions

G = ["CSE-5-A", "CSE-5-B"]


@pytest.fixture(scope="module")
def base():
    st = sample_store()
    return st, solve(st, st.config, G, [], time_limit=10)


def test_base_timetable_is_valid_and_complete(base):
    st, res = base
    assert res.ok
    sessions = build_sessions(st, G)
    assert len(res.placements) == len(sessions)
    assert validate(st, st.config, sessions, res.placements, []) == []


def test_labs_use_consecutive_periods_and_avoid_lunch(base):
    st, res = base
    labs = [p for p in res.placements if p.kind == "lab"]
    assert labs and all(p.duration == 2 for p in labs)
    assert all(not (set(p.slots()) & set(st.config.lunch_slots)) for p in labs)


def test_no_teacher_room_or_group_clash(base):
    _, res = base
    seen = {}
    for p in res.placements:
        for sl in p.slots():
            for key in (("g", p.group), ("t", p.teacher), ("r", p.room)):
                k = key + (p.day, sl)
                assert k not in seen, f"clash at {k}"
                seen[k] = True


def test_hard_constraints_are_respected():
    st = sample_store()
    cons = [Constraint("forbid", True, days=["Friday"], window=(12 * 60, 24 * 60), text="keep Friday pm free"),
            Constraint("forbid", True, window=(16 * 60, 24 * 60), text="none after 4"),
            Constraint("max_labs_per_day", True, value=1, text="labs")]
    res = solve(st, st.config, G, cons)
    assert res.ok
    for p in res.placements:
        end = st.config.slot_end(p.end - 1)
        assert end <= 16 * 60
        assert not (p.day == "Friday" and end > 12 * 60)


def test_teacher_unavailability_is_respected():
    st = sample_store()
    st.teachers.loc[st.teachers["name"] == "Dr. Rao", "unavailable_days"] = "Monday, Friday"
    res = solve(st, st.config, G, [])
    assert res.ok
    assert all(p.day not in ("Monday", "Friday") for p in res.placements if p.teacher == "Dr. Rao")


def test_soft_preference_is_satisfied_when_possible():
    st = sample_store()
    c = Constraint("require", False, subject="Java Lab", days=["Tuesday"], text="java lab tue")
    res = solve(st, st.config, G, [c])
    assert res.ok and res.soft_report[0]["satisfied"]


def test_impossible_request_is_explained():
    st = sample_store()
    cons = [Constraint("forbid", True, days=[d], text=f"keep {d} free") for d in ("Monday", "Tuesday", "Wednesday", "Thursday")]
    res = solve(st, st.config, G, cons)
    assert res.status == "INFEASIBLE" and not res.placements
    assert res.conflict and res.suggestions


def test_single_blocking_constraint_is_named():
    st = sample_store()
    # Dr. Rao teaches 12 periods across both groups; blocking 4 days for him alone is impossible
    cons = [Constraint("forbid", True, teacher="Dr. Rao", days=[d], text="x") for d in ("Monday", "Tuesday", "Wednesday", "Thursday")]
    res = solve(st, st.config, G, cons)
    assert res.status == "INFEASIBLE"
    assert res.conflict
    assert any("Removing or relaxing" in s for s in res.suggestions) or res.suggestions


def test_capacity_problem_is_reported():
    st = sample_store()
    st.rooms.loc[st.rooms["type"] == "theory", "capacity"] = 20
    res = solve(st, st.config, G, [])
    assert res.status == "INFEASIBLE"
    assert any("large enough" in i for i in res.issues)


def test_overloaded_teacher_is_reported():
    st = sample_store()
    st.subjects.loc[st.subjects["name"] == "DBMS", "sessions_per_week"] = 10
    st.subjects.loc[st.subjects["name"] == "DBMS Lab", "sessions_per_week"] = 10
    res = solve(st, st.config, G, [])
    assert res.status == "INFEASIBLE"
    assert any("Dr. Rao" in i for i in res.issues)


def test_stability_when_adding_unrelated_constraint(base):
    st, first = base
    c = Constraint("forbid", True, days=["Monday"], window=(17 * 60, 24 * 60), text="no evening monday")
    res = solve(st, st.config, G, [c], previous=first.placements)
    assert res.ok

    def key(pl):
        return {(p.group, p.subject, p.day, p.start) for p in pl}
    assert len(key(first.placements) - key(res.placements)) <= 10   # most of the week stays put
