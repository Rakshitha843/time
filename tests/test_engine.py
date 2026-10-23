import pytest
from models import sample_store
from engine import TimetableAssistant
from solver.scheduler import validate, build_sessions


@pytest.fixture()
def asst():
    a = TimetableAssistant(sample_store())
    r = a.handle("Create a timetable for CSE 5th semester with six subjects.")
    assert r.status == "ok"
    return a


def check_valid(a):
    sessions = build_sessions(a.store, a.scope_groups, a.num_subjects)
    assert validate(a.store, a.cfg, sessions, a.timetable, a.constraints) == []


def test_follow_up_edits_keep_context(asst):
    r = asst.handle("Keep Friday afternoon free.")
    assert r.status == "ok" and len(asst.constraints) == 1
    r = asst.handle("Move the Java class from Monday to Wednesday.")
    assert r.status == "ok"
    assert all(p.day != "Monday" for p in asst.timetable if p.subject == "Java Programming")
    assert any(p.day == "Wednesday" for p in asst.timetable if p.subject == "Java Programming")
    check_valid(asst)


def test_edits_update_existing_timetable_with_few_changes(asst):
    before = {(p.group, p.subject, p.day, p.start) for p in asst.timetable}
    asst.handle("Don't schedule classes after 5 PM.")
    after = {(p.group, p.subject, p.day, p.start) for p in asst.timetable}
    assert len(before - after) <= 8


def test_infeasible_request_changes_nothing_and_can_be_relaxed(asst):
    snapshot = list(asst.timetable)
    r = asst.handle("Keep Monday free. Keep Tuesday free. Keep Wednesday free. Keep Thursday free.")
    assert r.status == "infeasible"
    assert asst.timetable == snapshot
    assert not asst.constraints
    assert asst.rejected
    r2 = asst.apply_rejected_as_soft()
    assert r2.status == "ok"                      # as preferences they can be partly met
    assert all(not c.hard for c in asst.constraints)
    check_valid(asst)


def test_move_when_already_true_explains(asst):
    # find a day DBMS already uses, then ask to "move" to it
    day = next(p.day for p in asst.timetable if p.subject == "DBMS")
    r = asst.handle(f"Move DBMS to {day}")
    assert r.status == "info" and "already" in r.message


def test_view_commands_do_not_resolve(asst):
    before = list(asst.timetable)
    r = asst.handle("Show only laboratory sessions.")
    assert r.status == "view" and asst.view == {"kind": "lab"}
    assert asst.timetable == before
    asst.handle("Show everything")
    assert asst.view == {}


def test_lunch_extension_changes_grid(asst):
    n = len(asst.cfg.lunch_slots)
    r = asst.handle("Give us a longer lunch break.")
    assert r.status == "ok" and len(asst.cfg.lunch_slots) == n + 1
    assert not any(set(p.slots()) & set(asst.cfg.lunch_slots) for p in asst.timetable)


def test_teacher_context_for_pronoun(asst):
    assert asst.handle("Avoid the first period for this teacher.").status == "clarify"
    asst.handle("Show Dr. Rao's timetable")
    r = asst.handle("Avoid the first period for this teacher.")
    assert r.status == "ok"
    assert asst.constraints[-1].teacher == "Dr. Rao"


def test_unknown_subject_is_rejected_politely(asst):
    r = asst.handle("Schedule Data Structures on Monday")      # belongs to CSE-3-A, not in scope
    assert r.status == "clarify" and "not part of" in r.message


def test_undo(asst):
    asst.handle("Keep Friday free.")
    assert asst.constraints
    asst.undo()
    assert not asst.constraints


def test_regenerate_returns_valid_timetable(asst):
    r = asst.handle("regenerate")
    assert r.status == "ok"
    check_valid(asst)
