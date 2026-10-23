import pytest
from models import sample_store
from nlp.parser import Parser, ParseContext


@pytest.fixture(scope="module")
def env():
    st = sample_store()
    return Parser(st), ParseContext(st, st.config), st


def first(env, text):
    p, ctx, _ = env
    return p.parse(text, ctx).items[0]


def test_create_extracts_scope(env):
    it = first(env, "Create a timetable for CSE 5th semester with six subjects.")
    assert it.intent == "create"
    assert it.scope["groups"] == ["CSE-5-A", "CSE-5-B"]
    assert it.scope["num_subjects"] == 6


def test_soft_require_with_day_and_morning(env):
    c = first(env, "Schedule DBMS on Monday morning.").constraints[0]
    assert (c.type, c.hard, c.subject, c.days) == ("require", False, "DBMS", ["Monday"])
    assert c.window == (0, 13 * 60)          # morning ends where lunch starts


def test_hard_forbid_after_time(env):
    c = first(env, "Don't schedule classes after 4 PM.").constraints[0]
    assert c.type == "forbid" and c.hard and c.window == (16 * 60, 24 * 60)


def test_keep_free_is_hard_forbid_for_day_part(env):
    c = first(env, "Keep Friday afternoon free.").constraints[0]
    assert c.type == "forbid" and c.hard and c.days == ["Friday"]


def test_labs_per_day(env):
    c = first(env, "Avoid scheduling two lab sessions on the same day.").constraints[0]
    assert c.type == "max_labs_per_day" and c.value == 1 and c.hard


def test_move_creates_forbid_and_require(env):
    cs = first(env, "Move the Java class from Monday to Wednesday.").constraints
    assert [(c.type, c.days) for c in cs] == [("forbid", ["Monday"]), ("require", ["Wednesday"])]
    assert all(c.subject == "Java Programming" and c.hard for c in cs)


def test_view_filters(env):
    assert first(env, "Show the timetable for Section A.").view["groups"] == ["CSE-5-A", "CSE-3-A"]
    assert first(env, "Show only laboratory sessions.").view == {"kind": "lab"}


def test_lunch_change(env):
    assert first(env, "Give us a longer lunch break.").config_changes == [{"lunch": 1}]
    assert first(env, "Make the lunch break shorter").config_changes == [{"lunch": -1}]


def test_typo_is_corrected(env):
    it = first(env, "Keep Fridy free")
    assert it.entities.days == ["Friday"] and it.notes


def test_pronoun_needs_context(env):
    it = first(env, "Avoid the first period for this teacher.")
    assert it.clarification


def test_pronoun_resolved_from_context(env):
    p, _, st = env
    ctx = ParseContext(st, st.config, last_teacher="Dr. Rao")
    c = p.parse("Avoid the first period for this teacher", ctx).items[0].constraints[0]
    assert c.teacher == "Dr. Rao" and c.periods == [0]


def test_multi_clause_split(env):
    p, ctx, _ = env
    req = p.parse("Move DBMS to Tuesday and keep Friday free", ctx)
    assert req.intents == ["move", "modify"]


def test_hard_soft_cues(env):
    assert first(env, "I prefer no classes after 3 PM").constraints[0].hard is False
    assert first(env, "No classes should be after 3 PM").constraints[0].hard is True
    assert first(env, "Labs should be on different days").constraints[0].hard is False


def test_unknown_gets_helpful_message(env):
    it = first(env, "blah blah")
    assert it.intent == "unknown" and it.clarification


def test_evaluation_thresholds():
    from eval.evaluate import evaluate
    out, _ = evaluate(verbose=False)
    assert out["intent_accuracy"] >= 0.9
    assert out["constraint_type_accuracy"] >= 0.9
