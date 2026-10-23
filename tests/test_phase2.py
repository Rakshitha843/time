import json
import pytest
from models import sample_store
from nlp.parser import Parser, ParseContext
from nlp.constraints import Constraint, check_constraint
from nlp.llm_fallback import validate_payload, LLMValidationError, parse_with_llm
from engine import TimetableAssistant
from solver.scheduler import solve, validate


@pytest.fixture(scope="module")
def env():
    st = sample_store()
    return Parser(st), ParseContext(st, st.config), st


def one(env, text):
    p, ctx, _ = env
    return p.parse(text, ctx).items[0]


# ---- trained intent classifier ------------------------------------------------
def test_classifier_beats_rules_on_unseen_phrasings():
    from eval.evaluate import evaluate_intents
    r = evaluate_intents(verbose=False)
    assert r["model"]["accuracy"] >= 0.80
    assert r["hybrid"]["accuracy"] >= 0.80
    assert r["model"]["accuracy"] > r["rules"]["accuracy"] + 0.2


@pytest.mark.parametrize("text", ["What's the weather like today", "Tell me a joke", "blah blah", "How are you"])
def test_off_topic_text_is_rejected(env, text):
    assert one(env, text).intent == "unknown"


def test_paraphrases_are_understood(env):
    assert one(env, "Bring DBMS forward to Tuesday").intent == "move"
    assert one(env, "Wipe all the rules").intent == "reset"
    assert one(env, "Run the scheduler once more").intent == "regenerate"


def test_intent_modes(env):
    _, ctx, st = env
    for mode in ("rules", "model", "hybrid"):
        assert Parser(st, mode).parse_clause("Keep Friday free", ctx).intent == "modify"


# ---- unknown names ----------------------------------------------------------------
def test_unknown_teacher_is_reported(env):
    it = one(env, "Dr. Patel is not available on Friday")
    assert it.entities.unknown_teachers == ["Dr Patel"] and it.clarification and not it.constraints


def test_misspelt_teacher_gets_suggestion(env):
    assert "Dr. Rao" in one(env, "Dr. Roa cannot teach on Monday").clarification


# ---- vague expressions -------------------------------------------------------------
def test_early_means_first_period(env):
    c = one(env, "No early classes").constraints[0]
    assert c.type == "forbid" and c.periods == [0] and c.hard


def test_late_means_after_four(env):
    c = one(env, "Don't schedule late classes on Friday").constraints[0]
    assert c.window == (16 * 60, 24 * 60) and c.days == ["Friday"]


def test_in_a_row_variants(env):
    c = one(env, "No more than 3 classes in a row").constraints[0]
    assert (c.type, c.value, c.hard) == ("max_consecutive", 3, True)
    c = one(env, "Avoid back-to-back classes").constraints[0]
    assert (c.value, c.hard) == (2, False)
    c = one(env, "Dr. Rao should not have more than 2 classes in a row").constraints[0]
    assert (c.teacher, c.value, c.hard) == ("Dr. Rao", 2, True)


def test_max_consecutive_checker():
    from models import Placement
    st = sample_store()
    pls = [Placement("G", "A", "T", "R", "Monday", s, 1, "theory") for s in (0, 1, 2)]
    ok, _ = check_constraint(Constraint("max_consecutive", True, value=2), pls, st.config)
    assert not ok
    ok, _ = check_constraint(Constraint("max_consecutive", True, value=3), pls, st.config)
    assert ok


def test_max_consecutive_in_solver():
    st = sample_store()
    c = Constraint("max_consecutive", True, value=2, text="no more than 2 in a row")
    res = solve(st, st.config, ["CSE-3-A"], [c])
    assert res.ok
    assert check_constraint(c, res.placements, st.config)[0]
    assert validate(st, st.config, res.sessions, res.placements, [c]) == []


def test_no_early_in_solver():
    st = sample_store()
    c = one((Parser(st), ParseContext(st, st.config), st), "No early classes").constraints[0]
    res = solve(st, st.config, ["CSE-3-A"], [c])
    assert res.ok and all(p.start != 0 for p in res.placements)


# ---- LLM fallback --------------------------------------------------------------------
GOOD = {"intent": "modify", "constraints": [{"type": "forbid", "hard": True, "teacher": "Dr. Rao", "days": ["Monday"]}]}


def test_llm_payload_is_converted_and_validated():
    st = sample_store()
    intent, cons, _ = validate_payload(GOOD, "x", st)
    assert intent == "modify" and cons[0].teacher == "Dr. Rao" and cons[0].hard


@pytest.mark.parametrize("bad", [
    {"intent": "modify", "constraints": [{"type": "forbid", "teacher": "Dr. Nobody", "days": ["Monday"]}]},
    {"intent": "modify", "constraints": [{"type": "forbid", "days": ["Funday"]}]},
    {"intent": "modify", "constraints": [{"type": "explode"}]},
    {"intent": "modify", "constraints": [{"type": "forbid", "start_minute": 99999}]},
    {"intent": "modify", "constraints": [{"type": "max_consecutive"}]},
    {"intent": "dance"}, {"intent": "unknown"},
])
def test_llm_rejects_invalid_payloads(bad):
    with pytest.raises(LLMValidationError):
        validate_payload(bad, "x", sample_store())


def test_llm_failures_fall_back_silently():
    st = sample_store()
    assert parse_with_llm("x", st, lambda p: "not json") is None
    def boom(p): raise RuntimeError("network down")
    assert parse_with_llm("x", st, boom) is None


def test_llm_result_requires_confirmation_before_applying():
    st = sample_store()
    a = TimetableAssistant(st, llm_fn=lambda p: json.dumps(GOOD))
    assert a.handle("Create a timetable for CSE 5th semester").status == "ok"
    r = a.handle("zxq wibble flarp")
    assert r.status == "confirm" and not a.constraints
    r.parsed.confirmed = True
    assert a.apply(r.parsed).status == "ok"
    assert len(a.constraints) == 1
    assert all(not (p.teacher == "Dr. Rao" and p.day == "Monday") for p in a.timetable)
