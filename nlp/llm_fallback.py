"""Optional LLM fallback for sentences the rules + classifier cannot place.

The language model never touches the timetable.  It only has to fill in the same
JSON schema the rule parser produces; that JSON is then *validated* against your
data (unknown subjects, teachers, groups or days are rejected), converted into
ordinary `Constraint` objects, and shown to the user for confirmation before
anything is applied.

You plug in any callable ``call_fn(prompt: str) -> str`` that returns JSON text,
so this module has no network code and no API-key handling of its own.
"""
from __future__ import annotations

import json
import re

from models import ALL_DAYS, DataStore
from .constraints import Constraint

INTENTS = ["create", "modify", "move", "view", "regenerate", "reset"]
CTYPES = ["forbid", "confine", "require", "max_labs_per_day", "max_consecutive"]

SCHEMA_HELP = """Return ONLY a JSON object:
{
  "intent": one of %s,
  "constraints": [ {
      "type": one of %s,
      "hard": true|false,            // true only for rules that must never be broken
      "subject": exact subject name or null,
      "teacher": exact teacher name or null,
      "groups": [exact group names] or null,
      "days": [day names] or null,
      "start_minute": minutes after midnight or null,   // window start, e.g. 960 = 4 PM
      "end_minute": minutes after midnight or null,     // window end
      "periods": [0-based teaching period numbers, -1 = last] or null,
      "value": integer or null       // for max_labs_per_day / max_consecutive
  } ],
  "view": {"kind": "lab"|"theory"|null, "teacher": name|null, "subject": name|null, "days": [..]|null} or null
}
Use only names from the lists below. If the request cannot be expressed, return {"intent": "unknown"}."""


def build_prompt(text: str, store: DataStore) -> str:
    return (
        "You convert a college timetable instruction into structured scheduling constraints.\n"
        + SCHEMA_HELP % (INTENTS, CTYPES)
        + f"\n\nSubjects: {list(store.subjects['name'].unique())}"
        + f"\nTeachers: {list(store.teachers['name'])}"
        + f"\nGroups: {store.group_names()}"
        + f"\nDays: {store.config.days}"
        + f"\n\nInstruction: {text}\nJSON:"
    )


class LLMValidationError(ValueError):
    pass


def _extract_json(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise LLMValidationError("no JSON object in the model output")
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise LLMValidationError(f"invalid JSON: {e}")


def validate_payload(data: dict, text: str, store: DataStore) -> tuple:
    """Return (intent, constraints, view).  Raises LLMValidationError on anything dubious."""
    intent = data.get("intent")
    if intent == "unknown":
        raise LLMValidationError("the model could not interpret the sentence")
    if intent not in INTENTS:
        raise LLMValidationError(f"unknown intent {intent!r}")
    subjects = set(store.subjects["name"])
    teachers = set(store.teachers["name"])
    groups = set(store.group_names())
    constraints = []
    for raw in data.get("constraints") or []:
        ctype = raw.get("type")
        if ctype not in CTYPES:
            raise LLMValidationError(f"unknown constraint type {ctype!r}")
        if raw.get("subject") and raw["subject"] not in subjects:
            raise LLMValidationError(f"unknown subject {raw['subject']!r}")
        if raw.get("teacher") and raw["teacher"] not in teachers:
            raise LLMValidationError(f"unknown teacher {raw['teacher']!r}")
        g = raw.get("groups")
        if g is not None and (not isinstance(g, list) or not set(g) <= groups):
            raise LLMValidationError(f"unknown group in {g!r}")
        days = raw.get("days")
        if days is not None and (not isinstance(days, list) or not set(days) <= set(ALL_DAYS)):
            raise LLMValidationError(f"invalid days {days!r}")
        s, e = raw.get("start_minute"), raw.get("end_minute")
        for v in (s, e):
            if v is not None and not (isinstance(v, int) and 0 <= v <= 24 * 60):
                raise LLMValidationError(f"time out of range: {v!r}")
        window = (s if s is not None else 0, e if e is not None else 24 * 60) if (s is not None or e is not None) else None
        periods = raw.get("periods")
        if periods is not None and not (isinstance(periods, list) and all(isinstance(p, int) for p in periods)):
            raise LLMValidationError("periods must be a list of integers")
        value = raw.get("value")
        if ctype in ("max_labs_per_day", "max_consecutive") and not (isinstance(value, int) and value >= 1):
            raise LLMValidationError(f"{ctype} needs a positive integer value")
        constraints.append(Constraint(
            type=ctype, hard=bool(raw.get("hard", False)), text=text, groups=g, subject=raw.get("subject"),
            teacher=raw.get("teacher"), days=days, window=window, periods=periods, value=value or 1))
    view = data.get("view")
    if view:
        view = {k: v for k, v in view.items() if v}
        if view.get("kind") and view["kind"] not in ("lab", "theory"):
            raise LLMValidationError("invalid view kind")
    return intent, constraints, view or None


def parse_with_llm(text: str, store: DataStore, call_fn):
    """Return a ParsedInstruction (source='llm', needing confirmation) or None."""
    from .parser import ParsedInstruction          # local import avoids a circular dependency
    try:
        raw = call_fn(build_prompt(text, store))
        intent, constraints, view = validate_payload(_extract_json(raw), text, store)
    except Exception:                                # network errors, bad JSON, invalid content: all fall back
        return None
    item = ParsedInstruction(text=text, intent=intent, confidence=0.5, constraints=constraints, view=view)
    item.source = "llm"
    item.notes.append("Interpreted by the language-model fallback. Please check the constraints before applying.")
    return item
