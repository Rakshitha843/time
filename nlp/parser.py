"""Natural-language understanding for timetable instructions.

Method (hybrid, rule-based, no training data needed)
----------------------------------------------------
* spaCy tokenizer + ``PhraseMatcher`` (case-insensitive gazetteers built from the
  academic data) -> subjects, teachers, rooms, days, departments, groups,
  parts of the day, and intent cue phrases.  Longest match wins
  (``spacy.util.filter_spans``), so "DBMS Lab" beats "DBMS".
* Regular expressions for numeric / temporal patterns (4 PM, between 10 and 12,
  5th semester, section B, first period, "six subjects").
* A small cue lexicon decides hard vs soft ("must", "don't" -> hard;
  "prefer", "if possible", "avoid" -> soft).
* Weighted cue scoring classifies the intent.
* A tiny typo-corrector (difflib) fixes misspelt day names / day parts.
* Session context resolves "this teacher", "it", "us".

Why not a big model?  The vocabulary is closed and known (it comes from the
user's own data), so gazetteer matching is more accurate, explainable and
needs no downloads.  The design leaves room to add a statistical NER / LLM
fallback for unseen phrasings.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

import spacy
from spacy.matcher import PhraseMatcher
from spacy.util import filter_spans

from models import Config, DataStore, ALL_DAYS
from .constraints import Constraint, new_id

# --------------------------------------------------------------------------
# Lexicons
# --------------------------------------------------------------------------
DAY_WORDS = {
    "monday": "Monday", "mondays": "Monday", "mon": "Monday",
    "tuesday": "Tuesday", "tuesdays": "Tuesday", "tue": "Tuesday", "tues": "Tuesday",
    "wednesday": "Wednesday", "wednesdays": "Wednesday", "wed": "Wednesday",
    "thursday": "Thursday", "thursdays": "Thursday", "thu": "Thursday", "thur": "Thursday", "thurs": "Thursday",
    "friday": "Friday", "fridays": "Friday", "fri": "Friday",
    "saturday": "Saturday", "saturdays": "Saturday",
    "sunday": "Sunday", "sundays": "Sunday",
}
PART_WORDS = {
    "morning": "morning", "mornings": "morning", "forenoon": "morning", "first half": "morning",
    "afternoon": "afternoon", "afternoons": "afternoon", "second half": "afternoon", "post lunch": "afternoon",
    "evening": "evening", "evenings": "evening",
}
NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
             "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
ORDINALS = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2, "fourth": 3, "4th": 3,
            "fifth": 4, "5th": 4, "last": -1, "final": -1}
SEM_WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8}

TYPO_TARGETS = list(dict.fromkeys(
    [d.lower() for d in ALL_DAYS] + ["morning", "afternoon", "evening", "semester", "section", "laboratory"]))

# intent cue phrases: (phrase, intent, weight)
INTENT_CUES = [
    ("create", "create", 3), ("generate", "create", 3), ("build", "create", 3), ("prepare", "create", 3),
    ("produce", "create", 3), ("make", "create", 1), ("draft", "create", 1), ("set up", "create", 2),
    ("come up with", "create", 2),
    ("regenerate", "regenerate", 5), ("redo", "regenerate", 5), ("reshuffle", "regenerate", 5),
    ("shuffle", "regenerate", 4), ("try again", "regenerate", 5), ("rerun", "regenerate", 5),
    ("re-run", "regenerate", 5), ("generate again", "regenerate", 6), ("another timetable", "regenerate", 6),
    ("another version", "regenerate", 6), ("different timetable", "regenerate", 6), ("new version", "regenerate", 4),
    ("show", "view", 4), ("display", "view", 4), ("view", "view", 4), ("see", "view", 2), ("list", "view", 3),
    ("print", "view", 2), ("what is", "view", 3), ("what's", "view", 3), ("when is", "view", 3),
    ("when does", "view", 3), ("which classes", "view", 3), ("open", "view", 1),
    ("move", "move", 5), ("shift", "move", 5), ("reschedule", "move", 5), ("relocate", "move", 5),
    ("swap", "move", 4), ("transfer", "move", 4), ("postpone", "move", 4), ("prepone", "move", 4),
    ("reset", "reset", 5), ("clear all", "reset", 6), ("start over", "reset", 6), ("start afresh", "reset", 6),
    ("remove all", "reset", 6), ("forget everything", "reset", 6), ("clear constraints", "reset", 6),
    ("schedule", "modify", 2), ("keep", "modify", 3), ("avoid", "modify", 3), ("prefer", "modify", 3),
    ("give", "modify", 2), ("ensure", "modify", 3), ("make sure", "modify", 3), ("add", "modify", 2),
    ("put", "modify", 2), ("should", "modify", 1), ("must", "modify", 2), ("limit", "modify", 2),
    ("restrict", "modify", 2), ("allow", "modify", 2), ("set", "modify", 1), ("extend", "modify", 3),
    ("increase", "modify", 2), ("reduce", "modify", 2), ("longer", "modify", 2), ("no", "modify", 2),
    ("don't", "modify", 3), ("do not", "modify", 3), ("never", "modify", 3), ("dont", "modify", 3),
    ("free", "modify", 2), ("not", "modify", 1), ("only", "modify", 1), ("need", "modify", 1),
]
CREATE_RE = re.compile(r"\b(create|generate|make|prepare|draft|build|produce|need|want)\b.*\b(timetable|time table|schedule|routine)s?\b")
NEG_RE = re.compile(r"\b(?:don't|do not|dont|never|no|not|avoid|without|free|off|unavailable|cannot|can't|shouldn't|should not|mustn't)\b")
HARD_RE = re.compile(r"\b(?:must|never|always|don't|do not|dont|cannot|can't|no|only|free|unavailable|not available|strictly|required|has to|have to|mustn't|without|should not|shouldn't)\b")
SOFT_RE = re.compile(r"\b(?:prefer|preferably|if possible|when possible|as far as possible|try to|try|ideally|would like|rather|would be nice|hopefully|better|mostly)\b")
AVOID_RE = re.compile(r"\bavoid\b")
SHOULD_RE = re.compile(r"\b(?:should|ought to|would prefer|could)\b")
LAB_RE = re.compile(r"\b(?:labs?|laboratory|laboratories|practicals?)\b")
THEORY_RE = re.compile(r"\b(?:theory|lectures?)\b")
SAME_DAY_RE = re.compile(r"\b(?:same day|one day|single day|per day|each day|every day|a day|different days|separate days|distinct days)\b")
LABS_SPREAD_RE = re.compile(r"\b(?:different|separate|distinct|spread|at most|no more than|not more than|max(?:imum)?|only one|one lab)\b")
CONSEC_RE = re.compile(r"\b(?:in a row|back[- ]to[- ]back|consecutive|continuous(?:ly)?|without a break|non-?stop|long stretch(?:es)?)\b")
MAXRUN_RE = re.compile(r"\b(?:at most|no more than|not more than|more than|max(?:imum)?(?: of)?|up to|only)\s+(\d{1,2}|one|two|three|four|five)\b")
EARLY_RE = re.compile(r"\bearly\b|\bfirst thing\b")
LATE_RE = re.compile(r"\blate(?:r)?\b(?!\s+than)|\blast thing\b")
GAPS_RE = re.compile(r"\b(?:gaps?|free periods?|idle(?: periods?| time)?|holes?|breaks? between (?:classes|lectures)|free hours? between|compact|continuous block)\b")
DAILY_RE = re.compile(r"\b(?:per day|a day|each day|every day|daily|in a day|in one day)\b")
LOAD_RE = re.compile(r"\b(?:periods?|classes|lectures?|hours?|sessions?)\b")
ROOMPREF_RE = re.compile(r"\b(?:in|inside|at|use|using|held in|conducted in|located in)\b")
TRAVEL_RE = re.compile(r"\b(?:travel|walk(?:ing)?|commute|buildings?|block to block|move between|changing buildings?|switch(?:ing)? buildings?)\b")
HALFDAY_RE = re.compile(r"\b(?:half[- ]day|short(?:er)?|ends? (?:early|at|by)|closes?|only (?:the )?first \d+ periods?|until noon|till noon)\b")
LUNCH_RE = re.compile(r"\blunch\b")
LUNCH_MORE_RE = re.compile(r"\b(?:longer|extend|extended|more|extra|bigger|increase|additional|add)\b")
LUNCH_LESS_RE = re.compile(r"\b(?:shorter|reduce|less|decrease|shrink|cut)\b")
SEM_RE = re.compile(r"\b(?:(\d{1,2})(?:st|nd|rd|th)?|(first|second|third|fourth|fifth|sixth|seventh|eighth))[\s-]*sem(?:ester)?\b|\bsem(?:ester)?[\s:-]*(\d{1,2})\b")
SEC_RE = re.compile(r"\bsec(?:tion)?[\s:-]+([a-z])\b")
NSUBJ_RE = re.compile(r"\b(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:subjects?|courses?|papers?)\b")
PERIOD_ORD_RE = re.compile(r"\b(first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th|last|final)\s+(?:period|class|lecture|slot|hour)\b")
PERIOD_NUM_RE = re.compile(r"\bperiod\s+(\d{1,2})\b")
MAXLABS_RE = re.compile(r"\b(?:at most|no more than|not more than|max(?:imum)?(?: of)?|only|up to)\s+(\d|one|two|three)\s+(?:lab|practical)")

_NUM = r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?(?![\w:])"
BETWEEN_RE = re.compile(rf"\b(?:between|from)\s+{_NUM}\s*(?:and|to|-|till|until)\s*{_NUM}")
AFTER_RE = re.compile(rf"\b(?:after|post|beyond|past|starting at|starting from|later than|from)\s+{_NUM}")
BEFORE_RE = re.compile(rf"\b(?:before|by|until|till|prior to|earlier than|ending by|up to)\s+{_NUM}")
AT_RE = re.compile(rf"\b(?:at|around|@)\s+{_NUM}")
NOON_RE = re.compile(r"\b(after|before|by|at|until|till)\s+(?:noon|midday)\b")
LUNCHREL_RE = re.compile(r"\b(after|before|post|pre)[\s-]+lunch\b")

DOMAIN_RE = re.compile(
    r"\b(?:timetables?|schedules?|routine|plan|class(?:es)?|lectures?|labs?|laboratory|period|lunch|slot|"
    r"constraints?|rules?|solver|scheduler|everything|again|start|begin|scratch|slate|version|layout|arrangement|regenerate|reshuffle|redo|reset|rerun)\b")

TITLE_RE = re.compile(r"\b(Dr|Prof|Mr|Ms|Mrs)\.", re.I)
TITLED_NAME_RE = re.compile(r"\b(Dr|Prof|Mr|Ms|Mrs)\s+([A-Z][a-z]{2,})")


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------
@dataclass
class ParseContext:
    store: DataStore
    cfg: Config
    active_groups: list | None = None
    last_subject: str | None = None
    last_teacher: str | None = None


@dataclass
class Entities:
    subjects: list = field(default_factory=list)
    teachers: list = field(default_factory=list)
    rooms: list = field(default_factory=list)
    days: list = field(default_factory=list)
    parts: list = field(default_factory=list)
    times: list = field(default_factory=list)       # [(kind, minutes_a, minutes_b)]
    periods: list = field(default_factory=list)
    department: str | None = None
    semester: int | None = None
    section: str | None = None
    groups: list = field(default_factory=list)
    num_subjects: int | None = None
    kind: str | None = None
    unknown_teachers: list = field(default_factory=list)

    def to_dict(self):
        d = {k: v for k, v in self.__dict__.items() if v not in (None, [], "")}
        return d


@dataclass
class ParsedInstruction:
    text: str
    intent: str = "unknown"
    confidence: float = 0.0
    scores: dict = field(default_factory=dict)
    entities: Entities = field(default_factory=Entities)
    constraints: list = field(default_factory=list)
    view: dict | None = None
    config_changes: list = field(default_factory=list)
    scope: dict | None = None            # for create: department/semester/section/groups/num_subjects
    notes: list = field(default_factory=list)
    clarification: str | None = None
    source: str = "rules"                # "rules" | "llm"

    def to_dict(self):
        return {
            "source": self.source, "text": self.text, "intent": self.intent, "confidence": round(self.confidence, 2),
            "entities": self.entities.to_dict(), "scope": self.scope, "view": self.view,
            "config_changes": self.config_changes,
            "constraints": [c.to_dict() for c in self.constraints],
            "notes": self.notes, "clarification": self.clarification,
        }


@dataclass
class ParsedRequest:
    text: str
    items: list = field(default_factory=list)
    confirmed: bool = False

    @property
    def needs_confirmation(self) -> bool:
        return any(i.source == "llm" for i in self.items) and not self.confirmed

    @property
    def intents(self):
        return [i.intent for i in self.items]

    @property
    def constraints(self):
        return [c for i in self.items for c in i.constraints]

    @property
    def clarification(self):
        for i in self.items:
            if i.clarification:
                return i.clarification
        return None

    def to_dict(self):
        return {"text": self.text, "items": [i.to_dict() for i in self.items]}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _hm_to_min(h, m, ap) -> int:
    h = int(h)
    m = int(m) if m else 0
    if ap == "pm" and h < 12:
        h += 12
    elif ap == "am" and h == 12:
        h = 0
    elif ap is None and 1 <= h < 8:      # college hours: "at 4" means 4 PM
        h += 12
    return h * 60 + m


def _num_value(tok: str):
    tok = tok.lower()
    return int(tok) if tok.isdigit() else NUM_WORDS.get(tok)


def normalize(text: str) -> str:
    t = text.strip().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = re.sub(r"\b([ap])\.\s?m\b\.?", lambda m: m.group(1).lower() + "m", t, flags=re.I)
    t = TITLE_RE.sub(lambda m: m.group(1), t)          # "Dr. Rao" -> "Dr Rao" (keeps sentence splitting sane)
    t = re.sub(r"\s+", " ", t)
    return t


def fix_typos(text: str):
    notes = []

    def repl(m):
        w = m.group(0)
        lw = w.lower()
        if len(lw) < 5 or lw in TYPO_TARGETS or lw in DAY_WORDS or lw in PART_WORDS:
            return w
        close = difflib.get_close_matches(lw, TYPO_TARGETS, n=1, cutoff=0.84)
        if close:
            notes.append(f"Interpreted '{w}' as '{close[0]}'")
            return close[0]
        return w

    return re.sub(r"[A-Za-z]+", repl, text), notes


CLAUSE_SPLIT_RE = re.compile(
    r"\s*(?:,|\band\b|\bthen\b|\balso\b|\bplus\b)\s+(?=(?:please\s+)?(?:move|shift|keep|schedule|avoid|don't|do not|dont|show|give|ensure|make|put|no|never|prefer|reschedule|set|limit)\b)",
    re.I)


def split_clauses(text: str) -> list:
    out = []
    for sent in re.split(r"(?<=[.!?;])\s+|\n+", text):
        sent = sent.strip().rstrip(".!?;")
        if not sent:
            continue
        out.extend(c.strip() for c in CLAUSE_SPLIT_RE.split(sent) if c.strip())
    return out


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------
class Parser:
    def __init__(self, store: DataStore, intent_mode: str = "hybrid", llm_fn=None):
        """intent_mode: 'rules' (cue lexicon only), 'model' (trained classifier only) or 'hybrid'."""
        self.store = store
        self.intent_mode = intent_mode
        self._intent_model = None
        self.llm_fn = llm_fn            # optional: callable(prompt) -> JSON text
        self.nlp = spacy.blank("en")
        self._build_gazetteers()

    # ---- gazetteers -----------------------------------------------------
    def _add(self, matcher, key, phrases):
        docs = [self.nlp.make_doc(normalize(p)) for p in dict.fromkeys(p for p in phrases if p.strip())]
        if docs:
            matcher.add(key, docs)

    def _build_gazetteers(self):
        st = self.store
        self.ent_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.intent_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")

        # subjects (name, explicit aliases, auto-acronym for multi-word non-lab names)
        for name in dict.fromkeys(st.subjects["name"]):
            aliases = [name]
            row_aliases = st.subjects[st.subjects["name"] == name]["aliases"].iloc[0]
            aliases += [a.strip() for a in re.split(r"[;,|]", row_aliases) if a.strip()]
            words = name.split()
            if len(words) >= 2 and words[-1].lower() != "lab":
                aliases.append("".join(w[0] for w in words))
            self._add(self.ent_matcher, f"SUBJECT|{name}", aliases)

        # teachers: full name, without title, surname
        for name in st.teachers["name"]:
            bare = re.sub(r"^(dr|prof|mr|ms|mrs)\.?\s+", "", name, flags=re.I)
            variants = [name, bare]
            if len(bare.split()[-1]) >= 3:
                variants.append(bare.split()[-1])
            self._add(self.ent_matcher, f"TEACHER|{name}", variants)

        for name in st.rooms["name"]:
            self._add(self.ent_matcher, f"ROOM|{name}", [name])
        for d in st.config.days + [d for d in ALL_DAYS if d not in st.config.days]:
            self._add(self.ent_matcher, f"DAY|{d}", [w for w, c in DAY_WORDS.items() if c == d])
        for part in set(PART_WORDS.values()):
            self._add(self.ent_matcher, f"PART|{part}", [w for w, c in PART_WORDS.items() if c == part])
        for dept in dict.fromkeys(st.groups["department"]):
            if dept:
                self._add(self.ent_matcher, f"DEPT|{dept}", [dept])
        for g in st.groups["name"]:
            self._add(self.ent_matcher, f"GROUP|{g}", [g])

        for phrase, intent, w in INTENT_CUES:
            self._add(self.intent_matcher, f"INTENT|{intent}|{w}", [phrase])

    # ---- entity extraction ----------------------------------------------
    def extract_entities(self, text: str, cfg: Config) -> tuple:
        """Returns (Entities, remaining_text) - remaining_text has entity spans blanked out."""
        doc = self.nlp(text)
        spans = []
        for mid, s, e in self.ent_matcher(doc):
            span = doc[s:e]
            span.label_ = self.nlp.vocab.strings[mid]
            spans.append(span)
        spans = filter_spans(spans)
        ents = Entities()
        blank = list(text)
        for sp in spans:
            kind, _, canon = sp.label_.partition("|")
            bucket = {"SUBJECT": ents.subjects, "TEACHER": ents.teachers, "ROOM": ents.rooms,
                      "DAY": ents.days, "PART": ents.parts}.get(kind)
            if bucket is not None:
                if canon not in bucket:
                    bucket.append(canon)
            elif kind == "DEPT":
                ents.department = canon
            elif kind == "GROUP":
                ents.groups.append(canon)
            for i in range(sp.start_char, sp.end_char):
                blank[i] = " "
        rest = "".join(blank)

        for m in TITLED_NAME_RE.finditer(text):          # "Dr Patel" that is not in the database
            full = f"{m.group(1)} {m.group(2)}"
            known = any(m.group(2).lower() in t.lower() for t in self.store.teachers["name"])
            if not known and full not in ents.unknown_teachers:
                ents.unknown_teachers.append(full)

        low = text.lower()
        m = SEM_RE.search(low)
        if m:
            ents.semester = int(m.group(1)) if m.group(1) else SEM_WORDS.get(m.group(2)) or int(m.group(3))
        m = SEC_RE.search(low)
        if m:
            ents.section = m.group(1).upper()
        m = NSUBJ_RE.search(low)
        if m:
            ents.num_subjects = _num_value(m.group(1))

        for m in PERIOD_ORD_RE.finditer(low):
            ents.periods.append(ORDINALS[m.group(1)])
        for m in PERIOD_NUM_RE.finditer(low):
            ents.periods.append(int(m.group(1)) - 1)

        ents.times = self._parse_times(low, cfg)
        rest_low = rest.lower()
        if LAB_RE.search(rest_low):
            ents.kind = "lab"
        elif THEORY_RE.search(rest_low):
            ents.kind = "theory"
        return ents, rest

    def _parse_times(self, low: str, cfg: Config) -> list:
        times = []
        work = low
        for m in BETWEEN_RE.finditer(work):
            a = _hm_to_min(m.group(1), m.group(2), m.group(3) or m.group(6))
            b = _hm_to_min(m.group(4), m.group(5), m.group(6))
            if m.group(3) is None and m.group(6) == "pm" and a + 12 * 60 < b:
                a += 12 * 60
            times.append(("between", a, b))
        work = BETWEEN_RE.sub(" ", work)
        for m in NOON_RE.finditer(work):
            kind = {"after": "after", "before": "before", "by": "before", "until": "before",
                    "till": "before", "at": "at"}[m.group(1)]
            times.append((kind, 720, 720))
        work = NOON_RE.sub(" ", work)
        lw = cfg.lunch_window()
        for m in LUNCHREL_RE.finditer(work):
            if lw:
                if m.group(1) in ("after", "post"):
                    times.append(("after", lw[1], lw[1]))
                else:
                    times.append(("before", lw[0], lw[0]))
        for rx, kind in ((AFTER_RE, "after"), (BEFORE_RE, "before"), (AT_RE, "at")):
            for m in rx.finditer(work):
                times.append((kind, _hm_to_min(m.group(1), m.group(2), m.group(3)), 0))
            work = rx.sub(" ", work)
        return times

    # ---- intent ---------------------------------------------------------
    def classify(self, text: str, ents: Entities) -> tuple:
        doc = self.nlp(text)
        low = text.lower()
        scores: dict = {}
        first_tok = doc[0].lower_ if len(doc) else ""
        for mid, s, e in self.intent_matcher(doc):
            _, intent, w = self.nlp.vocab.strings[mid].split("|")
            w = float(w)
            if s == 0:
                w *= 1.5              # imperative verbs open the sentence
            if intent == "create" and doc[s:e].text.lower() == "make":
                w = 0.0                # "make Friday free" is not a create request
            scores[intent] = scores.get(intent, 0.0) + w
        if CREATE_RE.search(low):
            scores["create"] = scores.get("create", 0.0) + 4
        if "regenerate" in scores or any(x in low for x in ("generate again", "another timetable")):
            scores["create"] = max(0.0, scores.get("create", 0.0) - 3)
        if re.search(r"\bchange\b.*\b(to|from)\b", low) and ents.days:
            scores["move"] = scores.get("move", 0.0) + 4
        if scores.get("view", 0) and not scores.get("create"):
            scores["modify"] = max(0.0, scores.get("modify", 0.0) - 2)
        if LUNCH_RE.search(low) and (LUNCH_MORE_RE.search(low) or LUNCH_LESS_RE.search(low)):
            scores["modify"] = scores.get("modify", 0.0) + 3
        if first_tok == "give" and "show" in low:
            scores["view"] = scores.get("view", 0.0) + 2
        # structural evidence
        if (ents.department or ents.semester) and scores.get("create"):
            scores["create"] += 1
        if ents.days and (NEG_RE.search(low) or ents.times or ents.parts):
            scores["modify"] = scores.get("modify", 0.0) + 1
        scores = {k: v for k, v in scores.items() if v > 0}
        has_domain = any(v >= 5 for k, v in scores.items() if k != "view") or DOMAIN_RE.search(low) or any(
            [ents.days, ents.parts, ents.times, ents.subjects, ents.teachers, ents.rooms, ents.groups,
             ents.department, ents.semester, ents.section, ents.kind, ents.periods])
        return self._decide_intent(text, scores, bool(has_domain))

    def _model(self):
        if self._intent_model is None:
            from .intent_model import IntentModel
            self._intent_model = IntentModel.shared()
        return self._intent_model

    PRIORITY = ["reset", "regenerate", "view", "move", "create", "modify"]

    def _decide_intent(self, text: str, rule_scores: dict, has_domain: bool = True) -> tuple:
        mode = self.intent_mode
        if not has_domain:
            # the classifier is over-confident on off-topic text ("what is the weather" -> view 0.97),
            # so require at least one timetable-related cue before trusting any intent
            return "unknown", 0.0, {}
        rules_ok = mode in ("rules", "hybrid")
        model_ok = mode in ("model", "hybrid")
        total = sum(rule_scores.values())
        r = {k: v / total for k, v in rule_scores.items()} if (rules_ok and total) else {}
        p = self._model().predict_proba(text) if model_ok else {}
        if mode == "rules":
            if not r:
                return "unknown", 0.0, {}
            best = max(rule_scores.values())
            top = [k for k in self.PRIORITY if rule_scores.get(k) == best][0]
            return top, best / total, rule_scores
        if mode == "model":
            top = str(max(p, key=p.get))
            return top, p[top], {str(k): round(v, 3) for k, v in p.items()}
        # hybrid: average the two distributions; with no rule evidence trust the model but demand more certainty
        w_rule = 0.25 if r else 0.0
        combined = {k: w_rule * r.get(k, 0.0) + (1 - w_rule) * p.get(k, 0.0) for k in set(r) | set(p)}
        top = str(max(combined, key=combined.get))
        threshold = 0.35
        if combined[top] < threshold:
            return "unknown", combined[top], {k: round(v, 3) for k, v in combined.items()}
        return top, combined[top], {k: round(v, 3) for k, v in sorted(combined.items(), key=lambda kv: -kv[1])}

    # ---- constraint building --------------------------------------------
    @staticmethod
    def _window(ents: Entities, cfg: Config):
        """Collapse time expressions / parts of day into (window, label, start_at)."""
        w0, w1, start_at = None, None, None
        for kind, a, b in ents.times:
            if kind == "between":
                w0, w1 = a, b
            elif kind == "after":
                w0 = a if w0 is None else max(w0, a)
            elif kind == "before":
                w1 = a if w1 is None else min(w1, a)
            elif kind == "at":
                start_at = a
        label = None
        if w0 is None and w1 is None and start_at is None and ents.parts:
            part = ents.parts[0]
            lw = cfg.lunch_window()
            if part == "morning":
                w0, w1 = 0, (lw[0] if lw else 12 * 60)
            elif part == "afternoon":
                w0, w1 = (lw[0] if lw else 12 * 60), 24 * 60
            else:
                w0, w1 = 16 * 60, 24 * 60
            label = part
        window = None
        if w0 is not None or w1 is not None:
            window = (w0 if w0 is not None else 0, w1 if w1 is not None else 24 * 60)
        return window, label, start_at

    def _resolve_scope_groups(self, ents: Entities):
        if ents.groups:
            return list(dict.fromkeys(ents.groups))
        if ents.department or ents.semester or ents.section:
            return self.store.resolve_groups(ents.department, ents.semester, ents.section)
        return None

    def build(self, text, low, intent, ents, ctx: ParseContext, item: ParsedInstruction):
        cfg = ctx.cfg

        # ----- view ------------------------------------------------------
        if intent == "view":
            view = {}
            groups = self._resolve_scope_groups(ents)
            if groups:
                view["groups"] = groups
            if ents.teachers:
                view["teacher"] = ents.teachers[0]
            elif re.search(r"\b(this|that|the) teacher\b", low) and ctx.last_teacher:
                view["teacher"] = ctx.last_teacher
            if ents.subjects:
                view["subject"] = ents.subjects[0]
            if ents.rooms:
                view["room"] = ents.rooms[0]
            if ents.days:
                view["days"] = ents.days
            if ents.kind:
                view["kind"] = ents.kind
            if re.search(r"\b(all|everything|full|entire|whole|complete)\b", low) and not view:
                view = {"clear": True}
            item.view = view or {"clear": True}
            return

        # ----- create / reset / regenerate scope ---------------------------
        if intent == "create":
            groups = self._resolve_scope_groups(ents)
            item.scope = {"department": ents.department, "semester": ents.semester,
                          "section": ents.section, "groups": groups, "num_subjects": ents.num_subjects}
            if groups is not None and not groups:
                item.clarification = ("I couldn't find a class group matching "
                                      f"{ents.department or ''} {('semester ' + str(ents.semester)) if ents.semester else ''} "
                                      f"{('section ' + ents.section) if ents.section else ''}. "
                                      f"Available groups: {', '.join(self.store.group_names())}.").replace("  ", " ")
                return
        if intent in ("reset", "regenerate"):
            return

        # ----- lunch break ---------------------------------------------------
        if LUNCH_RE.search(low) and not ents.days:
            if LUNCH_MORE_RE.search(low):
                item.config_changes.append({"lunch": +1})
                item.notes.append("Lunch break will be extended by one period.")
                return
            if LUNCH_LESS_RE.search(low):
                item.config_changes.append({"lunch": -1})
                item.notes.append("Lunch break will be shortened by one period.")
                return

        groups = self._resolve_scope_groups(ents) if intent != "create" else None
        neg = bool(NEG_RE.search(low))
        soft_cue = bool(SOFT_RE.search(low))
        avoid_cue = bool(AVOID_RE.search(low))
        hard_cue = bool(HARD_RE.search(low))
        should_cue = bool(SHOULD_RE.search(low))

        def decide(default_hard: bool, structural=False) -> bool:
            if soft_cue:
                return False
            if avoid_cue and not structural:
                return False
            if hard_cue:
                return True
            if should_cue:                 # "should" is a recommendation, "must" is a rule
                return False
            return default_hard

        # ----- Phase 3: settings and scheduling-depth sentences -----------------------
        if self._phase3(text, low, intent, ents, cfg, groups, item, decide, neg):
            return

        # ----- labs per day ------------------------------------------------
        if LAB_RE.search(low) and SAME_DAY_RE.search(low) and (neg or LABS_SPREAD_RE.search(low) or avoid_cue):
            m = MAXLABS_RE.search(low)
            value = _num_value(m.group(1)) if m else 1
            item.constraints.append(Constraint(
                type="max_labs_per_day", hard=decide(True, structural=True), text=text,
                groups=groups, value=value or 1))
            return

        # ----- vague "early" / "late" -> concrete periods / window --------------------
        vague_periods, vague_window, vague_label = None, None, None
        if EARLY_RE.search(low) and not ents.periods:
            vague_periods, vague_label = [0], "early"
        if LATE_RE.search(low) and not ents.times and not ents.parts:
            vague_window, vague_label = (16 * 60, 24 * 60), None
        if re.search(r"\blast thing\b", low) and not ents.periods:
            vague_periods, vague_window = [-1], None

        # ----- too many classes in a row --------------------------------------
        if CONSEC_RE.search(low) and (neg or soft_cue or avoid_cue or MAXRUN_RE.search(low) or "too many" in low):
            m = MAXRUN_RE.search(low)
            value = _num_value(m.group(1)) if m else (2 if re.search(r"back[- ]to[- ]back|consecutive", low) else 3)
            teacher = (ents.teachers or [None])[0]
            item.constraints.append(Constraint(
                type="max_consecutive", hard=decide(False, structural=True) if not ("too many" in low) else False,
                text=text, groups=groups, teacher=teacher, value=max(1, value or 3)))
            return

        # ----- pronoun resolution --------------------------------------------
        subjects, teachers = list(ents.subjects), list(ents.teachers)
        if not teachers and re.search(r"\b(this|that|the same) teacher\b|\bhim\b|\bher\b|\bthem\b", low):
            if ctx.last_teacher:
                teachers = [ctx.last_teacher]
                item.notes.append(f"'this teacher' resolved to {ctx.last_teacher}.")
            else:
                item.clarification = "Which teacher do you mean? I don't have one in context yet."
                return
        if not subjects and re.search(r"\b(this|that) (subject|course|class|lecture)\b", low) and not ents.kind:
            if ctx.last_subject:
                subjects = [ctx.last_subject]
                item.notes.append(f"'this subject' resolved to {ctx.last_subject}.")
        if not subjects and intent == "move" and re.search(r"\bit\b", low) and ctx.last_subject:
            subjects = [ctx.last_subject]
            item.notes.append(f"'it' resolved to {ctx.last_subject}.")

        window, wlabel, start_at = self._window(ents, cfg)
        periods = ents.periods or vague_periods or None
        if vague_window and not window:
            window, wlabel = vague_window, None
        if periods and wlabel == "morning":      # "first thing in the morning": the period already says it
            window = wlabel = None
        days = ents.days or None

        def base(**kw):
            return Constraint(text=text, groups=groups, window=window, window_label=wlabel,
                              start_at=start_at, periods=periods, room=ents.rooms[0] if ents.rooms else None,
                              kind=ents.kind, **kw)

        # ----- move -----------------------------------------------------------
        if intent == "move":
            frm = re.search(r"\bfrom\s+(\w+)", low)
            to = re.search(r"\b(?:to|on|onto)\s+(\w+)", low)
            from_day = DAY_WORDS.get(frm.group(1)) if frm else None
            to_day = DAY_WORDS.get(to.group(1)) if to else None
            if not to_day and days:
                to_day = [d for d in days if d != from_day][-1] if [d for d in days if d != from_day] else None
            if not subjects and not teachers:
                item.clarification = "Which subject (or teacher) should I move?"
                return
            if not to_day and not from_day:
                item.clarification = "Which day should I move it to?"
                return
            hard = decide(True)
            targets = [("subject", s) for s in subjects] or [("teacher", t) for t in teachers]
            for field_, val in targets:
                scope_kw = {field_: val}
                if from_day:
                    c = base(type="forbid", hard=hard, days=[from_day], origin="move", **scope_kw)
                    c.window = c.window_label = c.start_at = c.periods = None
                    item.constraints.append(c)
                if to_day:
                    c = base(type="require", hard=hard, days=[to_day], origin="move" if not from_day else "move_to", **scope_kw)
                    item.constraints.append(c)
            return

        # ----- generic constraint -------------------------------------------------
        has_target = bool(subjects or teachers or ents.rooms or ents.kind)
        has_where = bool(days or window or periods or start_at is not None)
        if not (has_target or has_where):
            return
        if intent == "create" and not neg and not has_where:
            return
        if not has_where and not neg:
            if has_target and intent == "create":
                return
            # e.g. "Schedule DBMS" with no time information: nothing to constrain
            item.notes.append("No day/time information found, so nothing to constrain.")
            return
        if not has_where and neg and not (subjects or teachers or ents.rooms):
            return

        ctype = "forbid" if neg else ("require" if subjects else "confine")
        if neg and not has_where and (subjects or teachers or ents.rooms):
            # "Don't schedule DBMS" with no time: too absolute, ask.
            item.clarification = "When should that not happen? Give me a day or a time range."
            return
        hard = decide(default_hard=(ctype == "forbid"))
        targets = [("subject", s) for s in subjects] if len(subjects) > 1 else None
        if len(teachers) > 1 and not targets:
            targets = [("teacher", t) for t in teachers]
        if targets:
            for field_, val in targets:
                item.constraints.append(base(type=ctype, hard=hard, days=days, **{field_: val}))
        else:
            item.constraints.append(base(type=ctype, hard=hard, days=days,
                                         subject=subjects[0] if subjects else None,
                                         teacher=teachers[0] if teachers else None))


    # ---- Phase 3 sentences ------------------------------------------------------------
    def _phase3(self, text, low, intent, ents, cfg, groups, item, decide, neg) -> bool:
        """Travel buffer, short days, idle gaps, daily teacher load, room preference.
        Returns True when the sentence was handled."""
        # travel buffer between buildings (a setting, not a per-sentence constraint)
        if TRAVEL_RE.search(low) and re.search(r"\b(?:buffer|gap|time|free period|break|allow|need|leave|keep)\b", low):
            off = bool(re.search(r"\b(?:no need|don't need|do not need|not needed|remove|disable|turn off|stop|ignore)\b", low))
            item.config_changes.append({"travel_gap": not off})
            item.notes.append("A free period will " + ("no longer " if off else "") + "be kept when a class or teacher changes building.")
            return True
        # short / half days:  "make Saturday a half day", "Friday ends at 1 PM", "Saturday normal again"
        if ents.days and (HALFDAY_RE.search(low) or re.search(r"\b(?:full|normal|regular)\s+(?:day|schedule)\b|\bnormal again\b", low)):
            full = bool(re.search(r"\b(?:full|normal|regular)\b", low)) and not HALFDAY_RE.search(low)
            end_min = None
            for kind, a, b in ents.times:
                if kind in ("before", "at", "after") and kind != "after":
                    end_min = a
                elif kind == "between":
                    end_min = b
            if full:
                for d in ents.days:
                    item.config_changes.append({"day_slots": (d, None)})
                item.notes.append(f"{', '.join(ents.days)} will use the full set of periods.")
                return True
            if end_min is not None:
                n = max(1, -(-(end_min - cfg.start_min()) // cfg.slot_minutes))
            else:
                n = min(cfg.lunch_slots) if cfg.lunch_slots else max(1, cfg.num_slots // 2)
            n = min(n, cfg.num_slots)
            for d in ents.days:
                item.config_changes.append({"day_slots": (d, n)})
            item.notes.append(f"{', '.join(ents.days)} will have only the first {n} period(s) open "
                              f"(until {cfg.fmt(cfg.slot_start(n))}).")
            return True
        # idle gaps between a group's classes
        if GAPS_RE.search(low) and (neg or SOFT_RE.search(low) or AVOID_RE.search(low) or re.search(
                r"\b(?:minimi[sz]e|reduce|fewer|less|compact|together|back[- ]to[- ]back|without)\b", low)):
            item.constraints.append(Constraint(type="minimize_gaps", hard=False, text=text, groups=groups, weight=8))
            return True
        # daily load: "Dr. Rao should teach at most 3 periods a day"
        m = MAXRUN_RE.search(low)
        if m and DAILY_RE.search(low) and LOAD_RE.search(low) and not LAB_RE.search(low) and not CONSEC_RE.search(low):
            teacher = (ents.teachers or [None])[0]
            item.constraints.append(Constraint(type="max_daily_load", hard=decide(False, structural=True), text=text,
                                               groups=groups, teacher=teacher, value=max(1, _num_value(m.group(1)) or 1)))
            return True
        # room preference: "Java Lab should be in Lab 2" (no day / time / negation)
        if ents.subjects and ents.rooms and ROOMPREF_RE.search(low) and not neg and not ents.days \
                and not ents.times and not ents.parts and intent != "move":
            hard = bool(HARD_RE.search(low)) and not SOFT_RE.search(low)
            for s in ents.subjects:
                item.constraints.append(Constraint(type="room_pref", hard=hard, text=text, groups=groups,
                                                   subject=s, room=ents.rooms[0], weight=8))
            return True
        return False

    # ---- public API -----------------------------------------------------------
    def parse_clause(self, clause: str, ctx: ParseContext) -> ParsedInstruction:
        cleaned, typo_notes = fix_typos(normalize(clause))
        item = ParsedInstruction(text=clause)
        item.notes.extend(typo_notes)
        ents, _rest = self.extract_entities(cleaned, ctx.cfg)
        item.entities = ents
        intent, conf, scores = self.classify(cleaned, ents)
        if intent == "unknown" and (ents.days or ents.times or ents.parts or ents.subjects
                                    or GAPS_RE.search(cleaned.lower()) or TRAVEL_RE.search(cleaned.lower())):
            intent, conf = "modify", 0.4
        item.intent, item.confidence, item.scores = intent, conf, scores
        if intent == "unknown" and self.llm_fn:
            from .llm_fallback import parse_with_llm
            llm_item = parse_with_llm(clause, self.store, self.llm_fn)
            if llm_item is not None:
                llm_item.notes = item.notes + llm_item.notes
                return llm_item
        if intent == "unknown":
            item.clarification = ("I didn't understand that. Try something like 'Keep Friday afternoon free' "
                                  "or 'Schedule DBMS on Monday morning'.")
            return item
        if ents.unknown_teachers and intent not in ("create", "reset", "regenerate"):
            name = ents.unknown_teachers[0]
            surname = name.split()[1].lower()
            close = difflib.get_close_matches(
                surname, [re.sub(r"^(dr|prof|mr|ms|mrs)\.?\s+", "", t, flags=re.I).lower() for t in self.store.teachers["name"]],
                n=1, cutoff=0.6)
            if close:
                match = next(t for t in self.store.teachers["name"] if close[0] in t.lower())
                item.clarification = f"I don't know a teacher called '{name}'. Did you mean {match}?"
            else:
                item.clarification = (f"I don't know a teacher called '{name}'. Known teachers: "
                                      f"{', '.join(self.store.teachers['name'])}. Add them under 'Manage Academic Data' first.")
            return item
        self.build(clause, cleaned.lower(), intent, ents, ctx, item)
        return item

    def parse(self, text: str, ctx: ParseContext) -> ParsedRequest:
        req = ParsedRequest(text=text)
        local = ParseContext(ctx.store, ctx.cfg, ctx.active_groups, ctx.last_subject, ctx.last_teacher)
        for clause in split_clauses(normalize(text)):
            item = self.parse_clause(clause, local)
            req.items.append(item)
            if item.entities.subjects:
                local.last_subject = item.entities.subjects[-1]
            if item.entities.teachers:
                local.last_teacher = item.entities.teachers[-1]
        if not req.items:
            req.items.append(ParsedInstruction(text=text, clarification="Please type an instruction."))
        return req
