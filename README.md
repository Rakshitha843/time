# NLP-Based Intelligent Timetable Generation and Optimization System

Type timetable instructions in plain English. They are parsed into **hard** and **soft** constraints and solved with **Google OR-Tools CP-SAT**. This version includes Phase 1 (core system), Phase 2 (NLP depth: trained intent classifier, unknown-name handling, vague phrases, LLM fallback hook) Phase 3 (scheduling depth: shared classes, electives, lab batches, teacher limits, gap minimisation, room preferences, building travel gaps, short days) and Phase 4 (product polish: manual edits with live conflicts, saved versions + compare, CSV/Excel import, Excel/PDF export, role views, Docker/cloud deployment). `PROJECT_PROMPT.md` lists the phases.

```
"Keep Friday afternoon free."  ->  forbid(days=[Friday], window=12:00-24:00, hard)
"Schedule DBMS on Monday morning."  ->  require(subject=DBMS, day=Monday, before lunch, soft)
"Move the Java class from Monday to Wednesday."  ->  forbid(Java, Monday) + require(Java, Wednesday), both hard
```

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py        # web app
pytest                      # ~95 tests (the solver tests take several minutes)
python eval/evaluate.py     # NLP evaluation (entities, constraints, intent: rules vs model vs hybrid)
python ml/make_dataset.py   # regenerate the intent training set (optional)
```

No spaCy model download is needed: the project uses spaCy's tokenizer and matchers on a blank English pipeline. The intent classifier (scikit-learn) trains itself in about a second on first use, so there is no model file to download or keep in sync.

## Try this conversation (Assistant page)

1. `Create a timetable for CSE 5th semester with six subjects.`
2. `Schedule DBMS on Monday morning. Keep Friday afternoon free.`
3. `Don't schedule classes after 4 PM.`
4. `Avoid scheduling two lab sessions on the same day.`
5. `Move the Java class from Monday to Wednesday.`
6. `Give us a longer lunch break.`
7. `Show Dr. Rao's timetable` then `Avoid the first period for this teacher.`
8. `Show only laboratory sessions.`
9. `Keep Monday free. Keep Tuesday free. Keep Wednesday free. Keep Thursday free.` - impossible; you get the conflicting constraints, and the "Use as preference" button re-applies them as soft preferences.

The **NLP Lab** page shows exactly how any sentence is understood (intent, entities, constraints) without applying it. Tick *Confirm parsed constraints before applying* in the sidebar to review every parse first.

## Phase 3: scheduling depth

Load the demo with **Manage Academic Data -> Load advanced demo data** (two sections, two buildings, six-day week with a half-day Saturday), then try:

1. `Create a timetable for CSE 5th semester.`
2. `Try to avoid gaps between classes.`  -> soft `minimize_gaps`
3. `Dr. Rao must teach at most 2 periods a day.`  -> hard `max_daily_load`
4. `Java Lab should be in Lab 1.`  -> soft `room_pref` (overrides the room in the subject table)
5. `Make Friday a half day.` / `Saturday ends at 11 AM.` / `Make Friday a full day.`  -> per-day open periods
6. `Allow travel time when changing buildings.`  -> hard `travel_gap` (a free period between buildings, for groups and teachers)

How the academic-data columns work (all editable in the UI):

| Column | Effect |
|---|---|
| `shared_with` (subjects) | One class attended by several groups. It is scheduled once and blocks all of them. List it on one group's row only; the other group's duplicate row is ignored |
| `elective_block` | Subjects with the same block name run at the **same time** in different rooms with different teachers; students pick one. Room size is the cohort divided by the number of options |
| `batches` (labs) | Splits a section into `SEC-B1..BN`. Each batch gets its own lab session; the whole section's theory classes block every batch |
| `room_pref` | Soft preference for a room (weight 5) |
| `max_per_day`, `max_consecutive` (teachers) | Hard limits, shown as constraints marked "from academic data" so they appear in conflict explanations |
| `building` (rooms) + travel setting | With the setting on, nobody may have back-to-back classes in different buildings |
| Settings -> *Shorter days* | Only the first N periods are open on that day |

Gap minimisation is always a light built-in preference (weight 1); saying "avoid gaps" raises it. Impossible data (for example a teacher limit lower than their weekly load) is reported with the data-derived constraint named in the explanation and a pointer to where to change it.

**Limits:** electives assume every student takes exactly one option; batches rotate implicitly (the solver decides each batch's slots, there is no explicit rotation table); solving the two-section demo takes about 10-25 s because it keeps improving soft preferences until the time limit, and the quality of soft preferences such as gap minimisation varies from run to run when the limit is short (raise the time limit in the sidebar for tighter days).

## Phase 4: product polish

| Feature | Where | Notes |
|---|---|---|
| Manual edit with live conflict highlighting | **Edit by hand** | Pick a session, a day, a period (and room). Clashes (teacher / room / group), lunch, closing time, room type and capacity, teacher availability and every hard rule are outlined **in red before** you apply. A blocked move is refused unless you tick *Apply even though it breaks rules*. Moves can be pinned (a hard rule is added so later chat edits keep them) and undone |
| Saved versions | **Versions** | Save the current timetable under a name (JSON in `data/versions/`), reopen it, delete it, and compare two: moved sessions, room changes, added / removed |
| Import | **Manage Academic Data -> Import** | Excel workbook (sheets `subjects`, `teachers`, `rooms`, `groups`) or one CSV. Checked before use: missing columns, unknown teacher / group / room, bad numbers, wrong `type`, unknown days. Replace a table or add/overwrite by name. Download the template workbook to see the layout |
| Export | **Export** | CSV, printable HTML, **Excel** (one sheet per group, teacher or room, plus an "All sessions" sheet) and **PDF** (one landscape page per group / teacher / room) |
| Role views | sidebar *View as* | Admin sees everything. Teacher picks their name and Student their section and get a read-only grid and their own exports |
| Deployment | `Dockerfile`, `DEPLOY.md` | Docker, Streamlit Community Cloud, lab PC |

**Honest limits.** Streamlit has no native drag and drop, so "drag" is done with pickers and a live-updating grid (a real drag-and-drop grid needs a custom JavaScript component). The roles are a **view filter, not security**: there is no login, and anyone who can open the app can switch to Admin (see `DEPLOY.md`). Saved versions and data live in files on the server, so on Streamlit Community Cloud they disappear on restart. Manual moves check rules but do not re-optimise the rest of the timetable.

## Architecture

```
app.py (Streamlit) --> engine.py (session context, undo, rollback)
                          |-- nlp/parser.py       text -> intent + entities + constraints
                          |-- nlp/constraints.py  Constraint model + independent checker
                          `-- solver/scheduler.py CP-SAT model, unsat core, validation
manual.py / versions.py / importers.py / exporters.py   Phase 4 features
models.py   academic data (subjects, teachers, rooms, groups) + grid config
views.py    HTML timetable grid, CSV / printable export
eval/       labelled instructions + metrics      tests/  pytest suite
data/       academic_data.json (created on first run, editable in the UI)
```

## NLP method and why

| Task | Method | Why |
|---|---|---|
| Subjects, teachers, rooms, days, groups, parts of day | spaCy `PhraseMatcher` gazetteers built from *your data* (longest match wins; aliases and auto-acronyms) | The vocabulary is closed and known, so this is accurate, explainable and needs no training data |
| Times, semesters, sections, periods, counts | Regular expressions with college-hour heuristics ("at 4" = 4 PM) | Numeric patterns are regular |
| Intent (create / modify / move / view / regenerate / reset) | **Hybrid**: TF-IDF (word + character n-grams) + logistic regression trained on ~660 template-generated sentences, blended 75/25 with the cue-phrase scores; off-topic text is rejected by a domain-cue gate | The classifier generalises to new wording and typos; the rules keep explicit verbs ("regenerate", "move") reliable |
| Unknown names | Titled-name detector ("Dr Patel") checked against your teachers, with a difflib "did you mean" suggestion | Stops a misspelt or new teacher from being silently ignored |
| Vague phrases | "early" / "first thing" -> first period, "late" -> after 4 PM, "not too many in a row" / "back-to-back" -> `max_consecutive` | Common human wording, mapped to concrete constraints |
| Anything else (optional) | LLM fallback: any `call_fn(prompt) -> JSON`; output is schema-validated against your data and shown for confirmation | The model proposes, your code validates, you approve |
| Hard vs soft | Cue lexicon: `must`, `don't`, `free`, `only` -> hard; `prefer`, `try`, `avoid`, `should`, `if possible` -> soft; sensible per-type defaults | Mirrors how people actually word rules vs wishes |
| Robustness | difflib typo correction for day names, clause splitting, pronoun resolution from session context ("this teacher", "it") | Handles messy input and follow-ups |

Entity extraction is still gazetteer + regex (not a trained NER), so a subject or room that is not in your data is not recognised; unknown titled teacher names are caught, other unknown names are not.

## Constraint model

`forbid` (region must stay empty), `confine` (everything in scope must stay inside a region), `require` (at least one matching session), `max_labs_per_day`, `max_consecutive` (at most N periods in a row for a group or a teacher). Each has a scope (groups, subject, teacher, room, lab/theory) and a time region (days, time window, periods). Hard ones are CP-SAT hard rules guarded by assumption literals; soft ones are weighted penalties.

Always-on hard rules: no teacher / room / group double-booking, lab blocks of consecutive periods that never cross lunch, room type and capacity, teacher unavailable days, exact weekly session counts. Built-in soft rules: avoid the same subject twice in a day, balance daily load, avoid the last period, and keep unrelated sessions where they were during edits.

Every solution is re-checked by an independent `validate()` before it is shown.

## Evaluation

Run `python eval/evaluate.py`.

**Intent, on 79 sentences the model was not trained on** (`eval/intent_heldout.jsonl`, `intent_heldout2.jsonl`):

| Method | Accuracy |
|---|---|
| Rules only (Phase 1) | 49.4% |
| Trained classifier only | 87.3% |
| Hybrid (default) | 86.1% |

The trained classifier is the real improvement (+37 points over rules). The hybrid is **not** better than the classifier alone on these sets, it is about the same; it is the default because the rules keep explicit commands reliable and the domain gate handles off-topic text. The script also prints per-class precision / recall / F1 and a confusion matrix. The weakest classes are `view` and `create`; see the matrix for which sentences get confused.

**Entities and constraints, on 51 labelled instructions** (`eval/eval_set.jsonl`): intent, constraint type, hard/soft and day / subject / teacher F1 are all 1.00.

**Read these numbers with care.** I wrote every evaluation sentence myself. The 51-sentence set was written while building the parser and the failures it exposed were fixed, so it is a regression test, not a benchmark. The first intent test set was seen while I chose the blend weight and the off-topic gate, so only the second set (30 sentences) was untouched when the final configuration was fixed. The training data is generated from templates, so real users' phrasing will be harder. For an honest score, have classmates or faculty write 100+ instructions without seeing the parser and add them to `eval/`.

## Using the LLM fallback (optional, not wired to any provider)

```python
from engine import TimetableAssistant
def call_fn(prompt: str) -> str:
    ...                      # call your LLM of choice here and return its text reply
assistant = TimetableAssistant(store, llm_fn=call_fn)
```
It only runs when rules and classifier give up. Anything it returns is validated (unknown subjects, teachers, groups, days, out-of-range times are rejected), shown to the user, and applied only after confirmation. The validation and confirmation flow is tested with fake callables; I have not tested it against a real LLM API.

## Known limitations

- English only; unfamiliar phrasing, deep negations ("anything but...") and ambiguous references may be misread. The parse is always shown so you can catch this.
- "Move X to Day" without a source day only guarantees one session of X that day; to relocate a particular session, name the source day.
- Elective and shared-class sizes are estimates (cohort / options); real enrolment counts are not modelled.
- Travel gaps are a fixed one-period buffer between any two different buildings; no distance matrix.
- Training data for the intent classifier is template-generated; collect real instructions to improve it.
- "Back-to-back" is interpreted as at most 2 periods in a row, "too many in a row" as 3; say a number to override ("no more than 4 in a row").
