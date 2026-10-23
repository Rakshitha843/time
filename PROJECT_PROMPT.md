# Project Prompt: NLP-Based Intelligent Timetable Generation and Optimization System

Use this prompt to continue the project with any coding assistant. It is your original prompt, completed (the original stopped mid-sentence at "Manage Academic Data") and organised into phases so the scope stays manageable.

---

Act as a senior Python developer, NLP engineer and software architect. Extend the existing project in this folder (`timetable_nlp/`): **NLP-Based Intelligent Timetable Generation and Optimization System**. Keep it beginner-friendly: small modules, docstrings, no unexplained magic.

## Goal
A Streamlit web app where administrators, faculty and students create, view and modify academic timetables by typing plain-English instructions. The text is converted to structured constraints (hard vs soft), solved with Google OR-Tools CP-SAT, and shown as a timetable grid.

## Already built (Phase 1 - do not rewrite, extend)
- `nlp/parser.py`: spaCy `PhraseMatcher` gazetteers (subjects, teachers, rooms, days, groups), regex for times / semesters / sections / periods, cue-based intent scoring, hard-vs-soft cue detection, typo correction, pronoun resolution, multi-clause splitting.
- `nlp/constraints.py`: `Constraint` (forbid / confine / require / max_labs_per_day) with an independent checker.
- `solver/scheduler.py`: CP-SAT model, assumption-based unsat cores, core minimisation, relaxation suggestions, independent `validate()`.
- `engine.py`: session context, incremental edits (stability objective), undo, rollback on infeasible requests.
- `app.py`: Dashboard, Assistant (chat + grid), Manage Academic Data, Constraints & Settings, NLP Lab, Export.
- `eval/`: labelled instruction set and metrics script; `tests/`: pytest suite.

## Phase 2 - NLP depth (largely done; see below for what remains)

Done: trained intent classifier + hybrid, unknown titled-teacher detection with suggestions, vague phrases (early / late / in a row), `max_consecutive`, schema-validated LLM fallback hook with confirmation, intent evaluation with confusion matrix. Remaining: collect real user instructions and retrain; statistical NER for names without titles and for unknown subjects; test the LLM fallback against a real provider.

Original Phase 2 list:
1. Add a trainable intent classifier (scikit-learn TF-IDF + logistic regression, or a small transformer) trained on an expanded labelled set (300+ sentences, including paraphrases and typos). Combine with the rule cues; report both accuracies.
2. Add a statistical NER fallback (spaCy `en_core_web_sm` or a fine-tuned model) for teacher names not in the database, and a clarifying question when an entity is unknown or ambiguous.
3. Support relative / vague expressions ("after lunch", "early", "late", "first thing in the morning", "not too many in a row").
4. Add an optional LLM fallback for sentences the rules cannot parse, constrained to emit the same JSON constraint schema, always shown to the user for confirmation.
5. Extend `eval/` with entity-level F1 per type and a confusion matrix; keep results in the README.

## Phase 3 - Scheduling depth (DONE: shared/elective/batch sessions, teacher limits, gaps, room preferences, travel gaps, per-day slot templates; see README)
1. Per-teacher daily load limits, maximum consecutive classes, and gap minimisation for student groups as soft objectives.
2. Electives and shared classes (one session attended by several groups).
3. Lab batches (split a section into batches that rotate through lab slots).
4. Room preferences per subject; building/travel-time constraints.
5. Time-slot templates per day (e.g. short Saturday) instead of one grid for all days.

## Phase 4 - Product polish (DONE except true drag-and-drop; see README)
1. Drag-and-drop manual edits with live conflict highlighting.
2. Save / load named timetable versions and compare two versions.
3. Import academic data from CSV/Excel; export timetables to PDF and Excel per group, per teacher and per room.
4. Role views: student (my section), teacher (my week), admin (everything).
5. Deployment notes (Streamlit Community Cloud or Docker).

## Non-negotiable rules
- Hard constraints are never violated. Every returned timetable passes `validate()`.
- Impossible requests are explained: name the conflicting constraints and suggest relaxations. Never silently drop a constraint.
- Follow-up edits update the existing timetable with minimal movement instead of regenerating from scratch.
- Explain which NLP method is used for each feature and why.
- Every new feature ships with tests and, for NLP changes, updated evaluation numbers.

## Deliverables per phase
Working code, updated README, tests passing (`pytest`), a short demo script of example instructions, and a changelog entry.
