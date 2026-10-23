"""Conversation engine: keeps session context and ties NLP + solver together.

    text --Parser--> ParsedRequest --apply()--> constraints / view / config
                                                   |
                                         CP-SAT solve (stable w.r.t. previous timetable)
                                                   |
                                         Response (message, timetable, explanation)
"""
from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass, field

from models import DataStore, Placement
from nlp.constraints import Constraint, check_constraint
from nlp.parser import Parser, ParseContext, ParsedRequest
from solver.scheduler import solve, scope_warnings, SolveResult


@dataclass
class Response:
    status: str                    # ok | infeasible | clarify | view | info | error
    message: str
    parsed: ParsedRequest | None = None
    result: SolveResult | None = None
    changes: list = field(default_factory=list)


class TimetableAssistant:
    def __init__(self, store: DataStore, time_limit: float = 10.0, llm_fn=None, intent_mode: str = "hybrid"):
        self.store = store
        self.time_limit = time_limit
        self.llm_fn = llm_fn
        self.intent_mode = intent_mode
        self._parser = None
        self._sig = None
        self.reset()

    # ---- state ------------------------------------------------------------
    def reset(self):
        self.cfg = self.store.config.copy()
        self.scope_groups = None
        self.num_subjects = None
        self.constraints: list = []
        self.timetable: list | None = None
        self.last_result: SolveResult | None = None
        self.view: dict = {}
        self.last_subject = None
        self.last_teacher = None
        self.rejected: list = []
        self.history: list = []
        self.seed = 0

    def _snapshot(self):
        return copy.deepcopy((self.cfg, self.scope_groups, self.num_subjects, self.constraints,
                              self.timetable, self.last_result, self.rejected))

    def undo(self) -> Response:
        if not self.history:
            return Response("info", "Nothing to undo.")
        (self.cfg, self.scope_groups, self.num_subjects, self.constraints,
         self.timetable, self.last_result, self.rejected) = self.history.pop()
        return Response("ok", "Reverted to the previous timetable.")

    @property
    def parser(self) -> Parser:
        sig = (self.store.signature(), self.intent_mode, id(self.llm_fn))
        if self._parser is None or sig != self._sig:
            self._parser, self._sig = Parser(self.store, self.intent_mode, self.llm_fn), sig
        return self._parser

    def context(self) -> ParseContext:
        return ParseContext(self.store, self.cfg, self.scope_groups, self.last_subject, self.last_teacher)

    def remove_constraint(self, cid: str):
        self.constraints = [c for c in self.constraints if c.id != cid]

    # ---- public API -------------------------------------------------------------
    def interpret(self, text: str) -> ParsedRequest:
        return self.parser.parse(text, self.context())

    def handle(self, text: str) -> Response:
        return self.apply(self.interpret(text))

    def apply_rejected_as_soft(self) -> Response:
        if not self.rejected:
            return Response("info", "There is nothing to relax.")
        for c in self.rejected:
            c.hard = False
        req = ParsedRequest(text="(relax)")
        self.constraints.extend(self.rejected)
        added = self.rejected
        self.rejected = []
        return self._solve(added, req, note="Applied as soft preferences instead.")

    # ---- core -------------------------------------------------------------------
    def apply(self, req: ParsedRequest) -> Response:
        if req.clarification:
            return Response("clarify", req.clarification, parsed=req)
        if req.needs_confirmation:
            lines = ["The language-model fallback interpreted this as:"]
            lines += [f"- {c.describe()}" for c in req.constraints] or [f"- intent: {i.intent}" for i in req.items]
            lines.append("Nothing was applied. Confirm to continue.")
            return Response("confirm", "\n".join(lines), parsed=req)
        self.history.append(self._snapshot())
        self.history = self.history[-20:]

        need_solve, regenerate = False, False
        added: list = []
        notes: list = []
        view_msgs: list = []
        for item in req.items:
            notes.extend(item.notes)
            if item.entities.subjects:
                self.last_subject = item.entities.subjects[-1]
            if item.entities.teachers:
                self.last_teacher = item.entities.teachers[-1]

            if item.intent == "reset":
                self.reset()
                return Response("info", "Cleared everything. Tell me what to create, e.g. "
                                        "'Create a timetable for CSE 5th semester'.", parsed=req)
            if item.intent == "create":
                sc = item.scope or {}
                groups = sc.get("groups") or self.store.group_names()
                if self.timetable is not None:
                    self.constraints = []
                self.scope_groups = groups
                self.num_subjects = sc.get("num_subjects")
                self.cfg = self.store.config.copy()
                self.timetable, self.view, self.rejected = None, {}, []
                need_solve = True
                notes.append(f"Creating a timetable for {', '.join(groups)}"
                             + (f" using the first {self.num_subjects} theory subjects of each group (plus their labs)." if self.num_subjects else "."))
            elif item.intent == "regenerate":
                regenerate, need_solve = True, True
            elif item.intent == "view":
                self.view = {} if (item.view or {}).get("clear") else {**self.view, **(item.view or {})}
                view_msgs.append("Showing: " + (self._view_text() or "the full timetable") + ".")
            elif item.intent in ("modify", "move"):
                for ch in item.config_changes:
                    self._apply_config(ch)
                    need_solve = True
                for c in item.constraints:
                    added.append(c)
                    need_solve = True

        # a "move X to Day" with no source day that is already true -> explain instead of silently doing nothing
        skip = []
        for c in added:
            if c.type == "require" and c.origin == "move" and self.timetable:
                ok, _ = check_constraint(c, self.timetable, self.cfg)
                if ok:
                    where = [f"{self.cfg.slot_label(p.start)} ({p.group})" for p in self.timetable if c.matches_require(p, self.cfg)]
                    notes.append(f"{c.subject or c.teacher} already has a session on {'/'.join(c.days)}: {', '.join(where[:3])}. "
                                 f"To relocate one, say which day to move it from, e.g. "
                                 f"'Move {c.subject or c.teacher} from Monday to {c.days[0]}'.")
                    skip.append(c)
        added = [c for c in added if c not in skip]
        if skip and not added and not regenerate and not need_solve_other(req):
            return Response("info", "\n".join(notes), parsed=req)

        if self.scope_groups is None and (need_solve or added):
            self.scope_groups = self.store.group_names()
            notes.append("No class group was chosen, so I'm scheduling all groups.")

        if need_solve:
            self.constraints.extend(added)
            return self._solve(added, req, regenerate=regenerate, notes=notes)

        msg = " ".join(view_msgs) or "Done."
        return Response("view" if view_msgs else "info", msg + ("\n" + "\n".join(notes) if notes else ""), parsed=req)

    # ---- helpers ------------------------------------------------------------------
    # ---- manual edits -----------------------------------------------------------------
    def preview_move(self, index: int, day: str, start: int, room: str | None = None):
        """What would happen if one session were moved by hand?  Returns (new_timetable, messages, bad_indexes)."""
        from manual import move_session, find_conflicts
        new = move_session(self.timetable, index, day, start, room)
        msgs, bad = find_conflicts(self.store, self.cfg, new, self.constraints, self.scope_groups, self.num_subjects)
        return new, msgs, bad

    def apply_move(self, index: int, day: str, start: int, room: str | None = None,
                   pin: bool = True, force: bool = False) -> Response:
        new, msgs, _bad = self.preview_move(index, day, start, room)
        if msgs and not force:
            return Response("infeasible", "Not applied - it would break:\n- " + "\n- ".join(msgs))
        p = self.timetable[index]
        self.history.append(self._snapshot())
        self.timetable, self.last_result = new, None
        text = f"Moved {p.subject} ({p.label or p.group}) to {day} {self.cfg.slot_label(start)}."
        if pin:
            from nlp.constraints import Constraint
            self.constraints.append(Constraint(
                type="require", hard=True, text=f"manual move of {p.subject}", groups=[p.group], subject=p.subject,
                days=[day], start_at=self.cfg.slot_start(start), origin="manual"))
            text += " Pinned there, so later changes keep it (remove the rule under Constraints & Settings to free it)."
        if msgs:
            text += " WARNING, forced despite: " + "; ".join(msgs)
        return Response("ok", text)

    def _apply_config(self, ch: dict):
        if "lunch" in ch:
            self._change_lunch(ch["lunch"])
        if "travel_gap" in ch:
            self.cfg.travel_gap = bool(ch["travel_gap"])
        if "day_slots" in ch:
            day, n = ch["day_slots"]
            ds = dict(self.cfg.day_slots)
            if n is None:
                ds.pop(day, None)
            else:
                ds[day] = [0, int(n)]
            self.cfg.day_slots = ds

    def _change_lunch(self, delta: int):
        ls = sorted(self.cfg.lunch_slots)
        if delta > 0:
            nxt = ls[-1] + 1
            if nxt < self.cfg.num_slots:
                ls.append(nxt)
            elif ls[0] > 0:
                ls.insert(0, ls[0] - 1)
        elif len(ls) > 1:
            ls.pop()
        self.cfg.lunch_slots = sorted(ls)

    def _view_text(self) -> str:
        bits = []
        for k, v in self.view.items():
            bits.append(f"{k}: {', '.join(v) if isinstance(v, list) else v}")
        return "; ".join(bits)

    def _solve(self, added, req, regenerate=False, notes=None, note=None) -> Response:
        notes = list(notes or [])
        if note:
            notes.append(note)
        previous = None if regenerate or self.timetable is None else self.timetable
        if regenerate:
            self.seed += 1
        warns = scope_warnings(build_sessions_for(self), added)
        if warns:
            bad = [w[0] for w in warns]
            self.constraints = [c for c in self.constraints if c not in bad]
            msg = "I couldn't apply: " + "; ".join(f"'{c.describe()}' ({why})" for c, why in warns) + "."
            self.history.pop() if self.history else None
            return Response("clarify", msg, parsed=req)

        res = solve(self.store, self.cfg, self.scope_groups, self.constraints, self.num_subjects,
                    previous=previous, time_limit=self.time_limit, seed=self.seed)
        if res.ok:
            changes = diff(previous, res.placements, self.cfg) if previous else []
            self.timetable, self.last_result, self.rejected = res.placements, res, []
            lines = [f"Timetable ready ({res.status.lower()}, {res.seconds:.1f}s, {len(res.placements)} sessions)."]
            lines += notes
            if added:
                lines.append("Applied: " + "; ".join(c.describe() for c in added))
            miss = [r for r in res.soft_report if not r["satisfied"]]
            hit = [r for r in res.soft_report if r["satisfied"]]
            if hit:
                lines.append("Preferences met: " + "; ".join(r["constraint"].describe() for r in hit))
            if miss:
                lines.append("Preferences I could not meet: "
                             + "; ".join(f"{r['constraint'].describe()} ({r['detail']})" for r in miss))
            if changes:
                lines.append("Changes: " + "; ".join(changes[:8]) + (" ..." if len(changes) > 8 else ""))
            elif previous and not regenerate:
                lines.append("Nothing else had to move.")
            return Response("ok", "\n".join(lines), parsed=req, result=res, changes=changes)

        # failure: roll back the constraints that were just added; keep the old timetable
        self.rejected = [c for c in added if c in self.constraints]
        self.constraints = [c for c in self.constraints if c not in added]
        if res.status == "INFEASIBLE" and self.history:
            (self.cfg, *_rest) = self.history[-1]      # roll back lunch edits too
        self.last_result = res if self.timetable is None else self.last_result
        lines = ["I could not build a valid timetable with that request, so nothing was changed."]
        if res.conflict:
            lines.append("Conflicting requirements: " + "; ".join(c.describe() for c in res.conflict))
        lines += [f"- {i}" for i in res.issues]
        lines += [f"- {s}" for s in res.suggestions]
        return Response("infeasible", "\n".join(lines), parsed=req, result=res)


def need_solve_other(req) -> bool:
    return any(i.config_changes or i.intent in ("create", "regenerate") for i in req.items)


def build_sessions_for(assistant: "TimetableAssistant"):
    from solver.scheduler import build_sessions
    return build_sessions(assistant.store, assistant.scope_groups, assistant.num_subjects)


def diff(old: list, new: list, cfg) -> list:
    """Human-readable list of sessions that moved."""
    def bucket(pl):
        d = defaultdict(set)
        for p in pl:
            d[(p.group, p.subject)].add((cfg.days.index(p.day), p.start, p.day))
        return d
    a, b = bucket(old), bucket(new)
    out = []
    for key in sorted(b):
        gone = sorted(a.get(key, set()) - b[key])
        came = sorted(b[key] - a.get(key, set()))
        for x, y in zip(gone, came):
            out.append(f"{key[1]} ({key[0]}): {x[2][:3]} {cfg.slot_label(x[1]).split('-')[0]} -> "
                       f"{y[2][:3]} {cfg.slot_label(y[1]).split('-')[0]}")
    return out


# ---- timetable views ---------------------------------------------------------------
def filter_placements(placements: list, view: dict) -> list:
    out = []
    for p in placements or []:
        if view.get("groups") and not (set(view["groups"]) & set(p.all_groups)):
            continue
        if view.get("teacher") and p.teacher != view["teacher"]:
            continue
        if view.get("subject") and p.subject != view["subject"]:
            continue
        if view.get("room") and p.room != view["room"]:
            continue
        if view.get("kind") and p.kind != view["kind"]:
            continue
        if view.get("days") and p.day not in view["days"]:
            continue
        out.append(p)
    return out
