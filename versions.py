"""Save, load and compare named timetable versions (plain JSON files in data/versions/)."""
from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from models import Config, Placement
from nlp.constraints import Constraint

VERSION_DIR = Path(__file__).parent / "data" / "versions"


def _slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9 _.-]+", "", name).strip().replace(" ", "_")
    if not s:
        raise ValueError("Give the version a name.")
    return s[:60]


def snapshot(asst) -> dict:
    if not asst.timetable:
        raise ValueError("There is no timetable to save yet.")
    return {
        "saved_at": time.strftime("%Y-%m-%d %H:%M"),
        "groups": asst.scope_groups, "num_subjects": asst.num_subjects,
        "config": asdict(asst.cfg),
        "constraints": [asdict(c) for c in asst.constraints],
        "placements": [asdict(p) for p in asst.timetable],
    }


def _placement(d: dict) -> Placement:
    d = dict(d)
    for k in ("groups", "occupies"):
        d[k] = tuple(d.get(k) or ())
    return Placement(**d)


def _constraint(d: dict) -> Constraint:
    d = dict(d)
    if d.get("window"):
        d["window"] = tuple(d["window"])
    return Constraint(**d)


def save_version(asst, name: str, folder: Path = VERSION_DIR) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{_slug(name)}.json"
    path.write_text(json.dumps(snapshot(asst), indent=1))
    return path


def list_versions(folder: Path = VERSION_DIR) -> list:
    if not folder.exists():
        return []
    out = []
    for p in sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            d = json.loads(p.read_text())
            out.append({"name": p.stem, "saved_at": d.get("saved_at", ""), "sessions": len(d["placements"]),
                        "groups": ", ".join(d.get("groups") or [])})
        except Exception:
            continue
    return out


def load_data(name: str, folder: Path = VERSION_DIR) -> dict:
    path = folder / f"{_slug(name)}.json"
    if not path.exists():
        raise FileNotFoundError(f"No saved version called '{name}'.")
    return json.loads(path.read_text())


def restore_version(asst, name: str, folder: Path = VERSION_DIR):
    """Make a saved version the active timetable (the current one goes on the undo stack)."""
    d = load_data(name, folder)
    asst.history.append(asst._snapshot())
    asst.cfg = Config(**d["config"])
    asst.scope_groups, asst.num_subjects = d["groups"], d["num_subjects"]
    asst.constraints = [_constraint(c) for c in d["constraints"]]
    asst.timetable = [_placement(p) for p in d["placements"]]
    asst.last_result, asst.rejected = None, []


def delete_version(name: str, folder: Path = VERSION_DIR) -> bool:
    path = folder / f"{_slug(name)}.json"
    if path.exists():
        path.unlink()
        return True
    return False


def compare(a: dict, b: dict) -> dict:
    """Compare two saved snapshots.  Sessions are matched per (attendees, subject) in time order."""
    cfg = Config(**b["config"])

    def bucket(d):
        out = defaultdict(list)
        for p in d["placements"]:
            out[(p.get("label") or p["group"], p["subject"])].append(p)
        for v in out.values():
            v.sort(key=lambda p: (cfg.days.index(p["day"]) if p["day"] in cfg.days else 99, p["start"]))
        return out
    A, B = bucket(a), bucket(b)
    moved, room_changes, added, removed = [], [], [], []
    for key in sorted(set(A) | set(B)):
        la, lb = A.get(key, []), B.get(key, [])
        # drop sessions that are identical in both before pairing the rest
        sa = [p for p in la if (p["day"], p["start"], p["room"]) not in {(q["day"], q["start"], q["room"]) for q in lb}]
        sb = [q for q in lb if (q["day"], q["start"], q["room"]) not in {(p["day"], p["start"], p["room"]) for p in la}]
        for p, q in zip(sa, sb):
            when_a = f"{p['day'][:3]} {cfg.slot_label(p['start']).split('-')[0]}"
            when_b = f"{q['day'][:3]} {cfg.slot_label(q['start']).split('-')[0]}"
            if (p["day"], p["start"]) != (q["day"], q["start"]):
                moved.append(f"{key[1]} ({key[0]}): {when_a} -> {when_b}")
            if p["room"] != q["room"]:
                room_changes.append(f"{key[1]} ({key[0]}): {p['room']} -> {q['room']}")
        removed += [f"{key[1]} ({key[0]}) {p['day'][:3]}" for p in sa[len(sb):]]
        added += [f"{key[1]} ({key[0]}) {q['day'][:3]}" for q in sb[len(sa):]]
    same = not (moved or room_changes or added or removed)
    return {"moved": moved, "room_changes": room_changes, "added": added, "removed": removed, "identical": same}
