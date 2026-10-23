"""Import academic data from CSV / Excel and check it for mistakes before it is used."""
from __future__ import annotations

import io

import pandas as pd

from models import (DataStore, SUBJECT_COLS, TEACHER_COLS, ROOM_COLS, GROUP_COLS, ALL_DAYS, Config)

KINDS = {"subjects": SUBJECT_COLS, "teachers": TEACHER_COLS, "rooms": ROOM_COLS, "groups": GROUP_COLS}
REQUIRED = {"subjects": ["name", "group", "teacher"], "teachers": ["name"], "rooms": ["name"], "groups": ["name"]}


def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df.dropna(how="all").fillna("")


def read_table(file, kind: str, filename: str = "") -> tuple:
    """Read one table from a CSV/XLSX file-like object.  Returns (DataFrame, problems)."""
    problems = []
    name = (filename or getattr(file, "name", "")).lower()
    data = file.read() if hasattr(file, "read") else file
    try:
        if name.endswith((".xlsx", ".xlsm")):
            xl = pd.ExcelFile(io.BytesIO(data))
            sheet = next((s for s in xl.sheet_names if s.strip().lower() == kind), xl.sheet_names[0])
            df = xl.parse(sheet, dtype=str)
        else:
            df = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)
    except Exception as e:
        return pd.DataFrame(columns=KINDS[kind]), [f"Could not read the file: {e}"]
    df = _clean_cols(df)
    missing = [c for c in REQUIRED[kind] if c not in df.columns]
    if missing:
        problems.append(f"Missing required column(s) for {kind}: {', '.join(missing)}. "
                        f"Expected columns: {', '.join(KINDS[kind])}.")
    extra = [c for c in df.columns if c not in KINDS[kind]]
    if extra:
        problems.append(f"Ignoring unknown column(s): {', '.join(extra)}.")
    for c in KINDS[kind]:
        if c not in df.columns:
            df[c] = ""
    return df[KINDS[kind]].reset_index(drop=True), problems


def read_workbook(file) -> tuple:
    """An Excel workbook with sheets named subjects / teachers / rooms / groups -> dict of tables."""
    data = file.read() if hasattr(file, "read") else file
    xl = pd.ExcelFile(io.BytesIO(data))
    sheets = {s.strip().lower(): s for s in xl.sheet_names}
    tables, problems = {}, []
    for kind in KINDS:
        if kind not in sheets:
            problems.append(f"The workbook has no sheet called '{kind}'.")
            continue
        df, p = read_table(data, kind, "book.xlsx")
        tables[kind] = df
        problems += [f"[{kind}] {x}" for x in p]
    return tables, problems


def check_tables(subjects, teachers, rooms, groups) -> list:
    """Cross-table sanity checks.  Returns human-readable issues (empty = fine)."""
    issues = []
    for label, df in (("subjects", subjects), ("teachers", teachers), ("rooms", rooms), ("groups", groups)):
        names = df["name"].astype(str).str.strip() if "name" in df else []
        if label != "subjects":
            dups = sorted({n for n in names if n and list(names).count(n) > 1})
            if dups:
                issues.append(f"Duplicate {label} name(s): {', '.join(dups)}.")
    t_names, g_names, r_names = set(teachers["name"]), set(groups["name"]), set(rooms["name"])
    for _, r in subjects.iterrows():
        row = f"Subject '{r['name']}' ({r['group']})"
        if r["group"] not in g_names:
            issues.append(f"{row}: group is not in the groups table.")
        if r["teacher"] not in t_names:
            issues.append(f"{row}: teacher '{r['teacher']}' is not in the teachers table.")
        if r["room_pref"] and r["room_pref"] not in r_names:
            issues.append(f"{row}: preferred room '{r['room_pref']}' does not exist.")
        for o in [x.strip() for x in str(r["shared_with"]).replace(";", ",").split(",") if x.strip()]:
            if o not in g_names:
                issues.append(f"{row}: shared_with group '{o}' does not exist.")
        for col, lo in (("sessions_per_week", 1), ("duration", 1)):
            v = pd.to_numeric(r[col], errors="coerce")
            if pd.isna(v) or v < lo:
                issues.append(f"{row}: {col} must be a number >= {lo} (got '{r[col]}').")
        if str(r["type"]).strip().lower() not in ("theory", "lab", ""):
            issues.append(f"{row}: type must be 'theory' or 'lab' (got '{r['type']}').")
    for _, t in teachers.iterrows():
        bad = [d for d in str(t["unavailable_days"]).replace(";", ",").split(",")
               if d.strip() and d.strip().capitalize() not in ALL_DAYS]
        if bad:
            issues.append(f"Teacher '{t['name']}': unknown day(s) {', '.join(bad)} in unavailable_days.")
    if len(rooms) and not (rooms["type"].astype(str).str.lower() == "lab").any() \
            and (subjects["type"].astype(str).str.lower() == "lab").any():
        issues.append("There are lab subjects but no room of type 'lab'.")
    return issues


def merge_frames(tables: dict, base: DataStore, mode: str = "replace") -> dict:
    """Combine imported tables with the current store WITHOUT normalising them, so check_tables sees the raw
    values (DataStore silently turns an unknown type into 'theory', which would hide a typo)."""
    new = {}
    for kind in KINDS:
        cur = getattr(base, kind)
        imp = tables.get(kind)
        if imp is None:
            new[kind] = cur
        elif mode == "append":
            key = ["name", "group"] if kind == "subjects" else ["name"]
            merged = pd.concat([cur, imp], ignore_index=True)
            new[kind] = merged.drop_duplicates(subset=key, keep="last").reset_index(drop=True)
        else:
            new[kind] = imp
    return new


def build_store(tables: dict, base: DataStore, mode: str = "replace") -> DataStore:
    new = merge_frames(tables, base, mode)
    return DataStore(new["subjects"], new["teachers"], new["rooms"], new["groups"], base.config.copy())


def template_workbook(store: DataStore) -> bytes:
    """An Excel file with the four sheets filled with the current data - a template to edit and re-import."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for kind in KINDS:
            getattr(store, kind).to_excel(xw, sheet_name=kind, index=False)
    return buf.getvalue()
