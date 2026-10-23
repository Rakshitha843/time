"""Streamlit front end:  streamlit run app.py"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from engine import TimetableAssistant, filter_placements
from models import is_batch_key, DataStore, ALL_DAYS, sample_store, advanced_sample_store
from exporters import VIEWS, entities, select, to_excel, to_pdf
from importers import KINDS, build_store, check_tables, merge_frames, read_table, read_workbook, template_workbook
from versions import compare, delete_version, list_versions, load_data, restore_version, save_version
from views import grid_html, to_dataframe, standalone_html

DATA_PATH = Path(__file__).parent / "data" / "academic_data.json"
ROLE_PAGES = {
    "Admin": ["Dashboard", "Assistant", "Edit by hand", "Versions", "Manage Academic Data",
              "Constraints & Settings", "NLP Lab", "Export"],
    "Teacher": ["My Timetable", "Export"],
    "Student": ["My Timetable", "Export"],
}
EXAMPLES = [
    "Create a timetable for CSE 5th semester with six subjects.",
    "Schedule DBMS on Monday morning.",
    "Don't schedule classes after 4 PM.",
    "Keep Friday afternoon free.",
    "Avoid scheduling two lab sessions on the same day.",
    "Move the Java class from Monday to Wednesday.",
    "Give us a longer lunch break.",
    "Show only laboratory sessions.",
    "Show the timetable for Section A.",
]

st.set_page_config(page_title="NLP Timetable Generator", page_icon="🗓️", layout="wide")


# ---------------------------------------------------------------- state ----
def init_state():
    ss = st.session_state
    if "store" not in ss:
        if DATA_PATH.exists():
            ss.store = DataStore.load(DATA_PATH)
        else:
            ss.store = sample_store()
            DATA_PATH.parent.mkdir(exist_ok=True)
            ss.store.save(DATA_PATH)
    if "assistant" not in ss:
        ss.assistant = TimetableAssistant(ss.store)
    ss.setdefault("chat", [])
    ss.setdefault("pending", None)
    ss.setdefault("n_generated", 0)
    ss.setdefault("queued", None)
    ss.setdefault("page", "Dashboard")
    ss.setdefault("confirm", False)


def goto(page, prompt=None):
    st.session_state.page = page
    if prompt:
        st.session_state.queued = prompt


def run_prompt(prompt: str):
    ss = st.session_state
    asst: TimetableAssistant = ss.assistant
    asst.time_limit = ss.get("time_limit", 10)
    ss.chat.append({"role": "user", "content": prompt})
    req = asst.interpret(prompt)
    needs_confirm = not req.clarification and (req.needs_confirmation or (ss.confirm and any(
        i.intent in ("modify", "move", "create") for i in req.items)))
    if needs_confirm:
        ss.pending = req
        ss.chat.append({"role": "assistant", "content": "Here is how I understood that. Apply it?",
                        "parsed": req.to_dict()})
        return
    apply_request(req)


def apply_request(req):
    ss = st.session_state
    with st.spinner("Solving..."):
        resp = ss.assistant.apply(req)
    if resp.status == "ok" and resp.result is not None:
        ss.n_generated += 1
    ss.chat.append({"role": "assistant", "content": resp.message, "status": resp.status,
                    "parsed": resp.parsed.to_dict() if resp.parsed else None})


def save_store():
    st.session_state.store.save(DATA_PATH)


# ------------------------------------------------------------- sidebar ----
def sidebar():
    ss = st.session_state
    st.sidebar.title("🗓️ NLP Timetable")
    st.sidebar.selectbox("View as", list(ROLE_PAGES), key="role",
                         help="Admin sees everything. Teacher and Student see read-only timetables for one teacher / one section.")
    pages = ROLE_PAGES[ss.role]
    if ss.get("page") not in pages:
        ss.page = pages[0]
    st.sidebar.radio("Go to", pages, key="page", label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.checkbox("Confirm parsed constraints before applying", key="confirm")
    st.sidebar.slider("Solver time limit (s)", 3, 60, 10, key="time_limit")
    mode = st.sidebar.selectbox("Intent engine", ["hybrid", "model", "rules"], index=0,
                                help="hybrid = trained classifier + rule cues; model = classifier only; rules = cue lexicon only")
    ss.assistant.intent_mode = mode
    asst = ss.assistant
    if asst.timetable:
        st.sidebar.success(f"Active timetable: {len(asst.timetable)} sessions\n\n{', '.join(asst.scope_groups or [])}")
    else:
        st.sidebar.info("No timetable yet. Open the Assistant and describe what you need.")


# ---------------------------------------------------------------- pages ----
def page_dashboard():
    ss = st.session_state
    st.title("NLP-Based Intelligent Timetable Generation & Optimization")
    st.caption("Type timetable instructions in plain English. They are parsed into hard and soft constraints and solved with Google OR-Tools CP-SAT.")
    store = ss.store
    c = st.columns(5)
    c[0].metric("Subjects", len(store.subjects))
    c[1].metric("Teachers", len(store.teachers))
    c[2].metric("Classrooms & labs", len(store.rooms))
    c[3].metric("Class groups", len(store.groups))
    c[4].metric("Timetables generated", ss.n_generated)

    st.subheader("Quick actions")
    a, b, d = st.columns(3)
    a.button("▶ Create CSE 5th semester timetable", width="stretch", on_click=goto,
             args=("Assistant", EXAMPLES[0]))
    b.button("💬 Open the assistant", width="stretch", on_click=goto, args=("Assistant",))
    d.button("🗂 Manage academic data", width="stretch", on_click=goto, args=("Manage Academic Data",))

    st.subheader("How it works")
    st.markdown(
        "1. **Understand** - spaCy gazetteer matching + rules find the intent, subjects, teachers, days, times, rooms and groups.\n"
        "2. **Translate** - the sentence becomes structured constraints, each marked *hard* (never broken) or *soft* (optimised).\n"
        "3. **Solve** - CP-SAT builds a conflict-free timetable, staying close to the previous one when you edit.\n"
        "4. **Explain** - if your requests are impossible, you get the conflicting constraints and ways to relax them."
    )
    st.subheader("Try these")
    for e in EXAMPLES[:6]:
        st.code(e, language=None)


def page_assistant():
    ss = st.session_state
    asst: TimetableAssistant = ss.assistant
    st.header("Timetable assistant")

    # queued prompt (from quick actions / example buttons)
    if ss.queued:
        q, ss.queued = ss.queued, None
        run_prompt(q)

    left, right = st.columns([3, 2], gap="large")
    with left:
        if asst.timetable:
            shown = filter_placements(asst.timetable, asst.view)
            if asst.view:
                st.caption("View filter: " + asst._view_text() + "  -  say 'show everything' to clear it.")
            groups = sorted({k for p in shown for k in p.all_groups if not is_batch_key(k)})
            if len(groups) > 1:
                pick = st.segmented_control("Group", ["All"] + groups, default="All", key="grid_group")
                if pick and pick != "All":
                    shown = [p for p in shown if pick in p.all_groups]
            st.markdown(grid_html(shown, asst.cfg), unsafe_allow_html=True)
            soft = asst.last_result.soft_report if asst.last_result else []
            if soft:
                with st.expander("Preferences status"):
                    for r in soft:
                        st.write(("✅ " if r["satisfied"] else "⚠️ ") + r["constraint"].describe()
                                 + (f" - {r['detail']}" if r["detail"] else ""))
        else:
            st.info("No timetable yet. Start with an example below or type your own instruction.")
        cols = st.columns(3)
        for i, ex in enumerate(EXAMPLES[:6]):
            cols[i % 3].button(ex, key=f"ex{i}", width="stretch", on_click=goto, args=("Assistant", ex))

    with right:
        box = st.container(height=520, border=True)
        with box:
            if not ss.chat:
                st.caption("Conversation will appear here.")
            for m in ss.chat:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"].replace("\n", "  \n"))
                    if m.get("parsed"):
                        with st.expander("How I understood this"):
                            for it in m["parsed"]["items"]:
                                st.markdown(f"**Intent:** `{it['intent']}` ({it['confidence']:.0%})")
                                for c in it["constraints"]:
                                    st.write("• " + c["description"])
                                if it["entities"]:
                                    st.json(it["entities"], expanded=False)
        if ss.pending is not None:
            c1, c2 = st.columns(2)
            if c1.button("✅ Apply", width="stretch"):
                req, ss.pending = ss.pending, None
                req.confirmed = True
                apply_request(req)
                st.rerun()
            if c2.button("✖ Cancel", width="stretch"):
                ss.pending = None
                ss.chat.append({"role": "assistant", "content": "Cancelled."})
                st.rerun()
        b1, b2, b3 = st.columns(3)
        if b1.button("↩ Undo", width="stretch"):
            ss.chat.append({"role": "assistant", "content": asst.undo().message})
            st.rerun()
        if asst.rejected and b2.button("Use as preference", width="stretch",
                                       help="Re-apply the rejected request as a soft preference"):
            resp = asst.apply_rejected_as_soft()
            ss.chat.append({"role": "assistant", "content": resp.message})
            st.rerun()
        if b3.button("🗑 Start over", width="stretch"):
            asst.reset()
            ss.chat = []
            st.rerun()
        prompt = st.chat_input("e.g. Keep Friday afternoon free")
        if prompt:
            run_prompt(prompt)
            st.rerun()


def page_data():
    ss = st.session_state
    store: DataStore = ss.store
    st.header("Manage academic data")
    with st.expander("Import from CSV / Excel"):
        st.caption("Excel: one workbook with sheets named subjects, teachers, rooms, groups. CSV: one table at a time. "
                   "Your data is checked before anything is replaced.")
        st.download_button("⬇ Template workbook (your current data)", template_workbook(ss.store), "academic_data_template.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        up = st.file_uploader("File", type=["csv", "xlsx"], key="import_file")
        mode = st.radio("When names already exist", ["replace", "append"], horizontal=True,
                        format_func=lambda m: "Replace the whole table" if m == "replace" else "Add / overwrite by name",
                        key="import_mode")
        if up is not None:
            raw = up.getvalue()
            if up.name.lower().endswith(".csv"):
                kind = st.selectbox("This CSV contains", list(KINDS), key="import_kind")
                df, problems = read_table(raw, kind, up.name)
                tables = {kind: df}
            else:
                tables, problems = read_workbook(raw)
            for pr in problems:
                st.warning(pr)
            if tables:
                raw = merge_frames(tables, ss.store, mode)
                issues = check_tables(raw["subjects"], raw["teachers"], raw["rooms"], raw["groups"])
                for kd, df in tables.items():
                    st.write(f"**{kd}**: {len(df)} row(s)")
                    st.dataframe(df.head(8), hide_index=True, width="stretch")
                if issues:
                    st.error("Problems found - fix the file or accept them knowingly:\n\n" + "\n".join(f"- {x}" for x in issues[:25]))
                ok = st.checkbox("Import anyway", key="import_force") if issues else True
                if st.button("Import", type="primary", disabled=not ok):
                    cand = build_store(tables, ss.store, mode)
                    cand.save(DATA_PATH)
                    ss.store = cand
                    ss.assistant = TimetableAssistant(ss.store)
                    st.success("Imported. The assistant was reset; create a new timetable.")
                    st.rerun()
    st.caption("Add, edit or delete rows. Click **Save** to apply. Aliases (comma-separated) are what the NLP layer recognises, e.g. 'database; dbms'.")
    t1, t2, t3, t4 = st.tabs(["Subjects", "Teachers", "Classrooms & labs", "Class groups"])
    with t1:
        subj = st.data_editor(
            store.subjects, num_rows="dynamic", width="stretch", key="ed_subj",
            column_config={
                "type": st.column_config.SelectboxColumn(options=["theory", "lab"], required=True),
                "group": st.column_config.SelectboxColumn(options=store.group_names() or [""]),
                "teacher": st.column_config.SelectboxColumn(options=list(store.teachers["name"]) or [""]),
                "sessions_per_week": st.column_config.NumberColumn(min_value=1, max_value=10, step=1),
                "duration": st.column_config.NumberColumn("duration (periods)", min_value=1, max_value=4, step=1),
                "batches": st.column_config.NumberColumn(help="Labs only: split the section into this many batches (0 = no split)", min_value=0, max_value=6, step=1),
                "shared_with": st.column_config.TextColumn(help="Other groups that attend this same class, comma separated"),
                "elective_block": st.column_config.TextColumn(help="Subjects with the same block name run at the same time (students choose one)"),
                "room_pref": st.column_config.SelectboxColumn(help="Preferred room (soft)", options=[""] + list(store.rooms["name"])),
            })
    with t2:
        teach = st.data_editor(store.teachers, num_rows="dynamic", width="stretch", key="ed_teach",
                               column_config={"unavailable_days": st.column_config.TextColumn(help="e.g. Friday, Saturday"),
                                              "max_per_day": st.column_config.NumberColumn("max periods/day", help="0 = no limit", min_value=0, max_value=9, step=1),
                                              "max_consecutive": st.column_config.NumberColumn("max in a row", help="0 = no limit", min_value=0, max_value=9, step=1)})
    with t3:
        rooms = st.data_editor(store.rooms, num_rows="dynamic", width="stretch", key="ed_rooms",
                               column_config={"type": st.column_config.SelectboxColumn(options=["theory", "lab"], required=True)})
    with t4:
        groups = st.data_editor(store.groups, num_rows="dynamic", width="stretch", key="ed_groups")
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("💾 Save changes", type="primary"):
        new = DataStore(pd.DataFrame(subj), pd.DataFrame(teach), pd.DataFrame(rooms), pd.DataFrame(groups), store.config)
        ss.store = new
        ss.assistant.store = new
        save_store()
        st.success("Saved. The language model vocabulary was refreshed.")
    if c2.button("Load advanced demo data"):
        ss.store = advanced_sample_store()
        ss.assistant = TimetableAssistant(ss.store)
        save_store()
        st.rerun()
    if c2.button("Reset to sample data"):
        ss.store = sample_store()
        ss.assistant = TimetableAssistant(ss.store)
        save_store()
        st.rerun()


def page_constraints():
    ss = st.session_state
    asst: TimetableAssistant = ss.assistant
    st.header("Constraints & settings")
    st.subheader("Active constraints")
    if not asst.constraints:
        st.caption("None yet. Constraints appear here as you give instructions.")
    for c in asst.constraints:
        a, b = st.columns([6, 1])
        a.write(("🔒 " if c.hard else "💡 ") + c.describe())
        a.caption(f"“{c.text}”")
        if b.button("Remove", key=f"rm{c.id}"):
            asst.remove_constraint(c.id)
            ss.chat.append({"role": "assistant", "content": f"Removed: {c.describe()}. Say 'regenerate' to re-solve."})
            st.rerun()
    st.caption("🔒 hard = never violated · 💡 soft = optimised when possible")

    st.subheader("Timetable grid")
    cfg = ss.store.config
    c1, c2, c3 = st.columns(3)
    days = c1.multiselect("Working days", ALL_DAYS, default=cfg.days)
    start = c2.text_input("First period starts (HH:MM)", cfg.start_time)
    mins = c3.number_input("Period length (minutes)", 30, 180, cfg.slot_minutes, step=5)
    n = c1.number_input("Periods per day (including lunch)", 4, 14, cfg.num_slots)
    lunch = c2.multiselect("Lunch period index (0 = first)", list(range(int(n))), default=[i for i in cfg.lunch_slots if i < n])
    travel = c1.checkbox("Keep a free period when changing buildings", value=cfg.travel_gap,
                         help="Applies to student groups and teachers; needs a 'building' for each room.")
    short = {}
    with st.expander("Shorter days (e.g. half-day Saturday)"):
        for d in [x for x in ALL_DAYS if x in days]:
            cur = cfg.day_slots.get(d, [0, int(n)])[1]
            short[d] = st.number_input(f"{d}: number of open periods", 1, int(n), min(int(cur), int(n)), key=f"open_{d}")
    if c3.button("Save grid settings", type="primary"):
        try:
            h, m = start.split(":")
            assert 0 <= int(h) < 24 and 0 <= int(m) < 60
        except Exception:
            st.error("Start time must look like 09:00")
        else:
            cfg.days = [d for d in ALL_DAYS if d in days] or cfg.days
            cfg.start_time, cfg.slot_minutes, cfg.num_slots, cfg.lunch_slots = start, int(mins), int(n), sorted(lunch)
            cfg.travel_gap = bool(travel)
            cfg.day_slots = {d: [0, int(v)] for d, v in short.items() if int(v) < int(n)}
            save_store()
            st.success("Saved. Applies to the next 'Create a timetable ...'.")


def page_nlp_lab():
    ss = st.session_state
    asst: TimetableAssistant = ss.assistant
    st.header("NLP lab")
    st.caption("See exactly how a sentence is understood. Nothing is applied here.")
    text = st.text_input("Instruction", EXAMPLES[1])
    ex = st.selectbox("...or pick an example", [""] + EXAMPLES)
    text = ex or text
    if text:
        req = asst.interpret(text)
        for it in req.items:
            st.markdown(f"**Clause:** {it.text}")
            c1, c2 = st.columns(2)
            c1.metric("Intent", it.intent, f"{it.confidence:.0%} confidence")
            c2.write("Cue scores:")
            c2.json(it.scores, expanded=False)
            st.write("**Entities**")
            st.json(it.entities.to_dict(), expanded=True)
            if it.constraints:
                st.write("**Constraints**")
                st.dataframe(pd.DataFrame([
                    {"level": "hard" if c.hard else "soft", "type": c.type, "meaning": c.describe()}
                    for c in it.constraints]), width="stretch", hide_index=True)
            if it.view:
                st.write("**View filter**", it.view)
            if it.config_changes:
                st.write("**Config change**", it.config_changes)
            for n in it.notes:
                st.info(n)
            if it.clarification:
                st.warning(it.clarification)
            st.divider()


def _role_filter(asst):
    """For Teacher / Student roles: which view and entity they are allowed to see."""
    role = st.session_state.role
    if role == "Admin":
        return None
    by = "teacher" if role == "Teacher" else "group"
    options = entities(asst.timetable, by)
    label = "Which teacher are you?" if role == "Teacher" else "Which section are you in?"
    pick = st.selectbox(label, options, key="mine_pick")
    return by, pick


def page_mine():
    asst: TimetableAssistant = st.session_state.assistant
    st.header("My timetable")
    if not asst.timetable:
        st.info("No timetable has been published yet. Ask the administrator to generate one.")
        return
    by, pick = _role_filter(asst)
    mine = select(asst.timetable, by, pick)
    st.markdown(grid_html(mine, asst.cfg), unsafe_allow_html=True)
    st.caption(f"{len(mine)} session(s). Read-only view; use Export to download it.")


def page_export():
    ss = st.session_state
    asst: TimetableAssistant = ss.assistant
    st.header("Export")
    if not asst.timetable:
        st.info("Generate a timetable first.")
        return
    flt = _role_filter(asst)
    tt = select(asst.timetable, *flt) if flt else asst.timetable
    df = to_dataframe(tt, asst.cfg)
    st.dataframe(df, width="stretch", hide_index=True)
    c1, c2 = st.columns(2)
    c1.download_button("⬇ CSV", df.to_csv(index=False), "timetable.csv", "text/csv", width="stretch")
    c2.download_button("⬇ Printable HTML", standalone_html(tt, asst.cfg), "timetable.html",
                       "text/html", width="stretch")
    st.subheader("Excel and PDF")
    by_opts = [flt[0]] if flt else list(VIEWS)
    by = st.radio("One sheet / page per", by_opts, horizontal=True, key="exp_by")
    names = entities(tt, by)
    st.caption(f"{len(names)} {by}(s): {', '.join(names[:8])}{' ...' if len(names) > 8 else ''}")
    e1, e2 = st.columns(2)
    e1.download_button(f"⬇ Excel (one sheet per {by})", to_excel(tt, asst.cfg, by), f"timetable_by_{by}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")
    e2.download_button(f"⬇ PDF (one page per {by})", to_pdf(tt, asst.cfg, by), f"timetable_by_{by}.pdf",
                       "application/pdf", width="stretch")


def page_edit():
    ss = st.session_state
    asst: TimetableAssistant = ss.assistant
    st.header("Edit by hand")
    if not asst.timetable:
        st.info("Generate a timetable first (Assistant page).")
        return
    cfg = asst.cfg
    st.caption("Pick a session, choose where it should go, and watch the grid: clashes and broken rules are "
               "outlined in red before anything is changed. (Streamlit has no built-in drag and drop, so moving is done "
               "with these pickers.)")
    order = sorted(range(len(asst.timetable)), key=lambda i: (asst.timetable[i].label or asst.timetable[i].group,
                                                            cfg.days.index(asst.timetable[i].day), asst.timetable[i].start))
    def name(i):
        p = asst.timetable[i]
        return f"{p.label or p.group} · {p.subject} · {p.day[:3]} {cfg.slot_label(p.start).split('-')[0]}"
    idx = st.selectbox("Session", order, format_func=name, key="edit_idx")
    p = asst.timetable[idx]
    c1, c2, c3 = st.columns(3)
    day = c1.selectbox("Day", cfg.days, index=cfg.days.index(p.day), key=f"ed_day{idx}")
    start = c2.selectbox("Starts at", list(range(cfg.num_slots)), index=p.start,
                         format_func=lambda s: cfg.slot_label(s).split("-")[0], key=f"ed_start{idx}")
    rooms = list(ss.store.rooms[ss.store.rooms["type"] == ("lab" if p.kind == "lab" else "theory")]["name"])
    room = c3.selectbox("Room", rooms, index=rooms.index(p.room) if p.room in rooms else 0, key=f"ed_room{idx}",
                        disabled=bool(p.elective))
    new, msgs, bad = asst.preview_move(idx, day, start, None if p.elective or room == p.room else room)
    shown = new
    focus = st.toggle("Show only the groups this session belongs to", value=True, key="edit_focus")
    if focus:
        shown = [q for q in new if set(q.all_groups) & set(p.all_groups)]
    st.markdown(grid_html(shown, cfg, bad_ids={id(new[i]) for i in bad}), unsafe_allow_html=True)
    if msgs:
        st.error("This would break:\n\n" + "\n".join(f"- {m}" for m in msgs))
    else:
        st.success("No clashes or broken rules.")
    pin = st.checkbox("Pin it there (adds a hard rule so later changes keep it)", value=True)
    force = st.checkbox("Apply even though it breaks rules", value=False) if msgs else False
    if st.button("Apply move", type="primary", disabled=bool(msgs) and not force):
        r = asst.apply_move(idx, day, start, None if p.elective or room == p.room else room, pin=pin, force=force)
        ss.chat.append({"role": "assistant", "content": r.message})
        (st.success if r.status == "ok" else st.error)(r.message)
        st.rerun()


def page_versions():
    ss = st.session_state
    asst: TimetableAssistant = ss.assistant
    st.header("Saved versions")
    if asst.timetable:
        c1, c2 = st.columns([3, 1])
        name = c1.text_input("Name for the current timetable", placeholder="e.g. Draft 1 - before exam week")
        if c2.button("Save", type="primary", disabled=not name.strip()):
            try:
                save_version(asst, name)
                st.success(f"Saved '{name}'.")
            except ValueError as e:
                st.error(str(e))
    else:
        st.info("Generate a timetable to be able to save it.")
    versions = list_versions()
    if not versions:
        st.caption("No versions saved yet.")
        return
    st.dataframe(pd.DataFrame(versions), hide_index=True, width="stretch")
    names = [v["name"] for v in versions]
    pick = st.selectbox("Open / delete", names, key="ver_pick")
    b1, b2 = st.columns(2)
    if b1.button("Make this the active timetable"):
        restore_version(asst, pick)
        st.success(f"'{pick}' is now active. (Undo is available in the Assistant.)")
    if b2.button("Delete this version"):
        delete_version(pick)
        st.rerun()
    st.subheader("Compare two versions")
    c1, c2 = st.columns(2)
    a_name = c1.selectbox("Before", names, index=min(1, len(names) - 1), key="cmp_a")
    b_name = c2.selectbox("After", names, index=0, key="cmp_b")
    if st.button("Compare"):
        d = compare(load_data(a_name), load_data(b_name))
        if d["identical"]:
            st.success("The two versions are identical.")
        for title, key in (("Moved", "moved"), ("Room changes", "room_changes"), ("Only in 'After'", "added"),
                           ("Only in 'Before'", "removed")):
            if d[key]:
                st.markdown(f"**{title} ({len(d[key])})**")
                st.write("\n".join(f"- {x}" for x in d[key]))


def main():
    init_state()
    sidebar()
    {"Dashboard": page_dashboard, "Assistant": page_assistant, "Manage Academic Data": page_data,
     "Constraints & Settings": page_constraints, "NLP Lab": page_nlp_lab, "Export": page_export,
     "Edit by hand": page_edit, "Versions": page_versions, "My Timetable": page_mine}[st.session_state.page]()


main()
