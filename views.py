"""Rendering helpers: timetable grid (HTML) and tabular exports."""
from __future__ import annotations

import colorsys
import html
from collections import defaultdict

import pandas as pd

from models import Config, is_batch_key


def _colour(name: str):
    h = (sum(ord(c) * (i + 3) for i, c in enumerate(name)) % 360) / 360
    r, g, b = colorsys.hls_to_rgb(h, 0.5, 0.55)
    solid = f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"
    soft = f"rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},0.20)"
    return solid, soft


CSS = """
<style>
.tt{border-collapse:separate;border-spacing:4px;width:100%;table-layout:fixed;font-size:0.82rem}
.tt th{padding:6px 4px;text-align:center;font-weight:600;opacity:.85}
.tt td{vertical-align:top;padding:6px 8px;border-radius:6px;min-height:44px;line-height:1.25}
.tt td.time{width:92px;font-variant-numeric:tabular-nums;opacity:.7;text-align:right;white-space:nowrap}
.tt td.empty{background:rgba(128,128,128,.07)}
.tt td.lunch{background:rgba(128,128,128,.16);text-align:center;font-style:italic;opacity:.8}
.tt td.cls small{opacity:.8;display:block}
.tt td.cont{opacity:.55;font-size:.75rem}
.tt td.bad{outline:3px solid #ef4444;outline-offset:-3px}
</style>
"""


def grid_html(placements: list, cfg: Config, bad_ids: set = frozenset()) -> str:
    """bad_ids: id(placement) of sessions to outline in red (used for live conflict highlighting)."""
    if not placements:
        return CSS + "<p><i>No sessions to show.</i></p>"
    cells = defaultdict(list)
    for p in placements:
        for sl in p.slots():
            cells[(p.day, sl)].append((p, sl == p.start))
    bad_cells = {(p.day, sl) for p in placements if id(p) in bad_ids for sl in p.slots()}
    multi_group = len({p.label or p.group for p in placements}) > 1
    skip = set()
    out = [CSS, '<table class="tt"><thead><tr><th></th>']
    out += [f"<th>{html.escape(d)}</th>" for d in cfg.days]
    out.append("</tr></thead><tbody>")
    for slot in range(cfg.num_slots):
        out.append(f'<tr><td class="time">{cfg.slot_label(slot)}</td>')
        if slot in cfg.lunch_slots:
            out.append(f'<td class="lunch" colspan="{len(cfg.days)}">Lunch break</td></tr>')
            continue
        for day in cfg.days:
            if (day, slot) in skip:
                continue
            items = cells.get((day, slot))
            if not items:
                out.append('<td class="empty"></td>')
                continue
            if not multi_group and len(items) == 1 and items[0][1] and items[0][0].duration > 1:
                p = items[0][0]
                for k in range(1, p.duration):
                    skip.add((day, slot + k))
                out.append(_cell(p, rowspan=p.duration, show_group=False, bad=id(p) in bad_ids))
                continue
            parts = []
            for p, is_start in items:
                if is_start or multi_group is False:
                    parts.append(_inner(p, multi_group))
                else:
                    parts.append(f'<div class="cont">↳ {html.escape(p.subject)} ({html.escape(p.group)})</div>')
            solid, soft = _colour(items[0][0].subject)
            klass = "cls bad" if (day, slot) in bad_cells else "cls"
            out.append(f'<td class="{klass}" style="background:{soft};border-left:4px solid {solid}">{"".join(parts)}</td>')
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def _inner(p, show_group):
    icon = "🧪 " if p.kind == "lab" else ""
    who = p.label or p.group
    special = is_batch_key(p.group) or "+" in who
    grp = f" · {html.escape(who)}" if (show_group or special) else ""
    return (f"<div><b>{icon}{html.escape(p.subject)}</b>"
            f"<small>{html.escape(p.teacher)} · {html.escape(p.room)}{grp}</small></div>")


def _cell(p, rowspan, show_group, bad=False):
    solid, soft = _colour(p.subject)
    return (f'<td class="cls{" bad" if bad else ""}" rowspan="{rowspan}" style="background:{soft};border-left:4px solid {solid}">'
            f"{_inner(p, show_group)}</td>")


def to_dataframe(placements: list, cfg: Config) -> pd.DataFrame:
    rows = []
    for p in sorted(placements, key=lambda p: (p.group, cfg.days.index(p.day), p.start)):
        rows.append({"Group": p.group, "Day": p.day, "Start": cfg.fmt(cfg.slot_start(p.start)),
                     "End": cfg.fmt(cfg.slot_end(p.end - 1)), "Subject": p.subject, "Type": p.kind, "Attended by": p.label or p.group,
                     "Teacher": p.teacher, "Room": p.room})
    return pd.DataFrame(rows)


def standalone_html(placements: list, cfg: Config, title="Timetable") -> str:
    """Printable page: one grid per group."""
    body = []
    for g in sorted({k for p in placements for k in p.all_groups if not is_batch_key(k)}):
        body.append(f"<h2>{html.escape(g)}</h2>" + grid_html([p for p in placements if g in p.all_groups], cfg))
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
            f"<style>body{{font-family:system-ui,sans-serif;margin:24px;color:#222}}</style></head>"
            f"<body><h1>{html.escape(title)}</h1>{''.join(body)}</body></html>")
