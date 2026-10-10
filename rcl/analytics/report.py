"""Markdown / HTML benchmark report with dependency-free SVG charts.

Deterministic: no timestamps, fixed float formatting, so the same summary always produces the
byte-identical report (re-running a benchmark with the same seeds can be diffed).
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass
from typing import Any

from rcl.analytics.tables import SummaryRow, by_metric
from rcl.metrics import MetricInfo

_BAR, _INK, _MUTED = "#3b6ea5", "#1f2933", "#7b8794"


@dataclass(frozen=True)
class Report:
    markdown: str
    html: str
    plots: dict[str, str]  # file name -> SVG text (the Markdown links to ``plots/<name>``)


def _num(v: float | None) -> str:
    return "n/a" if v is None or math.isnan(v) else f"{v:.4g}"


def _interval(r: SummaryRow) -> str:
    return f"[{_num(r.ci_low)}, {_num(r.ci_high)}]"


def _notes(r: SummaryRow) -> str:
    notes = []
    if r.low_n:
        notes.append("<5 seeds")
    if r.n_censored:
        notes.append(f"{r.n_censored} censored")
    return ", ".join(notes)


def bar_chart_svg(title: str, unit: str, bars: list[SummaryRow]) -> str:
    """Bars of the cross-seed mean per controller with 95% interval whiskers."""
    w, h, left, right, top, bottom = 460, 280, 64, 16, 34, 58
    finite = [v for b in bars for v in (b.mean, b.ci_low, b.ci_high, 0.0) if not math.isnan(v)]
    lo, hi = min(finite, default=0.0), max(finite, default=1.0)
    if hi == lo:
        hi = lo + 1.0
    pad = 0.06 * (hi - lo)
    lo, hi = (lo - pad if lo < 0 else 0.0), hi + pad

    def y(v: float) -> float:
        return top + (hi - v) / (hi - lo) * (h - top - bottom)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'font-family="sans-serif" font-size="11" role="img" aria-label="{html.escape(title)}">',
        f'<text x="{left}" y="18" font-size="13" fill="{_INK}">{html.escape(title)}</text>',
    ]
    for i in range(5):  # y axis ticks
        v = lo + (hi - lo) * i / 4
        parts.append(
            f'<line x1="{left}" x2="{w - right}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="#e4e7eb"/>'
            f'<text x="{left - 6}" y="{y(v) + 4:.1f}" text-anchor="end" fill="{_MUTED}">{v:.3g}</text>'
        )
    slot = (w - left - right) / max(len(bars), 1)
    for k, b in enumerate(bars):
        cx, x0 = left + slot * (k + 0.5), left + slot * (k + 0.5) - slot * 0.3
        if not math.isnan(b.mean):
            top_y, base_y = sorted((y(b.mean), y(0.0)))
            parts.append(
                f'<rect x="{x0:.1f}" y="{top_y:.1f}" width="{slot * 0.6:.1f}" '
                f'height="{max(base_y - top_y, 0.5):.1f}" fill="{_BAR}"/>'
            )
        if not (math.isnan(b.ci_low) or math.isnan(b.ci_high)):
            parts.append(
                f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{y(b.ci_low):.1f}" y2="{y(b.ci_high):.1f}" '
                f'stroke="{_INK}" stroke-width="1.5"/>'
                f'<line x1="{cx - 5:.1f}" x2="{cx + 5:.1f}" y1="{y(b.ci_low):.1f}" y2="{y(b.ci_low):.1f}" stroke="{_INK}"/>'
                f'<line x1="{cx - 5:.1f}" x2="{cx + 5:.1f}" y1="{y(b.ci_high):.1f}" y2="{y(b.ci_high):.1f}" stroke="{_INK}"/>'
            )
        parts.append(
            f'<text x="{cx:.1f}" y="{h - bottom + 16}" text-anchor="middle" fill="{_INK}">'
            f"{html.escape(b.controller_id)}</text>"
        )
    unit_label = f" [{html.escape(unit)}]" if unit else ""
    parts.append(
        f'<text x="{left}" y="{h - 8}" fill="{_MUTED}">mean with 95% interval across seeds{unit_label}</text>'
        f'<line x1="{left}" x2="{left}" y1="{top}" y2="{h - bottom}" stroke="{_MUTED}"/></svg>'
    )
    return "\n".join(parts)


def _spec_lines(spec: dict[str, Any]) -> list[str]:
    envs = ", ".join(e["id"] for e in spec["environments"])
    ctls = ", ".join(c["id"] for c in spec["controllers"])
    lines = [
        f"- Environments: {envs}",
        f"- Controllers: {ctls}",
        f"- Seeds ({len(spec['seeds'])}): {', '.join(str(s) for s in spec['seeds'])}",
        f"- Evaluation episodes per run: {spec['episodes']}",
    ]
    if spec.get("training"):
        lines.append(
            f"- Training budget (RL controllers): {spec['training']['total_timesteps']} timesteps"
        )
    return lines


CAVEAT = (
    "Headline metrics ({names}) say whether the task was achieved. The reward sum (`return`) is a "
    "training signal, not the headline metric: success ends an episode early, so a controller that "
    "succeeds can score a lower return than one that never does."
)


def build_report(
    title: str,
    spec: dict[str, Any],
    summary: list[SummaryRow],
    infos: list[MetricInfo],
    skipped: list[dict[str, str]] | None = None,
) -> Report:
    """Render the comparison: per environment and metric a table plus a chart.

    ``skipped`` lists incompatible pairs (``env_id``, ``controller_id``, ``reason``) so the
    reader can see which cells of the grid were not run."""
    plots: dict[str, str] = {}
    md = [f"# {title}", "", "## Setup", "", *_spec_lines(spec), ""]
    body = [f"<h1>{html.escape(title)}</h1>", "<h2>Setup</h2><ul>"]
    body += [f"<li>{html.escape(line[2:])}</li>" for line in _spec_lines(spec)] + ["</ul>"]
    for env in dict.fromkeys(r.env_id for r in summary):
        md += [f"## Environment: {env}", ""]
        body.append(f"<h2>Environment: {html.escape(env)}</h2>")
        table = by_metric(summary, env)
        ordered = sorted((i for i in infos if i.name in table), key=lambda i: not i.headline)
        heads = [i.name for i in ordered if i.headline]
        if heads:
            intro = CAVEAT.format(names=", ".join(heads))
            md += [f"> {intro}", ""]
            body.append(f"<p><em>{html.escape(intro)}</em></p>")
        for info in ordered:
            rows = sorted(table[info.name].values(), key=lambda r: r.controller_id)
            unit = rows[0].unit
            direction = "higher is better" if info.higher_is_better else "lower is better"
            name = f"{env}_{info.name}.svg"
            plots[name] = bar_chart_svg(f"{info.name} ({env})", unit, rows)
            head = "| controller | mean | 95% interval | seeds (valid/total) | unit | notes |"
            md += [
                f"### {info.name}",
                "",
                f"{info.description} ({direction}; interval: {rows[0].ci_method}.)",
                "",
            ]
            md += [head, "|---|---|---|---|---|---|"]
            md += [
                f"| {r.controller_id} | {_num(r.mean)} | {_interval(r)} | {r.n_valid}/{r.n_total} | {r.unit} | {_notes(r)} |"
                for r in rows
            ]
            md += ["", f"![{info.name} on {env}](plots/{name})", ""]
            body += [
                f"<h3>{html.escape(info.name)}</h3>",
                f"<p>{html.escape(info.description)} ({direction})</p>",
            ]
            body.append(
                "<table><tr><th>controller</th><th>mean</th><th>95% interval</th><th>seeds</th><th>unit</th><th>notes</th></tr>"
            )
            body += [
                f"<tr><td>{html.escape(r.controller_id)}</td><td>{_num(r.mean)}</td><td>{_interval(r)}</td>"
                f"<td>{r.n_valid}/{r.n_total}</td><td>{html.escape(r.unit)}</td><td>{html.escape(_notes(r))}</td></tr>"
                for r in rows
            ]
            body += ["</table>", plots[name]]
    notes = [
        "Intervals are two-sided 95% across seeds (the seed is the unit of replication): Student t, "
        "or a seeded percentile bootstrap where the metric table says `bootstrap`.",
        "`censored` counts runs whose value is a bound (for example never settled); they are included in the mean.",
        "Rows marked `<5 seeds` have wide intervals: no significance claim should be made from them.",
    ]
    if skipped:
        lines = [f"{k['controller_id']} on {k['env_id']}: {k['reason']}" for k in skipped]
        md += ["## Skipped pairs", "", *[f"- {line}" for line in lines], ""]
        body += ["<h2>Skipped pairs</h2><ul>"] + [f"<li>{html.escape(x)}</li>" for x in lines]
        body.append("</ul>")
    md += ["## Notes", "", *[f"- {n}" for n in notes], ""]
    body += ["<h2>Notes</h2><ul>"] + [f"<li>{html.escape(n)}</li>" for n in notes] + ["</ul>"]
    style = (
        "body{font-family:sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#1f2933}"
        "table{border-collapse:collapse;margin:.5rem 0}td,th{border:1px solid #cbd2d9;padding:.25rem .6rem;text-align:left}"
    )
    page = (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
        f'content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title>'
        f"<style>{style}</style></head><body>\n" + "\n".join(body) + "\n</body></html>\n"
    )
    return Report("\n".join(md), page, plots)
