import math
import xml.etree.ElementTree as ET

from rcl.analytics import SummaryRow, build_report
from rcl.metrics import METRIC_INFO

SPEC = {
    "name": "b",
    "environments": [{"id": "env_a"}],
    "controllers": [{"id": "c1"}, {"id": "c2"}],
    "seeds": [0, 1, 2],
    "episodes": 2,
    "training": {"total_timesteps": 100},
}


def row(ctl, mean, lo, hi, metric="return", **over):
    base = dict(env_id="env_a", controller_id=ctl, metric_name=metric, unit="reward", n_valid=3,
                n_total=3, n_censored=0, mean=mean, std=1.0, ci_low=lo, ci_high=hi,
                ci_method="t", low_n=True)  # fmt: skip
    return SummaryRow(**(base | over))


SUMMARY = [
    row("c1", 10.0, 8.0, 12.0),
    row("c2", -4.0, -9.0, 1.0),
    row("c1", 0.5, math.nan, math.nan, metric="settling_time", unit="s", n_censored=3,
        ci_method="bootstrap"),
]  # fmt: skip


def test_markdown_has_setup_tables_intervals_notes_and_plot_links():
    r = build_report("Cartpole shoot-out", SPEC, SUMMARY, METRIC_INFO)
    md = r.markdown
    assert md.startswith("# Cartpole shoot-out") and "- Seeds (3): 0, 1, 2" in md
    assert "Training budget (RL controllers): 100 timesteps" in md
    assert "| c1 | 10 | [8, 12] | 3/3 | reward | <5 seeds |" in md
    assert "| c2 | -4 | [-9, 1] |" in md
    assert "| c1 | 0.5 | [n/a, n/a] | 3/3 | s | <5 seeds, 3 censored |" in md
    assert "![return on env_a](plots/env_a_return.svg)" in md and "no significance claim" in md
    assert set(r.plots) == {"env_a_return.svg", "env_a_settling_time.svg"}


def test_svgs_are_well_formed_and_scale_negative_values():
    r = build_report("t", SPEC, SUMMARY, METRIC_INFO)
    for svg in r.plots.values():
        ET.fromstring(svg)  # raises if malformed
    assert "c1" in r.plots["env_a_return.svg"] and r.plots["env_a_return.svg"].count("<rect") == 2


def test_html_is_self_contained_and_escapes_text():
    evil = [row("<b>x</b>", 1.0, 0.0, 2.0)]
    r = build_report("A & B", SPEC, evil, METRIC_INFO)
    assert "&lt;b&gt;x&lt;/b&gt;" in r.html and "<b>x</b>" not in r.html
    assert (
        "<title>A &amp; B</title>" in r.html
        and "<svg" in r.html
        and "http://" not in r.html.replace("http://www.w3.org/2000/svg", "")
    )


def test_report_is_deterministic_and_handles_empty_summary():
    a, b = (
        build_report("t", SPEC, SUMMARY, METRIC_INFO),
        build_report("t", SPEC, SUMMARY, METRIC_INFO),
    )
    assert (a.markdown, a.html, a.plots) == (b.markdown, b.html, b.plots)
    empty = build_report("t", SPEC, [], METRIC_INFO)
    assert empty.plots == {} and "## Notes" in empty.markdown


def test_skipped_pairs_section_in_markdown_and_html():
    skipped = [{"env_id": "disc", "controller_id": "c1", "reason": "not <compatible>"}]
    r = build_report("t", SPEC, SUMMARY, METRIC_INFO, skipped)
    assert "## Skipped pairs" in r.markdown and "- c1 on disc: not <compatible>" in r.markdown
    assert "<li>c1 on disc: not &lt;compatible&gt;</li>" in r.html
    assert "Skipped pairs" not in build_report("t", SPEC, SUMMARY, METRIC_INFO).markdown


def test_headline_metrics_lead_the_report_and_return_is_caveated():
    summary = [
        row("c1", 12.0, 10.0, 14.0),
        row("c1", 0.25, 0.0, 0.5, metric="success_rate", unit="fraction", ci_method="bootstrap"),
        row("c2", 3.0, 2.0, 4.0, metric="time_to_success", unit="s", ci_method="bootstrap"),
    ]
    r = build_report("t", SPEC, summary, METRIC_INFO)
    md = r.markdown
    assert md.index("### success_rate") < md.index("### time_to_success") < md.index("### return")
    assert "> Headline metrics (success_rate, time_to_success)" in md
    assert "not the headline metric" in md and "not the headline metric" in r.html
    no_headline = build_report("t", SPEC, [row("c1", 12.0, 10.0, 14.0)], METRIC_INFO)
    assert "Headline metrics" not in no_headline.markdown
