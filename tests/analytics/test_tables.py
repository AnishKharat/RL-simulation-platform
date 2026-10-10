import math

import pytest

from rcl.analytics import by_metric, summarize


def row(env, ctl, seed, metric, value, status="ok", unit="u"):
    return {
        "env_id": env,
        "controller_id": ctl,
        "seed": seed,
        "run_id": f"{env}{ctl}{seed}",
        "metric_name": metric,
        "value": value,
        "unit": unit,
        "status": status,
    }


ROWS = [
    row("e", "a", 0, "m", 1.0),
    row("e", "a", 1, "m", 3.0),
    row("e", "a", 2, "m", None, "undefined"),
    row("e", "b", 0, "m", 10.0, "censored"),
    row("e", "b", 1, "m", 20.0, "censored"),
    row("e", "a", 0, "k", 5.0),
    row("f", "a", 0, "m", 7.0),
]


def test_mean_std_counts_and_undefined_exclusion():
    s = {(r.env_id, r.controller_id, r.metric_name): r for r in summarize(ROWS)}
    a = s[("e", "a", "m")]
    assert (a.n_valid, a.n_total, a.n_censored) == (2, 3, 0)
    assert a.mean == pytest.approx(2.0) and a.std == pytest.approx(math.sqrt(2.0))
    b = s[("e", "b", "m")]
    assert (b.n_valid, b.n_censored, b.mean, b.unit) == (2, 2, pytest.approx(15.0), "u")


def test_single_value_has_no_std_and_all_undefined_is_nan():
    s = {(r.env_id, r.metric_name): r for r in summarize(ROWS) if r.controller_id == "a"}
    assert math.isnan(s[("f", "m")].std) and s[("f", "m")].mean == 7.0
    only_bad = summarize([row("e", "c", 0, "m", None, "undefined")])[0]
    assert only_bad.n_valid == 0 and math.isnan(only_bad.mean) and only_bad.n_total == 1


def test_by_metric_pivots_one_environment():
    table = by_metric(summarize(ROWS), "e")
    assert list(table) == ["m", "k"]  # first-seen order
    assert sorted(table["m"]) == ["a", "b"] and sorted(table["k"]) == ["a"]
    assert by_metric(summarize(ROWS), "nope") == {}
