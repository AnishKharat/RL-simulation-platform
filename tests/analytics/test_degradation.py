import math

import pytest

from rcl.analytics import degradation_curves


def row(ctl, level, seed, value, metric="success_rate"):
    return {
        "env_id": "e",
        "controller_id": ctl,
        "seed": seed,
        "run_id": f"{ctl}{level}{seed}",
        "metric_name": metric,
        "value": value,
        "unit": "fraction",
        "status": "ok",
        "perturbations": [{"id": "p", "params": {"std": level, "other": 1}}],
    }


ROWS = [
    row(c, lv, s, v)
    for c, vals in {
        "a": {0.0: (1, 1), 0.1: (1, 0.8), 0.2: (0.2, 0.4)},
        "b": {0.0: (0, 0), 0.1: (0, 0), 0.2: (0, 0)},
    }.items()
    for lv, pair in vals.items()
    for s, v in enumerate(pair)
] + [row("a", 0.0, 0, 9.0, metric="other")]


def curves():
    return {c.controller_id: c for c in degradation_curves(ROWS, "p", "std", 0.0)}


def test_curve_has_mean_ci_and_relative_value_per_level_in_ascending_order():
    a = curves()["a"]
    assert [p.level for p in a.points] == [0.0, 0.1, 0.2]
    assert [p.summary.mean for p in a.points] == pytest.approx([1.0, 0.9, 0.3])
    assert [p.relative for p in a.points] == pytest.approx([1.0, 0.9, 0.3])
    assert a.points[1].summary.n_valid == 2 and a.points[1].summary.ci_method == "bootstrap"


def test_breakdown_is_first_level_below_half_nominal_or_nan():
    assert curves()["a"].breakdown == 0.2
    b = curves()["b"]  # nominal 0: relative and breakdown undefined
    assert math.isnan(b.breakdown) and all(math.isnan(p.relative) for p in b.points)


def test_rows_without_the_perturbation_or_of_other_metrics_are_ignored():
    plain = [{**r, "perturbations": []} for r in ROWS]
    assert degradation_curves(plain, "p", "std", 0.0) == []
    assert {c.metric_name for c in degradation_curves(ROWS, "p", "std", 0.0, "other")} == {"other"}
