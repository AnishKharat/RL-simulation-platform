import math

import numpy as np
import pytest

from rcl.analytics import summarize
from rcl.analytics.stats import bootstrap_interval, t_critical, t_interval


def test_t_critical_known_values_and_series_continuity():
    assert t_critical(1) == 12.706 and t_critical(4) == 2.776 and t_critical(30) == 2.042
    assert t_critical(31) == pytest.approx(2.0395, abs=2e-4)
    assert t_critical(120) == pytest.approx(1.9799, abs=2e-4)
    assert t_critical(10_000) == pytest.approx(1.9602, abs=2e-4)
    with pytest.raises(ValueError):
        t_critical(0)


def test_t_critical_matches_scipy_when_available():
    stats = pytest.importorskip("scipy.stats")
    for df in (1, 2, 5, 12, 30, 31, 60, 500):
        assert t_critical(df) == pytest.approx(stats.t.ppf(0.975, df), abs=6e-4)


def test_t_interval_formula():
    low, high = t_interval([1.0, 2.0, 3.0, 4.0, 5.0])  # mean 3, s = sqrt(2.5), n = 5
    half = 2.776 * math.sqrt(2.5) / math.sqrt(5)
    assert (low, high) == (pytest.approx(3 - half), pytest.approx(3 + half))
    assert all(math.isnan(x) for x in t_interval([1.0])) and all(
        math.isnan(x) for x in t_interval([])
    )
    assert t_interval([2.0, 2.0, 2.0]) == (2.0, 2.0)  # zero variance -> degenerate interval


def test_bootstrap_is_seeded_bounded_and_nan_for_one_value():
    values = [1.0, 2.0, 4.0, 8.0, 16.0, 30.0]
    a, b = bootstrap_interval(values), bootstrap_interval(values)
    assert a == b and bootstrap_interval(values, seed=1) != a
    assert min(values) <= a[0] < np.mean(values) < a[1] <= max(values)
    assert all(math.isnan(x) for x in bootstrap_interval([3.0]))
    assert bootstrap_interval([5.0, 5.0, 5.0]) == (5.0, 5.0)


def _rows(metric, values, status="ok"):
    return [
        {"env_id": "e", "controller_id": "c", "seed": i, "run_id": str(i), "metric_name": metric,
         "value": v, "unit": "u", "status": status}
        for i, v in enumerate(values)
    ]  # fmt: skip


def test_summarize_attaches_method_per_catalogue_and_low_n_flag():
    t_row = summarize(_rows("return", [1.0, 2.0, 3.0, 4.0, 5.0]))[0]
    assert t_row.ci_method == "t" and not t_row.low_n
    assert t_row.ci_low == pytest.approx(t_interval([1, 2, 3, 4, 5])[0])
    boot = summarize(_rows("settling_time", [1.0, 2.0, 3.0], status="censored"))[0]
    assert boot.ci_method == "bootstrap" and boot.low_n and boot.n_censored == 3
    lone = summarize(_rows("return", [1.0]))[0]
    assert math.isnan(lone.ci_low) and math.isnan(lone.ci_high) and lone.low_n
    assert summarize(_rows("unknown_companion", [1.0, 2.0]))[0].ci_method == "t"
