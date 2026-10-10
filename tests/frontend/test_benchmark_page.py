from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from rcl.api.main import create_app
from rcl.frontend import client, plots

PAGE = str(Path(__file__).parents[2] / "rcl" / "frontend" / "pages" / "1_Benchmark.py")


def srow(ctl, mean, lo, hi, **over):
    base = {
        "env_id": "e", "controller_id": ctl, "metric_name": "m", "unit": "u", "n_valid": 3,
        "n_total": 3, "n_censored": 0, "mean": mean, "std": 1.0, "ci_low": lo, "ci_high": hi,
        "ci_method": "t", "low_n": True,
    }  # fmt: skip
    return base | over


def test_comparison_figure_has_asymmetric_interval_whiskers_and_handles_missing_ci():
    rows = [
        srow("b", 5.0, 3.0, 9.0),
        srow("a", 2.0, None, None),
        srow("a", 1.0, 0, 2, metric_name="x"),
    ]
    fig = plots.comparison_figure(rows, "m")
    bar = fig.data[0]
    assert list(bar.x) == ["a", "b"] and list(bar.y) == [2.0, 5.0]  # sorted, other metric ignored
    assert list(bar.error_y.array) == [0.0, 4.0] and list(bar.error_y.arrayminus) == [0.0, 2.0]
    assert (
        plots.comparison_figure([], "m").data[0].x is None
        or len(plots.comparison_figure([], "m").data[0].x) == 0
    )


def test_summary_table_formats_interval_counts_and_warnings():
    t = plots.summary_table(
        [srow("a", 2.0, 1.0, 3.0, n_censored=2), srow("b", None, None, None)], "m"
    )
    assert t[0]["95% interval"] == "[1, 3] (t)" and t[0]["seeds (valid/total)"] == "3/3"
    assert t[0]["notes"] == "fewer than 5 seeds, 2 censored" and t[1]["mean"] == "n/a"
    assert plots.summary_table([srow("a", 1, 0, 2)], "other") == []


@pytest.fixture
def api(tmp_path):
    app = create_app(
        db_url=f"sqlite:///{tmp_path / 'db.sqlite'}", artifact_root=str(tmp_path / "a")
    )
    with TestClient(app) as c:
        client.override = c
        yield c
    client.override = None


def test_page_runs_a_benchmark_and_shows_comparison_from_registry_data(api):
    at = AppTest.from_file(PAGE, default_timeout=300).run()
    assert not at.exception
    at.multiselect(key="bench-envs").select("cartpole").select("cartpole_discrete")
    at.multiselect(key="bench-ctls").select("pid").select("lqr")
    at.number_input(key="bench-seeds").set_value(3)
    at.number_input(key="bench-episodes").set_value(1)
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert any("Skipped" in c.value for c in at.caption)  # discrete env x PID/LQR
    assert at.selectbox(key="bench-env").options == ["cartpole"]
    assert "return" in at.selectbox(key="bench-metric").options
    assert len(at.dataframe) >= 2
    at.selectbox(key="bench-metric").select("settling_time").run()
    assert not at.exception and any("Fewer than 5 seeds" in i.value for i in at.info)


def test_page_shows_422_problems_instead_of_creating(api):
    at = AppTest.from_file(PAGE, default_timeout=60).run()
    at.multiselect(key="bench-envs").select("cartpole_discrete")
    at.multiselect(key="bench-ctls").select("pid")
    at.button[0].click().run()
    assert not at.exception and any("no compatible" in e.value for e in at.error)
    assert api.get("/benchmarks").json() == []
