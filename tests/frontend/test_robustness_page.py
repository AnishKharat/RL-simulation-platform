from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from rcl.api.main import create_app
from rcl.frontend import client, plots

PAGE = str(Path(__file__).parents[2] / "rcl" / "frontend" / "pages" / "2_Robustness.py")


def curve(ctl, means):
    pts = [
        {
            "level": lv, "mean": m, "ci_low": m - 0.1, "ci_high": m + 0.1, "ci_method": "bootstrap",
            "n_valid": 3, "n_total": 3, "relative": m / means[0],
        }
        for lv, m in zip((0.0, 0.5), means, strict=True)
    ]  # fmt: skip
    return {"env_id": "e", "controller_id": ctl, "metric_name": "success_rate",
            "breakdown": None, "points": pts}  # fmt: skip


def test_degradation_figure_and_table_from_api_payloads():
    curves = [curve("b", [1.0, 0.5]), curve("a", [1.0, 0.25])]
    fig = plots.degradation_figure(curves, "std")
    assert [t.name for t in fig.data] == ["a", "b"] and list(fig.data[0].y) == [1.0, 0.25]
    assert list(fig.data[0].error_y.array) == pytest.approx([0.1, 0.1])
    rows = plots.degradation_table(curves)
    assert rows[1]["relative to nominal"] == "0.25" and rows[1]["breakdown level"] == "n/a"
    assert plots.degradation_figure([]).data == ()


@pytest.fixture
def api(tmp_path):
    app = create_app(
        db_url=f"sqlite:///{tmp_path / 'db.sqlite'}", artifact_root=str(tmp_path / "a")
    )
    with TestClient(app) as c:
        client.override = c
        yield c
    client.override = None


def test_page_runs_a_sweep_and_plots_curves_from_registry_data(api):
    at = AppTest.from_file(PAGE, default_timeout=300).run()
    assert not at.exception
    at.selectbox(key="rob-env-pick").select("cartpole").run()
    at.selectbox(key="rob-pert").select("action_noise")
    at.multiselect(key="rob-ctls").select("pid")
    at.text_input(key="rob-levels").set_value("0.5, 2")
    at.number_input(key="rob-seeds").set_value(2)
    at.number_input(key="rob-episodes").set_value(1)
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert at.selectbox(key="rob-env").options == ["cartpole"]
    assert len(at.dataframe) == 1 and len(at.dataframe[0].value) == 3  # 3 levels, 1 controller


def test_page_reports_bad_levels_without_creating(api):
    at = AppTest.from_file(PAGE, default_timeout=60).run()
    at.selectbox(key="rob-env-pick").select("cartpole").run()
    at.multiselect(key="rob-ctls").select("pid")
    at.text_input(key="rob-levels").set_value("abc")
    at.button[0].click().run()
    assert not at.exception and any("levels" in e.value for e in at.error)
    assert api.get("/benchmarks").json() == []
