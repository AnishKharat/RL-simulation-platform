from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from rcl.api.main import create_app
from rcl.frontend import client

APP = str(Path(__file__).parents[2] / "rcl" / "frontend" / "app.py")


@pytest.fixture
def api(tmp_path):
    app = create_app(
        db_url=f"sqlite:///{tmp_path / 'db.sqlite'}", artifact_root=str(tmp_path / "a")
    )
    with TestClient(app) as c:
        client.override = c
        yield c
    client.override = None


def test_dropdowns_are_filled_from_the_api(api):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert at.selectbox(key="env").options == [
        "CartPole (continuous)",
        "CartPole (discrete)",
        "Inverted Pendulum",
    ]
    assert at.selectbox(key="env").value == "cartpole"
    assert at.selectbox(key="ctl-cartpole").options == ["LQR", "PID", "PPO", "SAC"]
    at.selectbox(key="env").select("cartpole_discrete").run()
    assert at.selectbox(key="ctl-cartpole_discrete").options == ["PPO"]


def test_run_from_the_form_reaches_done(api):
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.selectbox(key="ctl-cartpole").select("pid").run()
    at.number_input(key="controller-pid.kp").set_value(55.0)
    at.number_input(key="episodes").set_value(2)
    at.button[0].click().run()
    assert not at.exception
    assert any("done" in s.value for s in at.success)
    assert "Trajectory" in [h.value for h in at.subheader] and len(at.dataframe) >= 2
    run_id = at.session_state["run_id"]
    body = api.get(f"/runs/{run_id}").json()
    assert body["status"] == "done" and len(body["episodes"]) == 2
    assert (
        api.get(f"/experiments/{body['experiment_id']}").json()["config"]["controller"]["params"][
            "kp"
        ]
        == 55.0
    )


def test_server_validation_error_shown_next_to_the_widget(api):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.selectbox(key="ctl-cartpole").select("pid").run()
    at.number_input(key="controller-pid.integral_limit").set_value(0.0)  # schema: > 0 (exclusive)
    at.button[0].click().run()
    assert not at.exception and at.session_state["run_id"] is None
    assert any(":red[" in c.value for c in at.caption)
