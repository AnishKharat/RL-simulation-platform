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
    assert at.selectbox(key="ctl-cartpole").options == ["PID", "PPO"]
    at.selectbox(key="env").select("cartpole_discrete").run()
    assert at.selectbox(key="ctl-cartpole_discrete").options == ["PPO"]
