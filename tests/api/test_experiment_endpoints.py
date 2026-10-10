import time

import pytest

from rcl.core.config import ExperimentConfig

CONFIG = {
    "name": "api-test",
    "seed": 5,
    "environment": {"id": "cartpole"},
    "controller": {"id": "pid", "params": {"kp": 55.0}},
    "evaluation": {"episodes": 2},
}


def wait(client, run_id, timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        run = client.get(f"/runs/{run_id}").json()
        if run["status"] in ("done", "failed"):
            return run
        time.sleep(0.1)
    pytest.fail("run did not finish")


def post(client, **over):
    return client.post("/experiments", json={**CONFIG, **over})


def values(run):
    rows = [(m["metric_name"], m["episode_index"], m["value"]) for m in run["metrics"]]
    return sorted(rows, key=str)


def test_pid_flow_create_poll_results_trajectory(client):
    r = post(client)
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "queued" and len(body["run_ids"]) == 1
    run = wait(client, body["run_ids"][0])
    assert run["status"] == "done" and run["progress"] == 1.0 and run["error"] is None
    assert run["seed"] == 5 and set(run["derived_seeds"]) == {"train", "eval"}
    assert [e["index"] for e in run["episodes"]] == [0, 1] and run["train_curve"] == []
    names = {m["metric_name"] for m in run["metrics"]}
    assert {"return", "angle_error", "max_abs_angle_error"} <= names

    exp = client.get(f"/experiments/{body['experiment_id']}").json()
    assert exp["env_id"] == "cartpole" and exp["config"]["controller"]["params"]["kd"] == 5.0
    assert [x["run_id"] for x in exp["runs"]] == body["run_ids"]

    t = client.get(f"/runs/{run['run_id']}/trajectory/0").json()
    n = len(t["time"])
    assert len(t["states"]) == n and len(t["actions"]) == len(t["rewards"]) == n - 1
    assert t["state_labels"] == ["x", "x_dot", "theta", "theta_dot"] and t["action_units"] == ["N"]
    assert client.get(f"/runs/{run['run_id']}/trajectory/9").status_code == 404


def test_same_seed_gives_identical_metrics_over_http(client):
    runs = [wait(client, post(client).json()["run_ids"][0]) for _ in range(2)]
    assert values(runs[0]) == values(runs[1])


def test_validation_errors_are_422_with_body_locs(client):
    def errs(**over):
        r = post(client, **over)
        assert r.status_code == 422
        return {(tuple(e["loc"]), e["type"]) for e in r.json()["detail"]}

    assert errs(environment={"id": "nope"}) == {(("body", "environment", "id"), "unknown_id")}
    assert errs(environment={"id": "cartpole_discrete"}) == {
        (("body", "controller", "id"), "incompatible")
    }
    bad = errs(controller={"id": "pid", "params": {"kp": "x"}})
    assert (("body", "controller", "params", "kp"), "float_parsing") in bad
    assert errs(controller={"id": "ppo"}) == {(("body", "training"), "missing")}
    assert post(client, evaluation={"episodes": 0}).status_code == 422  # schema-level error
    assert client.get("/runs/x").status_code == 404
    assert client.get("/experiments/x").status_code == 404


def test_ppo_run_reports_versions_and_curve(client):
    r = post(
        client,
        controller={"id": "ppo", "params": {"n_steps": 32, "batch_size": 32, "n_epochs": 1}},
        training={"total_timesteps": 64},
        evaluation={"episodes": 1},
    )
    assert r.status_code == 202
    run = wait(client, r.json()["run_ids"][0])
    assert run["status"] == "done" and run["versions"]["device"] == "cpu"
    assert all(len(p) == 2 for p in run["train_curve"])


def test_failed_run_is_200_with_error_and_trajectory_409(client):
    cfg = ExperimentConfig.model_validate({**CONFIG, "environment": {"id": "gone"}})
    _, run_id = client.app.state.repo.create_experiment(cfg, 1, {"train": 1, "eval": 2})
    client.app.state.jobs.submit(run_id, cfg, 1).result(timeout=30)
    resp = client.get(f"/runs/{run_id}")
    assert resp.status_code == 200
    run = resp.json()
    assert run["status"] == "failed" and run["error"]["type"] == "ConfigError"
    assert client.get(f"/runs/{run_id}/trajectory/0").status_code == 409
