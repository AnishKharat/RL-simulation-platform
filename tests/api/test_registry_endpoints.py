def ids(resp):
    assert resp.status_code == 200
    return [e["id"] for e in resp.json()]


def test_lists_are_sorted_and_carry_forms_and_labels(client):
    envs = ids(client.get("/environments"))
    assert envs == sorted(envs)  # optional-extra robots (e.g. PyBullet) may add entries
    assert {"cartpole", "cartpole_discrete", "inverted_pendulum"} <= set(envs)
    assert ids(client.get("/controllers")) == ["lqr", "pid", "ppo", "sac"]
    env = next(e for e in client.get("/environments").json() if e["id"] == "cartpole")
    assert env["kind"] == "environment" and env["metadata"]["state_labels"]
    assert env["params_schema"]["properties"]["force_mag"]["default"] == 10.0
    ppo = next(c for c in client.get("/controllers").json() if c["id"] == "ppo")
    assert ppo["metadata"]["requires_training"] is True and ppo["params_schema"]["properties"]


def test_compatible_controllers(client):
    assert ids(client.get("/controllers/compatible", params={"env_id": "cartpole"})) == [
        "lqr",
        "pid",
        "ppo",
        "sac",
    ]
    assert ids(client.get("/controllers/compatible", params={"env_id": "cartpole_discrete"})) == [
        "ppo"
    ]
    assert client.get("/controllers/compatible", params={"env_id": "nope"}).status_code == 404
    assert client.get("/controllers/compatible").status_code == 422


def test_metrics_catalogue(client):
    m = {x["name"]: x for x in client.get("/metrics").json()}
    assert {"return", "angle_error", "settling_time", "control_effort"} <= set(m)
    assert m["angle_error"]["companions"] == [{"name": "max_abs_angle_error", "unit": None}]
    assert m["return"]["higher_is_better"] is True
