import time

import pytest

SPEC = {
    "name": "api-bench",
    "environments": [{"id": "cartpole"}, {"id": "cartpole_discrete"}],
    "controllers": [{"id": "pid"}, {"id": "lqr"}],
    "seeds": [0, 1, 2],
    "episodes": 1,
}


def wait(client, suite_id, timeout=300):
    end = time.time() + timeout
    while time.time() < end:
        d = client.get(f"/benchmarks/{suite_id}").json()
        if d["benchmark_runs"][0]["status"] in ("done", "failed"):
            return d
        time.sleep(0.2)
    pytest.fail("benchmark did not finish")


def post(client, **over):
    return client.post("/benchmarks", json={**SPEC, **over})


def test_create_reports_skipped_pairs_and_runs_to_completion(client):
    r = post(client)
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "queued"
    assert {(k["env_id"], k["controller_id"]) for k in body["skipped"]} == {
        ("cartpole_discrete", "pid"),
        ("cartpole_discrete", "lqr"),
    }
    detail = wait(client, body["suite_id"])
    assert detail["benchmark_runs"][0]["status"] == "done" and detail["spec"]["seeds"] == [0, 1, 2]
    assert [(e["controller_id"], len(e["runs"])) for e in detail["experiments"]] == [
        ("pid", 3),
        ("lqr", 3),
    ]
    assert [b["suite_id"] for b in client.get("/benchmarks").json()] == [body["suite_id"]]


def test_summary_has_cis_counts_and_json_safe_nulls(client):
    suite = post(client).json()["suite_id"]
    wait(client, suite)
    rows = client.get(f"/benchmarks/{suite}/summary", params={"env_id": "cartpole"}).json()
    ret = {r["controller_id"]: r for r in rows if r["metric_name"] == "return"}
    assert set(ret) == {"pid", "lqr"}
    for r in ret.values():
        assert (r["n_valid"], r["n_total"], r["ci_method"], r["low_n"]) == (3, 3, "t", True)
        assert r["ci_low"] <= r["mean"] <= r["ci_high"]
    assert {r["ci_method"] for r in rows if r["metric_name"] == "settling_time"} == {"bootstrap"}
    assert client.get(f"/benchmarks/{suite}/summary", params={"env_id": "nope"}).json() == []
    assert len(client.get(f"/benchmarks/{suite}/summary").json()) == len(
        rows
    )  # skipped env has no rows


def test_paired_difference_and_not_found(client):
    suite = post(client).json()["suite_id"]
    wait(client, suite)
    q = {"env_id": "cartpole", "metric": "return", "a": "pid", "b": "lqr"}
    d = client.get(f"/benchmarks/{suite}/paired", params=q).json()
    assert d["controller_id"] == "pid-lqr" and d["n_valid"] == 3
    for path in ("", "/summary"):
        assert client.get(f"/benchmarks/nope{path}").status_code == 404
    assert client.get("/benchmarks/nope/paired", params=q).status_code == 404


def test_invalid_specs_are_422_with_body_locs_and_nothing_is_created(client):
    r = post(client, environments=[{"id": "nope"}])
    assert r.status_code == 422
    assert [e["loc"] for e in r.json()["detail"]] == [["body", "environments", 0, "id"]]
    assert post(client, controllers=[{"id": "ppo"}]).json()["detail"][0]["type"] == "missing"
    assert post(client, environments=[{"id": "cartpole_discrete"}]).status_code == 422
    assert post(client, seeds=[1, 1]).status_code == 422  # schema-level
    assert client.get("/benchmarks").json() == []


def test_detail_lists_skipped_pairs_after_the_post_response(client):
    suite = post(client).json()["suite_id"]
    wait(client, suite)
    skipped = client.get(f"/benchmarks/{suite}").json()["skipped"]
    assert {(k["env_id"], k["controller_id"]) for k in skipped} == {
        ("cartpole_discrete", "pid"),
        ("cartpole_discrete", "lqr"),
    }


def test_perturbations_listing_and_robustness_endpoint(client):
    perts = client.get("/perturbations").json()
    noise = next(p for p in perts if p["id"] == "action_noise")
    assert noise["kind"] == "perturbation"
    assert noise["metadata"]["magnitude_param"] == "std"
    assert "std" in noise["params_schema"]["properties"]

    sweep = {"id": "action_noise", "levels": [0.5]}
    body = post(client, environments=[{"id": "cartpole"}], seeds=[0, 1], perturbation=sweep).json()
    suite = body["suite_id"]
    assert wait(client, suite)["benchmark_runs"][0]["status"] == "done"

    curves = client.get(f"/benchmarks/{suite}/robustness", params={"metric": "angle_error"}).json()
    assert {c["controller_id"] for c in curves} == {"pid", "lqr"}
    assert [p["level"] for p in curves[0]["points"]] == [0.0, 0.5]
    assert curves[0]["points"][0]["relative"] == 1.0
    # comparison tables use the nominal level only: 2 seeds, not 4 pooled runs
    rows = client.get(f"/benchmarks/{suite}/summary", params={"env_id": "cartpole"}).json()
    assert {r["n_total"] for r in rows if r["metric_name"] == "return"} == {2}
    paired = client.get(
        f"/benchmarks/{suite}/paired",
        params={"env_id": "cartpole", "metric": "return", "a": "pid", "b": "lqr"},
    ).json()
    assert paired["n_total"] == 2


def test_robustness_endpoint_errors(client):
    assert client.get("/benchmarks/nope/robustness").status_code == 404
    suite = post(client, seeds=[0], environments=[{"id": "cartpole"}]).json()["suite_id"]
    wait(client, suite)
    assert client.get(f"/benchmarks/{suite}/robustness").status_code == 422
    bad = post(client, perturbation={"id": "nope", "levels": [1.0]})
    assert bad.status_code == 422
    assert bad.json()["detail"][0]["loc"] == ["body", "perturbation", "id"]
