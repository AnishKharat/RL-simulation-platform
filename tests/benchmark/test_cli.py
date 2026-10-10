import json
import subprocess
import sys

from rcl.benchmark.cli import main

PPO = {"n_steps": 32, "batch_size": 32, "n_epochs": 1}
SAC = {"learning_starts": 16, "buffer_size": 1000, "batch_size": 32}


def four_way_spec(tmp_path):
    spec = {
        "name": "pid-lqr-ppo-sac",
        "environments": [{"id": "cartpole"}],
        "controllers": [
            {"id": "pid"},
            {"id": "lqr"},
            {"id": "ppo", "params": PPO},
            {"id": "sac", "params": SAC},
        ],
        "seeds": [0, 1],
        "episodes": 1,
        "training": {"total_timesteps": 64},
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return path


def test_grid_options_write_report_plots_and_csv(tmp_path, capsys):
    out = tmp_path / "out"
    code = main(
        ["--envs", "cartpole", "--controllers", "pid", "lqr", "--seeds", "2", "--episodes", "1",
         "--out", str(out), "--name", "demo"]
    )  # fmt: skip
    assert code == 0
    md = (out / "report.md").read_text(encoding="utf-8")
    assert capsys.readouterr().out == md  # the table is also printed
    assert md.startswith("# demo") and "| pid |" in md and "| lqr |" in md
    assert (out / "report.html").read_text(encoding="utf-8").startswith("<!doctype html>")
    assert (out / "plots" / "cartpole_return.svg").exists()
    header, *rows = (out / "summary.csv").read_text(encoding="utf-8").splitlines()
    assert header.startswith("env_id,controller_id,metric_name") and len(rows) > 10
    assert (out / "rcl.db").exists()


def test_exit_criterion_four_controllers_table_with_intervals_reproduces(tmp_path):
    spec = four_way_spec(tmp_path)
    outs = [tmp_path / "run1", tmp_path / "run2"]
    for out in outs:
        assert main(["--spec", str(spec), "--out", str(out)]) == 0
    md = (outs[0] / "report.md").read_text(encoding="utf-8")
    for ctl in ("pid", "lqr", "ppo", "sac"):
        assert f"| {ctl} |" in md
    assert "95% interval" in md and "[" in md and "![return on cartpole]" in md
    for name in ("report.md", "report.html", "summary.csv", "plots/cartpole_return.svg"):
        assert (outs[0] / name).read_bytes() == (outs[1] / name).read_bytes(), name  # same seeds


def test_bad_specs_exit_2_without_running(tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["--controllers", "nope", "--out", str(out)]) == 2
    assert main(["--envs", "cartpole_discrete", "--controllers", "pid", "--out", str(out)]) == 2
    assert main(["--envs", "cartpole", "--controllers", "ppo", "--timesteps", "64", "--seeds", "0",
                 "--out", str(out)]) == 2  # fmt: skip  # zero seeds is invalid
    err = capsys.readouterr().err
    assert "nope" in err and "no compatible" in err
    assert not (out / "report.md").exists()


def test_defaults_come_from_the_registry_and_incompatible_pairs_are_skipped(tmp_path):
    out = tmp_path / "out"
    spec = {"name": "all", "seeds": [0], "episodes": 1, "training": {"total_timesteps": 64}}
    # no ids given: every registered environment x controller; ppo/sac use tiny params via --spec
    spec.update(
        environments=[{"id": "cartpole"}, {"id": "cartpole_discrete"}],
        controllers=[{"id": "pid"}, {"id": "ppo", "params": PPO}],
    )
    path = tmp_path / "s.json"
    path.write_text(json.dumps(spec))
    assert main(["--spec", str(path), "--out", str(out)]) == 0
    csv = (out / "summary.csv").read_text(encoding="utf-8")
    assert "cartpole_discrete,ppo" in csv and "cartpole_discrete,pid" not in csv


def test_module_entry_point_runs_as_a_subprocess(tmp_path):
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, "-m", "rcl.benchmark", "--envs", "cartpole", "--controllers", "pid",
         "--seeds", "2", "--episodes", "1", "--out", str(out)],
        capture_output=True, text=True, timeout=300,
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    assert "| pid |" in r.stdout and (out / "report.md").exists()
