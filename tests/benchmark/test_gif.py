import subprocess
import sys

import pytest

pytest.importorskip("PIL")

from rcl.benchmark.gif import main  # noqa: E402


def test_unknown_id_and_non_rendering_env_exit_2(tmp_path, capsys):
    out = tmp_path / "x.gif"
    assert main(["--env", "nope", "--controller", "pid", "--out", str(out)]) == 2
    assert main(["--env", "cartpole", "--controller", "pid", "--out", str(out)]) == 2
    err = capsys.readouterr().err
    assert "nope" in err and "does not support rgb_array" in err and not out.exists()


def test_invalid_params_exit_2(tmp_path, capsys):
    pytest.importorskip("pybullet")
    args = ["--env", "balance_bot", "--controller", "pid", "--params", '{"kp": "x"}']
    assert main([*args, "--out", str(tmp_path / "x.gif")]) == 2
    assert "kp" in capsys.readouterr().err


def test_classical_episode_becomes_an_animated_gif(tmp_path):
    pytest.importorskip("pybullet")
    from PIL import Image

    out = tmp_path / "bb.gif"
    args = ["--env", "balance_bot", "--controller", "lqr", "--env-params", '{"max_steps": 12}']
    assert main([*args, "--every", "3", "--out", str(out)]) == 0
    with Image.open(out) as gif:
        assert gif.size == (320, 240) and gif.n_frames == 5  # reset frame + 12 steps / 3
        assert gif.info["duration"] == 60  # real time: 3 steps x 0.02 s, in 10 ms units


def test_rl_controller_trains_then_renders_via_module_entry_point(tmp_path):
    pytest.importorskip("pybullet")
    out = tmp_path / "arm.gif"
    cmd = [
        sys.executable, "-m", "rcl.benchmark.gif", "--env", "planar_arm", "--controller", "ppo",
        "--params", '{"n_steps": 32, "batch_size": 32, "n_epochs": 1}', "--timesteps", "64",
        "--env-params", '{"max_steps": 4}', "--out", str(out),
    ]  # fmt: skip
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert out.exists() and out.stat().st_size > 0 and "frames" in r.stderr
