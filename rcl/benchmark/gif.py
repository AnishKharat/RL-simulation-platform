"""``python -m rcl.benchmark.gif``: render one evaluation episode of a controller to a GIF.

The episode is exactly episode 0 of a benchmark run with the same seed: same derived seeds, same
training (RL), same initial state (it goes through ``execute`` with a frame-recording evaluation
wrapper that does not change dynamics). Needs an environment with ``rgb_array`` rendering and the
optional ``render`` extra (Pillow).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rcl.core.config import ExperimentConfig
from rcl.core.environment import BaseEnvironment, EnvWrapper
from rcl.experiments import ConfigError, execute
from rcl.registry import UnknownIdError, discover, get_registry

log = logging.getLogger(__name__)


class FrameRecorder(EnvWrapper):
    """Keeps an RGB frame after reset and after every ``every``-th step. Read-only."""

    def __init__(self, env: BaseEnvironment, every: int = 1) -> None:
        super().__init__(env)
        self.every, self._k = max(1, every), 0
        self.frames: list[np.ndarray] = []

    def _grab(self) -> None:
        frame = self._inner.render()
        if frame is None:
            raise RuntimeError("environment returned no frame")
        self.frames.append(np.asarray(frame, dtype=np.uint8))

    def reset(self, **kwargs: Any) -> Any:
        out = super().reset(**kwargs)
        if not self.frames:  # first episode only
            self._grab()
        return out

    def step(self, action: Any) -> Any:
        out = super().step(action)
        self._k += 1
        if self._k % self.every == 0:
            self._grab()
        return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m rcl.benchmark.gif", description=__doc__.split("\n")[0]
    )
    p.add_argument("--env", required=True, help="environment id (must support rgb_array)")
    p.add_argument("--controller", required=True, help="controller id")
    p.add_argument("--params", default="{}", help="controller params as JSON")
    p.add_argument("--env-params", default="{}", help="environment params as JSON")
    p.add_argument("--seed", type=int, default=0, help="run seed (same as a benchmark seed)")
    p.add_argument("--timesteps", type=int, default=20_000, help="training budget for RL")
    p.add_argument("--every", type=int, default=1, help="keep every n-th control step")
    p.add_argument("--fps", type=float, help="GIF frame rate (default: real time)")
    p.add_argument("--out", type=Path, default=Path("episode.gif"))
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """0 = GIF written; 2 = bad request (unknown id, no rendering, invalid params, no Pillow)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    try:
        from PIL import Image
    except ImportError:
        sys.stderr.write('error: Pillow is required: pip install -e ".[render]"\n')
        return 2
    discover()
    reg = get_registry()
    try:
        env_entry, ctl_entry = (
            reg.get("environment", args.env),
            reg.get("controller", args.controller),
        )
    except UnknownIdError as exc:
        sys.stderr.write(f"error: {exc.args[0]}\n")
        return 2
    if "rgb_array" not in getattr(env_entry.cls, "metadata", {}).get("render_modes", []):
        sys.stderr.write(f"error: environment {args.env!r} does not support rgb_array rendering\n")
        return 2
    trains = ctl_entry.metadata.requires_training
    try:
        config = ExperimentConfig.model_validate(
            {
                "name": f"gif:{args.env}:{args.controller}",
                "seed": args.seed,
                "environment": {"id": args.env, "params": json.loads(args.env_params)},
                "controller": {"id": args.controller, "params": json.loads(args.params)},
                "evaluation": {"episodes": 1},
                "training": {"total_timesteps": args.timesteps} if trains else None,
            }
        )
        recorder: list[FrameRecorder] = []

        def wrap(env: BaseEnvironment) -> BaseEnvironment:
            recorder.append(FrameRecorder(env, args.every))
            return recorder[0]

        result = execute(config, eval_wrapper=wrap)
    except ConfigError as exc:
        for e in exc.errors:
            sys.stderr.write(f"error: {'.'.join(str(p) for p in e['loc'])}: {e['msg']}\n")
        return 2
    except ValueError as exc:  # malformed JSON or config schema
        sys.stderr.write(f"error: {exc}\n")
        return 2
    frames = recorder[0].frames
    (traj,) = result.trajectories
    fps = args.fps or min(50.0, 1.0 / (traj.dt * args.every))
    images = [Image.fromarray(f) for f in frames]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        args.out,
        save_all=True,
        append_images=images[1:],
        duration=10 * max(1, round(100 / fps)),
        loop=0,
    )
    outcome = "success" if traj.success else "failure" if traj.terminated else "time limit"
    log.info("%d frames, %d steps (%s) -> %s", len(frames), traj.n_steps, outcome, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
