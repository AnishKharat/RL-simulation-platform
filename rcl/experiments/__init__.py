"""Experiment orchestration: rollout, run execution."""

from rcl.experiments.jobs import JobManager
from rcl.experiments.manager import ConfigError, RunResult, derive_seeds, execute, resolve_config
from rcl.experiments.rollout import rollout
from rcl.metrics import METRIC_INFO, MetricInfo

__all__ = [
    "METRIC_INFO",
    "JobManager",
    "MetricInfo",
    "ConfigError",
    "RunResult",
    "derive_seeds",
    "execute",
    "resolve_config",
    "rollout",
]
