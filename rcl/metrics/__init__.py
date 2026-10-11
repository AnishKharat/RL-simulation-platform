"""Pure metric functions over trajectories. Definitions: rcl/metrics/DEFINITIONS.md."""

from rcl.metrics.dynamics import overshoot, rise_time, settling_time, steady_state_error
from rcl.metrics.effort import control_effort, energy_usage
from rcl.metrics.metrics import (
    METRIC_INFO,
    METRICS,
    MetricInfo,
    MetricValue,
    aggregate,
    angle_error,
    compute_metrics,
    episode_return,
    wrap_angle,
)
from rcl.metrics.outcome import success_rate, time_to_success
from rcl.metrics.robustness import failure_count, recovery_time

# Phase 2 metrics join the manager's fixed list (selected by the manager, never by users).
METRICS.update(
    success_rate=success_rate,  # same order as METRIC_INFO below
    time_to_success=time_to_success,
    settling_time=settling_time,
    rise_time=rise_time,
    overshoot=overshoot,
    steady_state_error=steady_state_error,
    energy_usage=energy_usage,
    control_effort=control_effort,
    failure_count=failure_count,
    recovery_time=recovery_time,
)
METRIC_INFO.extend(
    [
        MetricInfo(
            name="success_rate",
            description="Fraction of episodes in which the task tolerance was held (headline).",
            unit="fraction",
            higher_is_better=True,
            ci_method="bootstrap",
            headline=True,
        ),
        MetricInfo(
            name="time_to_success",
            description="Time at which the successful in-tolerance streak began (censored: T_max).",
            unit="s",
            higher_is_better=False,
            ci_method="bootstrap",
            headline=True,
        ),
        MetricInfo(
            name="settling_time",
            description="Time until the error stays within 2% of the step size (censored: T_max).",
            unit="s",
            higher_is_better=False,
            ci_method="bootstrap",
            companions=[{"name": "settled", "unit": "bool"}],
        ),
        MetricInfo(
            name="rise_time",
            description="Time from 10% to 90% of the commanded step.",
            unit="s",
            higher_is_better=False,
            ci_method="bootstrap",
        ),
        MetricInfo(
            name="overshoot",
            description="Percent of the step size beyond the setpoint.",
            unit="%",
            higher_is_better=False,
        ),
        MetricInfo(
            name="steady_state_error",
            description="Mean error over the final 10% of the episode.",
            unit=None,
            higher_is_better=False,
        ),
        MetricInfo(
            name="energy_usage",
            description="Actuation energy proxy: sum of squared action norm times dt.",
            unit="action^2*s",
            higher_is_better=False,
        ),
        MetricInfo(
            name="control_effort",
            description="Time-integrated action magnitude.",
            unit="action*s",
            higher_is_better=False,
            companions=[{"name": "control_effort_rate", "unit": "action"}],
        ),
        MetricInfo(
            name="failure_count",
            description="Fraction of episodes that ended in failure (terminated).",
            unit="fraction",
            higher_is_better=False,
            ci_method="bootstrap",
        ),
        MetricInfo(
            name="recovery_time",
            description="Time from the first perturbation event till the state stays in tolerance.",
            unit="s",
            higher_is_better=False,
            ci_method="bootstrap",
        ),
    ]
)

__all__ = [
    "METRIC_INFO",
    "METRICS",
    "MetricInfo",
    "MetricValue",
    "aggregate",
    "angle_error",
    "compute_metrics",
    "episode_return",
    "control_effort",
    "energy_usage",
    "overshoot",
    "rise_time",
    "settling_time",
    "steady_state_error",
    "wrap_angle",
]
