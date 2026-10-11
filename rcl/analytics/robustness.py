"""Degradation curves of a robustness sweep (DEFINITIONS.md, Phase 4 metrics).

Input is ``Repository.suite_metric_rows`` of a sweep suite: the level of a row is the sweep
perturbation's ``magnitude_param`` in the row's stack. Per level the cross-seed summary is the
usual :func:`summarize` (same seeds, same interval rules). Nothing here names a perturbation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from rcl.analytics.tables import SummaryRow, summarize
from rcl.metrics.robustness import breakdown_magnitude, relative_curve


@dataclass(frozen=True)
class CurvePoint:
    level: float
    summary: SummaryRow
    relative: float  # mean / mean at the nominal level; NaN if that is 0 or missing


@dataclass(frozen=True)
class Curve:
    env_id: str
    controller_id: str
    metric_name: str
    points: list[CurvePoint]  # ascending level
    breakdown: float  # first level above nominal where the mean falls below half; else NaN


def _level(row: dict[str, Any], perturbation_id: str, magnitude_param: str) -> float | None:
    for p in row["perturbations"]:
        if p["id"] == perturbation_id:
            return float(p["params"][magnitude_param])
    return None


def degradation_curves(
    rows: list[dict[str, Any]],
    perturbation_id: str,
    magnitude_param: str,
    nominal: float,
    metric: str = "success_rate",
) -> list[Curve]:
    """One :class:`Curve` per (environment, controller) for ``metric``. ``breakdown`` is only
    meaningful for a higher-is-better fraction such as ``success_rate``."""
    by_level: dict[float, list[dict[str, Any]]] = {}
    for r in rows:
        lv = _level(r, perturbation_id, magnitude_param)
        if lv is not None and r["metric_name"] == metric:
            by_level.setdefault(lv, []).append(r)
    cells: dict[tuple[str, str], dict[float, SummaryRow]] = {}
    for lv in sorted(by_level):
        for s in summarize(by_level[lv]):
            cells.setdefault((s.env_id, s.controller_id), {})[lv] = s
    curves = []
    for (env_id, ctl), by_lv in cells.items():
        means = {lv: s.mean for lv, s in by_lv.items()}
        has_nominal = nominal in means and not math.isnan(means[nominal])
        rel = relative_curve(means, nominal) if has_nominal else dict.fromkeys(means, math.nan)
        curves.append(
            Curve(
                env_id,
                ctl,
                metric,
                [CurvePoint(lv, s, rel[lv]) for lv, s in by_lv.items()],
                breakdown_magnitude(means, nominal) if has_nominal else math.nan,
            )
        )
    return curves
