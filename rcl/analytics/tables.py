"""Comparison tables over a benchmark suite (unit of replication = the seed, DEFINITIONS.md).

Input is ``Repository.suite_metric_rows``: one run-level value per (env, controller, seed,
metric). Nothing here names a concrete environment, controller or metric.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SummaryRow:
    """Cross-seed summary of one metric for one (environment, controller) cell.

    ``n_total`` runs reported the metric, ``n_valid`` of them have a value (``undefined`` runs
    are excluded from the statistics), ``n_censored`` of the valid ones are censored bounds.
    """

    env_id: str
    controller_id: str
    metric_name: str
    unit: str
    n_valid: int
    n_total: int
    n_censored: int
    mean: float
    std: float  # sample standard deviation (ddof = 1); NaN if n_valid < 2


def summarize(rows: list[dict[str, Any]]) -> list[SummaryRow]:
    """Group run-level rows by (env, controller, metric) and summarise across seeds.

    Group order follows first appearance in ``rows`` (the repository orders by env, controller,
    seed), so metrics keep the order the manager computed them in.
    """
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["env_id"], r["controller_id"], r["metric_name"])].append(r)
    out = []
    for (env_id, controller_id, metric), g in groups.items():
        valid = np.array(
            [r["value"] for r in g if r["value"] is not None and r["status"] != "undefined"]
        )
        out.append(
            SummaryRow(
                env_id=env_id,
                controller_id=controller_id,
                metric_name=metric,
                unit=g[0]["unit"],
                n_valid=len(valid),
                n_total=len(g),
                n_censored=sum(
                    1 for r in g if r["status"] == "censored" and r["value"] is not None
                ),
                mean=float(valid.mean()) if len(valid) else math.nan,
                std=float(valid.std(ddof=1)) if len(valid) > 1 else math.nan,
            )
        )
    return out


def by_metric(summary: list[SummaryRow], env_id: str) -> dict[str, dict[str, SummaryRow]]:
    """``{metric: {controller: row}}`` for one environment: the shape of a comparison table
    (one table per metric, one row per controller)."""
    table: dict[str, dict[str, SummaryRow]] = {}
    for row in summary:
        if row.env_id == env_id:
            table.setdefault(row.metric_name, {})[row.controller_id] = row
    return table
