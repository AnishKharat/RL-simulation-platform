"""Cross-run comparison: summary tables and statistics."""

from rcl.analytics.report import Report, build_report
from rcl.analytics.tables import SummaryRow, by_metric, paired_difference, summarize

__all__ = ["Report", "build_report", "SummaryRow", "by_metric", "paired_difference", "summarize"]
