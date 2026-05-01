"""Reporting helpers for Phase 6 result artifacts."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path

DEFAULT_FIELD_MAP: tuple[tuple[str, str], ...] = (
    ("condition_b", "mean_condition_b"),
    ("b_match", "mean_b_match"),
    ("idlekv", "mean_idlekv"),
    ("random_k", "mean_random_k"),
    ("oldest_k", "mean_oldest_k"),
    ("oracle_k", "mean_oracle_k"),
    ("selection_lift", "mean_selection_lift"),
    ("oracle_lift", "mean_oracle_lift"),
    ("pct_idlekv_gt_b_match", "pct_idlekv_gt_b_match"),
    ("pct_idlekv_lt_b_match", "pct_idlekv_lt_b_match"),
    ("repair_ms", "mean_idlekv_repair_ms"),
)


def _k_value(k_label: str) -> int:
    if not k_label.startswith("k"):
        raise ValueError(f"Expected k-prefixed label, got: {k_label}")
    return int(k_label[1:])


def frontier_rows(
    aggregate_section: Mapping[str, Mapping[str, float]],
    field_map: Iterable[tuple[str, str]] = DEFAULT_FIELD_MAP,
) -> list[dict[str, float | int]]:
    """Return numerically sorted frontier rows for one aggregate section."""
    rows: list[dict[str, float | int]] = []
    for k_label in sorted(aggregate_section, key=_k_value):
        metrics = aggregate_section[k_label]
        row: dict[str, float | int] = {"k": _k_value(k_label)}
        for out_key, metric_key in field_map:
            if metric_key in metrics:
                row[out_key] = metrics[metric_key]
        rows.append(row)
    return rows


def split_rows(
    by_task_section: Mapping[str, Mapping[str, Mapping[str, float]]],
    field_map: Iterable[tuple[str, str]] = DEFAULT_FIELD_MAP,
) -> list[dict[str, float | int | str]]:
    """Flatten per-task aggregates into task-tagged frontier rows."""
    rows: list[dict[str, float | int | str]] = []
    for task_name in sorted(by_task_section):
        for row in frontier_rows(by_task_section[task_name], field_map=field_map):
            rows.append({"task": task_name, **row})
    return rows


def write_csv(rows: list[Mapping[str, object]], path: Path) -> None:
    """Write rows to CSV with the field order defined by the first row."""
    if not rows:
        raise ValueError("Cannot write empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
