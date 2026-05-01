"""Unit tests for Phase 6 reporting helpers."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from phases.phase6_repair.src.reporting import frontier_rows, split_rows, write_csv


class Phase6ReportingTests(unittest.TestCase):
    def test_frontier_rows_sort_k_labels_and_map_metrics(self) -> None:
        rows = frontier_rows(
            {
                "k32": {"mean_idlekv": 0.6, "mean_b_match": 0.1},
                "k8": {"mean_idlekv": 0.3, "mean_b_match": 0.1},
            }
        )
        self.assertEqual([row["k"] for row in rows], [8, 32])
        self.assertEqual(rows[0]["idlekv"], 0.3)
        self.assertEqual(rows[1]["b_match"], 0.1)

    def test_split_rows_attach_task_name(self) -> None:
        rows = split_rows(
            {
                "task_a": {"k8": {"mean_idlekv": 0.25}},
                "task_b": {"k16": {"mean_idlekv": 0.5}},
            }
        )
        self.assertEqual(rows[0]["task"], "task_a")
        self.assertEqual(rows[1]["task"], "task_b")
        self.assertEqual(rows[1]["k"], 16)

    def test_write_csv_uses_first_row_field_order(self) -> None:
        rows = [
            {"k": 8, "b_match": 0.1, "idlekv": 0.3},
            {"k": 16, "b_match": 0.1, "idlekv": 0.4},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "frontier.csv"
            write_csv(rows, out_path)
            with out_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, ["k", "b_match", "idlekv"])
                loaded = list(reader)
        self.assertEqual(loaded[1]["idlekv"], "0.4")


if __name__ == "__main__":
    unittest.main()
