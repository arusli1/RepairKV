"""Unit tests for the Phase 6 runner helpers."""

from __future__ import annotations

import unittest

import torch

from phases.phase2_kv_cache.src.kv_utils import PositionTrackedCache
from phases.phase6_repair.src.runner import (
    _artifact_path,
    _condition_label,
    _restore_positions,
    build_config,
    summarize_rows,
)


def _make_cache(positions: list[int]) -> PositionTrackedCache:
    seq_len = len(positions)
    keys = torch.arange(seq_len, dtype=torch.float16).reshape(1, 1, seq_len, 1)
    values = (keys + 100).clone()
    return PositionTrackedCache(((keys, values),), positions)


class Phase6RunnerTests(unittest.TestCase):
    def test_build_config_uses_stage_defaults_and_overrides(self) -> None:
        config = build_config(
            stage="smoke",
            task="clean_suite",
            num_samples=7,
            k_values=[24],
            conditions=["A", "B", "B_match", "IdleKV", "WrongQ-K", "Oracle-K"],
            base_context_budget=640,
            recency_window=64,
        )
        self.assertEqual(config.stage, "smoke")
        self.assertEqual(config.task, "clean_suite")
        self.assertEqual(len(config.split_specs), 3)
        self.assertEqual(config.num_samples, 7)
        self.assertEqual(config.k_values, (24,))
        self.assertEqual(config.conditions, ("A", "B", "B_match", "IdleKV", "WrongQ-K", "Oracle-K"))
        self.assertEqual(config.base_context_budget, 640)
        self.assertEqual(config.recency_window, 64)

    def test_build_config_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            build_config(stage="smoke", task="mq_niah_4q_split_14_to_23", k_values=[0])
        with self.assertRaises(ValueError):
            build_config(stage="smoke", task="mq_niah_4q_split_14_to_23", conditions=["A", "Bad"])
        with self.assertRaises(ValueError):
            build_config(stage="smoke", task="mq_niah_4q_split_14_to_23", base_context_budget=0)
        with self.assertRaises(ValueError):
            build_config(stage="smoke", task="mq_niah_4q_split_14_to_23", recency_window=-1)

    def test_artifact_path_includes_budget_and_recency(self) -> None:
        config = build_config(stage="smoke", task="clean_suite", num_samples=3, k_values=[12], base_context_budget=768, recency_window=64)
        self.assertIn("clean_suite_b768_r64_n3_k12_c", str(_artifact_path(config)))

    def test_condition_label_strips_symbols(self) -> None:
        label = _condition_label(["A", "B_match", "IdleKV", "WrongQ-K", "Oracle-K"])
        self.assertEqual(label, "a-bmatch-idlekv-wrongqk-oraclek")

    def test_restore_positions_injects_selected_tokens_in_sorted_order(self) -> None:
        active = _make_cache([0, 3])
        evicted = _make_cache([1, 2, 4])
        repaired, timing = _restore_positions(
            active_cache=active,
            evicted_cache=evicted,
            selected_positions=[4, 1],
        )

        self.assertEqual(repaired.positions, [0, 1, 3, 4])
        self.assertEqual(int(timing["restored_count"]), 2)
        self.assertGreaterEqual(float(timing["transfer_ms"]), 0.0)
        self.assertGreaterEqual(float(timing["inject_ms"]), 0.0)

    def test_summarize_rows_reports_lifts_and_overlap(self) -> None:
        rows = [
            {
                "k": 100,
                "task": "mq_niah_4q_split_14_to_23",
                "q1_score": 1.0,
                "condition_a_score": 1.0,
                "condition_b_score": 0.0,
                "b_match_score": 0.0,
                "idlekv_score": 1.0,
                "idlekv_overlap_fraction": 0.5,
                "idlekv_selection_s": 0.001,
                "idlekv_transfer_ms": 2.0,
                "idlekv_inject_ms": 1.0,
                "wrong_q_k_score": 0.0,
                "oracle_k_score": 1.0,
            },
            {
                "k": 100,
                "task": "mq_niah_4q_split_14_to_23",
                "q1_score": 1.0,
                "condition_a_score": 1.0,
                "condition_b_score": 0.5,
                "b_match_score": 0.5,
                "idlekv_score": 0.5,
                "idlekv_overlap_fraction": 0.0,
                "idlekv_selection_s": 0.002,
                "idlekv_transfer_ms": 3.0,
                "idlekv_inject_ms": 1.0,
                "wrong_q_k_score": 0.5,
                "oracle_k_score": 1.0,
            },
        ]

        summary = summarize_rows(rows)
        k100 = summary["k100"]
        self.assertEqual(k100["mean_condition_b"], 0.25)
        self.assertEqual(k100["mean_b_match"], 0.25)
        self.assertEqual(k100["mean_idlekv"], 0.75)
        self.assertEqual(k100["mean_selection_lift"], 0.5)
        self.assertEqual(k100["pct_idlekv_gt_b_match"], 0.5)
        self.assertEqual(k100["pct_idlekv_lt_b_match"], 0.0)
        self.assertEqual(k100["mean_wrong_q_k"], 0.25)
        self.assertEqual(k100["mean_oracle_k"], 1.0)

    def test_summarize_rows_groups_multi_split_runs(self) -> None:
        rows = [
            {
                "k": 12,
                "task": "mq_niah_4q_split_14_to_23",
                "q1_score": 1.0,
                "condition_a_score": 1.0,
                "condition_b_score": 0.0,
                "b_match_score": 0.0,
                "idlekv_score": 0.5,
                "idlekv_overlap_fraction": 0.0,
                "idlekv_selection_s": 0.001,
                "idlekv_transfer_ms": 1.0,
                "idlekv_inject_ms": 1.0,
            },
            {
                "k": 12,
                "task": "mq_niah_4q_split_24_to_13",
                "q1_score": 1.0,
                "condition_a_score": 1.0,
                "condition_b_score": 0.0,
                "b_match_score": 0.0,
                "idlekv_score": 1.0,
                "idlekv_overlap_fraction": 1.0,
                "idlekv_selection_s": 0.001,
                "idlekv_transfer_ms": 1.0,
                "idlekv_inject_ms": 1.0,
            },
        ]

        summary = summarize_rows(rows)
        self.assertIn("overall", summary)
        self.assertIn("by_task", summary)
        self.assertEqual(summary["overall"]["k12"]["mean_idlekv"], 0.75)
        self.assertEqual(summary["by_task"]["mq_niah_4q_split_14_to_23"]["k12"]["mean_idlekv"], 0.5)


if __name__ == "__main__":
    unittest.main()
