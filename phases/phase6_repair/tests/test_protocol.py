"""Unit tests for the Phase 6 protocol helpers."""

from __future__ import annotations

import unittest

import torch

from phases.phase1_degradation.phase1.models import PrefillSegment, RelevantSpan, TaskExample
from phases.phase2_kv_cache.src.kv_utils import PositionTrackedCache
from phases.phase6_repair.src.protocol import (
    ALL_SPLIT_SPECS,
    CLEAN_SPLIT_SPECS,
    DEFAULT_SPLIT_SPEC,
    TAIL_LEAKY_SPLIT_SPECS,
    build_turn_n_keep_plan,
    materialize_context_partition,
    split_example_for_turn,
)


def _make_cache(seq_len: int) -> PositionTrackedCache:
    layer = (
        torch.zeros((1, 1, seq_len, 2), dtype=torch.float16),
        torch.zeros((1, 1, seq_len, 2), dtype=torch.float16),
    )
    return PositionTrackedCache((layer,), list(range(seq_len)))


class Phase6ProtocolTests(unittest.TestCase):
    def test_split_registry_covers_all_balanced_two_two_partitions(self) -> None:
        self.assertEqual(len(ALL_SPLIT_SPECS), 6)
        self.assertEqual(len(CLEAN_SPLIT_SPECS), 3)
        self.assertEqual(len(TAIL_LEAKY_SPLIT_SPECS), 3)
        self.assertEqual(len({spec.name for spec in ALL_SPLIT_SPECS}), 6)
        for spec in ALL_SPLIT_SPECS:
            self.assertEqual(set(spec.q1_indices) | set(spec.q2_indices), {0, 1, 2, 3})
            self.assertEqual(set(spec.q1_indices) & set(spec.q2_indices), set())
            self.assertEqual(len(spec.q1_indices), 2)
            self.assertEqual(len(spec.q2_indices), 2)

    def test_split_example_uses_balanced_indices_and_values_only_format(self) -> None:
        example = TaskExample(
            index=0,
            task_name="mq_niah_4q",
            task_family="niah",
            context="ctx",
            question="orig",
            answer_prefix="orig",
            outputs=["11", "22", "33", "44"],
            max_new_tokens=64,
            target_context_length=128,
            relevant_spans=[
                RelevantSpan(name=f"needle_{i}", kind="needle", char_start=i, char_end=i + 1, depth_fraction=0.1 * i)
                for i in range(1, 5)
            ],
            prefill_segments=[PrefillSegment(name="context", char_start=0, char_end=3)],
            metadata={"query_keys": ["k1", "k2", "k3", "k4"]},
        )

        q1 = split_example_for_turn(example, query_indices=DEFAULT_SPLIT_SPEC.q1_indices, split_name="q1")
        q2 = split_example_for_turn(example, query_indices=DEFAULT_SPLIT_SPEC.q2_indices, split_name="q2")

        self.assertEqual(q1.outputs, ["11", "44"])
        self.assertEqual(q2.outputs, ["22", "33"])
        self.assertEqual(q1.answer_prefix, " Answer:")
        self.assertIn("comma-separated", q1.question)
        self.assertEqual(q1.metadata["query_keys"], ["k1", "k4"])
        self.assertEqual(q2.metadata["query_keys"], ["k2", "k3"])

    def test_context_partition_keeps_context_budget_and_full_tail(self) -> None:
        cache = _make_cache(seq_len=7)
        q1_answer_ids = torch.tensor([1, 2], dtype=torch.long)
        keep_plan = build_turn_n_keep_plan(
            post_q1_cache=cache,
            q1_answer_ids=q1_answer_ids,
            context_len=5,
            sink_size=1,
            recency_window=1,
            pooling="max",
        )
        # Make the keep order deterministic for the unit test rather than relying on all-zero scores.
        keep_plan = keep_plan.__class__(
            context_len=keep_plan.context_len,
            tail_positions=keep_plan.tail_positions,
            mandatory_context_positions=keep_plan.mandatory_context_positions,
            ranked_candidate_positions=(2, 3, 1),
            importance_scores={0: 0.0, 1: 0.1, 2: 0.9, 3: 0.5, 4: 0.0},
        )

        partition = materialize_context_partition(
            full_post_q1_cache=cache,
            keep_plan=keep_plan,
            context_budget=3,
        )

        self.assertEqual(partition.kept_context_positions, (0, 2, 4))
        self.assertEqual(partition.evicted_context_positions, (1, 3))
        self.assertEqual(partition.compressed.positions, [0, 2, 4, 5, 6])
        self.assertEqual(partition.evicted.positions, [1, 3])


if __name__ == "__main__":
    unittest.main()
