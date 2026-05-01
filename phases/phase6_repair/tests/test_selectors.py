"""Unit tests for the Phase 6 selector helpers."""

from __future__ import annotations

import unittest

import torch

from phases.phase2_kv_cache.src.kv_utils import PositionTrackedCache
from phases.phase6_repair.src.selectors import (
    pack_anchor_bursts,
    score_evicted_positions,
    select_idlekv_positions,
    select_oldest_positions,
    select_oracle_positions,
)


def _make_evicted_cache() -> PositionTrackedCache:
    keys = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]]],
        dtype=torch.float16,
    )
    values = torch.zeros_like(keys)
    return PositionTrackedCache(((keys, values),), [10, 11, 12])


class Phase6SelectorTests(unittest.TestCase):
    def test_score_evicted_positions_prefers_matching_key(self) -> None:
        cache = _make_evicted_cache()
        query_rows = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32)
        scores = score_evicted_positions(query_rows=query_rows, evicted_cache=cache, pooling="max")
        self.assertGreater(scores[10], scores[11])
        self.assertGreater(scores[10], scores[12])

    def test_pack_anchor_bursts_hits_exact_k_then_stops(self) -> None:
        selected = pack_anchor_bursts(
            anchor_positions=[10],
            available_positions=list(range(8, 16)),
            k=5,
            left=2,
            right=2,
            backfill_positions=list(range(8, 16)),
        )
        self.assertEqual(selected, [8, 9, 10, 11, 12])

    def test_selectors_respect_relevant_positions(self) -> None:
        evicted = [10, 11, 12, 13, 14, 15]
        q2_scores = {10: 0.1, 11: 0.2, 12: 0.9, 13: 0.8, 14: 0.05, 15: 0.04}
        turn_scores = {position: float(position) for position in evicted}

        idlekv = select_idlekv_positions(
            evicted_positions=evicted,
            q2_scores=q2_scores,
            turn_n_scores=turn_scores,
            k=3,
            left=0,
            right=0,
        )
        oldest = select_oldest_positions(
            evicted_positions=evicted,
            k=3,
            left=0,
            right=0,
        )
        oracle = select_oracle_positions(
            evicted_positions=evicted,
            relevant_positions=[13],
            q2_scores=q2_scores,
            turn_n_scores=turn_scores,
            k=3,
            left=1,
            right=1,
        )

        self.assertEqual(idlekv, [12, 13, 11])
        self.assertEqual(oldest, [10, 11, 12])
        self.assertEqual(oracle, [12, 13, 14])


if __name__ == "__main__":
    unittest.main()
