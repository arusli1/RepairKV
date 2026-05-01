"""Selection and burst-packing helpers for the Phase 6 repair study."""

from __future__ import annotations

import math
import random
from typing import Sequence

import torch

from phases.phase2_kv_cache.src.kv_utils import PositionTrackedCache


def score_evicted_positions(
    *,
    query_rows: torch.Tensor,
    evicted_cache: PositionTrackedCache,
    pooling: str = "max",
) -> dict[int, float]:
    """Score evicted positions by pooled Q2 attention over the evicted pool only."""
    if pooling not in {"max", "mean"}:
        raise ValueError(f"pooling must be 'max' or 'mean', got {pooling!r}.")
    if len(evicted_cache) == 0:
        return {}
    if not isinstance(query_rows, torch.Tensor) or query_rows.ndim != 4:
        raise ValueError("query_rows must have shape [n_layers, n_kv_heads, q_len, head_dim].")

    n_layers = len(evicted_cache.kv)
    if int(query_rows.shape[0]) != n_layers:
        raise ValueError(f"query_rows layer count {int(query_rows.shape[0])} does not match cache layers {n_layers}.")

    layer_scores: list[torch.Tensor] = []
    for layer_index, (key, _) in enumerate(evicted_cache.kv):
        query_layer = query_rows[layer_index]
        key_float = key.detach().to("cpu", dtype=torch.float32)[0]
        scores = torch.matmul(query_layer, key_float.transpose(-2, -1)) / math.sqrt(float(key_float.shape[-1]))
        scores = torch.softmax(scores, dim=-1)
        if pooling == "max":
            pooled = scores.amax(dim=1).mean(dim=0)
        else:
            pooled = scores.mean(dim=(0, 1))
        layer_scores.append(pooled)

    importance = torch.stack(layer_scores, dim=0).mean(dim=0)
    return {
        int(position): float(importance[dense_index].item())
        for dense_index, position in enumerate(evicted_cache.positions)
    }


def rank_positions(
    positions: Sequence[int],
    *,
    primary_scores: dict[int, float],
    secondary_scores: dict[int, float] | None = None,
) -> list[int]:
    """Sort positions by descending primary/secondary score and ascending position."""
    return sorted(
        (int(position) for position in positions),
        key=lambda position: (
            -float(primary_scores.get(int(position), 0.0)),
            -float((secondary_scores or {}).get(int(position), 0.0)),
            int(position),
        ),
    )


def pack_anchor_bursts(
    *,
    anchor_positions: Sequence[int],
    available_positions: Sequence[int],
    k: int,
    left: int,
    right: int,
    backfill_positions: Sequence[int] | None = None,
) -> list[int]:
    """Expand anchors into local bursts, then backfill singles until exactly K positions are selected."""
    target_k = max(int(k), 0)
    if target_k == 0:
        return []

    available = {int(position) for position in available_positions}
    selected: list[int] = []
    selected_set: set[int] = set()

    def _try_add(position: int) -> bool:
        candidate = int(position)
        if candidate not in available or candidate in selected_set:
            return False
        selected.append(candidate)
        selected_set.add(candidate)
        return len(selected) >= target_k

    for anchor in anchor_positions:
        anchor_int = int(anchor)
        for position in range(anchor_int - int(left), anchor_int + int(right) + 1):
            if _try_add(position):
                return selected

    for position in backfill_positions or ():
        if _try_add(int(position)):
            return selected

    return selected


def select_idlekv_positions(
    *,
    evicted_positions: Sequence[int],
    q2_scores: dict[int, float],
    turn_n_scores: dict[int, float],
    k: int,
    left: int,
    right: int,
) -> list[int]:
    """Select K positions by Q2 score, then turn-N score, then burst-pack locally."""
    ranked = rank_positions(
        evicted_positions,
        primary_scores=q2_scores,
        secondary_scores=turn_n_scores,
    )
    return pack_anchor_bursts(
        anchor_positions=ranked,
        available_positions=evicted_positions,
        k=int(k),
        left=int(left),
        right=int(right),
        backfill_positions=ranked,
    )


def select_random_positions(
    *,
    evicted_positions: Sequence[int],
    k: int,
    left: int,
    right: int,
    seed: int,
) -> list[int]:
    """Deterministic random ablation with the same burst-packing rule."""
    rng = random.Random(int(seed))
    anchors = [int(position) for position in evicted_positions]
    rng.shuffle(anchors)
    return pack_anchor_bursts(
        anchor_positions=anchors,
        available_positions=evicted_positions,
        k=int(k),
        left=int(left),
        right=int(right),
        backfill_positions=anchors,
    )


def select_oldest_positions(
    *,
    evicted_positions: Sequence[int],
    k: int,
    left: int,
    right: int,
) -> list[int]:
    """Recency-free baseline: prefer the oldest evicted positions first."""
    anchors = sorted(int(position) for position in evicted_positions)
    return pack_anchor_bursts(
        anchor_positions=anchors,
        available_positions=evicted_positions,
        k=int(k),
        left=int(left),
        right=int(right),
        backfill_positions=anchors,
    )


def select_oracle_positions(
    *,
    evicted_positions: Sequence[int],
    relevant_positions: Sequence[int],
    q2_scores: dict[int, float],
    turn_n_scores: dict[int, float],
    k: int,
    left: int,
    right: int,
) -> list[int]:
    """Gold hindsight selector with the same burst-packing rule as IdleKV."""
    available = {int(position) for position in evicted_positions}
    relevant_anchors = rank_positions(
        (int(position) for position in relevant_positions if int(position) in available),
        primary_scores=q2_scores,
        secondary_scores=turn_n_scores,
    )
    backfill = rank_positions(
        evicted_positions,
        primary_scores=q2_scores,
        secondary_scores=turn_n_scores,
    )
    return pack_anchor_bursts(
        anchor_positions=relevant_anchors,
        available_positions=evicted_positions,
        k=int(k),
        left=int(left),
        right=int(right),
        backfill_positions=backfill,
    )

