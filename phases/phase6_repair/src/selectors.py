"""Selection and burst-packing helpers for the Phase 6 repair study."""

from __future__ import annotations

import itertools
import math
import random
from typing import Sequence

import torch

from phases.phase2_kv_cache.src.kv_utils import PositionTrackedCache


def score_evicted_positions(
    *,
    query_rows: torch.Tensor,
    evicted_cache: PositionTrackedCache,
    active_cache: PositionTrackedCache | None = None,
    pooling: str = "max",
) -> dict[int, float]:
    """Score evicted positions against the active+evicted key set.

    `query_rows` may contain either proxy rows or exact extracted Q projections.
    When `active_cache` is provided, the attention normalization is performed over
    the concatenated active and evicted keys so the returned scores better match
    true decode-time competition for attention mass.
    """
    if pooling not in {"max", "mean"}:
        raise ValueError(f"pooling must be 'max' or 'mean', got {pooling!r}.")
    if len(evicted_cache) == 0:
        return {}
    if not isinstance(query_rows, torch.Tensor) or query_rows.ndim != 4:
        raise ValueError("query_rows must have shape [n_layers, n_query_heads, q_len, head_dim].")

    n_layers = len(evicted_cache.kv)
    if int(query_rows.shape[0]) != n_layers:
        raise ValueError(f"query_rows layer count {int(query_rows.shape[0])} does not match cache layers {n_layers}.")
    if active_cache is not None and len(active_cache.kv) != n_layers:
        raise ValueError(
            f"active_cache layer count {len(active_cache.kv)} does not match evicted cache layers {n_layers}."
        )

    layer_scores: list[torch.Tensor] = []
    for layer_index, (key, _) in enumerate(evicted_cache.kv):
        query_layer = query_rows[layer_index]
        query_heads = int(query_layer.shape[0])
        evicted_key_float = key.detach().to("cpu", dtype=torch.float32)[0]

        def _repeat_heads(key_rows: torch.Tensor) -> torch.Tensor:
            key_heads = int(key_rows.shape[0])
            if query_heads == key_heads:
                return key_rows
            if query_heads % key_heads != 0:
                raise ValueError(
                    f"query_rows head count {query_heads} is not compatible with cache heads {key_heads}."
                )
            return key_rows.repeat_interleave(query_heads // key_heads, dim=0)

        evicted_key_float = _repeat_heads(evicted_key_float)
        score_keys = evicted_key_float
        active_len = 0
        if active_cache is not None and len(active_cache) > 0:
            active_key_float = active_cache.kv[layer_index][0].detach().to("cpu", dtype=torch.float32)[0]
            active_key_float = _repeat_heads(active_key_float)
            active_len = int(active_key_float.shape[1])
            score_keys = torch.cat((active_key_float, evicted_key_float), dim=1)

        scores = torch.matmul(query_layer, score_keys.transpose(-2, -1)) / math.sqrt(float(score_keys.shape[-1]))
        scores = torch.softmax(scores, dim=-1)
        if active_len:
            scores = scores[:, :, active_len:]
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


def contrastive_position_scores(
    positions: Sequence[int],
    *,
    positive_scores: dict[int, float],
    negative_scores: dict[int, float],
    alpha: float = 1.0,
) -> dict[int, float]:
    """Return standardized positive-minus-negative scores over the candidate set."""
    candidate_positions = [int(position) for position in positions]
    if not candidate_positions:
        return {}

    def _z_scores(scores: dict[int, float]) -> dict[int, float]:
        values = [float(scores.get(position, 0.0)) for position in candidate_positions]
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        std = math.sqrt(variance)
        if std <= 1e-12:
            return {position: 0.0 for position in candidate_positions}
        return {
            position: (float(scores.get(position, 0.0)) - mean_value) / std
            for position in candidate_positions
        }

    positive_z = _z_scores(positive_scores)
    negative_z = _z_scores(negative_scores)
    return {
        position: float(positive_z[position] - float(alpha) * negative_z[position])
        for position in candidate_positions
    }


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


def _score_with_secondary(
    position: int,
    *,
    primary_scores: dict[int, float],
    secondary_scores: dict[int, float],
    secondary_weight: float = 1e-6,
) -> float:
    return float(primary_scores.get(position, 0.0)) + float(secondary_weight) * float(
        secondary_scores.get(position, 0.0)
    )


def select_coverage_aware_positions(
    *,
    evicted_positions: Sequence[int],
    q2_scores: dict[int, float],
    turn_n_scores: dict[int, float],
    k: int,
    left: int,
    right: int,
) -> list[int]:
    """Greedily choose burst windows by marginal score coverage.

    The default selector ranks anchors first and then expands each anchor into a
    burst. This variant ranks each candidate burst by the score it adds beyond
    already selected rows, which avoids wasting restore slots when the highest
    scoring anchors are adjacent and their bursts mostly overlap.
    """
    target_k = max(int(k), 0)
    if target_k == 0:
        return []

    available = sorted(dict.fromkeys(int(position) for position in evicted_positions))
    if not available:
        return []
    available_set = set(available)
    selected: list[int] = []
    selected_set: set[int] = set()

    def _window(anchor: int) -> list[int]:
        return [
            position
            for position in range(int(anchor) - int(left), int(anchor) + int(right) + 1)
            if position in available_set
        ]

    ranked_backfill = rank_positions(
        available,
        primary_scores=q2_scores,
        secondary_scores=turn_n_scores,
    )
    anchors = ranked_backfill
    while len(selected) < target_k:
        best_anchor: int | None = None
        best_new_positions: list[int] = []
        best_key: tuple[float, int, float, float, int] | None = None
        for anchor in anchors:
            new_positions = [position for position in _window(anchor) if position not in selected_set]
            if not new_positions:
                continue
            marginal_score = sum(
                _score_with_secondary(
                    position,
                    primary_scores=q2_scores,
                    secondary_scores=turn_n_scores,
                )
                for position in new_positions
            )
            key = (
                float(marginal_score),
                len(new_positions),
                float(q2_scores.get(anchor, 0.0)),
                float(turn_n_scores.get(anchor, 0.0)),
                -int(anchor),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_anchor = int(anchor)
                best_new_positions = new_positions
        if best_anchor is None:
            break
        del best_anchor
        for position in best_new_positions:
            if position in selected_set:
                continue
            selected.append(position)
            selected_set.add(position)
            if len(selected) >= target_k:
                return selected

    for position in ranked_backfill:
        if position in selected_set:
            continue
        selected.append(int(position))
        selected_set.add(int(position))
        if len(selected) >= target_k:
            break
    return selected


def select_mmr_positions(
    *,
    evicted_positions: Sequence[int],
    q2_scores: dict[int, float],
    turn_n_scores: dict[int, float],
    k: int,
    left: int,
    right: int,
    diversity_weight: float = 0.25,
) -> list[int]:
    """Select burst anchors with a small diversity bonus between anchors."""
    target_k = max(int(k), 0)
    if target_k == 0:
        return []

    available = sorted(dict.fromkeys(int(position) for position in evicted_positions))
    if not available:
        return []
    base_rank = rank_positions(
        available,
        primary_scores=q2_scores,
        secondary_scores=turn_n_scores,
    )
    if len(base_rank) <= 1:
        return pack_anchor_bursts(
            anchor_positions=base_rank,
            available_positions=available,
            k=target_k,
            left=int(left),
            right=int(right),
            backfill_positions=base_rank,
        )

    score_values = [float(q2_scores.get(position, 0.0)) for position in available]
    score_min = min(score_values)
    score_range = max(score_values) - score_min
    distance_scale = max(max(available) - min(available), 1)
    selected_anchors: list[int] = []
    remaining = set(available)
    selected_positions: list[int] = []

    def _normalized_score(position: int) -> float:
        if score_range <= 1e-12:
            return 0.0
        return (float(q2_scores.get(position, 0.0)) - score_min) / score_range

    while remaining and len(selected_positions) < target_k:
        best_position: int | None = None
        best_key: tuple[float, float, float, int] | None = None
        for position in remaining:
            if selected_anchors:
                nearest = min(abs(int(position) - anchor) for anchor in selected_anchors)
                diversity = min(float(nearest) / float(distance_scale), 1.0)
            else:
                diversity = 0.0
            key = (
                _normalized_score(position) + float(diversity_weight) * diversity,
                float(q2_scores.get(position, 0.0)),
                float(turn_n_scores.get(position, 0.0)),
                -int(position),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_position = int(position)
        assert best_position is not None
        selected_anchors.append(best_position)
        remaining.remove(best_position)
        selected_positions = pack_anchor_bursts(
            anchor_positions=selected_anchors,
            available_positions=available,
            k=target_k,
            left=int(left),
            right=int(right),
            backfill_positions=(),
        )

    return pack_anchor_bursts(
        anchor_positions=selected_anchors,
        available_positions=available,
        k=target_k,
        left=int(left),
        right=int(right),
        backfill_positions=base_rank,
    )


def select_refresh_positions(
    *,
    context_positions: Sequence[int],
    mandatory_positions: Sequence[int],
    q2_scores: dict[int, float],
    turn_n_scores: dict[int, float],
    context_budget: int,
    left: int,
    right: int,
) -> list[int]:
    """Reselect the full resumed context budget using the Q2-time signal.

    This is a stronger bounded comparator than IdleKV: it may drop stale rows
    from the base compressed cache and choose any buffered context row, while
    still preserving mandatory sink/recency rows and the same total active
    context budget.
    """
    available = sorted(dict.fromkeys(int(position) for position in context_positions))
    available_set = set(available)
    target_budget = min(max(int(context_budget), 0), len(available))
    if target_budget == 0:
        return []

    mandatory = [
        int(position)
        for position in dict.fromkeys(int(position) for position in mandatory_positions)
        if int(position) in available_set
    ][:target_budget]
    mandatory_set = set(mandatory)
    remaining_slots = max(0, target_budget - len(mandatory))
    if remaining_slots == 0:
        return sorted(mandatory)

    candidates = [position for position in available if position not in mandatory_set]
    ranked = rank_positions(
        candidates,
        primary_scores=q2_scores,
        secondary_scores=turn_n_scores,
    )
    selected = pack_anchor_bursts(
        anchor_positions=ranked,
        available_positions=candidates,
        k=remaining_slots,
        left=int(left),
        right=int(right),
        backfill_positions=ranked,
    )
    return sorted(set(mandatory) | set(selected))


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
    relevant_position_groups: Sequence[Sequence[int]] | None = None,
    q2_scores: dict[int, float],
    turn_n_scores: dict[int, float],
    k: int,
    left: int,
    right: int,
) -> list[int]:
    """Gold-span hindsight selector.

    When span groups are supplied, this performs an exact search over the small
    set of gold span groups to maximize complete recovered groups first, then
    recovered gold tokens, before score-ranked backfill.
    """
    del left, right
    available = {int(position) for position in evicted_positions}
    relevant_groups = [
        tuple(sorted(int(position) for position in group if int(position) in available))
        for group in (relevant_position_groups or ())
    ]
    relevant_groups = [group for group in relevant_groups if group]

    selected_relevant: list[int] = []
    if relevant_groups:
        best_value: tuple[int, int, float, float] = (-1, -1, float("-inf"), float("-inf"))
        best_selected: tuple[int, ...] = ()
        for subset_bits in itertools.product((0, 1), repeat=len(relevant_groups)):
            chosen_groups = [group for keep, group in zip(subset_bits, relevant_groups) if keep]
            chosen_positions = tuple(sorted({position for group in chosen_groups for position in group}))
            cost = len(chosen_positions)
            if cost > int(k):
                continue
            score_value = (
                len(chosen_groups),
                len(chosen_positions),
                float(sum(q2_scores.get(position, 0.0) for position in chosen_positions)),
                float(sum(turn_n_scores.get(position, 0.0) for position in chosen_positions)),
            )
            if score_value > best_value:
                best_value = score_value
                best_selected = chosen_positions
        selected_relevant = rank_positions(
            best_selected,
            primary_scores=q2_scores,
            secondary_scores=turn_n_scores,
        )
    else:
        selected_relevant = rank_positions(
            (int(position) for position in relevant_positions if int(position) in available),
            primary_scores=q2_scores,
            secondary_scores=turn_n_scores,
        )[: int(k)]

    selected_set = set(selected_relevant)
    remaining_relevant = rank_positions(
        (position for position in relevant_positions if int(position) in available and int(position) not in selected_set),
        primary_scores=q2_scores,
        secondary_scores=turn_n_scores,
    )
    backfill = rank_positions(
        (position for position in evicted_positions if int(position) not in selected_set),
        primary_scores=q2_scores,
        secondary_scores=turn_n_scores,
    )
    selected = list(selected_relevant)
    for position in itertools.chain(remaining_relevant, backfill):
        if len(selected) >= int(k):
            break
        if int(position) in selected_set:
            continue
        selected.append(int(position))
        selected_set.add(int(position))
    return selected
