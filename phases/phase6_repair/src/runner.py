"""Phase 6 runner: two-turn matched-footprint repair on the split MQ-NIAH task."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from pathlib import Path
import re
from statistics import fmean
from typing import Any, Iterable

import torch

from phases.phase1_degradation.phase1.evaluation import sample_score
from phases.phase2_kv_cache.src.kv_utils import PositionTrackedCache, inject_kv, slice_kv
from phases.phase2_kv_cache.src.runtime import load_model, load_tokenizer
from phases.phase3_eviction.src.runtime import build_position_tracked_cache, write_json

from .protocol import (
    CLEAN_SPLIT_SPECS,
    SPLIT_SPECS_BY_NAME,
    TAIL_LEAKY_SPLIT_SPECS,
    SplitTaskSpec,
    build_mismatched_question_ids,
    build_base_example,
    build_split_prepared_from_base_example,
    build_turn_n_keep_plan,
    compute_q2_query_rows,
    generate_turn,
    materialize_context_partition,
    relevant_positions_for_spans,
)
from .selectors import (
    score_evicted_positions,
    select_idlekv_positions,
    select_oldest_positions,
    select_oracle_positions,
    select_random_positions,
)

PHASE_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PHASE_ROOT / "results"
SCHEMA_VERSION = "phase6-two-turn-v1"
ALLOWED_CONDITIONS = ("A", "B", "B_match", "IdleKV", "WrongQ-K", "Random-K", "Oldest-K", "Oracle-K")

STAGE_DEFAULTS: dict[str, dict[str, Any]] = {
    "smoke": {
        "conditions": ("A", "B", "B_match", "IdleKV", "Oracle-K"),
        "num_samples": 8,
        "k_values": (8, 12, 24, 40, 48),
    },
    "pilot": {
        "conditions": ("A", "B", "B_match", "IdleKV", "Random-K", "Oldest-K", "Oracle-K"),
        "num_samples": 20,
        "k_values": (8, 12, 24, 40, 48),
    },
    "full": {
        "conditions": ("A", "B", "B_match", "IdleKV", "Oracle-K"),
        "num_samples": 100,
        "k_values": (8, 12, 24, 40, 48),
    },
}

TASK_ALIASES: dict[str, tuple[SplitTaskSpec, ...]] = {
    "clean_suite": CLEAN_SPLIT_SPECS,
    "diagnostic_suite": TAIL_LEAKY_SPLIT_SPECS,
}


@dataclass(frozen=True)
class Phase6Config:
    """Frozen configuration for one Phase 6 run."""

    stage: str
    task: str
    split_specs: tuple[SplitTaskSpec, ...]
    num_samples: int
    context_length: int = 32_768
    dataset_seed_offset: int = 0
    k_values: tuple[int, ...] = (8, 12, 24, 40, 48)
    conditions: tuple[str, ...] = ("A", "B", "B_match", "IdleKV", "Oracle-K")
    sink_size: int = 4
    recency_window: int = 128
    base_context_budget: int = 512
    pooling: str = "max"
    burst_left: int = 2
    burst_right: int = 20


def ensure_results_dirs(stage: str) -> Path:
    stage_dir = RESULTS_DIR / str(stage)
    stage_dir.mkdir(parents=True, exist_ok=True)
    return stage_dir


def _condition_label(conditions: Iterable[str]) -> str:
    parts = [re.sub(r"[^a-z0-9]+", "", str(condition).lower()) for condition in conditions]
    return "-".join(part for part in parts if part)


def _normalize_stage(stage: str) -> str:
    normalized = str(stage).strip().lower()
    if normalized not in STAGE_DEFAULTS:
        raise ValueError(f"Unsupported stage: {stage!r}.")
    return normalized


def _normalize_task(task: str) -> tuple[SplitTaskSpec, ...]:
    normalized = str(task).strip()
    if normalized in TASK_ALIASES:
        return TASK_ALIASES[normalized]
    split_spec = SPLIT_SPECS_BY_NAME.get(normalized)
    if split_spec is None:
        raise ValueError(f"Unsupported Phase 6 task: {task!r}.")
    return (split_spec,)


def build_config(
    *,
    stage: str,
    task: str,
    num_samples: int | None = None,
    context_length: int = 32_768,
    dataset_seed_offset: int = 0,
    k_values: Iterable[int] | None = None,
    conditions: Iterable[str] | None = None,
    base_context_budget: int = 512,
    recency_window: int = 128,
) -> Phase6Config:
    """Construct one run config with stage defaults unless overridden."""
    normalized_stage = _normalize_stage(stage)
    split_specs = _normalize_task(task)
    defaults = STAGE_DEFAULTS[normalized_stage]
    normalized_k = tuple(dict.fromkeys(int(value) for value in (k_values or defaults["k_values"])))
    if not normalized_k or any(value <= 0 for value in normalized_k):
        raise ValueError("k_values must contain at least one positive integer.")
    normalized_conditions = tuple(dict.fromkeys(str(value) for value in (conditions or defaults["conditions"])))
    invalid_conditions = tuple(value for value in normalized_conditions if value not in ALLOWED_CONDITIONS)
    if invalid_conditions:
        raise ValueError(f"Unsupported Phase 6 condition(s): {invalid_conditions}.")
    resolved_num_samples = int(num_samples or defaults["num_samples"])
    if resolved_num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if int(context_length) <= 0:
        raise ValueError("context_length must be positive.")
    if int(base_context_budget) <= 0:
        raise ValueError("base_context_budget must be positive.")
    if int(recency_window) < 0:
        raise ValueError("recency_window must be non-negative.")
    return Phase6Config(
        stage=normalized_stage,
        task=str(task).strip(),
        split_specs=tuple(split_specs),
        num_samples=resolved_num_samples,
        context_length=int(context_length),
        dataset_seed_offset=int(dataset_seed_offset),
        k_values=normalized_k,
        conditions=normalized_conditions,
        base_context_budget=int(base_context_budget),
        recency_window=int(recency_window),
    )


def _score_prediction(prediction: str, outputs: list[str]) -> float:
    return float(sample_score(prediction, outputs))


def _mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    return float(fmean(values)) if values else 0.0


def _pct(values: Iterable[bool]) -> float:
    values = [bool(value) for value in values]
    return float(sum(values) / len(values)) if values else 0.0


def _selected_overlap_fraction(selected_positions: Iterable[int], relevant_positions: Iterable[int]) -> float:
    selected = {int(position) for position in selected_positions}
    relevant = {int(position) for position in relevant_positions}
    if not relevant:
        return 0.0
    return float(len(selected & relevant) / len(relevant))


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _slice_fragment_by_positions(evicted_cache: PositionTrackedCache, positions: Iterable[int]) -> PositionTrackedCache | None:
    selected_positions = [int(position) for position in positions]
    if not selected_positions:
        return None
    position_to_dense = {int(position): dense_index for dense_index, position in enumerate(evicted_cache.positions)}
    dense_indices = [position_to_dense[int(position)] for position in selected_positions if int(position) in position_to_dense]
    if not dense_indices:
        return None
    fragment = slice_kv(evicted_cache, dense_indices)
    if not isinstance(fragment, PositionTrackedCache):
        raise RuntimeError("Phase 2 slice_kv did not preserve position tracking for the repair fragment.")
    return fragment


def _restore_positions(
    *,
    active_cache: PositionTrackedCache,
    evicted_cache: PositionTrackedCache,
    selected_positions: Iterable[int],
) -> tuple[PositionTrackedCache, dict[str, float]]:
    selected_fragment = _slice_fragment_by_positions(evicted_cache, selected_positions)
    if selected_fragment is None:
        return active_cache, {
            "transfer_ms": 0.0,
            "inject_ms": 0.0,
            "restored_count": 0.0,
        }

    target_device = active_cache.device
    _sync_if_cuda(target_device)
    transfer_start = time.perf_counter()
    selected_gpu = selected_fragment.to_device(target_device, non_blocking=True)
    _sync_if_cuda(target_device)
    transfer_ms = (time.perf_counter() - transfer_start) * 1000.0

    inject_start = time.perf_counter()
    repaired_cache = inject_kv(
        active_cache,
        selected_gpu,
        selected_gpu.positions,
    )
    _sync_if_cuda(target_device)
    inject_ms = (time.perf_counter() - inject_start) * 1000.0
    return repaired_cache, {
        "transfer_ms": transfer_ms,
        "inject_ms": inject_ms,
        "restored_count": float(len(selected_gpu.positions)),
    }


def _run_condition(
    *,
    model,
    tokenizer,
    prepared,
    cache: PositionTrackedCache,
) -> tuple[str, float, float]:
    generation_start = time.perf_counter()
    generated = generate_turn(model, tokenizer, prepared, cache)
    generation_s = time.perf_counter() - generation_start
    score = _score_prediction(generated.text, list(prepared.example.outputs))
    return generated.text, score, generation_s


def _run_one_split(
    *,
    model,
    tokenizer,
    config: Phase6Config,
    split,
    full_cache: PositionTrackedCache,
    index: int,
    wrong_q2_question_ids: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    example_start = time.perf_counter()
    q1_context_ids = split.q1_prepared.context_ids
    context_len = int(q1_context_ids.shape[1])

    q1_start = time.perf_counter()
    q1_turn = generate_turn(model, tokenizer, split.q1_prepared, full_cache)
    q1_generation_s = time.perf_counter() - q1_start
    q1_score = _score_prediction(q1_turn.text, list(split.q1_prepared.example.outputs))

    condition_a_output, condition_a_score, condition_a_generation_s = _run_condition(
        model=model,
        tokenizer=tokenizer,
        prepared=split.q2_prepared,
        cache=q1_turn.cache,
    )

    keep_plan_start = time.perf_counter()
    keep_plan = build_turn_n_keep_plan(
        post_q1_cache=q1_turn.cache,
        q1_answer_ids=q1_turn.token_ids,
        context_len=context_len,
        sink_size=config.sink_size,
        recency_window=config.recency_window,
        pooling=config.pooling,
    )
    keep_plan_s = time.perf_counter() - keep_plan_start

    base_partition = materialize_context_partition(
        full_post_q1_cache=q1_turn.cache,
        keep_plan=keep_plan,
        context_budget=config.base_context_budget,
    )
    condition_b_output, condition_b_score, condition_b_generation_s = _run_condition(
        model=model,
        tokenizer=tokenizer,
        prepared=split.q2_prepared,
        cache=base_partition.compressed,
    )

    q2_query_start = time.perf_counter()
    q2_query_rows = compute_q2_query_rows(
        model,
        active_cache=base_partition.compressed,
        question_ids=split.q2_prepared.question_ids,
    )
    q2_query_s = time.perf_counter() - q2_query_start

    q2_score_start = time.perf_counter()
    q2_scores = score_evicted_positions(
        query_rows=q2_query_rows,
        evicted_cache=base_partition.evicted,
        pooling=config.pooling,
    )
    q2_score_s = time.perf_counter() - q2_score_start

    wrong_q_scores: dict[int, float] | None = None
    wrong_q_query_s = 0.0
    wrong_q_score_s = 0.0
    if "WrongQ-K" in config.conditions:
        if wrong_q2_question_ids is None:
            raise ValueError("WrongQ-K requires donor Q2 question ids.")
        wrong_q_query_start = time.perf_counter()
        wrong_q_query_rows = compute_q2_query_rows(
            model,
            active_cache=base_partition.compressed,
            question_ids=wrong_q2_question_ids,
        )
        wrong_q_query_s = time.perf_counter() - wrong_q_query_start
        wrong_q_score_start = time.perf_counter()
        wrong_q_scores = score_evicted_positions(
            query_rows=wrong_q_query_rows,
            evicted_cache=base_partition.evicted,
            pooling=config.pooling,
        )
        wrong_q_score_s = time.perf_counter() - wrong_q_score_start

    q2_relevant_positions = relevant_positions_for_spans(split.q2_prepared, split.q2_span_names)
    evicted_positions = tuple(int(position) for position in base_partition.evicted.positions)

    rows: list[dict[str, Any]] = []
    for k in config.k_values:
        k_int = int(k)
        row: dict[str, Any] = {
            "example_id": f"{split.split_spec.name}:ex{index + 1:03d}",
            "task": split.split_spec.name,
            "suite_task": config.task,
            "stage": config.stage,
            "index": int(index),
            "k": k_int,
            "q1_indices": list(split.split_spec.q1_indices),
            "q2_indices": list(split.split_spec.q2_indices),
            "q1_score": round(q1_score, 6),
            "condition_a_score": round(condition_a_score, 6),
            "condition_b_score": round(condition_b_score, 6),
            "q1_output": q1_turn.text,
            "condition_a_output": condition_a_output,
            "condition_b_output": condition_b_output,
            "q1_answer_tokens": int(q1_turn.token_ids.numel()),
            "q2_relevant_positions": list(q2_relevant_positions),
            "q1_generation_s": round(q1_generation_s, 6),
            "condition_a_generation_s": round(condition_a_generation_s, 6),
            "turn_n_keep_plan_s": round(keep_plan_s, 6),
            "condition_b_generation_s": round(condition_b_generation_s, 6),
            "q2_query_rows_s": round(q2_query_s, 6),
            "q2_evicted_scoring_s": round(q2_score_s, 6),
            "wrong_q_query_rows_s": round(wrong_q_query_s, 6),
            "wrong_q_evicted_scoring_s": round(wrong_q_score_s, 6),
            "base_context_budget": int(config.base_context_budget),
            "context_length": context_len,
            "evicted_context_tokens": int(len(base_partition.evicted.positions)),
            "b_kept_context_positions": list(base_partition.kept_context_positions),
        }

        bmatch_partition = materialize_context_partition(
            full_post_q1_cache=q1_turn.cache,
            keep_plan=keep_plan,
            context_budget=config.base_context_budget + k_int,
        )
        bmatch_output, bmatch_score, bmatch_generation_s = _run_condition(
            model=model,
            tokenizer=tokenizer,
            prepared=split.q2_prepared,
            cache=bmatch_partition.compressed,
        )
        row.update(
            {
                "b_match_score": round(bmatch_score, 6),
                "b_match_output": bmatch_output,
                "b_match_generation_s": round(bmatch_generation_s, 6),
                "b_match_overlap_fraction": round(
                    _selected_overlap_fraction(bmatch_partition.kept_context_positions, q2_relevant_positions),
                    6,
                ),
            }
        )

        if "IdleKV" in config.conditions:
            select_start = time.perf_counter()
            idlekv_positions = select_idlekv_positions(
                evicted_positions=evicted_positions,
                q2_scores=q2_scores,
                turn_n_scores=keep_plan.importance_scores,
                k=k_int,
                left=config.burst_left,
                right=config.burst_right,
            )
            select_s = time.perf_counter() - select_start
            repaired_cache, restore_timing = _restore_positions(
                active_cache=base_partition.compressed,
                evicted_cache=base_partition.evicted,
                selected_positions=idlekv_positions,
            )
            idlekv_output, idlekv_score, idlekv_generation_s = _run_condition(
                model=model,
                tokenizer=tokenizer,
                prepared=split.q2_prepared,
                cache=repaired_cache,
            )
            row.update(
                {
                    "idlekv_score": round(idlekv_score, 6),
                    "idlekv_output": idlekv_output,
                    "idlekv_generation_s": round(idlekv_generation_s, 6),
                    "idlekv_selection_s": round(select_s, 6),
                    "idlekv_transfer_ms": round(restore_timing["transfer_ms"], 6),
                    "idlekv_inject_ms": round(restore_timing["inject_ms"], 6),
                    "idlekv_restored_count": int(restore_timing["restored_count"]),
                    "idlekv_selected_positions": idlekv_positions,
                    "idlekv_overlap_fraction": round(
                        _selected_overlap_fraction(idlekv_positions, q2_relevant_positions),
                        6,
                    ),
                }
            )

        if "WrongQ-K" in config.conditions:
            assert wrong_q_scores is not None
            select_start = time.perf_counter()
            wrong_q_positions = select_idlekv_positions(
                evicted_positions=evicted_positions,
                q2_scores=wrong_q_scores,
                turn_n_scores=keep_plan.importance_scores,
                k=k_int,
                left=config.burst_left,
                right=config.burst_right,
            )
            select_s = time.perf_counter() - select_start
            repaired_cache, restore_timing = _restore_positions(
                active_cache=base_partition.compressed,
                evicted_cache=base_partition.evicted,
                selected_positions=wrong_q_positions,
            )
            wrong_q_output, wrong_q_score, wrong_q_generation_s = _run_condition(
                model=model,
                tokenizer=tokenizer,
                prepared=split.q2_prepared,
                cache=repaired_cache,
            )
            row.update(
                {
                    "wrong_q_k_score": round(wrong_q_score, 6),
                    "wrong_q_k_output": wrong_q_output,
                    "wrong_q_k_generation_s": round(wrong_q_generation_s, 6),
                    "wrong_q_k_selection_s": round(select_s, 6),
                    "wrong_q_k_transfer_ms": round(restore_timing["transfer_ms"], 6),
                    "wrong_q_k_inject_ms": round(restore_timing["inject_ms"], 6),
                    "wrong_q_k_restored_count": int(restore_timing["restored_count"]),
                    "wrong_q_k_selected_positions": wrong_q_positions,
                    "wrong_q_k_overlap_fraction": round(
                        _selected_overlap_fraction(wrong_q_positions, q2_relevant_positions),
                        6,
                    ),
                }
            )

        if "Random-K" in config.conditions:
            select_start = time.perf_counter()
            random_positions = select_random_positions(
                evicted_positions=evicted_positions,
                k=k_int,
                left=config.burst_left,
                right=config.burst_right,
                seed=(index + 1) * 1000 + k_int,
            )
            select_s = time.perf_counter() - select_start
            repaired_cache, restore_timing = _restore_positions(
                active_cache=base_partition.compressed,
                evicted_cache=base_partition.evicted,
                selected_positions=random_positions,
            )
            random_output, random_score, random_generation_s = _run_condition(
                model=model,
                tokenizer=tokenizer,
                prepared=split.q2_prepared,
                cache=repaired_cache,
            )
            row.update(
                {
                    "random_k_score": round(random_score, 6),
                    "random_k_output": random_output,
                    "random_k_generation_s": round(random_generation_s, 6),
                    "random_k_selection_s": round(select_s, 6),
                    "random_k_transfer_ms": round(restore_timing["transfer_ms"], 6),
                    "random_k_inject_ms": round(restore_timing["inject_ms"], 6),
                    "random_k_restored_count": int(restore_timing["restored_count"]),
                    "random_k_selected_positions": random_positions,
                    "random_k_overlap_fraction": round(
                        _selected_overlap_fraction(random_positions, q2_relevant_positions),
                        6,
                    ),
                }
            )

        if "Oldest-K" in config.conditions:
            select_start = time.perf_counter()
            oldest_positions = select_oldest_positions(
                evicted_positions=evicted_positions,
                k=k_int,
                left=config.burst_left,
                right=config.burst_right,
            )
            select_s = time.perf_counter() - select_start
            repaired_cache, restore_timing = _restore_positions(
                active_cache=base_partition.compressed,
                evicted_cache=base_partition.evicted,
                selected_positions=oldest_positions,
            )
            oldest_output, oldest_score, oldest_generation_s = _run_condition(
                model=model,
                tokenizer=tokenizer,
                prepared=split.q2_prepared,
                cache=repaired_cache,
            )
            row.update(
                {
                    "oldest_k_score": round(oldest_score, 6),
                    "oldest_k_output": oldest_output,
                    "oldest_k_generation_s": round(oldest_generation_s, 6),
                    "oldest_k_selection_s": round(select_s, 6),
                    "oldest_k_transfer_ms": round(restore_timing["transfer_ms"], 6),
                    "oldest_k_inject_ms": round(restore_timing["inject_ms"], 6),
                    "oldest_k_restored_count": int(restore_timing["restored_count"]),
                    "oldest_k_selected_positions": oldest_positions,
                    "oldest_k_overlap_fraction": round(
                        _selected_overlap_fraction(oldest_positions, q2_relevant_positions),
                        6,
                    ),
                }
            )

        if "Oracle-K" in config.conditions:
            select_start = time.perf_counter()
            oracle_positions = select_oracle_positions(
                evicted_positions=evicted_positions,
                relevant_positions=q2_relevant_positions,
                q2_scores=q2_scores,
                turn_n_scores=keep_plan.importance_scores,
                k=k_int,
                left=config.burst_left,
                right=config.burst_right,
            )
            select_s = time.perf_counter() - select_start
            repaired_cache, restore_timing = _restore_positions(
                active_cache=base_partition.compressed,
                evicted_cache=base_partition.evicted,
                selected_positions=oracle_positions,
            )
            oracle_output, oracle_score, oracle_generation_s = _run_condition(
                model=model,
                tokenizer=tokenizer,
                prepared=split.q2_prepared,
                cache=repaired_cache,
            )
            row.update(
                {
                    "oracle_k_score": round(oracle_score, 6),
                    "oracle_k_output": oracle_output,
                    "oracle_k_generation_s": round(oracle_generation_s, 6),
                    "oracle_k_selection_s": round(select_s, 6),
                    "oracle_k_transfer_ms": round(restore_timing["transfer_ms"], 6),
                    "oracle_k_inject_ms": round(restore_timing["inject_ms"], 6),
                    "oracle_k_restored_count": int(restore_timing["restored_count"]),
                    "oracle_k_selected_positions": oracle_positions,
                    "oracle_k_overlap_fraction": round(
                        _selected_overlap_fraction(oracle_positions, q2_relevant_positions),
                        6,
                    ),
                }
            )

        row["example_wall_s"] = round(time.perf_counter() - example_start, 6)
        rows.append(row)

    return rows


def _summarize_condition(rows: list[dict[str, Any]], score_key: str) -> dict[str, float]:
    scores = [float(row[score_key]) for row in rows if score_key in row]
    return {
        "mean_score": round(_mean(scores), 6),
        "n_examples": len(scores),
    }


def _summarize_rows_by_k(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a homogeneous row set by K and condition."""
    by_k: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_k.setdefault(int(row["k"]), []).append(row)

    summary_by_k: dict[str, Any] = {}
    for k, group in sorted(by_k.items()):
        payload: dict[str, Any] = {
            "mean_q1_score": round(_mean(row["q1_score"] for row in group), 6),
            "mean_condition_a": round(_mean(row["condition_a_score"] for row in group), 6),
            "mean_condition_b": round(_mean(row["condition_b_score"] for row in group), 6),
            "mean_b_match": round(_mean(row["b_match_score"] for row in group), 6),
            "n_examples": len(group),
        }
        if "idlekv_score" in group[0]:
            payload.update(
                {
                    "mean_idlekv": round(_mean(row["idlekv_score"] for row in group), 6),
                    "mean_selection_lift": round(_mean(float(row["idlekv_score"]) - float(row["b_match_score"]) for row in group), 6),
                    "pct_idlekv_gt_b_match": round(
                        _pct(float(row["idlekv_score"]) > float(row["b_match_score"]) for row in group),
                        6,
                    ),
                    "pct_idlekv_lt_b_match": round(
                        _pct(float(row["idlekv_score"]) < float(row["b_match_score"]) for row in group),
                        6,
                    ),
                    "mean_idlekv_overlap_fraction": round(_mean(row["idlekv_overlap_fraction"] for row in group), 6),
                    "mean_idlekv_repair_ms": round(
                        _mean((float(row["idlekv_selection_s"]) * 1000.0) + float(row["idlekv_transfer_ms"]) + float(row["idlekv_inject_ms"]) for row in group),
                        6,
                    ),
                }
            )
        if "oracle_k_score" in group[0]:
            payload.update(
                {
                    "mean_oracle_k": round(_mean(row["oracle_k_score"] for row in group), 6),
                    "mean_oracle_lift": round(_mean(float(row["oracle_k_score"]) - float(row["b_match_score"]) for row in group), 6),
                    "pct_oracle_gt_b_match": round(
                        _pct(float(row["oracle_k_score"]) > float(row["b_match_score"]) for row in group),
                        6,
                    ),
                }
            )
        if "wrong_q_k_score" in group[0]:
            payload.update(
                {
                    "mean_wrong_q_k": round(_mean(row["wrong_q_k_score"] for row in group), 6),
                    "mean_wrong_q_lift": round(_mean(float(row["wrong_q_k_score"]) - float(row["b_match_score"]) for row in group), 6),
                    "pct_wrong_q_gt_b_match": round(
                        _pct(float(row["wrong_q_k_score"]) > float(row["b_match_score"]) for row in group),
                        6,
                    ),
                }
            )
        if "random_k_score" in group[0]:
            payload["mean_random_k"] = round(_mean(row["random_k_score"] for row in group), 6)
        if "oldest_k_score" in group[0]:
            payload["mean_oldest_k"] = round(_mean(row["oldest_k_score"] for row in group), 6)
        summary_by_k[f"k{k}"] = payload
    return summary_by_k


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the per-example rows by K, and by split when needed."""
    task_names = sorted({str(row["task"]) for row in rows})
    if len(task_names) <= 1:
        return _summarize_rows_by_k(rows)
    return {
        "overall": _summarize_rows_by_k(rows),
        "by_task": {
            task_name: _summarize_rows_by_k([row for row in rows if str(row["task"]) == task_name])
            for task_name in task_names
        },
    }


def _artifact_path(config: Phase6Config) -> Path:
    stage_dir = ensure_results_dirs(config.stage)
    k_label = "-".join(str(value) for value in config.k_values)
    condition_label = _condition_label(config.conditions)
    return stage_dir / (
        f"{config.task}_b{config.base_context_budget}_r{config.recency_window}"
        f"_n{config.num_samples}_k{k_label}_c{condition_label}.json"
    )


def _wrong_query_ids_by_split(split_views, *, tokenizer) -> dict[str, torch.Tensor]:
    """Use a task-matched decoy query with nonexistent keys as the negative control."""
    return {
        split.split_spec.name: build_mismatched_question_ids(
            base_example=split.base_example,
            split_spec=split.split_spec,
            tokenizer=tokenizer,
        )
        for split in split_views
    }


def run_experiment(config: Phase6Config) -> dict[str, Any]:
    """Run one full Phase 6 experiment and persist the JSON artifact."""
    artifact_path = _artifact_path(config)
    overall_start = time.perf_counter()
    model = load_model()
    tokenizer = load_tokenizer()

    rows: list[dict[str, Any]] = []
    for index in range(int(config.num_samples)):
        base_example = build_base_example(
            split_spec=config.split_specs[0],
            index=index,
            context_length=config.context_length,
            tokenizer=tokenizer,
            dataset_seed_offset=config.dataset_seed_offset,
        )
        split_views = tuple(
            build_split_prepared_from_base_example(
                base_example=base_example,
                split_spec=split_spec,
                tokenizer=tokenizer,
            )
            for split_spec in config.split_specs
        )
        wrong_q2_question_ids_by_split: dict[str, torch.Tensor] = {}
        if "WrongQ-K" in config.conditions:
            wrong_q2_question_ids_by_split = _wrong_query_ids_by_split(split_views, tokenizer=tokenizer)
        full_cache = build_position_tracked_cache(model, split_views[0].q1_prepared.context_ids)

        for split in split_views:
            example_rows = _run_one_split(
                model=model,
                tokenizer=tokenizer,
                config=config,
                split=split,
                full_cache=full_cache,
                index=index,
                wrong_q2_question_ids=wrong_q2_question_ids_by_split.get(split.split_spec.name),
            )
            rows.extend(example_rows)
            if not example_rows:
                continue
            first = example_rows[0]
            per_k_parts = []
            for row in sorted(example_rows, key=lambda item: int(item["k"])):
                part = f"k={row['k']}:Bm={row['b_match_score']:.3f}"
                if "idlekv_score" in row:
                    part += f"/I={row['idlekv_score']:.3f}"
                if "wrong_q_k_score" in row:
                    part += f"/W={row['wrong_q_k_score']:.3f}"
                if "random_k_score" in row:
                    part += f"/R={row['random_k_score']:.3f}"
                if "oldest_k_score" in row:
                    part += f"/O={row['oldest_k_score']:.3f}"
                if "oracle_k_score" in row:
                    part += f"/Or={row['oracle_k_score']:.3f}"
                per_k_parts.append(part)
            print(
                f"[{first['example_id']}] "
                f"A={first['condition_a_score']:.3f} "
                f"B={first['condition_b_score']:.3f} "
                + " ".join(per_k_parts),
                flush=True,
            )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "config": asdict(config),
        "aggregate": summarize_rows(rows),
        "rows": rows,
        "elapsed_s": round(time.perf_counter() - overall_start, 6),
        "artifact_path": str(artifact_path),
    }
    write_json(artifact_path, payload)
    return payload
