#!/usr/bin/env python3
"""Audit Phase 15 repair artifacts against control comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from phases.phase15_real_repo_relevance_shift.src.bootstrap import paired_cluster_bootstrap


CONTROL_KEYS = (
    "b_match_score",
    "random_k_score",
    "oldest_k_score",
    "stale_q_k_score",
    "wrong_q_k_score",
    "tool_file_k_score",
    "anchor_window_k_score",
)
WRONG_EVENT_DONOR_KEYS = (
    "wrong_event_donor_example_id",
    "wrong_event_donor_repo_id",
    "wrong_event_donor_answer",
)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _fraction_true(values: list[Any]) -> float:
    return sum(bool(value) for value in values) / len(values) if values else 0.0


def _repo_lift_summary(rows: list[dict[str, Any]], *, baseline_field: str) -> dict[str, Any]:
    by_repo: dict[str, list[float]] = {}
    for row in rows:
        if baseline_field not in row or "idlekv_score" not in row:
            continue
        repo_id = str(row.get("repo_id", ""))
        by_repo.setdefault(repo_id, []).append(float(row["idlekv_score"]) - float(row[baseline_field]))
    lifts = sorted(_mean(values) for values in by_repo.values() if values)
    if not lifts:
        return {"repo_count": 0}
    return {
        "repo_count": len(lifts),
        "median": round(float(_median(lifts)), 6),
        "min": round(float(min(lifts)), 6),
        "max": round(float(max(lifts)), 6),
        "positive_repos": sum(value > 0.0 for value in lifts),
        "negative_repos": sum(value < 0.0 for value in lifts),
    }


def _summarize_k_results(
    rows: list[dict[str, Any]],
    *,
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    by_k: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_k.setdefault(int(row["k"]), []).append(row)
    k_results: dict[str, Any] = {}
    for k, group in sorted(by_k.items()):
        idlekv = [float(row.get("idlekv_score", 0.0)) for row in group]
        payload_k: dict[str, Any] = {
            "n_rows": len(group),
            "mean_idlekv": round(_mean(idlekv), 6),
            "repo_count": len({str(row.get("repo_id", "")) for row in group}),
        }
        payload_k["repo_lift_vs_b_match"] = _repo_lift_summary(group, baseline_field="b_match_score")
        for key in CONTROL_KEYS:
            if key not in group[0]:
                continue
            control = [float(row.get(key, 0.0)) for row in group]
            label = key[: -len("_score")]
            payload_k[f"mean_{label}"] = round(_mean(control), 6)
            payload_k[f"mean_idlekv_minus_{label}"] = round(
                _mean([left - right for left, right in zip(idlekv, control)]),
                6,
            )
            payload_k[f"wins_vs_{label}"] = sum(left > right for left, right in zip(idlekv, control))
            payload_k[f"losses_vs_{label}"] = sum(left < right for left, right in zip(idlekv, control))
            repo_lifts: dict[str, list[float]] = {}
            for row, left, right in zip(group, idlekv, control):
                repo_lifts.setdefault(str(row.get("repo_id", "")), []).append(left - right)
            repo_mean_lifts = [_mean(values) for values in repo_lifts.values()]
            payload_k[f"repo_positive_count_vs_{label}"] = sum(value > 0.0 for value in repo_mean_lifts)
            payload_k[f"repo_nonnegative_count_vs_{label}"] = sum(value >= 0.0 for value in repo_mean_lifts)
            payload_k[f"repo_median_lift_vs_{label}"] = round(_median(repo_mean_lifts), 6)
            payload_k[f"repo_min_lift_vs_{label}"] = round(min(repo_mean_lifts), 6)
            payload_k[f"repo_max_lift_vs_{label}"] = round(max(repo_mean_lifts), 6)
            ci = paired_cluster_bootstrap(
                group,
                repo_field="repo_id",
                example_field="example_id",
                treatment_field="idlekv_score",
                baseline_field=key,
                draws=int(bootstrap_draws),
                seed=int(bootstrap_seed),
            )
            payload_k[f"bootstrap_idlekv_minus_{label}"] = {
                "mean": round(ci.mean, 6),
                "low": round(ci.low, 6),
                "high": round(ci.high, 6),
                "draws": ci.draws,
            }
        tool_file_fractions = [
            float(row["tool_file_k_selected_from_file_fraction"])
            for row in group
            if "tool_file_k_selected_from_file_fraction" in row
        ]
        if tool_file_fractions:
            payload_k["mean_tool_file_k_selected_from_file_fraction"] = round(_mean(tool_file_fractions), 6)
            payload_k["min_tool_file_k_selected_from_file_fraction"] = round(min(tool_file_fractions), 6)
        for prefix in ("tool_file_k", "anchor_window_k"):
            budget_values = [row[f"{prefix}_budget_matched"] for row in group if f"{prefix}_budget_matched" in row]
            if budget_values:
                payload_k[f"fraction_{prefix}_budget_matched"] = round(_fraction_true(budget_values), 6)
        k_results[f"k{k}"] = payload_k
    return k_results


def _filter_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_result_rows": len(rows),
        "n_examples": len({str(row.get("example_id", "")) for row in rows}),
        "repo_count": len({str(row.get("repo_id", "")) for row in rows}),
    }


def _artifact_checks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_k: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_k.setdefault(int(row["k"]), []).append(row)
    duplicate_by_k = {
        str(k): len(group) - len({str(row.get("example_id", "")) for row in group})
        for k, group in sorted(by_k.items())
    }
    manifest_audits = [row.get("phase15_manifest_audit") for row in rows]
    rows_with_manifest_audit = sum(isinstance(value, dict) for value in manifest_audits)
    manifest_audit_failed = sum(
        isinstance(value, dict) and value.get("passed") is not True
        for value in manifest_audits
    )
    wrong_event_rows = [row for row in rows if "wrong_q_k_score" in row]
    wrong_event_donor_metadata_rows = sum(
        all(key in row for key in WRONG_EVENT_DONOR_KEYS)
        for row in wrong_event_rows
    )
    return {
        "n_rows": len(rows),
        "rows_with_manifest_audit": rows_with_manifest_audit,
        "manifest_audit_failed": manifest_audit_failed,
        "all_manifest_audits_passed": rows_with_manifest_audit == len(rows) and manifest_audit_failed == 0,
        "duplicate_example_rows_by_k": duplicate_by_k,
        "has_duplicate_example_rows_by_k": any(count > 0 for count in duplicate_by_k.values()),
        "wrong_event_rows": len(wrong_event_rows),
        "wrong_event_donor_metadata_rows": wrong_event_donor_metadata_rows,
        "wrong_event_donor_metadata_complete": (
            len(wrong_event_rows) == 0 or wrong_event_donor_metadata_rows == len(wrong_event_rows)
        ),
    }


def _is_no_cue_hit(row: dict[str, Any]) -> bool:
    return float(row.get("cue_only_score", 0.0)) == 0.0


def _is_no_answer_retention(row: dict[str, Any]) -> bool:
    return (
        float(row.get("b_answer_token_overlap_fraction", 0.0)) == 0.0
        and float(row.get("b_match_answer_token_overlap_fraction", 0.0)) == 0.0
    )


def _is_strict_repair_eligible(row: dict[str, Any]) -> bool:
    return (
        _is_no_cue_hit(row)
        and _is_no_answer_retention(row)
        and float(row.get("condition_a_score", 0.0)) == 1.0
        and float(row.get("b_match_score", 0.0)) == 0.0
        and float(row.get("q1_score", 0.0)) == 1.0
    )


def _sensitivity_summaries(
    rows: list[dict[str, Any]],
    *,
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    filters = {
        "exclude_cue_only_hits": [row for row in rows if _is_no_cue_hit(row)],
        "exclude_answer_retention": [row for row in rows if _is_no_answer_retention(row)],
        "exclude_cue_and_answer_retention": [
            row for row in rows if _is_no_cue_hit(row) and _is_no_answer_retention(row)
        ],
        "strict_repair_eligible": [row for row in rows if _is_strict_repair_eligible(row)],
    }
    summaries: dict[str, Any] = {}
    for name, selected_rows in filters.items():
        summaries[name] = _filter_metadata(selected_rows)
        if selected_rows:
            summaries[name]["k_results"] = _summarize_k_results(
                selected_rows,
                bootstrap_draws=bootstrap_draws,
                bootstrap_seed=bootstrap_seed,
            )
        else:
            summaries[name]["k_results"] = {}
    return summaries


def audit_repair_artifact(payload: dict[str, Any], *, bootstrap_draws: int = 2000, bootstrap_seed: int = 0) -> dict[str, Any]:
    rows = list(payload.get("rows", []))
    if not rows:
        raise ValueError("Artifact has no rows.")
    return {
        "manifest_hash": payload.get("manifest_hash"),
        "protocol_hash": payload.get("protocol_hash"),
        "stage": payload.get("stage"),
        "artifact_checks": _artifact_checks(rows),
        "k_results": _summarize_k_results(rows, bootstrap_draws=bootstrap_draws, bootstrap_seed=bootstrap_seed),
        "sensitivity": _sensitivity_summaries(
            rows,
            bootstrap_draws=bootstrap_draws,
            bootstrap_seed=bootstrap_seed,
        ),
    }


def repair_gate(
    audit: dict[str, Any],
    *,
    primary_k: int,
    adjacent_k: int,
    min_primary_lift: float,
    require_positive_ci: bool,
    require_sensitivity: bool,
    min_sensitivity_examples: int,
    toolfile_margin_rows: int,
    min_toolfile_file_fraction: float,
    anchor_window_margin_rows: int,
) -> dict[str, Any]:
    """Evaluate the predeclared repair gate from an audit payload."""
    primary = audit["k_results"].get(f"k{int(primary_k)}")
    adjacent = audit["k_results"].get(f"k{int(adjacent_k)}")
    if primary is None:
        raise ValueError(f"Missing primary K={primary_k}.")
    if adjacent is None:
        raise ValueError(f"Missing adjacent K={adjacent_k}.")
    ci_keys = (
        "bootstrap_idlekv_minus_b_match",
        "bootstrap_idlekv_minus_random_k",
        "bootstrap_idlekv_minus_oldest_k",
        "bootstrap_idlekv_minus_stale_q_k",
        "bootstrap_idlekv_minus_wrong_q_k",
        "bootstrap_idlekv_minus_tool_file_k",
    )
    ci_keys_present = all(key in primary for key in ci_keys)
    ci_ok = ci_keys_present and all(float(primary[key]["low"]) > 0.0 for key in ci_keys)
    toolfile_margin = int(primary.get("wins_vs_tool_file_k", 0)) - int(primary.get("losses_vs_tool_file_k", 0))
    anchor_window_margin = int(primary.get("wins_vs_anchor_window_k", 0)) - int(primary.get("losses_vs_anchor_window_k", 0))
    repo_lift_summary = primary.get("repo_lift_vs_b_match", {})
    repo_median_lift = float(repo_lift_summary.get("median", primary.get("repo_median_lift_vs_b_match", 0.0)))

    sensitivity_checks: dict[str, Any] = {}
    for name in (
        "exclude_cue_only_hits",
        "exclude_answer_retention",
        "exclude_cue_and_answer_retention",
    ):
        slice_payload = audit.get("sensitivity", {}).get(name, {})
        slice_primary = slice_payload.get("k_results", {}).get(f"k{int(primary_k)}", {})
        n_examples = int(slice_payload.get("n_examples", 0))
        lift = float(slice_primary.get("mean_idlekv_minus_b_match", 0.0))
        sensitivity_checks[name] = {
            "n_examples": n_examples,
            "mean_idlekv_minus_b_match": round(lift, 6),
            "enough_examples": n_examples >= int(min_sensitivity_examples),
            "positive_lift": lift > 0.0,
            "ok": n_examples >= int(min_sensitivity_examples) and lift > 0.0,
        }
    strict_payload = audit.get("sensitivity", {}).get("strict_repair_eligible", {})
    strict_primary = strict_payload.get("k_results", {}).get(f"k{int(primary_k)}", {})
    strict_diagnostic = {
        "n_examples": int(strict_payload.get("n_examples", 0)),
        "repo_count": int(strict_payload.get("repo_count", 0)),
        "mean_idlekv_minus_b_match": round(
            float(strict_primary.get("mean_idlekv_minus_b_match", 0.0)),
            6,
        ),
    }
    sensitivity_ok = all(item["ok"] for item in sensitivity_checks.values())
    gate_results = {
        "manifest_audit_ok": bool(audit.get("artifact_checks", {}).get("all_manifest_audits_passed", False)),
        "no_duplicate_example_rows_by_k": not bool(
            audit.get("artifact_checks", {}).get("has_duplicate_example_rows_by_k", True)
        ),
        "primary_lift_ok": float(primary.get("mean_idlekv_minus_b_match", 0.0)) >= float(min_primary_lift),
        "adjacent_nonnegative_ok": float(adjacent.get("mean_idlekv_minus_b_match", 0.0)) >= 0.0,
        "required_ci_keys_present": ci_keys_present,
        "positive_ci_ok": (not require_positive_ci) or ci_ok,
        "toolfile_margin_ok": toolfile_margin >= int(toolfile_margin_rows),
        "anchor_window_margin_ok": anchor_window_margin >= int(anchor_window_margin_rows),
        "toolfile_selection_ok": (
            float(primary.get("min_tool_file_k_selected_from_file_fraction", 0.0))
            >= float(min_toolfile_file_fraction)
        ),
        "toolfile_budget_ok": float(primary.get("fraction_tool_file_k_budget_matched", 0.0)) == 1.0,
        "anchor_window_budget_ok": float(primary.get("fraction_anchor_window_k_budget_matched", 0.0)) == 1.0,
        "sensitivity_ok": (not require_sensitivity) or sensitivity_ok,
        "repo_median_lift_ok": repo_median_lift > 0.0,
    }
    return {
        "primary_k": int(primary_k),
        "adjacent_k": int(adjacent_k),
        "toolfile_win_loss_margin": toolfile_margin,
        "anchor_window_win_loss_margin": anchor_window_margin,
        "repo_median_lift_vs_b_match": round(repo_median_lift, 6),
        "sensitivity_checks": sensitivity_checks,
        "strict_repair_eligible_diagnostic": strict_diagnostic,
        "anchor_window_reference_note": (
            "AnchorWindow-K is label-assisted and diagnostic; its win/loss margin is reported "
            "as a strict main-promotion locality-reference check, not as a deployable baseline. "
            "If AnchorWindow-K dominates, keep Phase 15 out of the main paper."
        ),
        "gate_results": gate_results,
        "passed": all(gate_results.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", help="Phase 15 repair artifact JSON.")
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260504)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--primary-k", type=int, default=192)
    parser.add_argument("--adjacent-k", type=int, default=96)
    parser.add_argument("--min-primary-lift", type=float, default=0.10)
    parser.add_argument("--no-require-positive-ci", action="store_true")
    parser.add_argument("--no-require-sensitivity", action="store_true")
    parser.add_argument("--min-sensitivity-examples", type=int, default=8)
    parser.add_argument("--toolfile-margin-rows", type=int, default=1)
    parser.add_argument("--min-toolfile-file-fraction", type=float, default=0.10)
    parser.add_argument("--anchor-window-margin-rows", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(Path(args.artifact).read_text(encoding="utf-8"))
    audit = audit_repair_artifact(
        payload,
        bootstrap_draws=int(args.bootstrap_draws),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    if args.gate:
        audit["repair_gate"] = repair_gate(
            audit,
            primary_k=int(args.primary_k),
            adjacent_k=int(args.adjacent_k),
            min_primary_lift=float(args.min_primary_lift),
            require_positive_ci=not bool(args.no_require_positive_ci),
            require_sensitivity=not bool(args.no_require_sensitivity),
            min_sensitivity_examples=int(args.min_sensitivity_examples),
            toolfile_margin_rows=int(args.toolfile_margin_rows),
            min_toolfile_file_fraction=float(args.min_toolfile_file_fraction),
            anchor_window_margin_rows=int(args.anchor_window_margin_rows),
        )
    print(json.dumps(audit, indent=2, sort_keys=True))
    if args.gate and not bool(audit["repair_gate"]["passed"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
