from __future__ import annotations

import json
import subprocess
import sys

from phases.phase15_real_repo_relevance_shift.scripts.audit_phase15_repair_artifact import (
    audit_repair_artifact,
    repair_gate,
)


def test_audit_repair_artifact_reports_control_lifts() -> None:
    payload = {
        "manifest_hash": "abc",
        "protocol_hash": "def",
        "stage": "repair_smoke",
        "rows": [
            {
                "k": 96,
                "example_id": "a",
                "repo_id": "repo1",
                "idlekv_score": 1.0,
                "condition_a_score": 1.0,
                "q1_score": 1.0,
                "b_match_score": 0.0,
                "random_k_score": 0.0,
                "tool_file_k_score": 0.0,
                "anchor_window_k_score": 0.0,
                "tool_file_k_selected_from_file_fraction": 1.0,
                "tool_file_k_budget_matched": True,
                "anchor_window_k_budget_matched": True,
                "phase15_manifest_audit": {"passed": True},
                "wrong_event_donor_example_id": "donor-a",
                "wrong_event_donor_repo_id": "repo3",
                "wrong_event_donor_answer": "OtherA",
            },
            {
                "k": 96,
                "example_id": "b",
                "repo_id": "repo2",
                "idlekv_score": 0.0,
                "condition_a_score": 1.0,
                "q1_score": 1.0,
                "b_match_score": 0.0,
                "random_k_score": 1.0,
                "tool_file_k_score": 0.0,
                "anchor_window_k_score": 0.0,
                "tool_file_k_selected_from_file_fraction": 0.5,
                "tool_file_k_budget_matched": True,
                "anchor_window_k_budget_matched": True,
                "phase15_manifest_audit": {"passed": True},
                "wrong_event_donor_example_id": "donor-b",
                "wrong_event_donor_repo_id": "repo4",
                "wrong_event_donor_answer": "OtherB",
            },
        ],
    }

    result = audit_repair_artifact(payload, bootstrap_draws=50, bootstrap_seed=1)

    assert result["manifest_hash"] == "abc"
    assert result["artifact_checks"]["all_manifest_audits_passed"]
    assert not result["artifact_checks"]["has_duplicate_example_rows_by_k"]
    assert result["artifact_checks"]["wrong_event_donor_metadata_complete"]
    k96 = result["k_results"]["k96"]
    assert k96["n_rows"] == 2
    assert k96["repo_count"] == 2
    assert k96["mean_idlekv"] == 0.5
    assert k96["mean_idlekv_minus_b_match"] == 0.5
    assert k96["wins_vs_b_match"] == 1
    assert k96["losses_vs_random_k"] == 1
    assert k96["repo_positive_count_vs_b_match"] == 1
    assert k96["repo_nonnegative_count_vs_b_match"] == 2
    assert k96["repo_median_lift_vs_b_match"] == 0.5
    assert k96["repo_min_lift_vs_b_match"] == 0.0
    assert k96["repo_max_lift_vs_b_match"] == 1.0
    assert k96["repo_lift_vs_b_match"]["median"] == 0.5
    assert k96["repo_lift_vs_b_match"]["positive_repos"] == 1
    assert k96["bootstrap_idlekv_minus_b_match"]["draws"] == 50
    assert k96["mean_idlekv_minus_anchor_window_k"] == 0.5
    assert k96["min_tool_file_k_selected_from_file_fraction"] == 0.5
    assert k96["fraction_tool_file_k_budget_matched"] == 1.0
    assert k96["fraction_anchor_window_k_budget_matched"] == 1.0
    assert result["sensitivity"]["exclude_cue_and_answer_retention"]["n_examples"] == 2
    assert "k96" in result["sensitivity"]["strict_repair_eligible"]["k_results"]


def test_repair_gate_requires_primary_lift_adjacent_and_toolfile_margin() -> None:
    audit = {
        "k_results": {
            "k96": {
                "mean_idlekv_minus_b_match": 0.0,
            },
            "k192": {
                "mean_idlekv_minus_b_match": 0.25,
                "wins_vs_tool_file_k": 3,
                "losses_vs_tool_file_k": 1,
                "wins_vs_anchor_window_k": 3,
                "losses_vs_anchor_window_k": 1,
                "repo_lift_vs_b_match": {"median": 0.25},
                "min_tool_file_k_selected_from_file_fraction": 0.5,
                "fraction_tool_file_k_budget_matched": 1.0,
                "fraction_anchor_window_k_budget_matched": 1.0,
                "bootstrap_idlekv_minus_b_match": {"low": 0.05},
                "bootstrap_idlekv_minus_random_k": {"low": 0.02},
                "bootstrap_idlekv_minus_oldest_k": {"low": 0.03},
                "bootstrap_idlekv_minus_stale_q_k": {"low": 0.04},
                "bootstrap_idlekv_minus_wrong_q_k": {"low": 0.04},
                "bootstrap_idlekv_minus_tool_file_k": {"low": 0.01},
            },
        },
        "artifact_checks": {
            "all_manifest_audits_passed": True,
            "has_duplicate_example_rows_by_k": False,
            "wrong_event_donor_metadata_complete": True,
        },
        "sensitivity": {
            "exclude_cue_only_hits": {
                "n_examples": 8,
                "k_results": {"k192": {"mean_idlekv_minus_b_match": 0.20}},
            },
            "exclude_answer_retention": {
                "n_examples": 8,
                "k_results": {"k192": {"mean_idlekv_minus_b_match": 0.20}},
            },
            "exclude_cue_and_answer_retention": {
                "n_examples": 8,
                "k_results": {"k192": {"mean_idlekv_minus_b_match": 0.20}},
            },
            "strict_repair_eligible": {
                "n_examples": 4,
                "repo_count": 4,
                "k_results": {"k192": {"mean_idlekv_minus_b_match": 0.25}},
            },
        },
    }

    result = repair_gate(
        audit,
        primary_k=192,
        adjacent_k=96,
        min_primary_lift=0.10,
        require_positive_ci=True,
        require_sensitivity=True,
        min_sensitivity_examples=8,
        toolfile_margin_rows=1,
        min_toolfile_file_fraction=0.10,
        anchor_window_margin_rows=0,
    )

    assert result["passed"]
    assert result["toolfile_win_loss_margin"] == 2
    assert result["anchor_window_win_loss_margin"] == 2
    assert result["repo_median_lift_vs_b_match"] == 0.25
    assert result["gate_results"]["sensitivity_ok"]
    assert "diagnostic" in result["anchor_window_reference_note"]


def test_repair_gate_fails_main_gate_when_label_assisted_anchor_window_dominates() -> None:
    audit = {
        "k_results": {
            "k96": {
                "mean_idlekv_minus_b_match": 0.0,
            },
            "k192": {
                "mean_idlekv_minus_b_match": 0.25,
                "wins_vs_tool_file_k": 3,
                "losses_vs_tool_file_k": 1,
                "wins_vs_anchor_window_k": 0,
                "losses_vs_anchor_window_k": 4,
                "repo_lift_vs_b_match": {"median": 0.25},
                "min_tool_file_k_selected_from_file_fraction": 0.5,
                "fraction_tool_file_k_budget_matched": 1.0,
                "fraction_anchor_window_k_budget_matched": 1.0,
                "bootstrap_idlekv_minus_b_match": {"low": 0.05},
                "bootstrap_idlekv_minus_random_k": {"low": 0.02},
                "bootstrap_idlekv_minus_oldest_k": {"low": 0.03},
                "bootstrap_idlekv_minus_stale_q_k": {"low": 0.04},
                "bootstrap_idlekv_minus_wrong_q_k": {"low": 0.04},
                "bootstrap_idlekv_minus_tool_file_k": {"low": 0.01},
            },
        },
        "artifact_checks": {
            "all_manifest_audits_passed": True,
            "has_duplicate_example_rows_by_k": False,
            "wrong_event_donor_metadata_complete": True,
        },
        "sensitivity": {
            "exclude_cue_only_hits": {
                "n_examples": 8,
                "k_results": {"k192": {"mean_idlekv_minus_b_match": 0.20}},
            },
            "exclude_answer_retention": {
                "n_examples": 8,
                "k_results": {"k192": {"mean_idlekv_minus_b_match": 0.20}},
            },
            "exclude_cue_and_answer_retention": {
                "n_examples": 8,
                "k_results": {"k192": {"mean_idlekv_minus_b_match": 0.20}},
            },
            "strict_repair_eligible": {"n_examples": 0, "repo_count": 0, "k_results": {}},
        },
    }

    result = repair_gate(
        audit,
        primary_k=192,
        adjacent_k=96,
        min_primary_lift=0.10,
        require_positive_ci=True,
        require_sensitivity=True,
        min_sensitivity_examples=8,
        toolfile_margin_rows=1,
        min_toolfile_file_fraction=0.10,
        anchor_window_margin_rows=0,
    )

    assert not result["passed"]
    assert not result["gate_results"]["anchor_window_margin_ok"]
    assert result["anchor_window_win_loss_margin"] == -4
    assert "AnchorWindow-K is label-assisted" in result["anchor_window_reference_note"]


def test_repair_gate_fails_when_toolfile_control_is_not_real_file_selection() -> None:
    audit = {
        "k_results": {
            "k96": {
                "mean_idlekv_minus_b_match": 0.0,
            },
            "k192": {
                "mean_idlekv_minus_b_match": 0.25,
                "wins_vs_tool_file_k": 3,
                "losses_vs_tool_file_k": 1,
                "wins_vs_anchor_window_k": 3,
                "losses_vs_anchor_window_k": 1,
                "repo_lift_vs_b_match": {"median": 0.25},
                "min_tool_file_k_selected_from_file_fraction": 0.0,
                "fraction_tool_file_k_budget_matched": 1.0,
                "fraction_anchor_window_k_budget_matched": 1.0,
                "bootstrap_idlekv_minus_b_match": {"low": 0.05},
                "bootstrap_idlekv_minus_random_k": {"low": 0.02},
                "bootstrap_idlekv_minus_oldest_k": {"low": 0.03},
            },
        },
        "artifact_checks": {
            "all_manifest_audits_passed": True,
            "has_duplicate_example_rows_by_k": False,
            "wrong_event_donor_metadata_complete": True,
        },
    }

    result = repair_gate(
        audit,
        primary_k=192,
        adjacent_k=96,
        min_primary_lift=0.10,
        require_positive_ci=True,
        require_sensitivity=False,
        min_sensitivity_examples=8,
        toolfile_margin_rows=1,
        min_toolfile_file_fraction=0.10,
        anchor_window_margin_rows=0,
    )

    assert not result["passed"]
    assert not result["gate_results"]["toolfile_selection_ok"]


def test_repair_gate_fails_when_required_ci_keys_are_missing() -> None:
    audit = {
        "k_results": {
            "k96": {
                "mean_idlekv_minus_b_match": 0.0,
            },
            "k192": {
                "mean_idlekv_minus_b_match": 0.25,
                "wins_vs_tool_file_k": 3,
                "losses_vs_tool_file_k": 1,
                "wins_vs_anchor_window_k": 3,
                "losses_vs_anchor_window_k": 1,
                "repo_lift_vs_b_match": {"median": 0.25},
                "min_tool_file_k_selected_from_file_fraction": 0.5,
                "fraction_tool_file_k_budget_matched": 1.0,
                "fraction_anchor_window_k_budget_matched": 1.0,
                "bootstrap_idlekv_minus_b_match": {"low": 0.05},
            },
        },
        "artifact_checks": {
            "all_manifest_audits_passed": True,
            "has_duplicate_example_rows_by_k": False,
            "wrong_event_donor_metadata_complete": True,
        },
    }

    result = repair_gate(
        audit,
        primary_k=192,
        adjacent_k=96,
        min_primary_lift=0.10,
        require_positive_ci=True,
        require_sensitivity=False,
        min_sensitivity_examples=8,
        toolfile_margin_rows=1,
        min_toolfile_file_fraction=0.10,
        anchor_window_margin_rows=0,
    )

    assert not result["passed"]
    assert not result["gate_results"]["required_ci_keys_present"]
    assert not result["gate_results"]["positive_ci_ok"]


def test_repair_gate_fails_when_manifest_static_audit_fails() -> None:
    audit = {
        "k_results": {
            "k96": {"mean_idlekv_minus_b_match": 0.0},
            "k192": {
                "mean_idlekv_minus_b_match": 0.25,
                "wins_vs_tool_file_k": 3,
                "losses_vs_tool_file_k": 1,
                "wins_vs_anchor_window_k": 3,
                "losses_vs_anchor_window_k": 1,
                "repo_lift_vs_b_match": {"median": 0.25},
                "min_tool_file_k_selected_from_file_fraction": 0.5,
                "fraction_tool_file_k_budget_matched": 1.0,
                "fraction_anchor_window_k_budget_matched": 1.0,
                "bootstrap_idlekv_minus_b_match": {"low": 0.05},
                "bootstrap_idlekv_minus_random_k": {"low": 0.02},
                "bootstrap_idlekv_minus_oldest_k": {"low": 0.03},
                "bootstrap_idlekv_minus_stale_q_k": {"low": 0.04},
                "bootstrap_idlekv_minus_wrong_q_k": {"low": 0.04},
                "bootstrap_idlekv_minus_tool_file_k": {"low": 0.01},
            },
        },
        "artifact_checks": {
            "all_manifest_audits_passed": False,
            "has_duplicate_example_rows_by_k": False,
            "wrong_event_donor_metadata_complete": True,
        },
    }

    result = repair_gate(
        audit,
        primary_k=192,
        adjacent_k=96,
        min_primary_lift=0.10,
        require_positive_ci=True,
        require_sensitivity=False,
        min_sensitivity_examples=8,
        toolfile_margin_rows=1,
        min_toolfile_file_fraction=0.10,
        anchor_window_margin_rows=0,
    )

    assert not result["passed"]
    assert not result["gate_results"]["manifest_audit_ok"]


def test_repair_gate_fails_when_wrong_event_donor_metadata_is_missing() -> None:
    audit = {
        "k_results": {
            "k96": {"mean_idlekv_minus_b_match": 0.0},
            "k192": {
                "mean_idlekv_minus_b_match": 0.25,
                "wins_vs_tool_file_k": 3,
                "losses_vs_tool_file_k": 1,
                "wins_vs_anchor_window_k": 3,
                "losses_vs_anchor_window_k": 1,
                "repo_lift_vs_b_match": {"median": 0.25},
                "min_tool_file_k_selected_from_file_fraction": 0.5,
                "fraction_tool_file_k_budget_matched": 1.0,
                "fraction_anchor_window_k_budget_matched": 1.0,
                "bootstrap_idlekv_minus_b_match": {"low": 0.05},
                "bootstrap_idlekv_minus_random_k": {"low": 0.02},
                "bootstrap_idlekv_minus_oldest_k": {"low": 0.03},
                "bootstrap_idlekv_minus_stale_q_k": {"low": 0.04},
                "bootstrap_idlekv_minus_wrong_q_k": {"low": 0.04},
                "bootstrap_idlekv_minus_tool_file_k": {"low": 0.01},
            },
        },
        "artifact_checks": {
            "all_manifest_audits_passed": True,
            "has_duplicate_example_rows_by_k": False,
            "wrong_event_donor_metadata_complete": False,
        },
    }

    result = repair_gate(
        audit,
        primary_k=192,
        adjacent_k=96,
        min_primary_lift=0.10,
        require_positive_ci=True,
        require_sensitivity=False,
        min_sensitivity_examples=8,
        toolfile_margin_rows=1,
        min_toolfile_file_fraction=0.10,
        anchor_window_margin_rows=0,
    )

    assert not result["passed"]
    assert not result["gate_results"]["wrong_event_donor_metadata_ok"]


def test_repair_audit_cli_returns_nonzero_when_gate_fails(tmp_path) -> None:
    artifact = {
        "stage": "unit",
        "manifest_hash": "abc",
        "protocol_hash": "def",
        "rows": [
            {
                "k": 96,
                "example_id": "a",
                "repo_id": "repo1",
                "idlekv_score": 0.0,
                "b_match_score": 0.0,
                "random_k_score": 0.0,
                "oldest_k_score": 0.0,
                "tool_file_k_score": 0.0,
                "anchor_window_k_score": 0.0,
                "tool_file_k_selected_from_file_fraction": 1.0,
                "tool_file_k_budget_matched": True,
                "anchor_window_k_budget_matched": True,
            }
        ],
    }
    artifact_path = tmp_path / "repair.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "phases.phase15_real_repo_relevance_shift.scripts.audit_phase15_repair_artifact",
            str(artifact_path),
            "--gate",
            "--primary-k",
            "96",
            "--adjacent-k",
            "96",
            "--bootstrap-draws",
            "20",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert '"passed": false' in completed.stdout


def test_repair_gate_fails_when_required_ci_is_missing() -> None:
    audit = {
        "k_results": {
            "k96": {
                "mean_idlekv_minus_b_match": 0.0,
            },
            "k192": {
                "mean_idlekv_minus_b_match": 0.25,
                "wins_vs_tool_file_k": 3,
                "losses_vs_tool_file_k": 1,
                "wins_vs_anchor_window_k": 3,
                "losses_vs_anchor_window_k": 1,
                "repo_lift_vs_b_match": {"median": 0.25},
                "min_tool_file_k_selected_from_file_fraction": 0.5,
                "fraction_tool_file_k_budget_matched": 1.0,
                "fraction_anchor_window_k_budget_matched": 1.0,
                "bootstrap_idlekv_minus_b_match": {"low": 0.05},
                "bootstrap_idlekv_minus_random_k": {"low": 0.02},
            },
        },
        "artifact_checks": {
            "all_manifest_audits_passed": True,
            "has_duplicate_example_rows_by_k": False,
            "wrong_event_donor_metadata_complete": True,
        },
    }

    result = repair_gate(
        audit,
        primary_k=192,
        adjacent_k=96,
        min_primary_lift=0.10,
        require_positive_ci=True,
        require_sensitivity=False,
        min_sensitivity_examples=8,
        toolfile_margin_rows=1,
        min_toolfile_file_fraction=0.10,
        anchor_window_margin_rows=0,
    )

    assert not result["passed"]
    assert not result["gate_results"]["positive_ci_ok"]
