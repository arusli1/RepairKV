from __future__ import annotations

from pathlib import Path

import torch

from phases.phase15_real_repo_relevance_shift.src.edge import (
    boundary_occurrences,
    extract_callsite_edges,
    extract_declarations,
)
from phases.phase15_real_repo_relevance_shift.src.manifest import (
    RepoSource,
    build_phase15_prepared_example,
    encode_repair_signal,
    stable_manifest_hash,
)
from phases.phase15_real_repo_relevance_shift.src.protocol import (
    Phase15Protocol,
    protocol_hash,
)
from phases.phase15_real_repo_relevance_shift.src.runner import (
    build_repair_signal,
    tool_file_positions,
    with_wrong_event,
)


class CharTokenizer:
    """Minimal tokenizer with chat-template and offset support for CPU audits."""

    def apply_chat_template(self, messages, *, add_generation_prompt: bool, tokenize: bool):
        assert not tokenize
        assert add_generation_prompt
        content = messages[0]["content"]
        return f"<user>\n{content}\n<assistant>\n"

    def encode(self, text: str, *, return_tensors: str, add_special_tokens: bool):
        assert return_tensors == "pt"
        assert not add_special_tokens
        return torch.tensor([[ord(char) for char in text]], dtype=torch.long)

    def __call__(self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool):
        assert not add_special_tokens
        assert return_offsets_mapping
        return {"offset_mapping": [(index, index + 1) for index in range(len(text))]}


def _write_repo(root: Path) -> None:
    (root / "src").mkdir()
    (root / "src" / "alpha.py").write_text(
        "\n".join(
            [
                "class AlphaWindow:",
                "    pass",
                "",
                "def alpha_probe():",
                "    return 1",
            ]
        ),
        encoding="utf-8",
    )
    (root / "src" / "beta.py").write_text(
        "\n".join(
            [
                "def run_beta(value):",
                "    return BuildRow(value)",
            ]
        ),
        encoding="utf-8",
    )


def test_extract_callsite_edges_keeps_single_leaf_callee() -> None:
    text = "def run_beta(value):\n    return BuildRow(value)\n"

    edges = extract_callsite_edges("src/beta.py", text)

    assert len(edges) == 1
    assert edges[0].answer == "BuildRow"
    assert edges[0].anchor_name == "run_beta"
    assert edges[0].edge_type == "callsite_leaf_callee"


def test_extract_declarations_finds_q1_candidate() -> None:
    text = "class AlphaWindow:\n    pass\n\ndef alpha_probe():\n    return 1\n"

    declarations = extract_declarations("src/alpha.py", text)

    assert [candidate.answer for candidate in declarations] == ["AlphaWindow", "alpha_probe"]


def test_build_phase15_prepared_example_has_event_only_signal_and_audit(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    repo = RepoSource(
        repo_id="toyrepo",
        repo_root=str(tmp_path),
        repo_url="https://example.invalid/toyrepo",
        commit_sha="abc123",
        license_spdx="MIT",
        archive_sha256="deadbeef",
    )

    prepared = build_phase15_prepared_example(
        repo=repo,
        index=0,
        tokenizer=CharTokenizer(),
        target_context_length=1000,
        max_context_tokens=10_000,
        recency_window=0,
        k_max=0,
    )

    row = prepared.row
    assert row.source_task == "repodelta_edge"
    assert row.audit.passed
    assert row.answer == "BuildRow"
    assert row.answer not in row.tool_event
    assert row.answer not in row.q2_question
    assert row.tool_event in row.final_q2_prompt
    assert row.q2_question in row.final_q2_prompt
    assert row.metadata["repair_signal_mode"] == "event_only"
    assert prepared.repair_cue_ids.shape[1] < prepared.q2_prepared.question_ids.shape[1]
    assert prepared.q2_prepared.span_token_positions["repo_fact_2"]
    assert len(boundary_occurrences(prepared.q2_prepared.rendered_context, row.answer)) == 1


def test_repair_signal_modes_are_explicit(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    repo = RepoSource(repo_id="toyrepo", repo_root=str(tmp_path), commit_sha="abc123")
    prepared = build_phase15_prepared_example(
        repo=repo,
        index=0,
        tokenizer=CharTokenizer(),
        target_context_length=1000,
        max_context_tokens=10_000,
        recency_window=0,
        k_max=0,
    )

    event_only = build_repair_signal(CharTokenizer(), prepared.row, mode="event_only")
    event_plus_q2 = encode_repair_signal(CharTokenizer(), prepared.row, mode="event_plus_q2")

    assert event_only.mode == "event_only"
    assert event_only.decode_prompt_mode == "event_plus_q2"
    assert event_only.ids.shape[1] < event_plus_q2.shape[1]


def test_wrong_event_and_tool_file_control_helpers(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    repo = RepoSource(repo_id="toyrepo", repo_root=str(tmp_path), commit_sha="abc123")
    prepared = build_phase15_prepared_example(
        repo=repo,
        index=0,
        tokenizer=CharTokenizer(),
        target_context_length=1000,
        max_context_tokens=10_000,
        recency_window=0,
        k_max=0,
    )

    wrong = with_wrong_event(prepared.row, prepared.row)
    selected = tool_file_positions(
        evicted_positions=tuple(range(int(prepared.q2_prepared.context_ids.shape[1]))),
        segment_token_ranges=prepared.q2_prepared.segment_token_ranges,
        q2_path=str(prepared.row.q2["path"]),
        k=5,
    )

    assert wrong.metadata["wrong_event_source_example_id"] == prepared.row.example_id
    assert len(selected) == 5
    assert selected == sorted(selected)


def test_manifest_hash_is_stable(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    repo = RepoSource(repo_id="toyrepo", repo_root=str(tmp_path), commit_sha="abc123")
    prepared = build_phase15_prepared_example(
        repo=repo,
        index=0,
        tokenizer=CharTokenizer(),
        target_context_length=1000,
        max_context_tokens=10_000,
        recency_window=0,
        k_max=0,
    )

    assert stable_manifest_hash([prepared.row]) == stable_manifest_hash([prepared.row])


def test_protocol_hash_is_stable() -> None:
    protocol = Phase15Protocol(k_grid=(32, 96), conditions=("A", "IdleKV-EventOnly-K"))

    assert protocol_hash(protocol) == protocol_hash(protocol)
    assert protocol.to_dict()["k_grid"] == [32, 96]


def test_audit_flags_tail_leakage(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    repo = RepoSource(repo_id="toyrepo", repo_root=str(tmp_path), commit_sha="abc123")

    prepared = build_phase15_prepared_example(
        repo=repo,
        index=0,
        tokenizer=CharTokenizer(),
        target_context_length=1000,
        max_context_tokens=10_000,
        recency_window=10_000,
        k_max=128,
    )

    assert not prepared.row.audit.passed
    assert "q2_span_in_tail_reserve" in prepared.row.audit.flags
