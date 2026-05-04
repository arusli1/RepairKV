"""Build a CPU-audited Phase 15 manifest from pinned repository snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from phases.phase15_real_repo_relevance_shift.src.manifest import (
    RepoSource,
    build_phase15_prepared_example,
    stable_manifest_hash,
)
from phases.phase15_real_repo_relevance_shift.src.protocol import (
    Phase15Protocol,
    protocol_hash,
    read_protocol,
)


def _read_registry(path: Path) -> list[RepoSource]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    repos = payload.get("repos", payload if isinstance(payload, list) else None)
    if not isinstance(repos, list):
        raise ValueError("Registry must be a list or an object with a 'repos' list.")
    sources = [RepoSource(**repo) for repo in repos]
    if not sources:
        raise ValueError("Registry did not contain any repos.")
    return sources


def _validate_sources(sources: list[RepoSource], *, allow_unpinned_for_dev: bool) -> None:
    project_root = Path(__file__).resolve().parents[3]
    for source in sources:
        repo_root = Path(source.repo_root).resolve()
        if repo_root == project_root or project_root in repo_root.parents:
            raise ValueError(f"Phase 15 sources must not come from this repository: {source.repo_root}")
        missing = [
            field
            for field, value in {
                "repo_url": source.repo_url,
                "commit_sha": source.commit_sha,
                "license_spdx": source.license_spdx,
                "archive_sha256": source.archive_sha256,
            }.items()
            if not value
        ]
        if missing and not allow_unpinned_for_dev:
            raise ValueError(
                f"Repo {source.repo_id!r} is missing frozen source fields: {', '.join(missing)}. "
                "Use --allow-unpinned-for-dev only for local generator debugging."
            )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    protocol = read_protocol(args.protocol) if args.protocol else Phase15Protocol()
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer or protocol.tokenizer_dir), trust_remote_code=True)
    repos = _read_registry(Path(args.registry))
    _validate_sources(repos, allow_unpinned_for_dev=bool(args.allow_unpinned_for_dev))
    prepared_rows = []
    failures: list[dict[str, Any]] = []
    for repo in repos:
        repo_count = 0
        index = 0
        attempts = 0
        while repo_count < int(args.examples_per_repo) and attempts < int(args.max_attempts_per_repo):
            attempts += 1
            try:
                prepared = build_phase15_prepared_example(
                    repo=repo,
                    index=index,
                    tokenizer=tokenizer,
                    target_context_length=int(args.target_context_length),
                    max_context_tokens=int(protocol.context_tokens),
                    recency_window=int(protocol.recency_window),
                    k_max=max(int(value) for value in protocol.k_grid),
                    seed_offset=int(args.seed_offset),
                    max_files=int(args.max_files),
                )
            except Exception as exc:  # noqa: BLE001 - manifest builder should record bad rows.
                failures.append({"repo_id": repo.repo_id, "index": index, "error": repr(exc)})
                index += 1
                continue
            if prepared.row.audit.passed or args.include_failed:
                prepared_rows.append(prepared.row)
                repo_count += 1
            else:
                failures.append(
                    {
                        "repo_id": repo.repo_id,
                        "index": index,
                        "example_id": prepared.row.example_id,
                        "flags": list(prepared.row.audit.flags),
                    }
                )
            index += 1
    manifest_rows = [row.to_dict() for row in prepared_rows]
    _write_jsonl(Path(args.output), manifest_rows)
    summary = {
        "rows": len(prepared_rows),
        "repos": sorted({row.repo.repo_id for row in prepared_rows}),
        "manifest_hash": stable_manifest_hash(prepared_rows),
        "protocol_hash": protocol_hash(protocol),
        "failures": failures[: int(args.max_failures_in_summary)],
        "failure_count": len(failures),
    }
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, help="JSON repo registry path.")
    parser.add_argument("--output", required=True, help="Output JSONL manifest path.")
    parser.add_argument("--summary", help="Optional summary JSON path.")
    parser.add_argument("--protocol", help="Optional phase15_protocol.json path.")
    parser.add_argument("--tokenizer", help="Tokenizer path override.")
    parser.add_argument("--examples-per-repo", type=int, default=4)
    parser.add_argument("--max-attempts-per-repo", type=int, default=64)
    parser.add_argument("--target-context-length", type=int, default=80_000)
    parser.add_argument("--max-files", type=int, default=24)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--include-failed", action="store_true")
    parser.add_argument("--allow-unpinned-for-dev", action="store_true")
    parser.add_argument("--max-failures-in-summary", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    summary = build_manifest(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
