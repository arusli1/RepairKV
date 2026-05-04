from __future__ import annotations

from pathlib import Path

import pytest

from phases.phase15_real_repo_relevance_shift.scripts.build_phase15_manifest import _validate_sources
from phases.phase15_real_repo_relevance_shift.src.manifest import RepoSource


def test_manifest_cli_rejects_unpinned_sources_by_default(tmp_path: Path) -> None:
    source = RepoSource(repo_id="repo", repo_root=str(tmp_path))

    with pytest.raises(ValueError, match="missing frozen source fields"):
        _validate_sources([source], allow_unpinned_for_dev=False)

    _validate_sources([source], allow_unpinned_for_dev=True)


def test_manifest_cli_rejects_this_repository() -> None:
    project_root = Path(__file__).resolve().parents[3]
    source = RepoSource(
        repo_id="self",
        repo_root=str(project_root),
        repo_url="https://github.com/arusli1/IdleKV",
        commit_sha="abc123",
        license_spdx="MIT",
        archive_sha256="deadbeef",
    )

    with pytest.raises(ValueError, match="must not come from this repository"):
        _validate_sources([source], allow_unpinned_for_dev=False)

