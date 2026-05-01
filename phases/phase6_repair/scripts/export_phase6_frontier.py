#!/usr/bin/env python3
"""Export Phase 6 frontier CSVs for paper figures and tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phases.phase6_repair.src.reporting import frontier_rows, split_rows, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True, help="Path to a Phase 6 result JSON artifact")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory for generated CSV files")
    parser.add_argument("--prefix", type=str, default="phase6_frontier", help="Output filename prefix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    aggregate = artifact["aggregate"]

    overall_path = args.outdir / f"{args.prefix}_overall.csv"
    by_split_path = args.outdir / f"{args.prefix}_by_split.csv"

    write_csv(frontier_rows(aggregate["overall"]), overall_path)
    write_csv(split_rows(aggregate["by_task"]), by_split_path)

    print(overall_path)
    print(by_split_path)


if __name__ == "__main__":
    main()
