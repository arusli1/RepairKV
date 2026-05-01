#!/usr/bin/env python3
"""Run the Phase 6 two-turn matched-footprint IdleKV experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_ROOT.parents[1]
for root in (PHASE_ROOT, REPO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from phases.phase6_repair.src.runner import STAGE_DEFAULTS, build_config, run_experiment  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=sorted(STAGE_DEFAULTS), default="smoke")
    parser.add_argument("--task", default="clean_suite")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--context-length", type=int, default=32_768)
    parser.add_argument("--dataset-seed-offset", type=int, default=0)
    parser.add_argument("--k", nargs="+", type=int, default=None)
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--base-context-budget", type=int, default=512)
    parser.add_argument("--recency-window", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = build_config(
        stage=args.stage,
        task=args.task,
        num_samples=args.num_samples,
        context_length=args.context_length,
        dataset_seed_offset=args.dataset_seed_offset,
        k_values=args.k,
        conditions=args.conditions,
        base_context_budget=args.base_context_budget,
        recency_window=args.recency_window,
    )
    payload = run_experiment(config)
    print(payload["artifact_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
