#!/usr/bin/env python3
"""Summarize live Phase 14 controlled-proxy progress from a log file."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re


EVENT_RE = re.compile(r"\[(mq_niah_[^:]+):ex(\d+)\].*?(?=(?:\n\[mq_niah_)|\Z)", re.S)
K_RE = re.compile(
    r"k=(?P<k>\d+):Bm=(?P<b_match>[0-9.]+)/I=(?P<idlekv>[0-9.]+)/"
    r"R=(?P<random_k>[0-9.]+)/O=(?P<oldest_k>[0-9.]+)/Or=(?P<gold_k>[0-9.]+)"
)


@dataclass
class KSummary:
    count: int = 0
    b_match: float = 0.0
    idlekv: float = 0.0
    random_k: float = 0.0
    oldest_k: float = 0.0
    gold_k: float = 0.0

    def add(self, *, b_match: float, idlekv: float, random_k: float, oldest_k: float, gold_k: float) -> None:
        self.count += 1
        self.b_match += float(b_match)
        self.idlekv += float(idlekv)
        self.random_k += float(random_k)
        self.oldest_k += float(oldest_k)
        self.gold_k += float(gold_k)

    def mean(self, field: str) -> float:
        if self.count == 0:
            return 0.0
        return float(getattr(self, field)) / float(self.count)


def task_family(task: str) -> str:
    return str(task).split("_split_", 1)[0]


def parse_proxy_log(text: str) -> tuple[dict[str, int], dict[tuple[str, int], KSummary]]:
    progress: dict[str, int] = {}
    summaries: dict[tuple[str, int], KSummary] = {}
    for event in EVENT_RE.finditer(text):
        task = event.group(1)
        family = task_family(task)
        progress[family] = max(progress.get(family, 0), int(event.group(2)))
        body = event.group(0)
        for match in K_RE.finditer(body):
            k = int(match.group("k"))
            key = (family, k)
            summaries.setdefault(key, KSummary()).add(
                b_match=float(match.group("b_match")),
                idlekv=float(match.group("idlekv")),
                random_k=float(match.group("random_k")),
                oldest_k=float(match.group("oldest_k")),
                gold_k=float(match.group("gold_k")),
            )
    return progress, summaries


def latest_log(results_dir: Path) -> Path:
    logs = sorted((results_dir / "logs").glob("proxy_controlled_locked_*.log"))
    if not logs:
        raise FileNotFoundError(f"No proxy_controlled_locked_*.log under {results_dir / 'logs'}")
    return logs[-1]


def render_summary(progress: dict[str, int], summaries: dict[tuple[str, int], KSummary]) -> str:
    lines: list[str] = []
    for family in sorted(progress):
        lines.append(f"{family}: max_ex={progress[family]}")
        for family_key, k in sorted(summaries):
            if family_key != family:
                continue
            row = summaries[(family_key, k)]
            b_match = row.mean("b_match")
            idlekv = row.mean("idlekv")
            random_k = row.mean("random_k")
            oldest_k = row.mean("oldest_k")
            gold_k = row.mean("gold_k")
            max_control_lift = max(random_k - b_match, oldest_k - b_match)
            lines.append(
                "  "
                f"K={k}: n={row.count} "
                f"Bm={b_match:.3f} IdleKV={idlekv:.3f} "
                f"Rand={random_k:.3f} Old={oldest_k:.3f} Gold={gold_k:.3f} "
                f"lift={idlekv - b_match:.3f} max_control_lift={max_control_lift:.3f}"
            )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("phases/phase14_critical_flaw_closure/results"),
    )
    parser.add_argument("--log", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_path = args.log if args.log is not None else latest_log(args.results_dir)
    progress, summaries = parse_proxy_log(log_path.read_text(errors="ignore"))
    print(log_path)
    print(render_summary(progress, summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
