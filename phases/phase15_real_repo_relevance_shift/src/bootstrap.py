"""Cluster bootstrap utilities for Phase 15 locked-run audits."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class BootstrapCI:
    """Mean paired lift with percentile confidence interval."""

    mean: float
    low: float
    high: float
    draws: int


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile of an empty sequence.")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = min(max(float(q), 0.0), 1.0) * (len(ordered) - 1)
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_cluster_bootstrap(
    rows: Iterable[Mapping[str, object]],
    *,
    repo_field: str,
    example_field: str,
    treatment_field: str,
    baseline_field: str,
    draws: int = 2000,
    seed: int = 0,
) -> BootstrapCI:
    """Bootstrap paired treatment-baseline lift by repo then example."""
    grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
    lifts: list[float] = []
    for row in rows:
        treatment = float(row[treatment_field])
        baseline = float(row[baseline_field])
        lift = treatment - baseline
        repo = str(row[repo_field])
        example = str(row[example_field])
        grouped[repo].append((example, lift))
        lifts.append(lift)
    if not grouped or not lifts:
        raise ValueError("Bootstrap requires at least one row.")
    repos = sorted(grouped)
    rng = random.Random(seed)
    boot_means: list[float] = []
    for _ in range(int(draws)):
        sampled_lifts: list[float] = []
        for _repo_draw in repos:
            repo = rng.choice(repos)
            examples = grouped[repo]
            for _ in range(len(examples)):
                sampled_lifts.append(rng.choice(examples)[1])
        boot_means.append(sum(sampled_lifts) / len(sampled_lifts))
    mean = sum(lifts) / len(lifts)
    return BootstrapCI(
        mean=mean,
        low=_percentile(boot_means, 0.025),
        high=_percentile(boot_means, 0.975),
        draws=int(draws),
    )

