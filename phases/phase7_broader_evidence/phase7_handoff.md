# Phase 7 Handoff

## Current State

Phase 6 is done.

Locked main artifact:

- `phases/phase6_repair/results/full/clean_suite_b12288_r128_n100_k8-16-32-48-64_ca-b-bmatch-idlekv-randomk-oldestk-oraclek.json`

Locked main setup:

- task family: `mq_niah_4q`
- turn 1 / turn 2 split suite:
  - turn 1 asks `1 and 4`, turn 2 asks `2 and 3`
  - turn 1 asks `2 and 4`, turn 2 asks `1 and 3`
  - turn 1 asks `3 and 4`, turn 2 asks `1 and 2`
- `B_base = 12288`
- `R_ctx = 128`
- `K = {8, 16, 32, 48, 64}`
- `n = 100` per split
- selector: current default generic selector

Main pooled result:

| K | B | B_match | Random-K | Oldest-K | IdleKV | Oracle-K |
|---|---:|---:|---:|---:|---:|---:|
| 8  | 0.093 | 0.102 | 0.098 | 0.088 | 0.342 | 0.407 |
| 16 | 0.093 | 0.098 | 0.098 | 0.085 | 0.417 | 0.523 |
| 32 | 0.093 | 0.095 | 0.102 | 0.083 | 0.607 | 0.927 |
| 48 | 0.093 | 0.097 | 0.098 | 0.087 | 0.668 | 1.000 |
| 64 | 0.093 | 0.100 | 0.100 | 0.083 | 0.685 | 1.000 |

Takeaway:

- the matched no-repair baseline is nonzero
- `IdleKV` clearly beats it
- trivial restore controls stay near baseline
- there is still oracle headroom

## What We Know

- The basic claim is now supported: at the same final cache footprint, future-query-informed repair can beat no-repair retention.
- The experiment is valid, but the suite is heterogeneous:
  - one split is easy
  - one is moderate
  - one is hard
- The hardest split still has a large oracle gap, so the current selector is not saturating the available signal.

## Important Constraint

Do not change the main Phase 6 method for the paper result.

In particular:

- do not replace the default selector with benchmark-shaped heuristics
- do not change the 3 chosen turn splits
- do not treat within-turn order as a new condition

## Good Lanes From Here

### Lane A: More Rigor On The Same Result

Best if the goal is to make the current paper harder to question without changing scope.

Good runs:

- rerun the same main suite with a different dataset seed
- run a small `B_base` sensitivity around `12288`
- optionally rerun the locked main setup at larger `n` for tighter uncertainty

Why:

- this strengthens the existing `mq_niah_4q` story directly
- it is cheap and easy to interpret

### Lane B: Broader Evidence, Same Family

Best if the goal is more evidence without changing benchmark family yet.

Possible directions:

- stay on `mq_niah_4q` and add more robustness runs only
- or add a second split-query NIAH family later, but only after the current result is replicated cleanly

Why:

- same-family breadth is easier to explain than jumping immediately to a very different task

### Lane C: Selector Research

Best if the goal is algorithm improvement rather than just stronger evidence.

Current diagnostic insight:

- a benchmark-shaped probe on the hardest split showed that if the selector is forced to localize the right key-bearing sentence, the hard split can reach oracle
- that should **not** become the main paper method
- but it suggests the next real selector should be better at span localization, not just global reranking

So if this lane is pursued, the right target is a more general span-localizing selector, not a NIAH-specific string-match trick

## Recommended Near-Term Order

If the goal is the paper:

1. replicate the locked main suite with another seed
2. run a small `B_base` sensitivity
3. only then decide whether more breadth or selector work is the next best use of compute

## Paper Framing

The current paper can already say:

- idle-window repair is feasible
- there is recoverable headroom after compression
- a matched-footprint two-turn benchmark shows real gains over no-repair retention

What additional compute should buy now is not a new core claim.
It should buy either:

- stronger confidence in the current claim, or
- broader evidence that the claim is not narrow to one calibrated slice
