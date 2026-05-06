# Phase 18 pre-registration: outcome bands BEFORE queued reruns

**Committed at 2026-05-06 17:47 UTC, before any queued rerun has read its
data**. Cite this hash alongside the original `601d807` /
`55e8bda` / `af2fd93` chain.

This document binds expected outcomes for the four queued reruns. If
the actual result falls outside the band, we report it unchanged and
discuss the deviation in §Limitations rather than retro-fitting the
plan.

## Why these bands now

Round-3 hostile-reviewer critique flagged that:
- The PageSummary fusion fix (`c1f08a7`) was committed AFTER the
  K-sweep's PageSummary score of 0.292 was visible to the team.
- A reviewer can read the chain as data-aware design changes.
- Pre-registering expected post-fix outcomes before the rerun runs
  defuses this critique by demonstrating the rerun is not a
  fishing expedition.

## Tight-budget sweep (RUNNING, ETA 18:37 UTC)

K=96, 4Q, n=12 × 3 partitions, multipliers ∈ {0.05, 0.10, 0.30, 1.05}.

Conditions and predictions:

| Multiplier | Budget | RepairKV | Refresh-K-budgeted | PageSummary-Quest-inspired | RepairKV-no-burst |
|---|---|---|---|---|---|
| **1.05 (loose)** | ~1.5 s | 0.91 ± 0.04 | 0.95 ± 0.05 (cap rarely fires; effectively unbudgeted) | 0.40 ± 0.15 (with fusion fix; was 0.292 with bug) | 0.65 ± 0.10 |
| **0.30** | ~450 ms | 0.91 ± 0.04 | 0.85 ± 0.10 (cap fires for ~10% of cells) | 0.30 ± 0.10 | 0.65 ± 0.10 |
| **0.10** | ~150 ms | 0.91 ± 0.04 | 0.50 ± 0.15 (cap fires for ~70% of cells; partial scoring) | 0.20 ± 0.10 | 0.65 ± 0.10 |
| **0.05** | ~75 ms | 0.91 ± 0.04 | 0.25 ± 0.10 (cap fires immediately; near-tiebreaker fallback) | 0.20 ± 0.10 | 0.65 ± 0.10 |

**Pass conditions:**
- Refresh-K-budgeted MUST drop monotonically as multiplier decreases.
  If it doesn't, the cap is not actually binding -- problem.
- RepairKV stays approximately constant across multipliers (its
  scoring is outside the budget loop).
- At multiplier 0.10 (deployment-runtime budget),
  RepairKV − Refresh-K-budgeted ≥ 0.30. If smaller, the
  "approaches budgeted Q2-aware reselection at smaller
  wall-clock" claim weakens.

**Fail handling:** if Refresh-K-budgeted at multiplier 0.10 is ≥ 0.85,
the cap is still not binding tightly enough and we either (a) drop
multiplier further or (b) honestly reframe to "approaches at all
budgets we tested."

## K-sweep redo (QUEUED, ETA 19:15 UTC)

n=12, K∈{32, 64, 80, 96, 128}, 4Q, multiplier 1.05.

Predictions for the PageSummary fix only changes PageSummary's
numbers; other conditions should be within 0.02 of the original
K-sweep numbers (Wilcoxon noise floor).

| K | Original (buggy) PageSummary | Predicted (post-fix) PageSummary band |
|---|---|---|
| 32 | 0.264 | 0.30–0.50 |
| 64 | 0.250 | 0.30–0.55 |
| 80 | 0.264 | 0.30–0.55 |
| 96 | 0.292 | 0.30–0.55 |
| 128 | 0.278 | 0.30–0.55 |

**Pass condition:** PageSummary post-fix is HIGHER than buggy across
all K. If lower, the fusion change had unexpected effect; we report
unchanged and document.

**RepairKV vs PageSummary post-fix at K=96:** Δ ≥ 0.30 with HL CI
strictly above 0.20. If Δ falls below 0.20, the headline claim
weakens and we soften the abstract to "RepairKV outperforms" rather
than "RepairKV approaches X *and* outperforms Y."

## Chunk-size sensitivity sweep (QUEUED, ETA 19:55 UTC)

K=96, 4Q, n=12, multiplier 1.05, PageSummary-only condition at
chunk_size ∈ {32, 64, 256}. Also serves as Attack-2 partial defuse
(see §Attack 2 in RESULTS_AUDIT addendum).

| chunk_size | Stage 2 chunks visited (predicted) | PageSummary score (predicted) |
|---|---|---|
| 32 | ~2 chunks (cap 1.5 s, ~750 ms/chunk) | 0.40 ± 0.15 |
| 64 | ~1 chunk | 0.40 ± 0.15 |
| 128 (default) | ~0 chunks (post-fusion-fix: Stage 1 ranking) | 0.30–0.55 (from K-sweep redo) |
| 256 | ~0 chunks (Stage 1 ranking only) | 0.30 ± 0.15 |

**Pass condition:** PageSummary score is monotone non-decreasing as
chunk_size decreases (smaller chunks → more Stage-2 visits → better
selection within visited chunks). If non-monotone, document the
non-monotonicity in §Appendix and honestly report.

## Llama low-K (QUEUED, ETA 20:25 UTC)

K∈{32, 48} on Llama-3.1-8B-Instruct, 4Q, n=12.

Predictions:

| K | A | B_match | RepairKV | RepairKV-no-burst | Refresh-K-budgeted | PageSummary |
|---|---|---|---|---|---|---|
| 32 | 1.0 | 0.30–0.55 | 0.45–0.75 | 0.45–0.75 (Llama saturates fast) or 0.30–0.55 if burst matters | 0.85–1.00 | 0.30–0.55 |
| 48 | 1.0 | 0.35–0.60 | 0.65–0.90 | 0.45–0.85 | 0.90–1.00 | 0.35–0.60 |

**Disambiguation question:** Does RepairKV-no-burst on Llama at low
K stay close to RepairKV (saturation) or drop materially below
(burst contributes)? The K=96 result (RepairKV-no-burst = 1.000)
was uninformative because of saturation.

**Pass condition (for cross-model robustness):** Δ(RepairKV vs
B_match) > 0.10 at K=32 on Llama. If smaller, the cross-model
generalization claim weakens to "method works on Qwen."

## Outcome reporting protocol

After each rerun lands:
1. Compare actual to predicted band.
2. If WITHIN band: report normally.
3. If OUTSIDE band: report unchanged + add a §Discussion sentence
   acknowledging the deviation.
4. Do NOT modify the plan or the gate after seeing the data.

This document plus the prior pre-reg commits (601d807, 55e8bda,
af2fd93, c1f08a7, 853dfb1) constitutes the full pre-registration
chain for Phase 18. The paper §Method should cite all six.
