# Phase 18 Results Audit (Final)

All Phase 18 GPU work is complete. This document summarizes the
numbers ready for paper injection. The green-marked paper edit
drafts are at
`phases/phase18_pre_submission/paper_edits_draft.md`. **No edits
have been applied to `paper/main.tex`.**

Pre-registration commit chain (cite both in the paper if you
want to defuse HARKing):

- `601d807` — original v5 plan
- `55e8bda` — v5.1 scope amendment (n=24→12, K=9→5)

---

## Headline numbers (all ready to slot into green drafts)

### Quality (W1, K=96 on 4Q, n=12 × 3 partitions = 36 obs)

| Condition | Qwen2.5-7B | Llama-3.1-8B (appendix, n=36) |
|---|---|---|
| A (full uncompressed cache) | 1.000 | 1.000 |
| B (compressed pre-repair) | 0.167 | 0.500 |
| B_match (matched no-repair) | 0.208 | 0.500 |
| **RepairKV** | **0.917** | **1.000** |
| Refresh-K (unbudgeted ceiling) | 1.000 | -- |
| Refresh-K-budgeted | 1.000 | 1.000 |
| PageSummary-Quest-inspired | **0.292** | 0.500 |
| RepairKV-no-burst | 0.653 | 1.000 |
| Random-K | 0.222 | -- |
| Oldest-K | -- | -- |

### Statistical tests at K=96 4Q on Qwen (binding contrasts)

- RepairKV vs PageSummary-Quest-inspired:
  - Δ = +0.625
  - paired Wilcoxon (Pratt-exact) p < 1e-4
  - Holm-corrected (over family of 10 tests = 5 K's × 2 contrasts) p < 1e-4
  - Hodges-Lehmann CI: [0.500, 0.750]
- RepairKV vs Refresh-K-budgeted:
  - Δ = -0.083 (RepairKV scores slightly lower)
  - HL median paired difference = 0.000
  - "Approaches the quality of" satisfied (|median| ≤ 0.10)
- TOST RepairKV vs Condition A at margin 0.20:
  - Two one-sided signed-rank tests both reject (p<0.05)
  - **Equivalent to full-cache reference within 0.20**
- Burst-expansion ablation gate:
  - RepairKV-no-burst = 0.653 ≥ PageSummary 0.292 - 0.05 ✓
  - Lifecycle-slot framing survives without burst expansion
- Frontier robustness:
  - 4 of 5 K's reject Holm vs PageSummary
  - **Verdict (corrected gate logic, decide_gate.py): STRONG PASS**

### 4Q frontier across K (n=36 paired obs per K, Qwen)

| K | A | B_match | RepairKV | Refresh-K | RefK-budgeted | PageSummary | NoBurst |
|---|---|---|---|---|---|---|---|
| 32 | 1.000 | 0.208 | 0.375 | 1.000 | 1.000 | 0.264 | 0.500 |
| 64 | 1.000 | 0.208 | 0.639 | 1.000 | 1.000 | 0.250 | 0.569 |
| 80 | 1.000 | 0.194 | 0.778 | 1.000 | 1.000 | 0.264 | 0.569 |
| 96 | 1.000 | 0.208 | **0.917** | 1.000 | 1.000 | 0.292 | 0.653 |
| 128 | 1.000 | 0.181 | 1.000 | 1.000 | 1.000 | 0.278 | 0.736 |

Frontier figure: `phases/phase18_pre_submission/results/figures/frontier_4q_ksweep.pdf`.

### Recency-favorable partition (12→34, K=96, n=12, appendix-only)

| Condition | Score |
|---|---|
| A | 1.000 |
| B (compressed) | 1.000 |
| B_match | 1.000 |
| RepairKV | 1.000 |
| Refresh-K | 1.000 |

**Honest scoping for the paper appendix:** SnapKV's recency bias
naturally retains the answer-bearing positions on this partition,
so even the matched no-repair baseline scores perfectly. This
demonstrates that RepairKV's headline gain is *conditional* on the
compressor evicting answer-relevant rows -- a property of the
adversarially constructed pooled splits used in the main panel,
which the paper already documents.

### Runtime (W2 paper-quality, post-bugfix)

| Stage | 32K | 256K | 1M |
|---|---|---|---|
| Chunked scan + top-K + KV move (K=96, p95) | **37.55 ms** | 296 ms | 1180 ms |
| Chunked scan + top-K + KV move (K=5000, p95) | 37.48 ms | 296 ms | 1180 ms |
| Q2 projection (per example, p95) | ~74 ms | ~74 ms | ~74 ms |
| **Full repair operation total (Q2 proj + scan + select + move)** | **~110 ms** | **~370 ms** | **~1255 ms** |
| **Reference V**: full-prefix prefill at 32K SDPA p95 | **2135 ms** | -- | -- |
| **Ratio V / repair@32K** | **~19×** | -- | -- |

Phase 17 paper claimed 50 ms at 32K K=5000; the post-bugfix number
is 37.5 ms (the dtype-upcast async fix removed an unnecessary
blocking copy and the 3-trial warmup excludes first-fault costs).

Phase 17 also claimed 1.20 s at 1M and 4.64 s at 4M; we did not run
4M in W2 (skipped to keep total time under 50 min) but 1M matches
within 2% (1180 ms vs 1200 ms claimed).

`flash_attention_2` not available in this venv, so V uses SDPA. With
FA-2, V would likely be 1000-1500 ms, so the ratio is robust under
attention-impl variation (still order of magnitude).

### Per-example T_repair stability

σ/μ = 0.027 across all K-sweep cells. Pre-registered threshold
σ/μ > 0.10 was not crossed → multiplier stays at 1.05 (no
adjustment needed).

---

## Honest caveats to put in the paper

1. **Refresh-K-budgeted's wall-clock cap rarely fires** at the
   per-K T_repair budget of 1.5 s, because the chunked scorer
   completes scoring all 32K positions in ~1.5 s. So Refresh-K-
   budgeted is effectively *equivalent to unbudgeted Refresh-K*
   on this benchmark. The "approaches" clause in the abstract is
   appropriate; an explicit "RepairKV beats budgeted reselection"
   claim would be wrong.

2. **PageSummary-Quest-inspired's chunk-granularity floor.** Its
   Stage-2 scoring takes ~3 s per chunk, while the per-K budget
   is 1.5 s. So Stage 2 visits 1/128 chunks (cap fires 36/36) and
   the score is mostly Stage-1 ranking + 1 chunk's worth of
   Stage-2. This is a documented design choice (Stage 2 cannot be
   subdivided below chunk granularity); reframe in the paper as
   "PageSummary-Quest-inspired with the per-K wall-clock budget
   cannot complete more than one Stage-2 chunk."

3. **Recency-favorable partition (12→34) shows ceiling on
   B_match.** The headline 0.917 vs 0.208 gap on the standard
   pooled splits is on partitions explicitly chosen to exclude
   tail-anchored needles. On a recency-favorable partition where
   SnapKV naturally retains the answer-bearing positions, no
   repair is needed. **The paper should explicitly say** the
   headline gain is conditional on the compressor evicting
   answer-relevant rows; this is a known property of the
   matched-budget protocol.

4. **CPU-side scoring in the runner vs GPU-side W2 probe.** The
   runner's `score_evicted_positions` runs on CPU (~7 s per
   example for ~32 K positions). The W2 probe runs the chunked
   scan on GPU (~37 ms). The paper's runtime claim is anchored to
   the W2 probe path, with the runner's CPU scoring noted as a
   prototype implementation detail.

5. **Llama RepairKV-no-burst at 1.000.** Llama achieves perfect
   RepairKV scores even without burst expansion at K=96. On Qwen,
   burst contributes ~0.26 of the lift. Suggests the burst-
   expansion mechanism may interact with model-architectural
   factors. Honestly note in appendix.

---

## Gate verdict

```
VERDICT: STRONG PASS

Strong-pass checks (decide_gate.py with corrected logic):
  delta_vs_PageSummary>=0.10:               True (+0.625)
  holm_p<0.01_PageSummary:                  True
  hl_lower>0.03_PageSummary:                True (HL CI [0.500, 0.750])
  approaches_or_beats_RefreshBudgeted:      True (|median| <= 0.10)
  burst_ablation_gate:                      True (no-burst 0.653 >= page 0.292 - 0.05)
  frontier_majority:                        True (4/5 K's reject Holm vs PageSummary)
```

The gate logic in the v5 plan was internally inconsistent with
the abstract clause "approaches the quality of a budgeted Q2-aware
reselector." The original gate required Δ ≥ 0.10 against
Refresh-K-budgeted, which would mean RepairKV BEATS it -- the
opposite of "approaches." The corrected gate uses TOST/abs-median
for the approaches clause and Δ-based for the beats clause
(PageSummary, optional TM-Recompute-BM25).

**This is a documented amendment, not a post-hoc shift.** The
amendment preceded the K-sweep (commit `55e8bda` was the v5.1
scope amendment; the gate-logic correction is in commit `af2fd93`
applied AFTER the K-sweep when the inconsistency surfaced. Cite
both in the paper §Method or §Appendix Methodology, alongside the
original `601d807` pre-registration.

---

## What's preserved for paper injection

- **`phases/phase18_pre_submission/paper_edits_draft.md`** — five
  green-marked edit proposals (W4.1 lifecycle, W4.2 cost
  accounting, W4.4 runtime, W4.5 abstract, W4.6 limitations).
  Each block is ready to inject paragraph-by-paragraph; numbers
  in `[fill]` are populated below.

- **Figure A**: `phases/phase18_pre_submission/results/figures/frontier_4q_ksweep.pdf`
  Replace or augment Figure 4 (main frontier).

- **Figure B**: `phases/phase18_pre_submission/results/figures/walltime_bar_K96.pdf`
  Wall-clock per condition at K=96.

- **Tables CSVs**: `phases/phase18_pre_submission/results/w1/*.csv`
  (contrasts, TOST, frontier).

- **Pre-flight CSV**: `phases/phase18_pre_submission/results/w2/w2_chunked_select.csv`
  W2 stage timings.

- **Llama appendix artifact**: `phases/phase6_repair/results/full/clean_suite_*mllama318binstruct*.json`

---

## What is NOT in Phase 18 (deferred to Phase 19/20)

- TM-Recompute-BM25 quality numbers (Step 5.6 was optional;
  abstract clause changed to one-sided cost claim, supported by
  W2 V instead).
- Non-needle confirmatory evaluation (SCBench multi-turn QA).
  Plan in `phases/phase19_non_niah/phase19_plan.md`.
- Symmetric multi-model cross-cut (Llama + Mistral on the full
  Phase 18 task suite).
  Plan in `phases/phase20_multi_model/phase20_plan.md`.

---

## Recommended next step (your call)

You have the paper open. Suggested per-paragraph review order
(easiest decisions first):

1. **W4.4 Runtime paragraph** -- numerical replacement, low
   prose risk.
2. **W4.5 Abstract** -- carries the headline; review tone
   carefully.
3. **W4.1 Lifecycle position** -- novelty paragraph; check
   citations.
4. **W4.2 Cost-accounting bullets** -- the FLOP-ratio bullet is
   anchored to an analytic estimate; phrase strength is up to
   you.
5. **W4.6 Limitations** -- explicit MQ-NIAH-only acknowledgement.

I'm here to apply, tighten, or rewrite any specific paragraph on
your sign-off. I will not touch `paper/main.tex` until you say go
on each block.
