# Phase 18 RESULTS FINAL (work in progress)

Last updated: 2026-05-06 18:27:52 UTC. NO PAPER EDITS APPLIED.

**Status of queued reruns:**
- Tight-budget sweep (mult 0.05/0.10/0.30/1.05): 3/4 complete
- K-sweep redo (post PageSummary fix): pending
- Chunk-size sensitivity (CS 32/64/256 + RepairKV-chunked): pending
- Llama low-K (K=32, 48): pending
- GPU verify (n=12, K=96, score_on_gpu=True): pending

**The K-sweep table below is from the BUGGY-FUSION artifact.** PageSummary
post-fix scores expected to drop from ~0.29 to ~0.19 (matching tight sweep
mult 0.10/0.30/1.05 result of 0.194). The K-sweep redo will replace this
table.

(see status block at top)


---

### K-sweep (Qwen, n=12 × 3 partitions = 36 obs/K)

| K | A | B_match | RepairKV | Refresh-K | Refresh-K-budgeted | PageSummary-Quest-inspired | RepairKV-no-burst | Oracle-K |
|---|---|---|---|---|---|---|---|---|
| 32 | 1.000 | 0.208 | 0.375 | 1.000 | 1.000 | 0.264 | 0.500 | nan |
| 64 | 1.000 | 0.208 | 0.639 | 1.000 | 1.000 | 0.250 | 0.569 | nan |
| 80 | 1.000 | 0.194 | 0.778 | 1.000 | 1.000 | 0.264 | 0.569 | nan |
| 96 | 1.000 | 0.208 | 0.917 | 1.000 | 1.000 | 0.292 | 0.653 | nan |
| 128 | 1.000 | 0.181 | 1.000 | 1.000 | 1.000 | 0.278 | 0.736 | nan |


---

### Tight-budget multiplier sweep (Qwen, K=96, n=36)

| mult | budget (ms) | RepairKV | Refresh-K-budgeted | Δ vs RKB | PageSummary | Δ vs PSum | RKB cap fires | RKB positions scored |
|---|---|---|---|---|---|---|---|---|
| 0.05 | 347 | 0.917 | 0.389 | +0.528 | 0.194 | +0.722 | 36/36 | 8107/32768 |
| 0.10 | 708 | 0.917 | 0.667 | +0.250 | 0.194 | +0.722 | 36/36 | 15986/32768 |
| 0.30 | 2110 | 0.917 | 1.000 | -0.083 | 0.194 | +0.722 | 0/36 | 32768/32768 |
