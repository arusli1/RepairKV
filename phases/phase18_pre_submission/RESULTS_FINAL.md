# Phase 18 RESULTS FINAL

Aggregated from queued reruns. NO PAPER EDITS APPLIED.


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
