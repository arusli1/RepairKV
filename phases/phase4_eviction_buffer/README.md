# Phase 4: Eviction Buffer and Runtime Capacity

This folder contains the Phase 4 eviction-buffer prototypes and the current
runtime-capacity profiler used by the paper.

The original Phase 4 goal was to turn Phase 3 evicted-token artifacts into a
CPU-backed buffer that can:

- store evicted token KV pairs on CPU pinned memory,
- score buffered candidates against recent active-cache context,
- move selected winners back to GPU,
- and profile the repair pipeline.

## Status

The paper-backed runtime evidence now comes from `runtime_capacity.py` and
`scripts/run_runtime_capacity_profile.py`.

- `runtime_capacity.py` is the active profiler for synthetic Qwen-shaped BF16
  KV move/reinsertion and chunked offloaded-store candidate scanning.
- Runtime CSVs used by the paper are copied into `paper/figures/` so figure
  rendering is reproducible from a stable snapshot.
- `instructions.md`, `eviction_buffer.py`, `profiling.py`, and
  `run_phase4_profiling.py` are retained as the historical log-backed Phase 4
  prototype. They are useful for context and tests, but they are not the source
  of the current paper's runtime table or appendix runtime figures.

## What Is Here

- `instructions.md`
  - historical Phase 4 implementation spec and planning notes
- `src/buffer/eviction_buffer.py`
  - prototype `BufferEntry`, `EvictionBuffer`, and `extract_recent_q_vecs`
- `src/buffer/profiling.py`
  - historical log-backed profiling helpers for transfer, scoring, attention
    overhead, and end-to-end repair
- `src/buffer/runtime_capacity.py`
  - active paper profiler for Qwen-shaped BF16 KV movement and chunked
    offloaded-store candidate scanning
- `src/buffer/feasibility.py`
  - feasibility-frontier computation from profiling outputs
- `scripts/run_phase4_profiling.py`
  - historical CLI for the log-backed prototype path
- `scripts/run_runtime_capacity_profile.py`
  - active CLI for paper runtime-capacity runs

## Design Notes

- The runtime-capacity profiler reuses the Phase 2 `PositionTrackedCache` and
  `inject_kv` mechanics. It does not import Phase 3 eviction logs, historical
  selector strategies, or benchmark-specific repair quality code.
- The active profiler defaults match the paper's local Qwen geometry:
  `28` layers, `28` query heads, `4` KV heads, and `head_dim = 128`.
- The chunked-selection profiler streams synthetic key rows from pinned host
  memory to GPU and performs real GPU scoring/top-K operations. Synthetic values
  are used only because random key contents do not affect latency.
- The historical log-ingestion helpers recurse through nested Phase 3 eviction-log trees,
  so they can point at the benchmark log root or one task-specific leaf
  directory.

## Current Runtime Outputs

Current runtime-capacity runs write generated artifacts under:

- `results/runtime_capacity/*.csv`
- `results/runtime_capacity/*_metadata.json`
- `results/runtime_capacity/logs/*.log`

Paper-frozen CSV snapshots live under:

- `paper/figures/runtime_capacity_8k_32k_100k.csv`
- `paper/figures/runtime_capacity_250k_500k_k5000.csv`
- `paper/figures/runtime_chunked_select_32k_1m.csv`

## Historical Prototype Outputs

The older log-backed prototype writes JSON artifacts under:

- `results/phase4_profiling/transfer_latency.json`
- `results/phase4_profiling/scoring_latency.json`
- `results/phase4_profiling/attention_overhead.json`
- `results/phase4_profiling/end_to_end_repair.json`
- `results/phase4_profiling/feasibility_frontier.json`
- `results/phase4_profiling/selection_quality.json`

Those JSON outputs are not used by the current paper unless a future pass
explicitly revives and validates that path.

## Validation

Run the active runtime-capacity tests with:

```bash
.venv/bin/python -m pytest phases/phase4_eviction_buffer/tests/test_runtime_capacity.py -q
```

Run the full Phase 4 test slice with:

```bash
.venv/bin/python -m pytest phases/phase4_eviction_buffer/tests -q
```
