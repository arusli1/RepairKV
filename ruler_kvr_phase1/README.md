# Phase 1: RULER-KVR

This directory contains the repo-local code for Phase 1 eviction degradation
measurement.

The purpose of this phase is narrower than the stock
[ruler_benchmark_initial_test](/home/ubuntu/IdleKV/ruler_benchmark_initial_test):

- use fixed-depth, attribution-friendly long-context tasks
- run a real eviction policy during context prefill
- measure degradation as a function of `k_budget`
- log enough cache decisions to explain failures

The Phase 1 code is isolated here so that:

- upstream [benchmark/RULER](/home/ubuntu/IdleKV/benchmark/RULER) stays mostly vendored
- the earlier stock RULER calibration run stays readable
- all Phase 1 scripts, logs, data, and results live in one folder

## Layout

- `run_phase1_ruler_kvr.py`
  - repo-local entrypoint
- `phase1/`
  - modular Python package for task generation, eviction execution, scoring, and aggregation
- `artifacts/`
  - generated datasets, raw predictions, and per-sample eviction traces
- `results/`
  - summarized outputs, including `phase1_condition_b.json`
- `logs/`
  - run logs and small smoke-test outputs

## Current backend split

Phase 1 uses HuggingFace + `kvpress` first.

That is intentional:

- `kvpress` already supports Qwen2-family models and real prefill-time cache compression
- it exposes `SnapKVPress` and `StreamingLLMPress`
- it gives a clean way to validate the benchmark logic before a later `vLLM` port

So this folder is the first correct implementation path for Condition B, not the
final high-throughput serving path.

## Main outputs

- `results/phase1_condition_a.json`
- `results/phase1_condition_b.json`
- `artifacts/.../predictions.jsonl`
- `artifacts/.../trace/<sample>.pt`

`trace/*.pt` files contain the layer-level eviction logs used for attribution:

- kept vs dropped token positions
- importance scores
- logged query vectors for the last observation window
- task-relevant token survival summaries
