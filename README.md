# IdleKV

IdleKV is a research prototype for repairing compressed key-value (KV)
caches during idle gaps between long-context agent turns. The core idea is
that a cache compressed after one turn can be stale for the next turn: once
new turn text reveals what matters, the system can use the idle window to
restore selected evicted KV rows before decoding resumes.

The paper studies this under matched resumed active-cache budgets. IdleKV
keeps evicted KV rows in an off-device evicted-KV store, scores candidates
after the next-turn relevance signal is known, restores a fixed budget `K`,
and compares against a no-repair baseline with the same active-cache budget.

## Current Status

- Paper draft: `paper/main.tex`; rebuilt PDF: `paper/main.pdf`.
- Main figure generation: `paper/scripts/render_paper_figures.py`.
- Paper guide and terminology/style rules: `paper_guide.md`.
- Locked main evidence currently covers MQ-NIAH-2Q/4Q/6Q/8Q on
  Qwen2.5-7B-Instruct, plus specificity, multi-turn, and first-stage
  retention-policy diagnostics.
- Cross-model and additional algorithm branches are tracked in
  `phases/phase13_iteration_framework/phase13_plan.md`; promote only locked
  runs that pass written gates.

## Repository Map

- `paper/`: ICML-style draft, figure assets, and rendering scripts.
- `paper_guide.md`: source-of-truth writing, terminology, and figure rules.
- `docs/`: current project status and result-retention notes.
- `phases/phase6_repair/`: matched-budget repair protocol, selectors,
  reporting, and tests.
- `phases/phase7_broader_evidence/`: locked 4Q/6Q evidence and export
  scripts.
- `phases/phase9_experiment_deepening/`: operating-regime and proxy-scorer
  experiments.
- `phases/phase10_expansion/`: query-count breadth, specificity, multi-turn,
  retention-policy, model-transfer, and quantization probes.
- `phases/phase11_main_robustness/`: Llama 4Q and accumulated-attention
  retention robustness runs.
- `phases/phase12_policy_breadth/`: sink-plus-recent retention breadth runs.
- `phases/phase13_iteration_framework/`: closure framework, promotion gates,
  and active next-run scripts.
- `saved_results/`: small retained summaries from earlier phases.
- `models/`: local model weights only; ignored by git.
- `ruler/`: vendored RULER checkout; treated as external benchmark code.

## Rebuild The Paper

From the repo root:

```bash
.venv/bin/python paper/scripts/render_paper_figures.py
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

`paper/main.pdf` should rebuild without LaTeX errors. Underfull box warnings
from float placement are acceptable; overfull text or figure overlap should be
fixed before a paper snapshot.

## Test Commands

Targeted paper/closure tests:

```bash
.venv/bin/python -m pytest \
  phases/phase13_iteration_framework/tests/test_paper_language.py \
  phases/phase13_iteration_framework/tests/test_framework.py \
  phases/phase10_expansion/tests/test_multiturn.py \
  phases/phase10_expansion/tests/test_multiturn_runner.py -q
```

Broader CPU-side suite:

```bash
.venv/bin/python -m pytest -q
```

Run unit tests after code changes. GPU experiments should be preceded by the
smallest smoke that can falsify the experiment design.

## Experiment Discipline

Every GPU run should be tied to one of two purposes:

- **Smoke run:** validate task design, implementation, budget calibration, and
  failure modes before spending more compute.
- **Locked run:** produce a pre-specified figure/table candidate with a written
  promotion gate.

Long GPU jobs should run in `tmux`, write explicit CSV/JSON outputs under the
owning phase directory, and be audited before paper integration. Do not use
smoke-only data in the main paper.

## Evidence Conventions

- `Matched` means no repair under the same resumed active-cache budget.
- `IdleKV` means restore conditioned on the current next-turn signal from the
  offloaded evicted-KV store.
- `Gold-K` is a benchmark-metadata hindsight reference, not an implementable
  algorithm.
- `Random-K` and `Oldest-K` are content-agnostic restore controls.
- `Refresh-buffered` reselects the whole resumed active budget from active plus
  offloaded rows using the next-turn signal; it is a method-boundary reference,
  not a deployable full-prefix recompute baseline.

## Git Hygiene

Generated phase outputs, local model weights, LaTeX intermediates, and rendered
plot binaries are ignored unless deliberately promoted. Keep paper-ready source
changes in tracked code, `.tex`, `.md`, and compact CSV artifacts that are
needed to regenerate figures.
