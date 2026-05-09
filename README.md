# RepairKV

**Cache You Later: Post-Compression KV Repair for Long-Context Agentic
LLM Inference.**

RepairKV is a research prototype for **post-compression KV cache repair** in
long-context, multi-turn LLM inference. Existing KV-cache compression methods
decide which past tokens stay active using only the information available at
compression time, and then treat that decision as one-way. In agentic
workflows, a later user turn, tool result, retrieved document, or browser
observation can shift which earlier context matters, but tokens that mattered
in later turns may already be evicted.

RepairKV treats the compressed cache as revisable. During the pause before
decoding resumes, it scores offloaded evicted KV rows against the next-turn
signal and promotes a small budgeted subset back into the active GPU cache.
The main comparisons use a **matched resumed active-cache budget**: RepairKV
is compared against no-repair and content-agnostic restore controls with the
same number of active context KV rows.

## Research Question

KV-cache compression decides which past tokens remain active before the next
turn is known. RepairKV asks a narrower question:

> If a future turn reveals new evidence about which past tokens matter, can a
> runtime revise the active KV state during a pause without increasing the
> resumed active-cache budget?

This is a mechanism study, not a production serving stack. Model weights stay
fixed; the runtime revises which historical context the next decode can
directly attend to.

## Headline Result

On Qwen2.5-7B-Instruct at 32K context, RepairKV reaches **91.0%** retrieval on
a four-query needle-in-a-haystack task versus **24.5%** for the matched
no-repair baseline at the same active-cache budget, with only **96** promoted
tokens.

## Scope and Prerequisites

- The matched budget is the resumed active context KV budget. RepairKV also
  keeps an offloaded evicted-KV store; storage, scoring, and transfer costs
  are reported separately as service costs.
- The runtime figure is a single-node capacity envelope for score/select/
  promote mechanics, not a trace-backed distribution of real tool-call wait
  times.
- Paper-grade GPU runs assume local model weights under `models/`, the
  vendored `ruler/` checkout, CUDA/PyTorch, and enough GPU memory for the
  selected model and context length.
- CPU tests and figure-rendering checks can be run without launching new GPU
  experiments.

## Evidence in This Repository

- Controlled benchmark: split-query multi-query needle-in-a-haystack
  (MQ-NIAH) derived from RULER, providing explicit cross-turn relevance
  shifts and annotated future-relevant spans.
- Primary model: Qwen2.5-7B-Instruct at 32K context on a single
  RTX PRO 6000 Blackwell GPU.
- Main experiments: MQ-NIAH-2Q/4Q/6Q/8Q matched-budget restore-budget
  sweeps (`K=8,16,24,32,48,64,80,96,128`), next-turn signal specificity
  controls (StaleQ-K, WrongQ-K, Refresh-buffered), a five-turn
  relevance-shift diagnostic, eviction-policy sensitivity (SnapKV-style,
  H2O-style, StreamingLLM-style, Scissorhands-style), and runtime-capacity
  probes.
- Breadth checks: same-protocol Llama-3.1-8B portability probes and
  alternative selector/retention variants.
- Preliminary external-validity check: a controlled real-repository
  relevance-shift diagnostic over 48 callsite examples from 12 pinned
  open-source repositories drawn from the SWE-bench repository pool. It is
  not a SWE-bench issue-resolution benchmark. At `K=192`, event-only
  RepairKV improves exact identifier accuracy from 18.8% (matched
  no-repair) to 72.9%; label-assisted references (File-gated RepairKV at
  83.3%, AnchorWindow-K at 89.6%) show remaining headroom.

## Paper

- Source: `paper/main.tex`
- Rebuilt PDF: `paper/main.pdf`
- Figure renderer: `paper/scripts/render_paper_figures.py`
- Writing and venue guide: `paper_guide.md`

Rebuild the paper from the repo root:

```bash
.venv/bin/python paper/scripts/render_paper_figures.py
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

LaTeX intermediates are written to `paper/aux/` by `paper/.latexmkrc`.
Undefined references, undefined citations, overfull boxes, or figure overlap
should be fixed before a paper snapshot.

## Reproducing Checks

Run the focused active diagnostic tests:

```bash
.venv/bin/python -m pytest \
  phases/phase15_real_repo_relevance_shift/tests \
  phases/phase6_repair/tests/test_runner.py -q
```

Run focused paper and closure tests:

```bash
.venv/bin/python -m pytest \
  phases/phase14_critical_flaw_closure/tests/test_audit_phase14_readiness.py \
  phases/phase13_iteration_framework/tests/test_paper_language.py \
  phases/phase13_iteration_framework/tests/test_framework.py \
  phases/phase10_expansion/tests/test_multiturn.py \
  phases/phase10_expansion/tests/test_multiturn_runner.py -q
```

Run the broader CPU-side suite:

```bash
.venv/bin/python -m pytest -q
```

GPU experiments should start with the smallest smoke test that can falsify the
design, then move to a locked run only after the smoke passes a written gate.

## Repository Map

- `paper/`: ICML-style paper draft, figure assets, and rendering scripts.
- `paper_guide.md`: terminology, venue constraints, figure rules, and
  editing guardrails.
- `phases/phase6_repair/`: core matched-budget repair protocol, selectors,
  reporting, and unit tests.
- `phases/phase9_experiment_deepening/` through
  `phases/phase14_critical_flaw_closure/`: completed experiment expansions,
  smoke evaluators, locked-run wrappers, and paper-readiness audits.
- `phases/phase15_real_repo_relevance_shift/`: completed appendix
  diagnostic for real-repository relevance shifts.
- `phases/phase18_pre_submission/`: pre-submission supplement (selector
  ablations and time-budgeted query-aware baselines).
- `docs/`: project status and result-retention notes.
- `models/`: local model weights; ignored by git.
- `ruler/`: vendored RULER checkout; treated as external benchmark code.

## Experiment Vocabulary

- `Matched no-repair` (`B_match`): no repair under the same resumed
  active-cache budget. Primary baseline for all main claims.
- `RepairKV`: scores evicted KV rows against the next-turn signal and
  promotes a `K`-budgeted subset back into the active cache before
  decoding resumes.
- `Random-K`, `Oldest-K`: content-agnostic restore controls that promote
  from the same evicted KV store as RepairKV without scoring against the
  next-turn signal.
- `StaleQ-K`: scores using the previous-turn query rather than the
  upcoming one. Specificity control.
- `WrongQ-K`: scores using another example's next-turn query.
  Specificity control.
- `Refresh-buffered`: reselects the full resumed active budget from active
  plus offloaded rows using the next-turn signal. A method-boundary
  reference for selector headroom, not a deployable full-prefix recompute
  baseline.
- `ToolFile-K`: file-name-assisted control for the real-repository
  diagnostic, with oldest-row backfill.
- `File-gated RepairKV`: label-assisted reference that restricts repair
  candidates to the event-named file.
- `AnchorWindow-K`: label-assisted locality reference for the
  real-repository diagnostic.
- `SpanRef-K`: appendix-only diagnostic over annotated future answer-span
  groups. It enumerates feasible annotated span-group subsets with cost at
  most `K`; it is not an implementable algorithm and is not a universal
  upper bound over all possible K-token repairs.

## Active Questions

- What stronger next-turn-aware selectors close the gap to Refresh-buffered
  and the label-assisted references (File-gated, AnchorWindow-K)?
- How should repair lift from token-row promotion to page- or block-level
  promotion in a production KV-tiering stack?
- What scheduler-aware policy decides when to repair across arbitrary turn,
  tool, retrieval, or state-change boundaries rather than only at the
  fixed two-turn pause boundary?
- What trace-scheduled repair experiment best connects the
  runtime-capacity envelope to real tool/environment wait distributions?

## Git Hygiene

Generated phase outputs, local model weights, LaTeX intermediates, and rendered
plot binaries are ignored unless deliberately promoted. Keep paper-ready source
changes in tracked code, `.tex`, `.md`, and compact CSV artifacts that are
needed to regenerate figures.
