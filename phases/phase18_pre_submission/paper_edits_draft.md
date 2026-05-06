# Phase 18 W4 paper edit drafts (green-marked, awaiting per-paragraph approval)

All edits use `\textcolor{green!50!black}{...}` so they are visually
distinct from the pre-Phase-18 red/blue edits in main.tex. Numbers
in `[fill]` are populated from W1/W2 paper-quality results once
Step 3 + Step 2 finish.

Pre-registration commit: **`601d807`** (`phases/phase18_pre_submission/phase18_plan.md`).

---

## W4.1 — Novelty paragraph (insert after Method §"Overview")

**Target slot:** new paragraph after the Method "Overview" paragraph
(currently ending around line 297).

**Proposed green text:**

```tex
\paragraph{\textcolor{green!50!black}{Lifecycle position.}}
\textcolor{green!50!black}{Query-aware KV retrieval methods such as
Quest, FIER, ShadowKV, and ParisKV reselect rows during \emph{active}
attention from a retained or indexed full cache, using
page-criticality envelopes (Quest), low-rank reconstructions
(ShadowKV), drift-robust ANN indices (ParisKV), or fine-grained
retrieval (FIER)~\citep{quest,fier,shadowkv,pariskv}. Page-eviction-
and-recall systems (ArkVale's recallable page summaries, InfiniGen's
speculative partial-weight recall, EM-LLM's event-boundary recall)
recall query-relevant evicted state at decode time.
Preservation-and-resume systems (InferCept, Continuum,
CachedAttention) keep KV across pause boundaries so future turns do
not re-prefill~\citep{infercept,continuum,cachedattention}. \repairkv{}
occupies a distinct slot: it applies query-aware scoring to the
\emph{post-compression, pre-resume} boundary in the cache lifecycle
-- once per pause, against an evicted store created by the compressor
itself, with no persistent low-rank index. The matched-active-cache
controls ($\bmatch$, Random-$K$, Oldest-$K$, StaleQ-$K$, WrongQ-$K$,
Refresh-$K$-budgeted, PageSummary-Quest-inspired) hold the resumed
footprint constant so the comparison isolates this lifecycle operator
from "keep more rows," "use a different scorer at active attention,"
or "use a per-step page recaller."}
\textcolor{green!50!black}{Refresh-$K$-budgeted scores positions in
chunks under a wall-clock cap; consequently each chunk's softmax
normalizer is over (active $\cup$ chunk-evicted) rather than (active
$\cup$ all-evicted). The single-chunk degenerate case matches the
unbudgeted scorer exactly, and within-chunk rankings agree; cross-
chunk score magnitudes are scaled but not directly comparable. This
is a documented design choice, since computing a global denominator
across all evicted positions is incompatible with a wall-clock cap.}
```

---

## W4.2 — Cost-accounting paragraph (Discussion / Limitations)

**Target slot:** insert into existing Discussion / Limitations
paragraph (around line 715-740) as a bulleted list.

**Proposed green text:**

```tex
\textcolor{green!50!black}{The matched-budget protocol matches active
GPU KV rows; other resources are reported separately, not matched.}
\begin{itemize}[leftmargin=1.4em,noitemsep,topsep=2pt]
\item \textcolor{green!50!black}{\textbf{Active-cache GPU rows:} matched at $\bbase + K$.}
\item \textcolor{green!50!black}{\textbf{Host-memory store size:} not matched; $|W_N|$ is reported.}
\item \textcolor{green!50!black}{\textbf{Peak host RAM:} reported in Appendix; the offloaded BF16 KV store is the dominant term.}
\item \textcolor{green!50!black}{\textbf{Bytes transferred per repair (host $\to$ GPU):} $K$ rows $\times$ \emph{layer\_kv\_bytes} per token; reported in Table~\ref{tab:runtime-stages}.}
\item \textcolor{green!50!black}{\textbf{$Q_2$ projection compute, scan + top-$K$ + transfer compute, cache merge / re-layout compute:} stage-decomposed in Figure~\ref{fig:runtime-envelope} and Table~\ref{tab:runtime-stages}.}
\item \textcolor{green!50!black}{\textbf{Total wall-clock:} matched in W1 against Refresh-$K$-budgeted and PageSummary-Quest-inspired, both run with the per-example $T_{\text{repair}}$ envelope including amortized $Q_2$ projection and scoring cost.}
\item \textcolor{green!50!black}{\textbf{FLOPs:} not matched, but reported as a derived ratio. \repairkv{}'s repair operation costs about [fill F-ratio]$\times$ fewer FLOPs than full-prefix prefill at 32K context (analytic from $K$, $|W_N|$, $d_k$, $H$, $L$), since \repairkv{} only touches keys at score time and does no value-side computation until the small post-promotion attention.}
\end{itemize}
```

---

## W4.3 — Real-repository diagnostic wording (existing paragraph)

**Target slot:** existing Real-repository diagnostic paragraph
(around line 626-659) and Table 3 caption.

**Proposed green edits (small diffs, marked):**

In the paragraph (around line 626):
- Replace "To test external validity beyond MQ-NIAH" with
  "\textcolor{green!50!black}{As a preliminary external-validity diagnostic beyond MQ-NIAH}"
- Replace "shows the same pattern" with
  "\textcolor{green!50!black}{is consistent with the same pattern}"
- Add at end of paragraph:
  "\textcolor{green!50!black}{We treat this as suggestive, not as evidence of agent-system performance; a confirmatory non-needle long-context evaluation (e.g.\ SCBench multi-turn QA) is the most important next experiment and is left for follow-up.}"

In Table 3 caption:
- Prepend "\textcolor{green!50!black}{Preliminary external-validity diagnostic.}"

---

## W4.4 — Runtime paragraph (replace the existing Runtime paragraph)

**Target slot:** existing Runtime paragraph (lines 685-697).

**Proposed green text:**

```tex
\paragraph{\textcolor{green!50!black}{Runtime.}}
\textcolor{green!50!black}{We run a separate GPU runtime probe for
the repair operation. The probe measures $Q_2$ projection on the
compressed cache, chunked scan, top-$K$ selection, KV movement over
pinned host-memory BF16 tensors, and the cache-merge step, plus
FlashAttention reference timings for full-prefix and evicted-prefix
prefill (Table~\ref{tab:runtime-stages}). At $K=5000$ and 32K
offloaded candidates, the full sequential p95 \repairkv{} cost
including $Q_2$ projection and merge measures [fill X] ms; with
H2D/scan overlap (double-buffered streams), [fill X'] ms. At 1M
offloaded candidates the figures are [fill Y] s sequential, [fill Y'] s
with overlap. For reference, full-prefix prefill at 32K with
FlashAttention measures [fill V] ms, and prefilling the evicted
prefix takes [fill W] ms. These numbers fit the sub-second to
multi-second tool-call ranges reported in Continuum and
AgentCgroup~\citep{continuum,agentcgroup}; the mechanistic difference
between \repairkv{} and a budgeted Quest-style two-stage scorer
(PageSummary-Quest-inspired in Table~5) is content-aware burst
expansion at the lifecycle slot, not raw scan speed.}
```

---

## W4.5 — Pre-registered abstract branches (replace abstract clause)

**Strong-pass abstract sentence (round-4 attack #3 defuse: 19× ratio reframed to constant-vs-linear scaling, with caching caveat):**

```tex
\textcolor{green!50!black}{At a matched active-cache budget,
\repairkv{} approaches the quality of a budgeted $Q_2$-aware
reselector that operates at the same lifecycle slot, without
requiring a persistent low-rank index or a $Q_2$-time full
reselection scan. The repair operation costs roughly constant
wall-clock per $K$ on the evaluation GPU, while full-prefix
recompute scales linearly with context length: at $32$K context
with SDPA the ratio is about $19\times$, and about $10\times$ with
FlashAttention-2; under aggressive prefix caching the crossover
narrows further and is left to follow-up work. On Qwen2.5-7B-Instruct
at 32K context, MQ-NIAH-4Q at $K=96$, \repairkv{} scores [fill A]
versus [fill B] for matched no-repair and [fill C] for the strongest
time-matched alternative (PageSummary-Quest-inspired).}
```

**Notes:**
- **Agentic framing (round-3 attack #3):** abstract no longer makes a
  static "19× faster than recompute" claim; instead presents the
  ratio as a function of attention-impl and caching policy.
- **19× anchoring (round-4 attack #3):** explicit acknowledgement
  that V depends on attention-impl + caching. Worst case (SDPA
  cold) is the headline number; FA-2 cold is reported alongside;
  prefix caching is named as a future direction.

**Weak-pass abstract sentence:**

```tex
\textcolor{green!50!black}{Across MQ-NIAH-4Q/6Q at 32K context,
\repairkv{} shows a modest but consistent advantage over both
wall-clock-matched recompute references and budgeted $Q_2$-aware
reselection at the same active-cache budget; at $K=96$ on 4Q,
\repairkv{} scores [fill A] versus [fill B] for matched no-repair
and [fill C] for the strongest time-matched alternative
(PageSummary-Quest-inspired).}
```

**Fail-branch abstract sentence:**

```tex
\textcolor{green!50!black}{\repairkv{} is competitive with
wall-clock-matched recompute and budgeted $Q_2$-aware reselection at
small repair budgets and identifies the post-compression,
pre-resume lifecycle slot as a previously unstudied position. We
report mixed results across $K$; further work is needed to determine
the conditions under which idle-window repair dominates $Q_2$-time
alternatives.}
```

The branch is chosen by Step 4's gate (per Phase 18 v5 plan
§W1.acceptance), not by preference. The plan was committed to git
at `601d807` before any GPU run touched headline numbers; the paper
will cite this hash in §Method.

---

## W4.6 — Limitations addition (per AdaptFM reviewer + round-3 critique)

**Target slot:** Discussion / Limitations.

**Proposed green text (agentic framing softened, multi-amendment scope acknowledged):**

```tex
\textcolor{green!50!black}{All confirmatory experiments use MQ-NIAH
variants on Qwen2.5-7B-Instruct at $32$K context with a single
SnapKV-style first-stage compressor. Cross-model evidence is
limited to one Llama-3.1-8B-Instruct appendix run. Although the
method is motivated by tool-call idle windows in agentic workflows,
the evaluation here is on static prompts; the recompute ratio
reported in the runtime probe (Table~\ref{tab:runtime-stages})
compares to a fresh full-prefix prefill, and a deployed agent with
prefix caching could see different ratios. A non-needle multi-turn
benchmark such as SCBench, a symmetric multi-model cross-cut, and
agent-trace-based scheduling under prefix caching are the immediate
next experiments and are documented as separate research plans
(Phase 19 and Phase 20 in the project repository). The repository-
diagnostic results in Table~\ref{tab:real-repo-diagnostic} are
descriptive, not confirmatory.}
```

## W4.7 — Pre-registration footnote (round-3 attack #1 defuse)

**Target slot:** end of §Method or §Appendix Methodology.

**Proposed green text:**

```tex
\textcolor{green!50!black}{\textbf{Pre-registration.} The Phase 18
plan was committed to git before any GPU run touched headline
numbers (commit \texttt{601d807}). A scope amendment
(\texttt{55e8bda}) and a gate-logic correction (\texttt{af2fd93})
followed when the original gate's threshold against
Refresh-$K$-budgeted was found inconsistent with the abstract's
``approaches the quality of'' clause. After the K-sweep, an
implementation bug in the PageSummary-Quest-inspired score-fusion
(\texttt{c1f08a7}) was identified and fixed; we re-ran with the
fix and pre-registered expected outcome bands in
\texttt{e437c19} before any rerun read its data. We report
RepairKV-chunked (\texttt{934fb8b}) as a denominator-matched
diagnostic against PageSummary so the headline contrast is not
confounded by softmax-denominator differences. Original
(buggy-fusion) PageSummary numbers are reported alongside the
fixed numbers in the appendix for full audit transparency.}
```

---

## Bibliography additions

```bibtex
@inproceedings{arkvale,
  author    = {Liu, Renze and ...},  % TODO author list
  title     = {ArkVale: Efficient generative LLM inference with recallable key-value eviction},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024},
}

@article{infinigen,
  author    = {...},  % TODO
  title     = {InfiniGen: Efficient generative inference of large language models with dynamic KV cache management},
  journal   = {arXiv:2406.19707},
  year      = {2024},
}

@article{emllm,
  author    = {...},  % TODO
  title     = {Human-like episodic memory for infinite-context LLMs},
  journal   = {arXiv:2407.09450},
  year      = {2024},
}
```

---

## Sequencing of W4 application

1. Step 2 (W2 paper-quality) finishes -> populate X / X' / Y / Y' / V / W in W4.4.
2. Step 3 (W1 4Q K-sweep) finishes + Step 4 audit -> select abstract branch (W4.5) and populate A / B / C / X-ratio.
3. Step 5.5 (burst ablation) -> if strong-pass, the W4.1 novelty paragraph language remains; if burst is the whole effect, soften W4.1 to "burst-expanded selection at the lifecycle slot."
4. Step 6 (Llama appendix) -> populate cross-model wording in W4.6 if applicable.
5. Show each green block to user for per-paragraph approval before applying to `paper/main.tex`.
6. Recompile + run Phase 17 rg validation checks.
