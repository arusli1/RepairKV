# RepairKV — AdaptFM 2026 Edit Plan (v5, finalized for audit)

**Live source:** `paper/main.tex` (1257 lines).  `draft1/` is stale.
**Deadline:** May 8 AoE.  **Venue:** AdaptFM @ ICML 2026, double-blind, 6 main pages excl. refs + unlimited appendix.
**Page state:** main body ends mid-page-7 → over by ~½ column.
**Anonymization:** handled at PDF level by user.

---

## Coordination boundary with Phase 18 agent

Phase 18 owns reviewer fix #2 (compute/latency comparison to recompute). This plan therefore excludes:

- §Runtime paragraph at lines 685–697 (Phase 18 will edit).
- Recompute-deferred disclaimer at lines 317–321 (Phase 18's call).
- Any insertion of recompute timing numbers.
- Any framing language using "full-prefix recompute" (Phase 18's vocabulary).

**Scope here:** reviewer fixes #1, #3, #4 + page-budget cuts.

---

## Color-markup policy

`paper/main.tex` contains 94 `\textcolor{red}{...}` and 27 `\textcolor{blue}{...}` wrappers marking the user's in-progress revisions. Rules for this pass:

- **NEW sentences I insert:** plain text, no color. The user can color-tag during audit.
- **Modifications inside an existing colored wrapper:** preserve the wrapper. Edit the contents in place.
- **Cuts (deletions):** remove the entire colored span if the wrapper is exactly around the deleted text; otherwise close/open wrappers cleanly.
- **Replacements of large colored blocks** (e.g., C6): drop the wrapper. Re-wrap is the user's call during audit.

---

## Edits — atomic, with exact text

Each row is one diff with verified line numbers and exact `before`/`after` text.

### A1. Sharper novelty boundary  *(reviewer #1)*

**Where:** §2 Related Work, lines 265–267 — the closing sentence of paragraph B, currently inside `\textcolor{blue}{...}`. The existing sentence already draws the InferCept/CachedAttention contrast; the sharpening adds (a) the lifecycle-position label and (b) the matched-budget framing.

**Before** (lines 265–267):
```
\textcolor{blue}{InferCept and CachedAttention show how preserved or preloaded KV can avoid
recompute; \repairkv{} instead studies which preserved rows should re-enter
the active cache when a later turn reveals a new relevance signal.}
```

**After:**
```
\textcolor{blue}{InferCept and CachedAttention show how preserved or preloaded KV can avoid
recompute, but they preserve rows verbatim and do not re-rank under a new
signal. \repairkv{} is therefore a \emph{post-compression, pre-resume}
lifecycle operator: not ``better query-aware retrieval,'' but an
active-cache adaptation step evaluated under a matched resumed-cache budget
that other dynamic-KV proposals do not control for.}
```

**Pay:** C5 (delete Discussion opener at lines 715–717). **Net:** ~+2 source lines, paid by C5's ~3-line cut.

**Why this lands:** in two sentences the paper's contribution is now positioned against three adjacent literatures (compression, query-aware retrieval, pause/resume preservation), with the *matched active-cache budget* surfaced as the protocol-level distinction. Highest-leverage edit in the plan.

---

### A3. Cost-accounting itemize  *(reviewer #3)*

**Where:** §3 Method "Matched-budget evaluation" paragraph. Insert after line 411, immediately following the existing sentence: *"\textcolor{blue}{Extra} store bytes, scorer time, and transfer time are \textcolor{blue}{not part of} the matched active-cache budget\textcolor{blue}{; they are reported separately as service costs.}"*

**Why §Method, not §Runtime:** Phase 18 owns §Runtime. Putting cost taxonomy in §Method makes it a *protocol* statement (what we account for) rather than a runtime measurement — a stronger response to "clarify cost accounting" — and avoids stepping on Phase 18.

**Insert** (plain text, no color):
```
Concretely, service costs comprise: (i) the $Q_2$ projection prefill on
$|Q_2|$ tokens, (ii) the host-memory store, which scales linearly with the
offload pool ($\approx 1.8$\,GB at 32K rows for Qwen2.5-7B-Instruct BF16
and $\approx 240$\,GB at 4M rows; see Appendix~\ref{app:additional-discussion}),
(iii) chunked scan plus top-$K$ selection, and (iv) host-to-device transfer
of the promoted $K$ rows. Magnitudes appear in §\ref{sec:results}.
```

**Pay:** C2 (delete two redundant sentences from same paragraph). **Net:** roughly neutral.

**Numbers verified:** the appendix at line 1095 already states "4M searchable rows imply about 240 GB before metadata." 32K rows scale linearly: 240 GB × (32K / 4M) ≈ 1.85 GB → I round to 1.8 GB. No new measurement introduced.

---

### A4a. Soften agentic claims (token-level)  *(reviewer #4)*

**Edit 1 — Abstract, line 60.** Modify *inside* the existing `\textcolor{blue}{the same pattern}` wrapper.

| | |
|---|---|
| **Before** | `shows \textcolor{blue}{the same pattern};` |
| **After** | `shows \textcolor{blue}{directionally consistent gains};` |

**Edit 2 — §Real-repo, lines 654–655.** Replace one sentence (currently inside the big `\textcolor{red}{...}` block spanning 627–659). Keep the red wrapper intact.

| | |
|---|---|
| **Before (line 654–655)** | `This remains a small external-validity diagnostic, with no claim to be a real-code benchmark.` |
| **After** | `This is preliminary external validity, not proof of agentic-workload effectiveness; we make no claim to a real-code or agent benchmark.` |

**Net:** 0. (Both edits replace existing text 1:1.)

---

### A4b. Intro framing tightening  *(reviewer #4 — root cause)*

The agent-heavy paragraph at lines **146–159** sets up agentic motivation, but the headline evidence is synthetic MQ-NIAH. That setup-payoff mismatch *is* what reviewer #4 is reacting to.

**Where:** §1 Introduction, lines 146–159. The block contains the "56–74% / 99% / 37%" citation cascade and is currently mostly inside `\textcolor{red}{...}` (with the closing sentence in `\textcolor{blue}{...}`).

**Before** (lines 146–159):
```
Coding-agent measurements reinforce why this interval is worth studying:
OS-level execution, including tool calls and setup, can account for
56--74\% of task latency~\citep{agentcgroup}.
\textcolor{red}{Furthermore, modern production serving stacks such as
vLLM~\citep{vllm} now deploy prefix caching by default, but re-deriving
evicted KV through a fresh prefill pass remains the fallback on cache
miss, eviction, or paused-request discard.
The cost is severe: prior work reports recompute consuming as much as
99\% of multi-turn prefill time when historical KV is repeatedly
recomputed~\citep{cachedattention}, and 37\% of end-to-end execution
time when paused-request KV is discarded in agentic
workloads~\citep{infercept}, which motivates tiered KV \textcolor{blue}{systems} that trade
recompute for offloaded reuse~\citep{lmcache,ttkv}.}
Together, \textcolor{blue}{these factors motivate repairing the active KV state before the next decode.}
```

**After** (plain text, drop the colored wrappers — user can re-color during audit):
```
Multi-turn settings — agentic workflows in particular — make these idle
windows operationally visible, with execution and tool-call gaps reported
as a substantial fraction of task latency~\citep{agentcgroup}. When
paused-request KV is recomputed rather than reused, the cost can dominate
end-to-end time~\citep{cachedattention,infercept}, motivating tiered-KV
approaches~\citep{lmcache,ttkv}. Together, these factors motivate
repairing the active KV state before the next decode.
```

**Net:** ~6 source lines saved. Removes the setup-payoff mismatch by reframing agents as the most-pronounced case of multi-turn relevance shift, not the central frame.

**Citations preserved:** agentcgroup, cachedattention, infercept, lmcache, ttkv. Dropped: vllm (still cited in §Related Work line 256 — no orphan).

---

### B1. Abstract task qualifier  *(serves A4)*

**Where:** Abstract, line 60.

| | |
|---|---|
| **Before** | `91.0\% retrieval on a four-query needle-in-a-haystack task` |
| **After** | `91.0\% retrieval on the synthetic split-query MQ-NIAH-4Q task` |

**Note:** "at 32K context" already appears earlier in the same sentence ("on Qwen2.5-7B-Instruct at 32K context, RepairKV…"), so the qualifier "at 32K context" is **not** added — would be a duplication.

**Net:** 0.

---

### B2. No-fetch-baseline defensive sentence  *(serves A1)*

**Where:** end of §2 Related Work, immediately after the A1 modification (extends paragraph B).

**Insert** (plain text, after the closing `}` of the A1-modified blue wrapper):
```
We do not directly compare to attention-time fetch methods (Quest, FIER,
ShadowKV) because they do not maintain a fixed active-cache budget; they
trade compute for adaptive top-$K$ access and are evaluated under
different protocols.
```

**Pay:** C2 (combined with A3's pay). **Net:** ~+1 line.

---

## Cuts (~26 source lines saved)

| # | Lines | Edit |
|---|---|---|
| C1 | 146–159 | Subsumed by A4b — already counted (~6 saved) |
| C2 | 423–426 | Delete `This retrieval score measures target recall, while unconstrained answer quality is outside the controlled benchmark. Stricter exact-set and precision-sensitive scoring serve as confirmatory checks.` (~5 lines) |
| C4 | 218–219 | Delete `\textcolor{blue}{Within this controlled scope, the results suggest} that long-context agents may need between-turn KV state updates in addition to pre-generation pruning.` (~2 lines) |
| C5 | 715–717 + 718 | Delete sentence at 715–717 (`\repairkv{} is a between-turn maintenance primitive ... resource-adaptive tiered-KV runtimes.`); on line 718, change `It complements` → `\repairkv{} complements` (~3 lines net) |
| C6 | 754–771 | Replace 17-line "Future work" block with 3-sentence compressed version (~10 lines saved) |
| C7 | 611–612 | Move `\textcolor{red}{Canonical reproductions of H$_2$O and StreamingLLM would require the full original online policies.}` into a `\footnote{...}` attached to the preceding sentence (~2 lines) |

**C6 replacement text** (drop colored wrappers; user re-colors during audit):
```
A production-facing repair benchmark would add multi-turn traces with
explicit pause boundaries, hidden future relevance, and controls that
separate active-cache budget from storage, transfer, and recompute cost,
evaluated across top long-context models and multiple compression policies.
It should also measure idle-window slack in real agent workloads (coding,
tool use, retrieval, browser interaction), since repair is only useful
when scoring and KV movement fit scheduler constraints. Preliminary
nonlinear variable-tracking diagnostics (Appendix~\ref{app:vt8hop-div2})
suggest branching dependency structure can cause sharp compression failures
even where standard tasks appear solved; \repairkv{} is robust in those
preliminary tests, but broader validation is needed.
```

**Total estimated cut:** ~26 source lines.
**Total estimated insert:** A1 ~2 + A3 ~6 + A4b 0 (covered by C1 cut) + B2 ~4 = ~12.
**Net:** ~14 source lines reclaimed → main body comfortably under 6 pages.

---

## Edit order (no time estimates per user request)

**Phase 1 — independent edits:**
1. C4 (line 218–219 delete).
2. B1 (abstract task qualifier).
3. A4a Edit 1 (abstract softening).

**Phase 2 — boundary + framing:**
4. A1 + C5 paired.
5. B2 (after A1).
6. A4b (replaces lines 146–159; subsumes C1).

**Phase 3 — cost accounting + remaining cuts:**
7. A3 + C2 paired.
8. A4a Edit 2.
9. C6 (future-work compression).
10. C7 (eviction-policy footnote).

**Page-count check** with `pdfinfo paper/main.pdf` after Phase 1 and Phase 2.

---

## Internal audit (what I verified)

- **Line numbers** verified by reading current `paper/main.tex` for every edit target.
- **Existing color wrappers** identified at every edit site; preservation policy applied.
- **Cross-references** `\ref{sec:results}`, `\ref{app:additional-discussion}`, `\ref{app:vt8hop-div2}` all confirmed to exist (lines 518, 1093, 1120).
- **Citation accuracy:**
  - `infercept` and `cachedattention` are accurately characterized as preserve-KV-without-re-rank (per the paper's own Related Work, lines 256–257, 263).
  - `quest`, `fier`, `shadowkv` are accurately characterized as attention-time fetch with no fixed active budget (per paper lines 240–241 and the Quest/FIER/ShadowKV literature).
  - `agentcgroup` 56–74%, `cachedattention` 99%, `infercept` 37% — all are real numbers from those papers, retained or paraphrased appropriately in A4b.
  - `lmcache`, `ttkv` correctly grouped as tiered-KV approaches.
- **No edits inside §Runtime (lines 685–697) or the recompute-deferred disclaimer (lines 317–321)** — Phase 18 territory respected.
- **A3 cost numbers** consistent with existing appendix accounting (4M rows ≈ 240 GB at line 1095). 32K rows ≈ 1.8 GB by linear scaling, no new measurement.
- **A1 + B2** placed in the same paragraph (paragraph B of Related Work) so both boundary statements live next to their targets. Reading order: pause/resume contrast → lifecycle label → matched-budget framing → no-fetch-baseline defense.
- **C5 follow-on**: confirmed line 718 needs `It` → `\repairkv{}` substitution after deleting 715–717, otherwise dangling antecedent.
- **A4b citation set:** every citation in the original 14-line block is either retained in the 6-line replacement or already present elsewhere in the paper (`vllm` at line 256). No orphans.
- **B1 duplicate-check:** "at 32K context" already in earlier clause of same sentence — qualifier dropped to avoid repetition.
- **No conflicts with Phase 18 vocabulary:** A3 does not use "full-prefix recompute" phrasing.

---

## Out of scope

- §Runtime paragraph (Phase 18).
- Recompute-deferred disclaimer at lines 317–321 (Phase 18).
- Anonymization in `.tex` (handled at PDF level).
- `draft1/` directory.
- Multi-model main-body content (`model_transfer_breadth.pdf` is stale).
- Title.
- Algorithm 1 / §Method / §Setup / §Results structural changes.
- New experimental claims.

---

## Open items

None blocking. Plan is ready for user audit.
