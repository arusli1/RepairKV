# Format Edit Plan — `draft1/main.tex`

Status: **proposal only, no edits committed**. Each item below is a discrete change you can approve / reject independently. Line numbers refer to the current `draft1/main.tex` (1257 lines, 13 PDF pages).

Driving constraints (from style references audit, 2026-05-08):
- AdaptFM 2026 main-text limit = **6 pages**, references uncapped, appendix optional. We are at **8 main pages**, so most edits below also serve length.
- ICML2026 example template uses **three numbered heading levels** (`\section / \subsection / \subsubsection`). It does **not** use `\paragraph{Foo.}` bold inline headers anywhere.
- 5 of 7 reference workshop papers use numbered subsections in Method (`3.1`, `3.2`, …); we use 0.
- We currently use **27 `\paragraph` headers** as the only organizational device — none of the references do this.

Every change is a structural / formatting move; **no scientific claim is touched**.

---

## P0 — Required for AdaptFM compliance

### P0.1 Add Impact Statement
**Where:** new unnumbered section before `\begin{thebibliography}` (current line 776), after the existing `\FloatBarrier` (line 773).

**Why:** ICML2026 template requires it; PredSched, Cache Saver, and Batch-Max all include one in this peer group. We currently have none, which is likely a gating issue, not just style.

**Proposal:** ~3-sentence boilerplate-style statement covering: (i) the work studies inference-time efficiency for long-context LLMs, (ii) downstream impact is reduced GPU memory pressure for local/dedicated agent serving, (iii) no novel ethical risks beyond those already inherent to long-context LLM deployment. Draft text to follow in a separate proposal once this slot is approved.

### P0.2 Reduce main text from 8 to 6 pages
**Where:** main text (lines 1–773, excluding bibliography).

**Why:** AdaptFM hard limit. Reference workshop papers in this peer group sit at 4–5 main pages. Cartridges (8 pages) is the only longer one and is an oral, not a 6-page workshop submission.

**Proposal — demotion candidates, ranked by lowest-cost first:**
1. **Eviction-policy sensitivity** (lines 601–624, ~1 figure + ~24 lines of prose) → demote to one-sentence main-text pointer + full content into appendix. Saves ~⅓ column + figure.
2. **Real-repository diagnostic** (lines 626–683, ~58 lines + 1 table) → keep one paragraph in main text, demote table + audit text to appendix. Saves ~⅔ column + table.
3. **Runtime** (lines 685–708, ~24 lines + 1 figure) → demote to appendix with one main-text sentence pointing at appendix figure. Saves ~⅓ column + figure.

If we keep all three, we cannot reach 6 pages without compressing Method or Results, which weakens the headline. Recommendation: demote (1) and (3) fully, keep (2) at 1 paragraph in main text. Final decision needs your call on which experiments are load-bearing for the AdaptFM reviewer.

---

## P1 — Section structure (the user-flagged issue)

### P1.1 Add numbered subsections to `\section{Method}`
**Where:** lines 270–430.

**Why:** 5 of 7 references use numbered Method subsections (Cartridges 3.1–3.5, Batch-Max 3.1–3.2, PredSched 3.1–3.4, PiKV 3.1–3.4, LATTICE 2.1–2.3). ICML example template demonstrates this layout. Our four `\paragraph` blocks already map cleanly.

**Proposal — replace 4 `\paragraph` headers with 4 `\subsection` headers:**

| Current line | Current `\paragraph` | Proposed `\subsection` |
|---|---|---|
| 274 | `\paragraph{Overview.}` | `\subsection{Overview}` (3.1) |
| 299 | `\paragraph{Two-turn protocol.}` | `\subsection{Two-turn protocol}` (3.2) |
| 330 | `\paragraph{Repair operation.}` | `\subsection{Repair operation}` (3.3) |
| 396 | `\paragraph{Matched-budget evaluation.}` | `\subsection{Matched-budget evaluation}` (3.4) |

### P1.2 Add numbered subsections to `\section{Experimental Setup}`
**Where:** lines 432–516.

**Proposal — replace 4 `\paragraph` headers with 4 `\subsection` headers:**

| Current line | Current `\paragraph` | Proposed `\subsection` |
|---|---|---|
| 436 | `\paragraph{Benchmark family.}` | `\subsection{Benchmark family}` (4.1) |
| 462 | `\paragraph{Evaluation configuration.}` | `\subsection{Evaluation configuration}` (4.2) |
| 486 | `\paragraph{Scorers.}` | `\subsection{Scorers}` (4.3) |
| 496 | `\paragraph{Baselines and references.}` | `\subsection{Baselines and references}` (4.4) |

### P1.3 Add numbered subsections to `\section{Results}`
**Where:** lines 517–708.

**Why:** Reference Results sections almost always use numbered subsections so reviewers can cite a finding ("Cartridges §5.1", "PredSched §4.1"). Our 6 unnumbered `\paragraph` blocks should become 6 subsections, or be grouped to 4.

**Proposal A — direct 1:1 mapping (6 subsections):**

| Current line | Current `\paragraph` | Proposed `\subsection` |
|---|---|---|
| 521 | `Matched-budget frontier.` | 5.1 Matched-budget frontier |
| 549 | `Next-turn signal specificity.` | 5.2 Next-turn signal specificity |
| 581 | `Repeated relevance shifts.` | 5.3 Repeated relevance shifts |
| 601 | `Eviction-policy sensitivity.` | 5.4 Eviction-policy sensitivity *(see P0.2 — likely demoted)* |
| 626 | `Real-repository diagnostic.` | 5.5 Real-repository diagnostic |
| 685 | `Runtime.` | 5.6 Runtime *(see P0.2 — likely demoted)* |

**Proposal B — grouped (4 subsections), recommended if main-text page budget is tight:**
- 5.1 Matched-budget frontier (current 521)
- 5.2 Robustness *(current 549 + 581 + 601 merged with bold inline phrase leadins)*
- 5.3 Real-repository diagnostic (current 626; one paragraph if P0.2 demotes)
- 5.4 Runtime envelope (current 685; pointer-only if P0.2 demotes)

Pick A or B based on the P0.2 outcome.

### P1.4 Rename `\section{Method}` → `\section{RepairKV}`
**Where:** line 270.

**Why:** Three references rename Method after the contribution itself (Cartridges = "The Cartridge paradigm", Batch-Max = "Batch-Max", LATTICE = "Compression Layer"). Reviewers reading the TOC see what the paper builds. "Method" is the most generic possible label.

**Proposal:** `\section{Method}` → `\section{\repairkv{}}`. The existing `\label{sec:method}` (line 271) stays — labels and titles are independent.

---

## P2 — Discussion / Limitations / Conclusion / Future Work

### P2.1 Restructure §6
**Where:** lines 711–771 (currently `\section{Discussion and Limitations}` with body text + `\paragraph{Conclusion.}` + `\paragraph{Future work.}`).

**Why:** Our current shape (one numbered section containing Discussion → Conclusion → Future work as three `\paragraph`s) doesn't match any reference. The two patterns reviewers expect:
- **Cartridges:** `\section{Discussion and conclusion}` — limitations folded into body, no separate Future Work.
- **Cache Saver:** `\section{Discussion}` + unnumbered `\section*{Limitations}` + `\section*{References}`.

**Proposal — Cartridges shape (recommended, lower edit cost):**
- Rename `\section{Discussion and Limitations}` → `\section{Discussion and conclusion}`.
- Keep current body (lines 715–742) as Discussion paragraphs.
- Keep `\paragraph{Conclusion.}` block (lines 743–752) as the closing paragraph; remove the `\paragraph{Conclusion.}` header itself — let the paragraph stand.
- **Drop the `\paragraph{Future work.}` block (lines 754–771)** as a separate paragraph. Compress to one closing sentence appended to the Conclusion (e.g., "Production-facing repair benchmarks with multi-turn traces, multiple compression policies, and idle-window slack measurement are the natural next step."). Move the rest into a new appendix subsection `Future work and benchmark axes`.

### P2.2 Drop the itemize block in Discussion
**Where:** lines 723–728 (the "should be most useful when" 3-item list).

**Why:** Reference papers use 0–1 itemize blocks per main text. We have 4. The list reads as well as inline prose: "The mechanism should be most useful when the next turn changes which old context matters, the compressed cache still leaves useful choices in the evicted host-memory store, and the idle window or scheduler has enough slack for scoring and KV movement."

**Proposal:** convert to one inline sentence.

---

## P3 — Itemize / list discipline

### P3.1 Remove itemize in Method Overview
**Where:** lines 284–292 (the `C_base / W_N / s` definition list).

**Proposal:** convert to inline prose: "Three objects participate at the pause boundary: the active evictable KV $C_{\mathrm{base}}$ with $|C_{\mathrm{base}}|=\bbase$; the offloaded evicted KV $W_N$, hidden from decoding unless promoted; and the signal $s$ at the pause boundary — any tokens the runtime uses to score evicted KV, such as a new user query, a tool result, the model's recent generation, or other tokens that signal upcoming attention or relevance."

### P3.2 Keep Contributions itemize
**Where:** lines 205–217.

**Why:** every reference workshop paper that lists contributions uses an itemize for it (Cartridges Intro, PiKV Intro, Cache Saver Intro). Keep as-is.

### P3.3 Keep Evaluation configuration itemize
**Where:** lines 469–477 (compressor / base budgets / restore budgets / replay).

**Decision needed:** this is a hyperparameter list. Inlining hurts readability. **Recommend keeping**. Reference precedent: Cartridges keeps similar setup lists as itemize.

**Net effect:** 4 itemize → 2 itemize, matching reference baseline.

---

## P4 — `\paragraph` reduction

After P1.1–P1.3 promote 14 of our 27 `\paragraph` headers to `\subsection`s, we still have 13 left:
- Intro: 3 (`Motivation`, `Setting`, `Contributions`) — lines 67, 175, 189
- Appendix C / D / E: 10

**Proposal P4.1 — Intro:** drop the three `\paragraph` headers in Intro. Reference Intros (Cartridges, FMP, PredSched, PiKV, LATTICE) use no inline headers. The first sentence of each block already topic-sentences itself; bolding adds nothing.

**Proposal P4.2 — Appendix:** keep appendix `\paragraph` headers as-is. Cache Saver and Cartridges both use heavy `\paragraph` in appendix; this is acceptable and matches references.

**Net effect:** 27 → ~10 `\paragraph` headers (all inside appendix), matching the ICML2026 template guidance and Cartridges/PredSched practice.

---

## P5 — Related Work organization

### P5.1 Keep Related Work in main text but consider thematic clustering
**Where:** lines 222–268 (currently 2 flat paragraphs, 47 lines).

**Decision point:** if we make 6 pages with budget room, keep as-is — 2 well-shaped paragraphs is fine (Cartridges shape). If we need to shed lines, two options:

- **Option A — defer to appendix entirely** (Cache Saver / LATTICE pattern): replace §2 with a 4-line pointer paragraph and move the body to a new appendix `\section{Extended Related Work}`. Saves ~⅔ column.
- **Option B — keep as 2 paragraphs but tighten** to ~30 lines. Saves ~½ column. Lower risk.

**Recommendation:** Option B unless P0.2 demotions don't get us under 6 pages.

---

## P6 — Appendix renaming

### P6.1 Mirror main text in appendix names
**Where:** lines 1002, 1092, 1147.

**Why:** Cartridges (`A. Extended Results / B. Extended Related Work / C. Extended method / D. Datasets`) and Cache Saver (`A. Related Works / B. Cache Saver: Additional Details / C. Additional Experimental Details`) both mirror main-text section names so reviewers can find supporting evidence.

**Proposal:**

| Current line | Current name | Proposed name |
|---|---|---|
| 1002 | `Additional Evaluation Details` | `Extended Experimental Setup` |
| 1092 | `Additional Discussion` | `Extended Discussion` |
| 1147 | `Supplementary Experimental Views` | `Extended Results` |

If P0.2 demotes Eviction-policy sensitivity / Runtime fully, add a fourth appendix section `Extended Robustness` (or similar) for those.

---

## P7 — Smaller line-level fixes

### P7.1 Section name capitalization consistency
ICML2026 example uses **sentence case with content words capitalized** (e.g., "Format of the Paper"). Our section names already follow this convention. **No change needed.**

### P7.2 Table caption placement
Tables 1–3 use `\caption{...}` placed above the tabular environment. Matches ICML convention. **No change needed.**

### P7.3 Figure caption length
Reference workshop figure captions are 2–4 lines. Ours are mostly OK; spot-check candidates if we want to shed lines:
- Figure 1 caption (lines 122–125, 4 lines): borderline, leave.
- Figure 7 (`fig:multiturn`) caption (lines 594–598, 5 lines): trim 1 line possible.
- Figure 8 (`fig:policy-breadth`) caption: OK.
**Low priority.**

### P7.4 Algorithm style
Algorithm 1 uses `\caption{General \repairkv{} Execution Framework}` — fine. Reference comparator: PiKV Algorithm 1 sits in the same style. **No change needed.**

---

## Suggested commit ordering (for when you approve)

1. **P0.1** add Impact Statement (smallest insertion, unblocks compliance audit).
2. **P1.1 / P1.2 / P1.3** add subsections — pure rename, low risk, but verifies layout still compiles cleanly.
3. **P1.4** rename Method → RepairKV.
4. **P4.1** drop Intro `\paragraph` headers.
5. **P3.1 / P2.2** convert two itemize blocks to prose.
6. **P2.1** restructure §6.
7. **P6.1** rename appendix sections.
8. **P0.2** main-text demotions to appendix — bigger surgery, do last so we re-measure page count after the cheaper edits.
9. **P5.1** Related Work decision — depends on P0.2 outcome.

After each step: recompile, recount pages, save the PDF as a checkpoint per the per-paragraph approval workflow.

---

## What is explicitly NOT in this plan

- No claim text is changed.
- No numbers / experimental results / citations are changed.
- No figure data or captions are rewritten (only captioned-line trims flagged in P7.3).
- No bibliography reordering / additions.
- No abstract changes (current 6-sentence abstract already matches ICML 4–6 guidance).
- No author / affiliation / template-command changes.
