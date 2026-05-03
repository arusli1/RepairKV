# Real Agentic Diagnostic Design

This note turns the "one real agentic task" gap into an executable experiment.
It should not be run until the active proxy scorer run is finished or idle GPU
capacity is clearly available.

## Reviewer Question

Does pre-resume KV repair help when the next turn is driven by a realistic
workflow event, not only by synthetic MQ-NIAH question text?

The target evidence is not full SWE-bench or WebArena task success. The target
is a narrower trace diagnostic: after a tool-like event shifts relevance to a
different part of a real code context, does IdleKV restore the needed context
better than matched no-repair and content-agnostic restore controls?

## Candidate Task: RepoDelta-Retrieval

Use real repository files as the long context and synthetic but verifiable
questions over exact spans in those files.

One example has:

- `context`: concatenated file cards from a real repository, with each card
  containing path, language, and source text.
- `Q1`: asks for an exact identifier, string literal, or configuration value in
  one file group.
- `tool event`: a short appended message such as "pytest failed in
  `path/to/file.py`; inspect `function_or_symbol`".
- `Q2`: asks for an exact identifier/value from a different file group named by
  the tool event.
- `gold spans`: token spans for the Q2 file/symbol region.

This is still controlled, but it uses real code/document tokens and a
tool-result-like relevance shift. It should be described as a realistic-content
diagnostic, not as end-to-end agent validation.

## Minimal Implementation

1. Add a small generator that emits `TaskExample` records from real repository
   files.
2. Reuse the Phase 6 two-turn protocol by adding a task alias whose Q1/Q2 spans
   point at different real file cards.
3. Treat the tool event as part of the turn-`N+1` relevance signal, appended
   before Q2 scoring and decoding.
4. Keep answer scoring exact-match over short outputs, as in MQ-NIAH, so the
   metric remains auditable.

## Unit Tests Before GPU

- The generator produces unique Q1 and Q2 answer strings.
- Q1 and Q2 spans come from different file cards.
- The Q2 answer string appears in the rendered context exactly once.
- Character spans map to nonempty token positions after chat-template rendering.
- The tool event text names the Q2 file or symbol and does not contain the Q2
  answer itself.
- The task can be generated under the intended context-length budget.

## Smoke Run

Run only after the unit tests pass.

- Model: Qwen2.5-7B-Instruct.
- Context: 32K.
- Samples: `n=2` or `n=3`.
- Restore budgets: `K={48,96}`.
- Conditions: `A/B/B_match/Random-K/Oldest-K/IdleKV/Oracle-K`.
- Scorer: exact Q2/tool-event signal first; proxy only after exact behavior is
  understood.

Smoke passes only if:

- full-context score is at least `0.80`;
- matched no-repair is at least `0.15` below full context;
- IdleKV beats matched and both content-agnostic controls at one or more K;
- Gold-K covers IdleKV;
- failures are not caused by answer-format truncation or duplicate answer
  strings.

## Locked Run If Smoke Passes

- Samples: `n=20-30`, depending on smoke variance.
- Restore budgets: `K={32,48,64,96,128}`.
- Same conditions as smoke.
- Bootstrap confidence intervals for IdleKV and matched no-repair.

The target paper object is one compact one-column plot or a short main-text
sentence plus appendix plot. It should replace weaker future-work prose rather
than add clutter.

## Promotion Gate

Main-paper promotion requires:

- a distinct workflow claim: tool/repo-style relevance shift, not another
  MQ-NIAH-shaped query split;
- full-context ability and nontrivial matched no-repair gap;
- IdleKV improvement over Random-K and Oldest-K;
- at least `n=20`;
- a caption that states this is a controlled trace diagnostic, not end-to-end
  SWE-bench/WebArena success.

If any gate fails, keep the paper's current limitation language and move this to
future work or an appendix negative result.
