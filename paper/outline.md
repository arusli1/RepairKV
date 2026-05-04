# IdleKV Paper Outline

## Title
- IdleKV: Repairing Active KV Memory Between Turns

## Abstract
- KV compression makes an early memory-allocation decision.
- Multi-turn workflows reveal relevance later.
- IdleKV adds a pre-resume repair point: buffer evicted KV, score with next-turn cue, restore K.
- Claim: better resumed quality than matched no-repair at same active KV budget.
- Scope: controlled setting; exact scorer is evidence, proxy/scoring work remains.

## 1. Introduction
- Long-context inference is KV-memory bound.
- KV compression is active-memory allocation, not just storage reduction.
- Problem: future relevance is unknown when compression happens.
- Multi-turn agents/tools/retrieval/user followups reveal new cues later.
- Failure: active cache can be misaligned with next-turn relevance.
- Opportunity: pre-resume repair window.
- Batch-1/dedicated: slack may be idle compute; shared serving: explicit tradeoff.
- Core question: can a cue repair compressed active memory under matched active budget?
- Contributions: problem/protocol, IdleKV primitive, matched-budget evidence, cost/scorer accounting.

## 2. Problem and System Model
- State: active GPU KV, evicted KV buffer, preserved prompt/answer tail.
- Timeline: prefill/Q1 -> compress -> cue arrives -> repair -> resume/Q2.
- Budgets: base active budget, restore budget K, matched resumed active budget.
- Accounting: active GPU KV matched; CPU buffer/latency/bytes reported separately.
- Repair window: idle/slack in dedicated settings; scheduled work in shared serving.
- Goal: improve active memory selection without full-prefix recompute.

## 3. Related Work
- Compression/eviction: SnapKV, H2O, StreamingLLM.
- Query-aware access: QUEST, ShadowKV.
- Serving/offload/reuse: vLLM, InferCept, paged/offloaded KV.
- Benchmarks/lifecycle: RULER, SCBench.
- External/persistent memory: still must pass through context/KV to affect decoding.
- Boundary: revising a prior compression decision after future cue arrives.

## 4. IdleKV Method
- Compress context after turn N; keep evicted rows recallable.
- Encode next-turn cue Q2 before answer generation.
- Score evicted rows against cue under active+evicted attention competition.
- Select local bursts around high-score anchors.
- Restore K context tokens; resume decoding.
- Scorers: exact Q2 query vectors; proxy Q2 cache rows for cheaper path.

## 5. Evaluation Protocol
- Task: split-query MQ-NIAH as controlled relevance-shift test.
- Model/context: Qwen2.5-7B-Instruct, 32K.
- Compressor: context-only SnapKV; appendix H2O-inspired accumulated-attention
  retention check.
- Conditions: full, base, matched no-repair, Random-K, Oldest-K, IdleKV, SpanRef-K.
- Specificity controls: stale cue, donor cue, refresh-buffered.
- Metrics: Q2 score, repair gain over matched, SpanRef-K recovery, latency.

## 6. Results
- Q1: Does repair beat matched no-repair?
  - Restore-budget frontier across 2Q/4Q/6Q/8Q.
- Q2: Is gain cue-specific?
  - Stale/donor cues near matched; true Q2 cue improves.
- Q3: Where does repair help?
  - Stale-but-recoverable regime; not all-zero or saturated regimes.
- Q4: What remains unsolved?
  - SpanRef-K gap shows selector headroom over annotated answer-span groups.
- Q5: What is the systems cost?
  - KV movement cheap; exact scoring slow; proxy path reduces latency.
- Q6: Is result robust?
  - Partition endpoints, content-agnostic controls, limited compressor/model checks.

## 7. Discussion and Limitations
- Main claim: deferred, cue-conditioned active-memory repair.
- Not generic KV compression; complementary to eviction/offload/query-aware access.
- Not free in saturated serving; requires slack or scheduling budget.
- Evidence limits: synthetic retrieval, explicit Q2 cue, one main model, one main compressor.
- Exact scorer is mechanistic; deployable repair needs faster scorer.
- External memory does not remove KV bottleneck; retrieved facts still need active context/KV.
- Future work: realistic multi-turn benchmarks, better selectors, proxy/page scorers, byte-budgeted repair, MLA/multi-architecture KV.

## 8. Conclusion
- Compression should not be a final memory decision.
- Future cues can reveal what active memory should contain.
- IdleKV shows pre-resume repair can recover quality under matched active KV budget.
- Broader direction: mutable KV memory for long-context inference.

## Appendix
- Protocol details, prompts, splits.
- SpanRef-K definition and caveats.
- Query-count breadth.
- Specificity/control details.
- Selection/overlap diagnostics.
- H2O-inspired accumulated-attention retention check.
- Partition endpoints.
- Proxy/runtime and model-transfer checks.
