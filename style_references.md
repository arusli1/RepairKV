# IdleKV Style References

Last verified: 2026-05-05.

Use this file as the quick reference sheet for writing, figure layout, and
framing decisions for the AdaptFM ICML 2026 submission. `paper_guide.md`
remains the active editing contract; this file records external style examples
and what to borrow from them.

## Venue Anchor

- AdaptFM CFP: https://adaptfm.gitlab.io/call-for-papers.html
- Submission portal: https://openreview.net/group?id=ICML.cc/2026/Workshop/AdaptFM
- ICML 2026 template/example: https://media.icml.cc/Conferences/ICML2026/Styles/example_paper.pdf

AdaptFM explicitly welcomes work on resource-efficient foundation model
inference, including adaptive test-time compute, token-level adaptation,
adaptive attention, dynamic KV cache compression, runtime systems,
hardware-software co-design, benchmarking across resource budgets, and
quality-resource tradeoff analysis. IdleKV should therefore be framed as
resource-adaptive inference-time state management, not as generic LLM serving,
fine-tuning, or a coding-agent benchmark.

AdaptFM 2026 appears to be a new workshop, so there are no prior AdaptFM
accepted papers to imitate. The closest style references are ICML efficient
foundation model workshops, especially ES-FoMo and long-context workshops.

## Highest-Priority Style References

### Cartridges

- OpenReview: https://openreview.net/forum?id=DuVSIWY5vC
- PDF: https://openreview.net/pdf?id=DuVSIWY5vC
- Venue/status: ICML 2025 ES-FoMo III oral.
- Why it matters: closest high-quality workshop example for long-context memory
  cost, KV-cache representation, and a systems-facing argument around serving
  cheaper long-context workloads.
- Borrow:
  - Clear abstract structure: cost problem, proposed representation, why the
    naive baseline fails, main quantitative win, implication.
  - Strong first-page framing that makes memory and serving cost central.
  - A confident but bounded tone: the paper sells a new runtime object without
    pretending it solves every long-context workload.
- Avoid:
  - Do not imply IdleKV has the same breadth of benchmarks or training-time
    evidence. Our current paper is more controlled and diagnostic.

### Batch-Max

- OpenReview: https://openreview.net/forum?id=j7OED598Kz
- PDF: https://openreview.net/pdf?id=j7OED598Kz
- Venue/status: ICML 2025 ES-FoMo III.
- Why it matters: very close AdaptFM-style framing around KV cache compression,
  batch size, throughput, and resource-constrained inference.
- Borrow:
  - Systems motivation that ties KV memory directly to practical runtime
    throughput.
  - A compact story around "same model quality, better resource use."
  - Figure/table style that emphasizes operational tradeoffs rather than only
    accuracy.
- Avoid:
  - Do not overclaim throughput. IdleKV's exact scorer is a diagnostic
    mechanism unless proxy or systems runs support deployment claims.

### Resource-Efficient Inference With Foundation Model Programs

- OpenReview: https://openreview.net/forum?id=MqJfoI6Yto
- PDF: https://openreview.net/pdf?id=MqJfoI6Yto
- Venue/status: ICML 2025 ES-FoMo III spotlight.
- Why it matters: good example of an agentic or compound-AI framing within an
  efficiency workshop.
- Borrow:
  - Framing around input-adaptive resource allocation.
  - Explicit cost-performance language.
  - Willingness to introduce a framework while still giving measured evidence.
- Avoid:
  - IdleKV should not sound like it has solved full agentic workflows. Say it
    studies a controlled split-query instance of a broader agent-state problem.

### Predictive Scheduling for Efficient Inference-Time Reasoning

- OpenReview: https://openreview.net/forum?id=Mn3lrAWy20
- PDF: https://openreview.net/pdf?id=Mn3lrAWy20
- Venue/status: ICML 2025 ES-FoMo III.
- Why it matters: similar adaptive-inference thesis, but for token budget
  allocation rather than KV state allocation.
- Borrow:
  - "Fixed budgets waste resources" framing.
  - Matched-budget evaluation language.
  - A concise abstract that names the adaptation signal, budgeted decision, and
    accuracy-resource result.
- Avoid:
  - Do not use test-time scaling terminology unless the text makes clear that
    our adaptation is active KV-state selection, not extra reasoning traces.

### PiKV

- OpenReview: https://openreview.net/forum?id=hHoK1kBPd9
- PDF: https://openreview.net/pdf?id=hHoK1kBPd9
- Venue/status: ICML 2025 ES-FoMo III.
- Why it matters: close style reference for a systems-facing KV cache paper
  with hardware/distributed implications.
- Borrow:
  - A system diagram style where KV storage, routing, scheduling, and
    compression are visible as runtime components.
  - Language connecting KV management to multi-GPU memory and communication
    bottlenecks.
- Avoid:
  - Do not imply IdleKV implements distributed serving or expert sharding.
    Mention multi-tenant/distributed generalization only as an implication or
    future runtime extension.

### Cache Saver

- OpenReview: https://openreview.net/forum?id=Ve2r5Bap1Q
- PDF: https://openreview.net/pdf?id=Ve2r5Bap1Q
- Venue/status: ICML 2025 ES-FoMo III.
- Why it matters: useful for non-intrusive, modular inference-system framing.
- Borrow:
  - "Plug-in runtime layer" language when describing IdleKV as a cache
    maintenance primitive.
  - Cost/reproducibility/benchmarking language in limitations and discussion.
- Avoid:
  - Do not present IdleKV as application-transparent production middleware.

## Secondary References

### CO2

- OpenReview: https://openreview.net/forum?id=02zPmtcZa0
- PDF: https://openreview.net/pdf?id=02zPmtcZa0
- Venue/status: ICML 2024 ES-FoMo-II poster.
- Use for: KV replacement framing, attention-score observation, and the
  difference between generation-time cache replacement and IdleKV's
  post-compression, pre-resume repair.

### SPECS

- OpenReview: https://openreview.net/forum?id=wRRtifTM5b
- PDF: https://openreview.net/pdf?id=wRRtifTM5b
- Venue/status: ICML 2025 ES-FoMo III.
- Use for: latency-aware test-time compute writing. This is not a KV-cache
  paper, but it is a useful style example for balancing accuracy gains against
  latency constraints.

### LATTICE

- OpenReview: https://openreview.net/forum?id=ijk4DRumSW
- PDF: https://openreview.net/pdf?id=ijk4DRumSW
- Venue/status: ICML 2025 ES-FoMo III.
- Use for: memory compression language and how to write a mechanism paper that
  introduces a different view of sequence memory.

### MMLongBench

- OpenReview: https://openreview.net/forum?id=zsdJSkeS9S
- PDF: https://openreview.net/pdf?id=zsdJSkeS9S
- Venue/status: ICML 2025 LCFM.
- Use for: benchmark-paper style, length-control language, and cautious claims
  around long-context evaluation coverage.

## Existing KV-Cache References Already In The Paper Guide

These are still important for technical positioning even if they are not ICML
workshop papers.

- H2O: https://papers.nips.cc/paper_files/paper/2023/file/6ceefa7b15572587b78ecfcebb2827f8-Paper-Conference.pdf
- Scissorhands: https://proceedings.neurips.cc/paper_files/paper/2023/file/a452a7c6c463e4ae8fbdc614c6e983e6-Paper-Conference.pdf
- StreamingLLM: https://arxiv.org/abs/2309.17453
- SnapKV: https://papers.nips.cc/paper_files/paper/2024/file/28ab418242603e0f7323e54185d19bde-Paper-Conference.pdf
- KIVI: https://proceedings.mlr.press/v235/liu24bz.html
- QUEST: https://arxiv.org/abs/2406.10774
- SCBench: https://proceedings.iclr.cc/paper_files/paper/2025/hash/a540b17fb2295c736d5afd6c507acf66-Abstract-Conference.html

Use these for related-work precision. Use the ES-FoMo/LCFM papers above for
workshop style and paper-shape decisions.

## Abstract Style Takeaways

Target: one paragraph, roughly 4-6 sentences.

Recommended structure:

1. State the deployment/resource pressure.
2. Name the missing decision: an already-compressed paused cache may need to be
   repaired after a later relevance signal appears.
3. Define IdleKV in one sentence: buffer evicted KV, score after the next-turn
   signal, promote under a matched active-cache budget, resume.
4. Give one quantitative headline result, not every result.
5. Close with scope and implication: controlled diagnostics support idle-window
   cache maintenance as a resource-adaptive inference primitive for future
   local, dedicated, and resource-constrained long-context agent runtimes.

Avoid:

- Listing every phase, figure, and ablation in the abstract.
- Claiming end-to-end agent performance.
- Saying "deployable" unless Phase 16 or later provides stronger latency and
  systems evidence.
- Using "adaptation" in a way that sounds like fine-tuning or PEFT.

## Figure Style Takeaways

Top adjacent workshop papers usually use figures to establish one of three
things quickly:

- A system object or runtime path.
- A quality-resource frontier.
- A benchmark or protocol that explains why the experiment is fair.

For IdleKV, the main figures should therefore prioritize:

- Figure 1: runtime/data-path abstraction for paused, tiered KV state.
- Figure 2: main matched-budget accuracy-resource result.
- Specificity controls: evidence that the gain is not generic buffering.
- Repeated relevance shifts: evidence that this is an agent-workflow problem,
  not only a single query.
- Phase 15 real-repo diagnostic: preliminary external-validity evidence, kept
  cautious.
- Runtime/proxy: enough to show there is a latency path, not enough to claim
  full serving readiness.

Figure layout rules for the final pass:

- Prefer dense single-column figures when each figure has one job.
- Use two-column figures only when a full-width comparison genuinely reduces
  cognitive load.
- Keep captions short: what was measured, what the reader should conclude, and
  one scope caveat if needed.
- Avoid low-density paired panels. If a panel is mostly whitespace, merge it
  into a table, combine it with another metric, or move it to the appendix.
- The first system/framework figure should appear early enough that a reviewer
  knows the object before reading results.

## Main-Text Framing Rules

Use this language:

- "resource-adaptive inference-time state management"
- "idle-window cache maintenance"
- "post-compression, pre-resume repair"
- "matched active-cache budget"
- "tiered KV runtime primitive"
- "controlled diagnostic evidence"
- "preliminary external-validity evidence"
- "interactive single-session or dedicated-tenant agent inference"

Avoid this language unless carefully qualified:

- "agent benchmark"
- "coding-agent evaluation"
- "production-ready serving system"
- "real-world code repair"
- "general KV compression method"
- "test-time adaptation" without clarifying the adaptation target is runtime
  cache state.
- "batch-1" as the main motivation; use it only as an implementation detail.
  For the paper narrative, prefer "interactive single-session" when the point
  is a user or agent program owning the latency path, and "dedicated-tenant"
  when the point is a reserved worker or isolated deployment.

## Appendix Style Takeaways

The appendix can be longer and figure-heavy, but reviewers are not required to
read it. It should behave like an audit trail:

- Put every claim needed for acceptance in the main paper.
- Use the appendix for exact hyperparameters, additional seeds, scorer audits,
  retention-rule sensitivity, prompt/instance examples, and layout-heavy
  diagnostics.
- Reference appendix figures from the main text only where they strengthen a
  claim already made in the main text.
- Do not promote appendix figures solely to fill space. Promote them only if
  they answer a likely reviewer objection.

Likely promotion candidates:

- Anything showing the method survives stronger baselines or additional models.
- A compact table/figure for Phase 16 if it adds Scissorhands, Mistral, or
  compute/time-matched reselection evidence.
- A real-repository diagnostic only if the caption and text are explicit that
  it is preliminary and custom.

Likely appendix-only material:

- Low-density sensitivity plots.
- Exact scorer implementation audits.
- Rows that require substantial caveating to interpret.
- Counterevidence that is important for honesty but would distract from the
  main paper unless resolved by a cleaner rerun.

## Practical Editing Checklist

Before submission:

- Abstract is 4-6 sentences and contains only one dense numeric sentence.
- Introduction makes the missing runtime decision obvious by the end of page 1.
- Figure 1 appears before the reader has to interpret the experimental protocol.
- Results are written as matched-budget evidence, not as raw accuracy wins.
- Phase 15 is described as preliminary external-validity evidence.
- Runtime/proxy text says "possible latency path" rather than deployment proof.
- Limitations explicitly mention synthetic MQ-NIAH, primary model/context
  scope, exact scorer cost, and small custom real-repo diagnostic.
- Appendix figures have no obvious unearned whitespace, low-density panels, or
  orphaned captions.
- All claims that reviewers must believe are in the main six pages, not only in
  the appendix.
