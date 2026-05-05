# Phase 17: Latency Reframe, Paper Edits Only

## Scope

Phase 17 is an editing pass only.

Do not run new experiments.
Do not launch GPU jobs.
Do not add new latency claims that require fresh measurements.
Use only existing evidence, and avoid making the paper look like RepairKV
depends on the current CPU exact-scorer implementation.

## Problem to fix

The current latency paragraph overemphasizes a 6--7 second exact scorer. That
number comes from an unoptimized research diagnostic that scores active plus
evicted KV on CPU/float32. It should not be presented as RepairKV runtime.

The intended systems story is:

> Exact scoring is used to validate the repair mechanism. A deployable repair
> path should use GPU-side or otherwise accelerator-side scoring during
> non-decoding slack, with warm-tier KV movement and chunked selection treated
> as systems work.

## Required paper edits

1. Rewrite the main latency paragraph.
   - Remove the main-text sentence that headlines exact scorer p50
     `6.08--7.62 s`.
   - Say exact Q2 scoring is a mechanism diagnostic used for the controlled
     quality curves.
   - Say proxy scoring shows a cheaper Q2-conditioned scoring path can preserve
     the repair effect on the benchmark.
   - Say production use would need a GPU-side/paged selector and scheduling
     integration.

2. Keep Figure 6 only as a capacity envelope.
   - Do not imply it is end-to-end RepairKV latency.
   - Caption should say generation and diagnostic exact scoring are excluded.
   - Text should say it measures a GPU-side chunked selector over pinned
     host-memory candidate keys.

3. Move or soften exact-scorer timing.
   - Preferred: omit the 6--7 second number entirely from the main paper.
   - If retained anywhere, put it in appendix/protocol as unoptimized CPU
     diagnostic timing, not a method runtime.

4. Update appendix wording.
   - Define exact scorer as diagnostic/mechanistic.
   - Define proxy as a cheaper benchmarked scoring variant.
   - Add one sentence that neither is a fused production serving kernel.

5. Update limitations.
   - Say the paper establishes the repair effect and matched-budget protocol.
   - Say a deployable selector remains future systems work.
   - Mention GPU-side chunking, paging, approximate filtering, and scheduler
     overlap as natural next steps.

## Text direction

Good wording:

> The exact Q2 scorer is a diagnostic used to establish the repair effect, not
> the proposed runtime implementation.

> The proxy scorer provides benchmark evidence that cheaper Q2-conditioned
> scoring can preserve most of the repair gain.

> A production RepairKV system should score warm-tier candidates with GPU-side
> chunking or paged selection during non-decoding intervals.

Avoid:

> RepairKV takes 6--7 seconds.

> CPU scoring is acceptable because the model is idle.

> The proxy is production-ready.

> Figure 6 proves deployment latency.

## Main-paper target

The main paper should say only:

- exact scorer: controlled diagnostic;
- proxy scorer: cheaper benchmark evidence;
- runtime figure: GPU capacity envelope;
- deployment: future systems work.

## Appendix target

The appendix can be more explicit:

- Exact Q2 scorer extracts post-RoPE query projections and scores active plus
  evicted keys.
- The current implementation is unfused diagnostic code.
- Proxy appends Q2 to the compressed cache and uses the appended Q2 cache rows
  as scoring rows.
- These measurements support the existence of cheaper scoring paths but do not
  constitute a production serving implementation.

## Acceptance criteria

Phase 17 is done when:

- The main paper no longer foregrounds the 6--7 second CPU scorer.
- No sentence implies RepairKV's intended implementation scores KV on CPU.
- Figure 6 and the proxy result are framed as feasibility evidence, not
  deployment proof.
- Limitations are honest without underselling the contribution.

