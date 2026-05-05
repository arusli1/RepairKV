# Figure 1 Hardware-Aware System Diagram Spec

Goal: replace the current small TikZ pipeline in `paper/main.tex` with a
polished one-column vector figure that makes IdleKV feel like a systems
primitive, not only an experiment protocol. Do not add a new main figure; replace
the existing Figure 1 in the Method section so the main-paper page budget stays
stable.

## Why This Figure Is Worth Doing

Adjacent KV-cache and efficient-inference papers usually use an early figure to
define the system category:

- SCBench Figure 1 defines the KV-cache lifecycle and names the missing stages.
- QUEST Figure 1 contrasts dense attention, query-agnostic sparsity, and
  query-aware sparsity.
- InferCept uses diagrams to explain interrupted generation and context
  handling across GPU/CPU memory.
- MInference uses a first-page figure to make long-context cost and sparse
  attention visually concrete.

IdleKV's analogous message is: compression chooses active GPU KV before the
future-turn signal exists; agent idle time opens a new place in the KV lifecycle
where a runtime can adapt active state by scoring a warm tier and promoting
selected rows.

## Recommended Figure

Title inside artwork, if any: **Idle-window KV state adaptation**.

Use a horizontal timeline with three vertical layers:

1. **Agent timeline** at the top
   - `Turn N: prefill + Q1`
   - `tool / user / test / file event`
   - `Turn N+1: resume decode`
   - Show the idle gap as a highlighted band between turns.

2. **GPU active KV** in the middle
   - Before compression: large active context.
   - After compression: `B_base` active rows.
   - After repair: `B_base + K` active rows, labeled "matched resumed budget".
   - Include a faint gray no-repair path that resumes with stale active KV under
     the same budget, to make the matched-budget claim visual.

3. **Host-memory warm tier** at the bottom
   - Evicted KV rows flow from GPU active KV to host memory after Q1.
   - During the idle window, next-turn signal plus warm-tier metadata/feed
     enters a `score / select / promote` box.
   - Selected `K` rows flow back to GPU active KV before decoding resumes.
   - Optional faint lower box: "cold tier / recompute / summaries", marked as
     future systems work, not evaluated. Keep this subtle or omit if it clutters
     the figure.

Visual grammar:

- Use GPU blue for active device KV.
- Use warm orange/gold for host-memory evicted KV.
- Use gray for no-repair/preserve-only baseline.
- Use green or teal only for the newly available next-turn signal.
- Keep arrows thick and directional; avoid many tiny labels.
- Use row/bar glyphs rather than literal token text. A few highlighted rows
  should move from host tier to GPU active cache.
- Keep it one-column width. If handmade in Figma/Keynote/diagrams.net, export as
  PDF/SVG and include with `\includegraphics[width=\columnwidth]{...}`.

## Exact Labels To Use

Primary labels:

- `active GPU KV`
- `host-memory warm tier`
- `idle window`
- `next-turn signal`
- `score / select`
- `promote selected KV units (budget K)`
- `matched resumed active budget`

Secondary labels, only if there is space:

- `compress to B_base`
- `no-repair: stale active state`
- `optional cold tier: compressed / recomputable / remote`

Avoid these labels:

- `oracle`
- `real-world validation`
- `SWE-bench`
- `production deployment`
- internal phase/run names

## What The Figure Should Not Pretend To Solve

The polished figure should be hardware-aware but still honest about scope. Modern
serving stacks usually manage KV through blocks/pages, block tables, batched
movement, and scheduler-owned memory budgets rather than isolated token rows. Use
row glyphs only as an intuition; label the movement as selected KV units/pages
where possible.

Do not imply that IdleKV already solves:

- block/page packing and fragmentation under PagedAttention-style allocators;
- continuous-batching scheduling or contention among many paused requests;
- async DMA overlap, prefetch timing, pinned-memory pressure, or PCIe/NVLink
  bandwidth scheduling;
- tensor/context parallel placement of K/V shards across devices;
- quantized or compressed warm-tier representations and dequantization cost;
- metadata indexing at production scale;
- security, isolation, or cache-lifetime policy across users/requests.

Those are future systems integration questions. The main claim is narrower:
turn-conditioned selection can improve which cached units should be active when
a paused request resumes.

## Caption Draft

`\idlekv{} treats the pause between turns as an active-state adaptation window.
After turn $N$, a base cache policy leaves $B_{\mathrm{base}}$ rows active on
GPU and retains evicted KV in a host-memory warm tier. When the next-turn signal
arrives, the runtime scores the warm tier, promotes selected KV units under
budget $K$, and resumes under the same active-cache budget used by matched
no-repair.`

If the caption needs to be shorter:

`\idlekv{} uses the idle gap between turns to score a host-memory warm tier and
promote $K$ evicted KV rows back into active GPU memory before resuming under a
matched active-cache budget.`

## Placement

Best placement is the existing Figure 1 slot in the Method section, immediately
after the data-path paragraph. It already has the right surrounding prose:
offload after Q1, score after Q2 signal, promote before Q2 decode, and no
full-prefix recompute.

Do not add this as Figure 7 or an appendix-only figure. The value of the diagram
is front-loaded orientation. If it cannot be made polished enough, keep the
current simple TikZ rather than adding a rough placeholder.

## Minimal Placeholder Layout

For a draft asset, make a one-column canvas with this layout:

```text
Agent timeline:
 [Turn N: prefill + Q1] -- [idle window: tool/user/test/file event] -- [Turn N+1: resume decode]
                                  | next-turn signal

GPU active KV:
 [large active KV] -> [B_base active rows]  ---- no-repair stale path ----> [B_base + K matched budget]
                                      ^                                  ^
                                      | promote K rows                   |

Host-memory warm tier:
        [evicted KV rows retained after Q1] -- score/select during idle --+

Optional subtle future layer:
        [cold tier: compressed / recomputable / remote]  (future systems work)
```

This placeholder is a design guide only. Do not submit a text-box placeholder in
the final PDF.
