# KV Cache Repair via GPU Idle Time: Research Development Plan

## Overview

**Motivation:** GPU idle time during tool calls in batch-1 edge-compute LLM agentic workloads is entirely unused. This plan develops a system that uses this idle time to repair the KV cache — restoring evicted tokens that have become newly relevant — thereby improving agent response quality without adding latency.

**Problem:** Token eviction algorithms (SnapKV, StreamingLLM, H2O) compress the KV cache by dropping tokens deemed unimportant by attention scores at eviction time. As shown in *The Pitfalls of KV Cache Compression* (arXiv:2510.00231), this causes important details to be forgotten when context needs shift mid-conversation. Agentic tool calls (script executions, MCP calls, pytest runs: often 2–60 seconds) create idle windows where recomputation is free in wall-clock terms.

**Proposed solution:** During the tool call idle window, recompute attention scores between the most recent query vectors and a CPU-resident buffer of evicted token KV pairs. Promote the highest-scoring evicted tokens back to the active GPU KV cache before the next LLM turn begins.

**Target venue:** 3–4 page workshop paper (e.g. NeurIPS Efficient Reasoning, ICLR Sparsity in LLMs, or similar). The bar for accepted workshop papers in this space is: one novel mechanism, one clean results table across 2–3 conditions on 1–2 models, one figure showing the budget tradeoff. Degradation from KV eviction on reasoning-heavy tasks is already established in the published literature (NeurIPS 2025: "token eviction methods struggle with tasks that rely heavily on in-context learning, passkey retrieval, and long-context reasoning because they tend to eject critical tokens") — cite this and focus experimental effort on the repair contribution.

**Evaluation framework:** Later repair phases use a matched-footprint RULER-KVR protocol:
- **Condition A** — monolithic pass (standard RULER, single forward pass, no eviction)
- **Condition B_onset** — compressed-cache baseline at base keep budget `B_base`; used in P1 to find the onset-of-regression regime
- **Condition B_match** — no-repair matched-footprint baseline at final active budget `B_match = B_base + K_slots`
- **Condition C** — IdleKV repair: prefill retains `B_base`, idle-time repair injects `K_slots` buffered tokens, and the resumed cache footprint is also `B_match`
- **Matched baselines** — `Random-K` and `Recency-K`, both filling the same `K_slots` and ending at the same `B_match` footprint

Primary later-phase metrics are **matched-footprint recovery** = `(C − B_match) / (A − B_match)` and absolute **selection lift** = `C − B_match`. This makes the core claim "repair picks better tokens at the same GPU footprint," not "repair gets more tokens."

---

## Research Arc

```
P0          P1              P2            P3              P4              P5              P6            P7
Baseline → Eviction      → KV access  → Eviction     → CPU eviction → [ORACLE     → Repair      → Appendix
RULER       degradation    layer         implementation  buffer          GATE]         algorithm     proof-of-life
            measurement    (infra)       + validation    + profiling     go/no-go      + ablation    (optional)
```

**Blocking dependencies:**
- P1 requires P0 (needs baseline scores to measure degradation against)
- P3 requires P2 (needs KV access API to implement eviction correctly)
- P4 requires P3 (needs eviction logs to know what to store in buffer)
- P5 requires P2 + P3 + P4 (needs the full stack to run oracle)
- P6 requires P5 passing (oracle must show recoverable signal before repair is worth building)
- P7 is optional after P6 (not a main-paper gate)

P2 can be developed in parallel with P1.

---

## Phase 0: Baseline RULER Validation

**Goal:** Reproduce published RULER scores with Qwen-7B before touching anything else.

Every later phase needs a trustworthy score to compare against. This is the only truly blocking gate — nothing proceeds until scores match within ~2% of reported baselines.

### Model Setup

- Install `Qwen/Qwen2.5-7B-Instruct` via HuggingFace
- Verify tokenizer output matches expected sequence lengths
- Confirm 32K context window is accessible (check `rope_scaling` config)
- Set up `vllm` serving locally — this will be the inference backend for all later phases
- GPU memory requirement: ~28GB at fp16 for Qwen-7B at 32K context

### RULER Setup

- Clone `hsiehjackson/RULER`, read generation scripts
- Run S-NIAH at 4K, 8K, 16K with 500 examples each
- Run VT (2-hop) and FWE at the same lengths
- Use exact chat template for Qwen-2.5-Instruct — wrong template is the most common source of score divergence
- Do not modify RULER in any way at this stage

### Steps

1. Install dependencies. Verify GPU memory.
2. Run RULER data generation for S-NIAH at 4K. Inspect 5 examples manually — verify needle is present, haystack fills correctly, query is sensible.
3. Run inference and score. If scores diverge by more than 2% from reported Qwen-2.5-7B numbers, debug: check chat template, system prompt, `max_tokens`, temperature (must be 0).
4. Repeat for 8K, 16K, 32K. Plot the degradation curve. This is your Figure 1 baseline.
5. Run VT-2hop and FWE at the same lengths. Save all raw model outputs, not just accuracy scores.

### Acceptance Criteria

| Task | Length | Minimum score |
|------|--------|---------------|
| S-NIAH | 4K | ≥ 98% |
| S-NIAH | 32K | ≥ 85% |
| VT-2hop | 16K | Within ±2% of paper |
| FWE | 16K | Within ±2% of paper |

### Deliverable

`results/baseline_ruler.json` — scores across all tasks × lengths. Git-tagged as `v0-baseline`. Never modified — all future experiments measure delta against this file.

---

## Phase 1: RULER-KVR — Eviction Degradation Measurement

**Goal:** Establish a trustworthy condition B baseline for the later repair experiment by measuring where SnapKV first causes meaningful degradation on the reduced 32K harness. Phase 1 is not about maximizing collapse; it is about locating the onset-of-regression regime where repair vs. no-repair is scientifically meaningful.

**Go/no-go question:** At realistic retained-cache budgets, does live SnapKV prefill degrade VT-4hop and MQ-NIAH-4q in a way that is both measurable and attributable to dropped middle-context spans?

---

### What Conditions A and B Actually Are

**Condition A — monolithic pass:** full context plus query in one pass, no eviction. This is the ceiling and comes from P0.

**Condition B — eviction during prefill, answer from the compressed cache:** SnapKV runs live while the 32K context is prefed, so by the time the query is asked only `k_budget` token positions remain in the cache. The query is then answered directly from that compressed cache.

Eviction happens during prefill, not after it. Do not delete the cache and reprefill from scratch. Reprefill measures a different system.

The key variable is `k_budget`. Gap duration between prefill and query is irrelevant in condition B and only matters later in condition C, where it determines repair time budget.

---

### Reduced-Harness Design

Use the reduced 32K harness only:

- **Context length:** 32K
- **Eviction algorithm:** SnapKV only
- **Main `k_budget` sweep:** `16384`, `8192`, `4096`
- **Examples per main cell:** `100`
- **Main tasks:** `VT-4hop`, `MQ-NIAH-4q`

These budgets are chosen to find the onset of damage, not the point where the cache is obviously unusable:

- `k=16384` is the sanity budget and should remain close to condition A.
- `k=8192` is the upper onset budget.
- `k=4096` is the lower onset budget and should show clear degradation without collapsing into a trivial failure regime.

`k=2048` and below are stress settings only. They are useful only after the higher-budget compressed path already behaves sensibly.

Before any full run, perform a short pilot on a few examples per task. Continue only if:

- condition A is correct on the spot-checked examples
- `k=16384` is near condition A and clearly non-pathological
- `k=8192` and/or `k=4096` show degradation relative to `k=16384`
- the detailed logs show nonzero survival for the tail-pinned control spans

If the pilot is inverted or near-all-zero even at `k=16384`, stop and debug the compressed-cache continuation path before doing a full sweep.

---

### Task Selection and Span Placement

The reduced harness uses fixed placements, not random depths, so attribution is stable across reruns.

**VT-4hop — primary result**

- hop 1 at `12%`
- hop 2 at `37%`
- hop 3 at `62%`
- hop 4 pinned into SnapKV's actual protected recent tail
- terminal value pinned into the protected recent tail after hop 4

This is the cleanest attribution task because it has single-point-of-failure structure: once an earlier hop disappears, the chain breaks completely.

**MQ-NIAH-4q — secondary result**

- needle 1 at `10%`
- needle 2 at `37%`
- needle 3 at `63%`
- needle 4 pinned into the protected recent tail

This gives graded recall rather than binary success and is useful for seeing whether degradation begins gradually.

**S-NIAH — optional spot check only**

Do not use `k=256` as the default sanity gate for this reduced harness. If you want a single-needle spot check, use `k=16384` as the non-pathological control and optionally compare to `k=4096`.

**FWE, CWE, and cross-turn MK-NIAH — defer to later phases**

These matter as later controls, not as the core P1 result.

---

### Logging and Attribution

For every compressed run, log:

- full keep/evict mask per layer/head
- per-token importance scores at eviction time
- Q vectors for the last `M` tokens at eviction conclusion
- raw model output
- task-relevant span metadata and positions
- task-relevant survival using the actual recorded `survival_fraction`, not only a boolean survived/dead flag

In the reduced harness, late control spans are physically inserted into the protected recent tail, and the summary's `eviction_survival_rate` must be computed from the underlying span-level `survival_fraction` values.

For VT failures, record the first broken hop and its depth. In the corrected setup, breaks should cluster on the middle hops rather than on the tail-pinned control spans.

### Deliverable

`results/phase1_condition_b.json` and `results/phase1_summary.json` with degradation curves for `VT-4hop` and `MQ-NIAH-4q` at 32K across `k_budget ∈ {4096, 8192, 16384}`.

**Go/no-go signal:** `k=16384` should remain close to condition A, while `k=4096` should show a clear attributable drop. If all budgets behave almost identically, or if the highest budget already collapses, the compressed-cache path is not yet trustworthy enough for repair work.

---

## Phase 2: KV Cache Access Layer

**Goal:** Build the infrastructure all later phases depend on — serialize, inspect, and modify KV tensors with a clean, tested API.

> **Critical test — round-trip identity:** Save KV cache after context prefill, reload it, continue generation. Output must be bit-for-bit identical to a run that never evicted. If this fails, nothing downstream can be trusted. This is the P2 acceptance gate.

### KV Cache Anatomy (Qwen-7B)

`past_key_values` is a Python tuple of length `n_layers` (32 for Qwen-7B). Each element is a `(key, value)` tuple where each tensor has shape `[batch=1, n_kv_heads, seq_len, head_dim]`.

Qwen-7B with GQA: 32 layers, 32 KV heads, head_dim=128.
Total KV size at 32K tokens: `32 × 2 × 32K × 128 × 2 bytes ≈ 16GB at fp16` — must be CPU-offloaded for full-cache experiments.

### The Five-Function API

Implement in `src/kv_utils.py`. Everything from P3 onward calls this interface:

```python
def save_kv(past_key_values, path: str) -> None:
    """Serialize KV cache to disk or CPU pinned memory."""

def load_kv(path: str) -> tuple:
    """Restore KV cache from disk to GPU."""

def slice_kv(past_key_values, token_indices: list[int]) -> tuple:
    """Extract a subset of token positions."""

def merge_kv(cache_a: tuple, cache_b: tuple) -> tuple:
    """Concatenate two KV caches along the sequence dimension."""

def inject_kv(past_key_values, new_pairs: tuple, positions: list[int]) -> tuple:
    """Insert repaired KV pairs at their original sequence positions."""
```

### Required Unit Tests

**Test 1 — Round-trip identity:** Run to position P → `save_kv` → `load_kv` → continue. Next-token logits must match a run with no save/load. Tolerance: fp16 rounding only (max absolute difference < 1e-3).

**Test 2 — Selective injection:** Remove token at position P via `slice_kv`. Generate answer. Inject back via `inject_kv`. Generate again. Second output must recover. Verifies inject is semantically correct.

**Test 3 — CPU → GPU transfer latency:** Profile time to move a 32K KV cache from CPU pinned memory to GPU. At PCIe 4.0 x16 (~32 GB/s): ~500ms. This sets the repair time budget ceiling.

### Attention Score Visualization

Use `output_attentions=True` in `model.generate()`. Visualize attention weight heatmaps: x = query position, y = key position, color = weight. Look for: attention sinks (positions 0–4 always high), recency bias (last ~1K tokens always high), and which positions genuinely attend to needle/hop-link tokens. Run at 4K, 16K, 32K. These go into the paper's motivation figures.

### Deliverable

`src/kv_utils.py` with full unit test suite. Attention heatmap visualizations at multiple context lengths.

---

## Phase 3: Eviction Algorithm Implementation and Validation

**Goal:** Implement SnapKV, StreamingLLM, and H2O under a unified interface; validate by reproducing known failure modes from the Pitfalls paper.

### Unified Interface

```python
class EvictionPolicy:
    def select_tokens_to_keep(
        self,
        past_key_values: tuple,
        attention_scores: torch.Tensor,
        budget: int
    ) -> list[int]:
        """Returns indices of tokens to keep in the active KV cache."""
```

### SnapKV

Pool attention over the observation window (last ~32 tokens) to identify which prior tokens receive the most attention. Always keep: attention sinks (positions 0–4), recency window (last ~1K), top-K by importance score.

```python
importance = mean(attn_weights[:, :, obs_window_tokens, :], dim=[0, 1, 2])
```

Hyperparameters: `k_budget`, `obs_window_size` (default 32), `sink_size` (default 4), `recency_window` (default 1024).

### StreamingLLM

Keep only attention sinks + sliding recency window. Simple but discards everything in the middle.

### H2O (Heavy Hitter Oracle)

Track cumulative attention scores per token across all generated tokens. Evict those with lowest cumulative score. More accurate than SnapKV's one-shot observation but requires running statistics.

### Validation Against the Pitfalls Paper

Reproduce at least two failure modes:

**Failure mode 1 — Mid-document information loss:** Place needle at depth 40–60% in a 32K document. Apply SnapKV with k_budget=512. Needle should be evicted and model returns wrong answer ~80%+ of the time.

**Failure mode 2 — Multi-hop chain breakage:** Run VT-3hop with SnapKV at 16K. Intermediate chain variable should be evicted before it is needed.

If you cannot reproduce these failures, the eviction implementation is incorrect.

### Logging Requirements

For every eviction run, log to disk: which positions were kept vs. evicted (per layer, per head), importance scores per token, and Q vectors for the last M tokens at eviction time. This log populates the CPU buffer in P4.

### RULER Under Eviction

Run RULER-KVR condition B with real SnapKV. Compare to P1 numbers. Key question: does real SnapKV produce more or less degradation than the MVP P1 measurements? This comparison matters for the paper.

### Deliverable

`src/eviction/` — SnapKV, StreamingLLM, H2O under unified interface. RULER scores at the onset budgets selected in P1 (default reduced-harness sweep: `k_budget ∈ {4096, 8192, 16384}`). Pitfalls failure mode reproductions as documented test cases.

---

## Phase 4: CPU Eviction Buffer — Storage and Retrieval Profiling

**Goal:** Store evicted KV pairs on CPU; profile all transfer and recompute operations to establish feasibility within real tool call idle windows.

> **Feasibility gate:** Profile this exhaustively before implementing any repair logic. If moving 1000 evicted tokens takes > 500ms and recomputing attention takes another 500ms, you are already consuming a 2-second tool call budget with nothing left.

### EvictionBuffer Design

```python
class EvictionBuffer:
    def __init__(self, max_tokens: int, selection_strategy: str):
        self.buffer: dict[int, tuple] = {}         # token_pos → (K, V) per layer, on CPU
        self.scores: dict[int, float] = {}         # importance score at eviction time
        self.q_vecs: dict[int, torch.Tensor] = {}  # query vectors at eviction time

    def push(self, pos: int, kv_pair: tuple, score: float, q_vec: torch.Tensor): ...
    def query(self, recent_q_vectors: torch.Tensor, top_k: int) -> list[tuple[int, tuple]]: ...
    def to_gpu(self, positions: list[int]) -> tuple: ...
```

**Memory per evicted token (Qwen-7B, fp16):**
`32 layers × 2 × 1 token × 32 heads × 128 dims × 2 bytes = 524KB`

Storing 1000 evicted tokens ≈ 524MB CPU RAM. Feasible.

### Selection Strategies

**Strategy A — L2 norm of Q vectors (primary proposal):** At eviction time, store tokens whose query vectors had high L2 norm. High norm ≈ "was strongly queried." Score = `‖q_vec‖₂`.

**Strategy B — Original attention score:** Store tokens with highest attention scores at eviction time. Simpler but stale.

**Strategy C — Random sample (ablation baseline):** Uniform random sample of evicted tokens. Isolates whether selection quality matters at all.

**Strategy D — Recency-inverse:** Store oldest evicted tokens preferentially. Targets attention sink victims — early-context tokens most likely to be in the eviction dead zone.

All four strategies implement the same `push` / `query` interface.

### Latency Profiling

Measure wall-clock time for each operation. Run 100 trials, report p50/p90/p99. Use `torch.cuda.synchronize()` before timing GPU operations.

| Operation | Parameter range | Notes |
|-----------|-----------------|-------|
| CPU → GPU KV transfer | N ∈ {100, 500, 1000, 2000} tokens | Use pinned memory |
| GPU attention recompute | N tokens × M query tokens, M ∈ {64, 128, 512} | Time per layer; can parallelize |
| Buffer dot-product search | N ∈ {1000, 5000, 10000} | CPU or GPU index? |

**Feasibility frontier:** For each tool call duration T, what is the maximum K (tokens repaired)? This table appears directly in the paper's system design section. For a 2-second tool call, budget is ~1.8 seconds total. For a 15-second pytest call, budget is ~14.8 seconds — substantially more headroom.

### Deliverable

`src/eviction_buffer.py` with all four selection strategies. Profiling table: operation × N_tokens → latency (p50/p90). Feasibility frontier: maximum repairable K vs. tool call duration T.

---

## Phase 5: Oracle Experiment — Recoverability and Fixed-Footprint Headroom

**Goal:** Answer two distinct questions before building the practical repair algorithm:

1. Is eviction damage recoverable at all if memory were free?
2. At the same final GPU footprint, is there headroom beyond what the base eviction policy gets by simply keeping `K_slots` more tokens?

> Do not proceed to P6 until both are measured. The first validates the repair hypothesis; the second removes the matched-budget confound.

### Shared notation

- `B_base`: onset keep budget selected in P1
- `K_slots`: repair slots feasible under the P4 latency frontier
- `B_match = B_base + K_slots`: final active-cache footprint used for all matched comparisons

### Oracle 1 — Full restore

Restore all evicted tokens during the idle window:

```python
all_evicted_positions = list(eviction_buffer.buffer.keys())
all_evicted_kv = eviction_buffer.to_gpu(all_evicted_positions)
full_oracle_cache = inject_kv(active_cache, all_evicted_kv, all_evicted_positions)
outputs = model(query_tokens, past_key_values=full_oracle_cache)
```

This still ignores compute and memory budget. It answers: "is the lost content in principle sufficient to recover the answer?"

### Oracle 2 — Matched-footprint Oracle-K

Restore only `K_slots` evicted tokens, but choose them with task hindsight so the final cache size is exactly `B_match`:

```python
oracle_positions = select_hindsight_relevant_positions(
    eviction_log=eviction_log,
    task_metadata=task_metadata,
    top_k=K_slots,
)
oracle_kv = eviction_buffer.to_gpu(oracle_positions)
oracle_k_cache = inject_kv(active_cache, oracle_kv, oracle_positions)
outputs = model(query_tokens, past_key_values=oracle_k_cache)
```

For RULER-KVR tasks, the generator metadata already tells you which spans are task-relevant. If more relevant spans are missing than `K_slots` allows, use a simple dependency-aware greedy order: earlier broken VT hops first, then remaining relevant spans.

This is the right ceiling for P6 because it keeps the final active footprint fixed. Compare it against `B_match`, the no-repair baseline where the base eviction policy simply runs at budget `B_base + K_slots`.

### Three possible outcome patterns

**Outcome 1 — Full restore high, Oracle-K high:**
Lost KV content is recoverable, and there is clear fixed-footprint headroom over the base policy. **→ Proceed to P6 as planned.**

**Outcome 2 — Full restore high, Oracle-K near `B_match`:**
Content is recoverable in principle, but not much of the gain survives the fixed-footprint constraint. The story becomes memory-budget-sensitive, and P6 should focus on selection efficiency rather than headline recovery. **→ Narrow P6 claims.**

**Outcome 3 — Full restore low:**
Degradation is not primarily from lost KV content. **→ Diagnose before P6.**

### Diagnostic tests if full restore fails

**Positional encoding test:** After full restore, shift all position IDs to match original absolute positions. Does recovery improve? Tests whether RoPE position shift is the culprit.

**Serialization path test:** Use exact KV serialization (`save_kv` / `load_kv`, no eviction). Does this produce condition A? If yes: eviction + restore pipeline has a bug. If no: the two-call structure itself degrades quality for a model-intrinsic reason.

**Attention pattern comparison:** Compare attention heatmaps between condition A and the full-oracle condition via `output_attentions=True`. Do patterns differ? Which layers?

### Oracle tables

Report two tables:

- **Full-oracle recovery** = `(Oracle_full − B_base) / (A − B_base)`
- **Matched-footprint oracle recovery** = `(Oracle_K − B_match) / (A − B_match)`

Break both out by:

- Task type: VT-4hop, MQ-NIAH-4q, S-NIAH
- Context length: 16K, 32K
- `B_base`: onset budgets selected in P1
- `K_slots`: feasible repair slots from P4

Per-task hypothesis: full restore may nearly solve S-NIAH while Oracle-K exposes the harder selector problem on VT and MQ-NIAH. That distinction directly shapes P6.

### Why P5 is still the paper's core contribution

Even if P6 delivers only modest practical gains, P5 now provides both the unconstrained recoverability ceiling and the matched-footprint ceiling. Together they quantify how much of eviction damage is repairable at all, and how much of that headroom survives when GPU footprint is held fixed.

### Deliverable

Two oracle tables: full restore and matched-footprint Oracle-K. Root-cause diagnosis if full restore is partial. This is the paper's central go/no-go phase.

---

## Phase 6: Repair Algorithm Implementation and Fixed-Footprint Ablation Study

**Goal:** Build a budget-constrained repair selector whose gain survives a matched-footprint comparison. P6 is about token quality, not token count.

### Matched-footprint protocol

For every repaired run, evaluate four systems at the same final active-cache size `B_match = B_base + K_slots`:

- **`B_match` no-repair baseline:** run the base eviction policy directly at budget `B_match`
- **`Random-K`:** run at `B_base`, then fill `K_slots` with random buffered tokens
- **`Recency-K`:** run at `B_base`, then fill `K_slots` with oldest evicted tokens
- **`IdleKV`:** run at `B_base`, then fill `K_slots` with the repair selector

Any gain by IdleKV over these baselines must come from which `K_slots` are chosen, not from a larger GPU footprint.

### Algorithm: Budget-constrained KV repair

Given idle budget `T_idle` and an eviction buffer of size `N`:

**Step 1 — Determine how many slots can be repaired**

```python
K_slots = min(
    floor(T_repair / t_per_token),          # time-limited from P4 profiling
    floor(gpu_free_memory / kv_per_token),  # memory-limited
    len(eviction_buffer),                   # buffer-limited
)
```

**Step 2 — Score evicted tokens**

```text
relevance(i) = max over layers of softmax(Q_recent · K_i^T / sqrt(head_dim))
```

where `Q_recent` comes from the last `M` active tokens.

**Step 3 — Fill exactly `K_slots`**

Move the top-`K_slots` selected KV pairs to GPU and inject them at their original sequence positions. The resumed cache must end at `B_match`, not exceed it.

**Step 4 — Continue**

The next LLM turn uses the matched-footprint augmented cache. No extra wall-clock latency beyond the tool call idle window.

### Comparison suite and ablations

Each ablation changes exactly one component while holding the matched-footprint protocol fixed:

| Ablation | Variable | Values | Default |
|----------|----------|--------|---------|
| Selection strategy | Idle-time scorer | L2-norm Q, dot-product, random, recency-inverse | L2-norm Q |
| `K_slots` | Tokens filled during idle time | 50, 100, 250, 500, 1000 | 250 |
| `B_base` | Base keep budget before repair | onset budgets from P1 | chosen onset budget |
| Query window `M` | Recent tokens used to score evicted tokens | 32, 64, 128, 256 | 64 |
| Eviction base | Base eviction algorithm | SnapKV, StreamingLLM, H2O | SnapKV |
| Layer selection | Which layers to repair | All layers, top-8, top-4 | All layers |

### Main plots

For each configuration:

- **X:** Repair compute cost in FLOPs = `K_slots × M × head_dim × n_layers × 2`
- **Y1:** Matched-footprint recovery = `(C − B_match) / (A − B_match)`
- **Y2:** Absolute selection lift = `C − B_match`

Plot the Pareto frontier and compare against:

- `B_match` no-repair baseline
- `Random-K`
- `Recency-K`
- `Oracle-K`

**Secondary plot:** Matched-footprint recovery vs. tool call duration `T_idle`, using the P4 feasibility frontier to translate duration into feasible `K_slots`.

### Negative controls

Run FWE and CWE under the same matched-footprint settings used for VT and MQ-NIAH. Expected: little or no lift over `B_match`, `Random-K`, and `Recency-K`. This shows the mechanism is selective rather than a generic context booster.

### Per-task analysis

Report matched-footprint recovery separately for each task type. Key paper claim: task-dependent gain is mechanistically predictable — high when a few localized spans matter, low when the answer depends on distributed global state.

### Multi-gap robustness

Test sequences with 3, 5, 10 gaps. Does the selector remain useful when each turn only gets `K_slots` matched repair capacity, or does noise accumulate?

### Deliverable

`src/repair/` with matched-footprint comparison hooks. Main tables: `B_match` vs. `Random-K` vs. `Recency-K` vs. `IdleKV` vs. `Oracle-K`. Pareto frontier plots and multi-gap robustness curves.

---

## Phase 7: Optional Appendix Proof-of-Life on Agentic Benchmarks

**Goal:** Sanity-check transfer to an end-to-end agent loop without making the main paper depend on a noisy 7B coding benchmark.

This phase is not on the critical path. The default decision is: **do not use Qwen2.5-7B SWE-bench Verified resolve rate as a main-text claim or go/no-go gate.** The base solve rate is too low for small deltas to be interpretable.

### Default branch — appendix-only SWE-bench proof-of-life

If you run SWE-bench at 7B at all, keep it in the appendix and treat it as descriptive only:

- fixed task subset
- same prompts, tools, and environment each run
- multiple seeds or repeated samples
- raw resolve counts plus 95% Wilson or bootstrap intervals

Interpret only large effects. Do not use small resolve-rate changes as evidence for the repair mechanism.

### Preferred 7B-friendly applied branch

If an applied result is still needed after P6, switch to a lower-noise benchmark where the 7B baseline is comfortably above zero and tool calls still create idle windows. Reasonable candidates are:

- `tau-bench`
- a curated SWE-bench Lite subset of tasks the 7B model can already sometimes solve
- a controlled long-context multi-turn tool benchmark derived from RULER-KVR traces

Only move SWE-bench back into the main text if you later rerun the applied evaluation on a larger model where the base success rate is high enough to resolve a signal.

### What P7 should measure if it is run

Use the same matched-footprint comparison suite from P6:

- `B_match` no-repair baseline
- `Random-K`
- `Recency-K`
- `IdleKV`

Measure:

- task success with confidence intervals
- tool duration distribution and resulting `K_slots`
- additional resume latency
- failure taxonomy: buffer miss vs. selector miss

### Deliverable

Optional appendix-only proof-of-life figure or table. Not a main-paper gate.

---

## Summary: Deliverables and Decision Points

| Phase | Key deliverable | Gate condition |
|-------|-----------------|----------------|
| P0 | `results/baseline_ruler.json` | Scores within ±2% of paper |
| P1 | Condition B degradation curves | Highest budget stays near A while onset budget shows a clear attributable drop |
| P2 | `src/kv_utils.py` + attention heatmaps | Round-trip identity test passes |
| P3 | `src/eviction/` + Pitfalls reproductions | Two failure modes reproducible |
| P4 | `src/eviction_buffer.py` + latency table | K≥100 feasible in 2s budget |
| P5 | Full-oracle + Oracle-K tables | Full restore shows recoverability and Oracle-K beats `B_match` on primary tasks |
| P6 | `src/repair/` + matched-footprint ablations | IdleKV beats `B_match`, `Random-K`, and `Recency-K` on primary tasks |
| P7 | Optional appendix proof-of-life | No gate |

**Critical path:** P0 → P1 → P3 → P4 → P5 → P6 (P2 overlaps with P1)

**Highest-risk phase:** P5. If full restore does not recover performance, or Oracle-K does not beat `B_match`, the core repair story weakens substantially.

**For the workshop paper:** P5 plus matched-footprint P6 is the full paper. P7 is appendix-only proof-of-life if it is run at all.

---

## Appendix: Key Code Interfaces

```python
# P2: KV utilities
save_kv(past_key_values, path)
load_kv(path) -> past_key_values
slice_kv(past_key_values, token_indices) -> past_key_values
merge_kv(cache_a, cache_b) -> past_key_values
inject_kv(past_key_values, new_pairs, positions) -> past_key_values

# P3: Eviction policies (unified interface)
class EvictionPolicy:
    select_tokens_to_keep(past_kv, attn_scores, budget) -> list[int]

# P4: Eviction buffer
class EvictionBuffer:
    push(pos, kv_pair, score, q_vec)
    query(recent_q_vectors, top_k) -> list[(pos, kv_pair)]
    to_gpu(positions) -> past_key_values fragment

# P6: Repair
class KVRepairAlgorithm:
    repair(active_cache, eviction_buffer, budget_seconds) -> repaired_cache
```

---

## Appendix: Metric Definitions

**Matched-footprint recovery:** `(C − B_match) / (A − B_match)`
- A = condition A score (monolithic pass, P0 baseline)
- `B_match` = no-repair baseline at final matched footprint `B_base + K_slots`
- C = IdleKV score at the same final matched footprint
- 1.0 = IdleKV closes the remaining gap at fixed footprint; 0.0 = no gain over the matched no-repair baseline

**Selection lift:** `C − B_match`
- Absolute gain from choosing better `K_slots` at the same GPU footprint

**Full-oracle recovery:** `(Oracle_full − B_base) / (A − B_base)`
- `Oracle_full` = score when all evicted tokens are restored with no footprint limit
- Measures whether lost content is recoverable at all

**Matched Oracle-K recovery:** `(Oracle_K − B_match) / (A − B_match)`
- `Oracle_K` = task-hindsight upper bound with only `K_slots` restored
- Sets the fixed-footprint ceiling for any selector

**Compute efficiency:** `selection_lift / repair_FLOPs`
- Compares selectors at equal compute budgets under the matched-footprint protocol

**Repair slots `K_slots`:** `min(floor(T_repair / t_per_token), gpu_memory_headroom / kv_per_token)`
- `t_per_token` from P4 profiling
- Determines how many buffered tokens can be promoted during the idle window
