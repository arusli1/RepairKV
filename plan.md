# KV Cache Repair via GPU Idle Time: Research Development Plan

## Overview

**Motivation:** GPU idle time during tool calls in batch-1 edge-compute LLM agentic workloads is entirely unused. This plan develops a system that uses this idle time to repair the KV cache — restoring evicted tokens that have become newly relevant — thereby improving agent response quality without adding latency.

**Problem:** Token eviction algorithms (SnapKV, StreamingLLM, H2O) compress the KV cache by dropping tokens deemed unimportant by attention scores at eviction time. As shown in *The Pitfalls of KV Cache Compression* (arXiv:2510.00231), this causes important details to be forgotten when context needs shift mid-conversation. Agentic tool calls (script executions, MCP calls, pytest runs: often 2–60 seconds) create idle windows where recomputation is free in wall-clock terms.

**Proposed solution:** During the tool call idle window, recompute attention scores between the most recent query vectors and a CPU-resident buffer of evicted token KV pairs. Promote the highest-scoring evicted tokens back to the active GPU KV cache before the next LLM turn begins.

**Evaluation framework:** Later repair phases use a matched-footprint RULER-KVR protocol:
- **Condition A** — monolithic pass (standard RULER, single forward pass, no eviction)
- **Condition B_onset** — compressed-cache baseline at base keep budget `B_base`; used in P1 to locate the onset-of-regression regime
- **Condition B_match** — no-repair matched-footprint baseline at final active budget `B_match = B_base + K_slots`
- **Condition C** — IdleKV repair: prefill retains `B_base`, idle-time repair injects `K_slots` buffered tokens, and the resumed cache footprint is also `B_match`
- **Matched baselines** — `Random-K` and `Recency-K`, both filling the same `K_slots` and ending at the same `B_match` footprint

Primary later-phase metrics are **matched-footprint recovery** = `(C − B_match) / (A − B_match)` and absolute **selection lift** = `C − B_match`. This makes the core claim "repair picks better tokens at the same GPU footprint," not "repair gets more tokens."

---

## Research Arc: Phase Overview

```
P0            P1              P2            P3              P4              P5              P6            P7
Baseline  →   Gap injection → KV access  → Eviction     → CPU eviction → [ORACLE     → Repair      → Appendix
RULER         + degradation   layer         implementation  buffer          GATE]         algorithm     proof-of-life
              measurement     (infra)       + validation    + profiling     go/no-go      + ablation    (optional)
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

**Duration:** ~1 week  
**Goal:** Reproduce published RULER scores with Qwen-7B before touching anything else.

Every later phase needs a trustworthy score to compare against. If P0 produces wrong RULER numbers, every downstream result is meaningless. This is the only truly blocking phase — nothing proceeds until scores match within ~2% of reported baselines.

### Model Setup

- Install `Qwen/Qwen2.5-7B-Instruct` via HuggingFace
- Verify tokenizer output matches expected sequence lengths
- Confirm 32K context window is accessible (requires `rope_scaling` config check)
- Set up `vllm` serving locally — this will be the inference backend for all later phases
- GPU memory requirement: ~28GB at fp16 for Qwen-7B at 32K context

### RULER Setup

- Clone `hsiehjackson/RULER`, read generation scripts carefully
- Run S-NIAH at 4K, 8K, 16K with 500 examples each
- Run VT (2-hop) and FWE at the same lengths
- Use exact chat template for Qwen-2.5-Instruct — wrong template is the most common source of score divergence
- Do not modify RULER in any way at this stage

### Step-by-step

1. Install dependencies: `pip install vllm transformers` and RULER's requirements. Verify GPU memory.
2. Run RULER's data generation script for S-NIAH at 4K tokens. Inspect 5 examples manually — verify the needle is present at the expected position, haystack fills to the correct length, and the query is sensible.
3. Run inference and score. Compare to reported Qwen-2.5-7B numbers in the RULER paper. If scores diverge by more than 2%, debug: check chat template, system prompt, `max_tokens` setting, and temperature (must be 0 for reproducibility).
4. Repeat for 8K, 16K, 32K lengths. Plot the degradation curve. This becomes your Figure 1 baseline.
5. Run VT (2-hop) and FWE at the same lengths. Save all raw model outputs — not just final accuracy scores — because you will need them for error analysis in later phases.

### Acceptance Criteria

| Task | Length | Minimum score |
|------|--------|---------------|
| S-NIAH | 4K | ≥ 98% |
| S-NIAH | 32K | ≥ 85% |
| VT-2hop | 16K | Within ±2% of paper |
| FWE | 16K | Within ±2% of paper |

### Deliverable

`results/baseline_ruler.json` — scores across all tasks × lengths. Git-tagged as `v0-baseline`. **This file is never modified.** All future experiments measure delta against it.

---

## Phase 1: RULER-KVR — Gap Injection and Degradation Measurement

**Duration:** ~1 week  
**Goal:** Modify RULER to prove the degradation problem exists and is measurable before building any solution.

**Go/no-go question:** Does inserting a synthetic gap (KV eviction between context delivery and query) actually degrade RULER scores? If degradation is < 3% even at 128K, the problem may not be severe enough under normal eviction policies to motivate a repair system.

### What "Gap Injection" Means in Code

Standard RULER runs context + query in a single call. This phase splits that into two calls with explicit KV cache handling between them:

- **Call 1:** Feed context tokens only → save KV cache state
- **Gap:** Evict the KV cache (or save to CPU and zero GPU), optionally sleep N seconds to simulate a real tool call duration
- **Call 2 (Condition B):** Reprefill from scratch — concatenate full context + query and run a new forward pass as if the gap never happened

```python
# Pseudocode for condition B
outputs_1 = model(context_tokens, use_cache=True)
kv_cache = outputs_1.past_key_values
# --- gap: evict ---
del kv_cache  # GPU memory freed
time.sleep(gap_duration)
# --- resume: full reprefill ---
outputs_2 = model(context_tokens + query_tokens, use_cache=False)
answer = decode(outputs_2)
```

### Key HuggingFace Hooks

- `model.generate(..., past_key_values=None)` — forces full reprefill from scratch
- `outputs.past_key_values` — tuple of `(K, V)` tensors per layer, shape `[batch, n_heads, seq_len, head_dim]`
- `torch.save(past_key_values, "cache.pt")` / `torch.load(...)` — checkpoint the cache between calls
- For vLLM: use `LLMEngine` with manual `step()` calls and `block_manager` access for finer-grained control

### Experimental Design

**Conditions to implement:**
- Condition A: already done in P0 (single forward pass scores serve directly)
- Condition B: implement here — context prefill → KV evicted → query reprefill from scratch

**Variables to sweep:**

| Variable | Values |
|----------|--------|
| Context length | 4K, 16K, 32K, 64K |
| Task type | S-NIAH, MK-NIAH, VT-2hop, FWE |
| Needle depth (for NIAH) | 5%, 25%, 50%, 75%, 95% |
| Gap duration | 0.5s, 2s, 8s, 30s |

### New Task: Cross-Turn MK-NIAH

Add a task variant where:
- Turn 1 delivers context containing the **target needle**
- Gap at turn 2 (simulated tool call)
- Turn 3 delivers a **distractor needle** (same key, different value)
- Query asks for the target

Without repair, the model should increasingly return the turn-3 distractor value (recency bias). With repair, the turn-1 needle should be recoverable. Score separately as:
- "Returned correct target value" (true positive)
- "Returned distractor value" (recency failure)
- "Returned neither" (hallucination)

This is the cleanest controlled test of gap-induced degradation because the failure mode is attributable to a specific mechanism.

### New Task: Depth-Stratified NIAH

Replace random needle depth with five fixed depths: 5%, 25%, 50%, 75%, 95% of context. This lets you measure whether repair preferentially restores attention to early-context tokens (attention sink victims) vs. late ones. Plot accuracy as a 2D heatmap: x = needle depth, y = gap duration.

### Deliverable

Degradation curves: score vs. context length × gap duration × task type. These become the "Condition B" column in your main results table.

**Expected pattern:** VT and FWE degrade more than S-NIAH; early-context needles (depth 5–25%) degrade more than late-context ones; longer gap durations produce more degradation (eviction policy more aggressive).

---

## Phase 2: KV Cache Access Layer

**Duration:** ~1 week  
**Goal:** Build the infrastructure that all later phases depend on — serialize, inspect, and modify KV tensors with a clean API.

> **Critical test — round-trip identity:** Save KV cache after context prefill, reload it, continue generation. The output must be bit-for-bit identical to a run that never evicted. If this test fails, nothing downstream can be trusted. This is the P2 acceptance gate.

### KV Cache Anatomy (Qwen-7B)

`past_key_values` is a Python tuple of length `n_layers` (32 for Qwen-7B). Each element is a `(key, value)` tuple where each tensor has shape:

```
[batch=1, n_kv_heads, seq_len, head_dim]
```

Qwen-7B specifics with GQA:
- 32 transformer layers
- 32 KV heads (grouped into 4 per query group)
- head_dim = 128
- Total KV size at 32K tokens: `32 layers × 2 tensors × 32K tokens × 128 dims × 2 bytes (fp16) ≈ 16GB`

This means full KV at 32K **must be CPU-offloaded** — it doesn't fit in GPU VRAM alongside the model weights.

### The Five-Function API

Implement these five functions in `src/kv_utils.py`. Everything from P3 onward calls into this interface:

```python
def save_kv(past_key_values, path: str) -> None:
    """Serialize KV cache to disk or CPU pinned memory."""

def load_kv(path: str) -> tuple:
    """Restore KV cache from disk to GPU. Returns past_key_values tuple."""

def slice_kv(past_key_values, token_indices: list[int]) -> tuple:
    """Extract a subset of token positions. Returns new past_key_values."""

def merge_kv(cache_a: tuple, cache_b: tuple) -> tuple:
    """Concatenate two KV caches along the sequence dimension."""

def inject_kv(past_key_values, new_pairs: tuple, positions: list[int]) -> tuple:
    """Insert repaired KV pairs at their original sequence positions."""
```

### Unit Tests (Required Before Proceeding)

**Test 1 — Round-trip identity:**
Run model to position P → `save_kv` → `load_kv` → continue generation. Next-token logits must match a run with no save/load. Tolerance: fp16 rounding only (max absolute difference < 1e-3).

**Test 2 — Selective injection:**
Remove token at position P from cache using `slice_kv`. Generate an answer. Inject it back using `inject_kv`. Generate again. Second output must recover. Verifies that inject is semantically correct (not just dimensionally correct).

**Test 3 — CPU → GPU transfer latency:**
Profile: time to move a 32K KV cache (16GB) from CPU pinned memory → GPU. At PCIe 4.0 x16 (~32 GB/s theoretical): expect ~500ms. This sets your repair time budget ceiling — you cannot use more time than this for full cache restoration.

### Attention Score Visualization

Use `output_attentions=True` in `model.generate()` to capture attention weight tensors. Visualize as heatmaps: x = query position, y = key position, color = attention weight.

Specifically look for:
- **Attention sinks:** Positions 0–4 always receive disproportionate attention regardless of content
- **Recency bias:** Last ~1K tokens receive high attention; everything before the recency window fades
- **Needle attention:** Which positions genuinely attend to the needle token

Run these visualizations at 4K, 16K, 32K context lengths. The pattern you see here motivates which tokens to prioritize in the CPU eviction buffer (P4) and which layers to focus repair compute on (P6).

### Deliverable

`src/kv_utils.py` — the 5-function KV access API, with full unit test suite. Also: attention heatmap visualizations at multiple context lengths showing the attention sink + recency structure. These go directly into the paper's motivation figures (Figure 2).

---

## Phase 3: Eviction Algorithm Implementation and Validation

**Duration:** ~2 weeks  
**Goal:** Implement SnapKV, StreamingLLM, and H2O under a unified interface; validate by reproducing known failure modes from the Pitfalls paper.

### Unified Interface

All eviction algorithms implement:

```python
class EvictionPolicy:
    def select_tokens_to_keep(
        self,
        past_key_values: tuple,
        attention_scores: torch.Tensor,
        budget: int
    ) -> list[int]:
        """
        Returns indices of tokens to keep in the active KV cache.
        Evicted tokens (all positions not in this list) go to the CPU buffer.
        """
```

### SnapKV Implementation

Key idea: pool attention over an "observation window" (last ~32 tokens of context) to identify which prior tokens receive the most cumulative attention. Evict those below the resulting importance threshold.

**Steps:**
1. Hook into attention computation after the observation window forward pass
2. Compute per-token importance: `importance = mean(attn_weights[:, :, obs_window_tokens, :], dim=[0, 1, 2])`
3. Keep top-K tokens by importance score, plus:
   - Attention sinks (positions 0–4): always kept
   - Recency window (last ~1K tokens): always kept
4. Slice KV cache using `slice_kv` from P2

**Hyperparameters to expose:**
- `k_budget` — number of tokens to keep (test: 256, 512, 1024, 2048)
- `obs_window_size` — size of observation window (test: 16, 32, 64)
- `sink_size` — number of attention sink positions (default: 4)
- `recency_window` — number of recent tokens always kept (default: 1024)

### StreamingLLM Implementation

Keep only attention sinks (positions 0–4) plus a sliding recency window of the last W tokens. Simple, fast, but discards everything in the "middle" of the context.

```python
def select_tokens_to_keep(self, past_kv, attn_scores, budget):
    n_tokens = past_kv[0][0].shape[2]
    sink_indices = list(range(self.sink_size))
    recency_indices = list(range(max(0, n_tokens - self.recency_window), n_tokens))
    return list(set(sink_indices + recency_indices))[:budget]
```

### H2O (Heavy Hitter Oracle) Implementation

Track cumulative attention scores per token across all generated tokens. Evict those with the lowest cumulative score. More accurate than SnapKV's one-shot observation window, but requires maintaining running statistics.

```python
# During each generation step:
self.cumulative_scores += attention_weights.mean(dim=[0, 1])  # [seq_len]

def select_tokens_to_keep(self, past_kv, attn_scores, budget):
    topk_indices = self.cumulative_scores.topk(budget - self.sink_size).indices
    sink_indices = torch.arange(self.sink_size)
    return torch.cat([sink_indices, topk_indices]).unique().tolist()
```

### Validation Against the Pitfalls Paper (arXiv:2510.00231)

Reproduce at least two failure modes to verify your implementations are realistic:

**Failure mode 1 — Mid-document information loss:**
- Place a needle at depth 40–60% in a 32K document
- Apply SnapKV with `k_budget=512`
- Verify the needle token is evicted and the model returns the wrong answer
- If your SnapKV is correct, this should fail consistently (~80%+ failure rate)

**Failure mode 2 — Multi-hop chain breakage:**
- Run VT-3hop with SnapKV at 16K context
- The intermediate chain variable (hop 2 link) should be evicted before it's needed
- Measure accuracy vs. full-cache baseline — expect significant degradation

If you cannot reproduce these failure modes, your eviction implementation is incorrect.

### RULER Under Eviction

Run RULER-KVR Condition B but with real SnapKV eviction (not synthetic gap). Compare these scores to the P1 synthetic-gap numbers. Key question: does real SnapKV produce more or less degradation than simply evicting everything and reprefilling? This comparison is important for the paper because it shows whether the token selection quality of SnapKV matters at all.

### Logging Requirements

For every eviction run, log to disk:
- Which token positions were kept vs. evicted, per layer, per head
- Importance scores for each token at eviction time
- The query vectors (Q) for the last M tokens at eviction time

This log is exactly what the CPU buffer (P4) needs to populate itself.

### Deliverable

`src/eviction/` — SnapKV, StreamingLLM, H2O under the unified interface. RULER scores under each method at `k_budget ∈ {256, 512, 1024, 2048}`. Pitfalls failure mode reproductions as documented, reproducible test cases. Eviction rate vs. budget tradeoff curves.

---

## Phase 4: CPU Eviction Buffer — Storage and Retrieval Profiling

**Duration:** ~1 week  
**Goal:** Store evicted KV pairs on CPU; profile all transfer and recompute operations to establish feasibility within real tool call idle windows.

> **Feasibility gate:** The repair algorithm only works if we can move evicted tokens from CPU → GPU and recompute attention within the tool call's idle window. If moving 1000 evicted tokens takes > 500ms and recomputing attention takes another 500ms, you're already consuming a 2-second tool call budget with nothing left for selection logic. Profile this exhaustively before implementing any repair logic.

### EvictionBuffer Class Design

```python
class EvictionBuffer:
    def __init__(self, max_tokens: int, selection_strategy: str):
        self.buffer: dict[int, tuple] = {}      # token_pos → (K, V) per layer, on CPU
        self.scores: dict[int, float] = {}      # importance score at eviction time
        self.q_vecs: dict[int, torch.Tensor] = {}  # query vectors at eviction time

    def push(self, pos: int, kv_pair: tuple, score: float, q_vec: torch.Tensor):
        """Add an evicted token to the buffer."""

    def query(self, recent_q_vectors: torch.Tensor, top_k: int) -> list[tuple[int, tuple]]:
        """
        Given the current query vectors (from the last M tokens),
        return top_k evicted tokens ranked by relevance.
        Returns: [(token_pos, kv_pair), ...]
        """

    def to_gpu(self, positions: list[int]) -> tuple:
        """Move selected tokens' KV pairs to GPU and return as past_key_values fragment."""
```

**Memory footprint per evicted token (Qwen-7B, fp16):**
`32 layers × 2 tensors × 1 token × 32 heads × 128 dims × 2 bytes = 524,288 bytes ≈ 0.5 MB`

Storing 1000 evicted tokens: ~524 MB CPU RAM. Feasible on any modern workstation.

### Selection Strategies to Implement and Compare

All four strategies must implement the same `push` / `query` interface so they're interchangeable in ablations:

**Strategy A — L2 norm of Q vectors (paper's primary proposal):**
Store tokens whose query vectors had high L2 norm at eviction time. High L2 norm indicates the position was "strongly queried" — it was an active lookup target at the moment it was dropped. Score = `‖q_vec‖₂`.

**Strategy B — Original attention score:**
Store tokens with the highest attention scores at the moment of eviction. Simpler than A but stale — the relevance estimate is from the eviction time query, not the future query.

**Strategy C — Random sample (ablation baseline):**
Store a uniform random sample of evicted tokens. This isolates whether selection quality matters at all, vs. simply having any evicted tokens available for reuse.

**Strategy D — Recency-inverse:**
Store oldest evicted tokens preferentially. Specifically targets attention sink victims — tokens from the early context that were evicted despite being potentially important. Cheapest strategy computationally.

### Latency Profiling — The Critical Table

Measure wall-clock time for each operation. Run 100 trials, report p50 / p90 / p99. Use `torch.cuda.synchronize()` before timing GPU operations.

**Operations to profile:**

| Operation | Parameter range | Notes |
|-----------|-----------------|-------|
| CPU → GPU KV transfer | N ∈ {100, 500, 1000, 2000} tokens | Use pinned memory for speed |
| GPU attention recompute | N evicted tokens × M query tokens | M ∈ {64, 128, 512}; time per layer |
| Buffer dot-product search | N ∈ {1000, 5000, 10000} | Can this stay on CPU or needs GPU index? |
| Buffer push | N tokens per eviction event | Should be negligible |

**Target budget:** All operations must fit within `T_idle − 200ms` where T_idle is the tool call duration. For a 2-second tool call: ~1.8-second repair budget. For an 8-second pytest call: ~7.8-second budget.

**Output: feasibility frontier table.** For each tool call duration T, what is the maximum number of tokens K that can be repaired? This table appears directly in the paper's system design section and determines the scope of all P6 ablations.

### Deliverable

`src/eviction_buffer.py` with all four selection strategies. Profiling table: operation × N_tokens → latency (p50/p90). The feasibility frontier plot: maximum repairable tokens K vs. tool call duration T for each hardware configuration.

---

## Phase 5: Oracle Experiment — Recoverability and Fixed-Footprint Headroom

**Duration:** ~1 week  
**Go/no-go decision:** Before implementing a practical selector, answer two questions:

1. Is eviction damage recoverable at all if memory were free?
2. At the same final GPU footprint, is there headroom beyond what the base eviction policy gets by simply keeping `K_slots` more tokens?

**This is still the highest-value phase in the paper.**

### Shared notation

- `B_base`: onset keep budget selected in P1
- `K_slots`: repair slots feasible under the P4 latency frontier
- `B_match = B_base + K_slots`: final active-cache footprint used for all matched comparisons

### Oracle 1 — Full restore

```python
# Full oracle: restore ALL evicted tokens
all_evicted_positions = list(eviction_buffer.buffer.keys())
all_evicted_kv = eviction_buffer.to_gpu(all_evicted_positions)
full_oracle_cache = inject_kv(active_cache, all_evicted_kv, all_evicted_positions)
outputs = model(query_tokens, past_key_values=full_oracle_cache)
```

This ignores compute and footprint limits. It answers: "is the lost content in principle enough to recover the answer?"

### Oracle 2 — Matched-footprint Oracle-K

```python
# Oracle-K: restore only K_slots tokens, chosen with task hindsight
oracle_positions = select_hindsight_relevant_positions(
    eviction_log=eviction_log,
    task_metadata=task_metadata,
    top_k=K_slots,
)
oracle_kv = eviction_buffer.to_gpu(oracle_positions)
oracle_k_cache = inject_kv(active_cache, oracle_kv, oracle_positions)
outputs = model(query_tokens, past_key_values=oracle_k_cache)
```

For RULER-KVR tasks, the generator metadata already identifies the task-relevant spans. If more relevant spans are missing than `K_slots` allows, use a dependency-aware greedy order, for example earlier broken VT hops first.

This is the ceiling that matters for P6. Compare it against `B_match`, the no-repair baseline where the base eviction policy simply runs at budget `B_base + K_slots`.

### Outcome patterns

**Outcome 1 — Full restore high, Oracle-K high:** lost KV content is recoverable, and there is real fixed-footprint headroom over the base policy. **→ Proceed to P6.**

**Outcome 2 — Full restore high, Oracle-K near `B_match`:** content is recoverable in principle, but little of that gain survives the fixed-footprint constraint. **→ Narrow P6 claims to selection efficiency.**

**Outcome 3 — Full restore low:** degradation is not primarily due to missing KV content. **→ Diagnose before P6.**

### Diagnostics if full restore fails

- **Positional encoding test:** after full restore, shift position IDs to match the single-pass run
- **Serialization path test:** use `save_kv` / `load_kv` without eviction and verify condition A behavior
- **Attention comparison:** compare condition A and full-oracle heatmaps via `output_attentions=True`

### Oracle tables

Report two tables:

- **Full-oracle recovery** = `(Oracle_full - B_base) / (A - B_base)`
- **Matched-footprint oracle recovery** = `(Oracle_K - B_match) / (A - B_match)`

Break both out by:

- task type: VT-4hop, MQ-NIAH-4q, S-NIAH
- context length: 16K, 32K
- `B_base`: onset budgets selected in P1
- `K_slots`: feasible repair slots from P4

### Deliverable

Two oracle tables plus root-cause diagnosis if full restore is partial. This phase now answers both "is repair possible?" and "is there a selector problem at fixed footprint?"

---

## Phase 6: Repair Algorithm Implementation and Fixed-Footprint Ablation Study

**Duration:** ~3 weeks  
**Goal:** Build a budget-constrained repair selector whose gain survives a matched-footprint comparison. The claim is about token quality, not token count.

### Matched-footprint protocol

For every repaired run, evaluate four systems at the same final active-cache size `B_match = B_base + K_slots`:

- **`B_match` no-repair baseline:** run the base eviction policy directly at budget `B_match`
- **`Random-K`:** run at `B_base`, then fill `K_slots` with random buffered tokens
- **`Recency-K`:** run at `B_base`, then fill `K_slots` with oldest evicted tokens
- **`IdleKV`:** run at `B_base`, then fill `K_slots` with the repair selector

If IdleKV wins here, it is because it chooses better tokens for the same footprint.

### Algorithm: budget-constrained KV repair

Given idle budget `T_idle` and an eviction buffer of size `N`:

**Step 1 — Determine how many slots are feasible**

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

The next generation turn uses the matched-footprint augmented cache. No extra wall-clock latency beyond the tool call idle window.

### Ablation axes

| Ablation | Variable | Values tested | Default |
|----------|----------|---------------|---------|
| Selection strategy | Idle-time scorer | L2-norm, dot-product, random, recency-inverse | L2-norm Q vectors |
| `K_slots` | Tokens filled during idle time | 50, 100, 250, 500, 1000 | 250 |
| `B_base` | Base keep budget before repair | onset budgets from P1 | chosen onset budget |
| Query window `M` | Recent tokens used to score evicted tokens | 32, 64, 128, 256 | 64 |
| Eviction base | Which eviction algorithm was used | SnapKV, StreamingLLM, H2O | SnapKV |
| Layer selection | Whether to repair all layers or only high-attention layers | All layers, top-8 layers, top-4 layers | All layers |

### Main plots

For each configuration:

- **X axis:** repair compute cost in FLOPs = `K_slots × M × head_dim × n_layers × 2`
- **Y1:** matched-footprint recovery = `(C - B_match) / (A - B_match)`
- **Y2:** absolute selection lift = `C - B_match`

Compare against:

- `B_match`
- `Random-K`
- `Recency-K`
- `Oracle-K`

**Secondary plot:** matched-footprint recovery vs. tool call duration `T_idle`, using the P4 feasibility frontier to map duration to feasible `K_slots`.

### Per-task analysis

Report matched-footprint recovery separately for each RULER-KVR task. The main claim is that gain should be highest when the answer depends on a small number of localized spans, and lowest when the task needs broad distributed context.

### Multi-gap robustness

Test sequences with multiple tool calls: 3, 5, 10 gaps. The question is whether matched-footprint repair remains useful across repeated gaps or whether selector noise compounds.

### Deliverable

`src/repair/` plus matched-footprint ablation tables: `B_match` vs. `Random-K` vs. `Recency-K` vs. `IdleKV` vs. `Oracle-K`. Pareto frontier plots and multi-gap robustness curves.

---

## Phase 7: Optional Appendix Proof-of-Life on Agentic Benchmarks

**Duration:** ~1 week  
**Goal:** Sanity-check transfer to an end-to-end agent loop without making the main paper depend on a noisy 7B coding benchmark.

Default decision: **do not use Qwen2.5-7B SWE-bench Verified resolve rate as a main-text claim or gate.** The base solve rate is too low for small deltas to be interpretable.

### Default branch — appendix-only SWE-bench proof-of-life

If SWE-bench is run at 7B at all:

- keep it in the appendix
- use a fixed task subset and identical tool environment each run
- run multiple seeds or repeated samples
- report raw resolve counts plus 95% Wilson or bootstrap intervals

Interpret only large effects. Do not use sub-point resolve-rate changes as evidence for the repair mechanism.

### Preferred 7B-friendly applied branch

If an applied result is still needed after P6, switch to a lower-noise benchmark where the 7B baseline is comfortably above zero and tool calls still create idle windows. Candidates include:

- `tau-bench`
- a curated SWE-bench Lite subset
- a controlled long-context multi-turn tool benchmark derived from RULER-KVR traces

Only move SWE-bench back into the main text if the applied evaluation is rerun on a larger model with a materially higher base success rate.

### What P7 should measure if it is run

Use the same matched-footprint comparison suite from P6:

- `B_match`
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

| Phase | Duration | Key deliverable | Gate condition |
|-------|----------|-----------------|----------------|
| P0 | ~1 week | `results/baseline_ruler.json` | Scores within ±2% of paper before proceeding |
| P1 | ~1 week | Degradation curves (Condition B) | Highest budget stays near A while onset budget shows a clear attributable drop |
| P2 | ~1 week | `src/kv_utils.py` + attention heatmaps | Round-trip identity test must pass |
| P3 | ~2 weeks | `src/eviction/` + Pitfalls reproductions | Must reproduce at least 2 Pitfalls failure modes |
| P4 | ~1 week | `src/eviction_buffer.py` + latency table | Profiling shows K≥100 feasible in 2s budget |
| P5 | ~1 week | Full-oracle + Oracle-K tables | Full restore shows recoverability and Oracle-K beats `B_match` on primary tasks |
| P6 | ~3 weeks | `src/repair/` + matched-footprint ablations | IdleKV beats `B_match`, `Random-K`, and `Recency-K` on primary tasks |
| P7 | ~1 week | Optional appendix proof-of-life | No gate |

**Total estimated duration:** ~12 weeks plus optional appendix work

**Critical path:** P0 → P1 → P3 → P4 → P5 → P6 (P2 can overlap with P1)

**Highest-risk phase:** P5. If full restore does not recover performance, or Oracle-K does not beat `B_match`, the core repair story weakens substantially.

---

## Appendix: Key Code Interfaces Summary

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

## Appendix: Evaluation Metric Definitions

**Matched-footprint recovery:** `(C − B_match) / (A − B_match)`
- A = Condition A score (monolithic pass, P0 baseline)
- `B_match` = no-repair baseline at final matched footprint `B_base + K_slots`
- C = IdleKV score at the same final matched footprint
- Interpretation: 1.0 = IdleKV closes the remaining gap at fixed footprint; 0.0 = no gain over the matched no-repair baseline

**Selection lift:** `C − B_match`
- Absolute gain from choosing better `K_slots` at the same GPU footprint

**Full-oracle recovery:** `(Oracle_full − B_base) / (A − B_base)`
- `Oracle_full` = score when all evicted tokens are restored with no footprint limit
- Measures whether lost content is recoverable at all

**Matched Oracle-K recovery:** `(Oracle_K − B_match) / (A − B_match)`
- `Oracle_K` = task-hindsight upper bound with only `K_slots` restored
- Sets the fixed-footprint ceiling for any selector

**Compute efficiency:** `selection_lift / repair_FLOPs`
- Used to compare selectors at equal compute budgets under the matched-footprint protocol

**Repair slots `K_slots`:** Number of buffered tokens promoted during the idle window. Determined by: `K_slots = min(floor(T_repair / t_per_token), gpu_memory_headroom / kv_per_token)`
- `t_per_token` from P4 profiling
- Determines maximum fixed-footprint headroom available during each idle window
