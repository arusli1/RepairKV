# KV Cache Repair via GPU Idle Time: Research Development Plan

## Overview

**Motivation:** GPU idle time during tool calls in batch-1 edge-compute LLM agentic workloads is entirely unused. This plan develops a system that uses this idle time to repair the KV cache — restoring evicted tokens that have become newly relevant — thereby improving agent response quality without adding latency.

**Problem:** Token eviction algorithms (SnapKV, StreamingLLM, H2O) compress the KV cache by dropping tokens deemed unimportant by attention scores at eviction time. As shown in *The Pitfalls of KV Cache Compression* (arXiv:2510.00231), this causes important details to be forgotten when context needs shift mid-conversation. Agentic tool calls (script executions, MCP calls, pytest runs: often 2–60 seconds) create idle windows where recomputation is free in wall-clock terms.

**Proposed solution:** During the tool call idle window, recompute attention scores between the most recent query vectors and a CPU-resident buffer of evicted token KV pairs. Promote the highest-scoring evicted tokens back to the active GPU KV cache before the next LLM turn begins.

**Evaluation framework:** A modified RULER benchmark (RULER-KVR) with three conditions:
- **Condition A** — monolithic pass (standard RULER, single forward pass, no gap)
- **Condition B** — gap, no repair (KV evicted between context delivery and query)
- **Condition C** — gap with repair (evicted tokens selectively restored during idle window)

Primary metric: **repair recovery rate** = `(C − B) / (A − B)`. A value of 1.0 means repair fully restores monolithic performance; 0.0 means no benefit over eviction + reprefill.

---

## Research Arc: Phase Overview

```
P0            P1              P2            P3              P4              P5              P6            P7
Baseline  →   Gap injection → KV access  → Eviction     → CPU eviction → [ORACLE     → Repair      → End-to-end
RULER         + degradation   layer         implementation  buffer          GATE]         algorithm     evaluation
              measurement     (infra)       + validation    + profiling     go/no-go      + ablation    (SWE-bench)
```

**Blocking dependencies:**
- P1 requires P0 (needs baseline scores to measure degradation against)
- P3 requires P2 (needs KV access API to implement eviction correctly)
- P4 requires P3 (needs eviction logs to know what to store in buffer)
- P5 requires P2 + P3 + P4 (needs the full stack to run oracle)
- P6 requires P5 passing (oracle must show recoverable signal before repair is worth building)
- P7 requires P6 (needs the repair algorithm to be complete)

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

## Phase 5: Oracle Experiment — The Go/No-Go Gate

**Duration:** ~1 week  
**Go/no-go decision:** Does theoretically perfect KV repair fully recover performance? If not, diagnose why before building anything else.

**This is the most important experiment in the paper.**

The oracle asks: if we perfectly restored *all* evicted tokens (no selection budget, no approximation — the theoretical upper bound) during the idle window, do RULER scores return to Condition A levels? This tests whether KV content loss is the actual cause of degradation, or whether something else is responsible.

### What the Oracle Does

```python
# Oracle: during idle window, restore ALL evicted tokens
all_evicted_positions = list(eviction_buffer.buffer.keys())  # all of them
all_evicted_kv = eviction_buffer.to_gpu(all_evicted_positions)  # move everything to GPU
repaired_cache = inject_kv(active_cache, all_evicted_kv, all_evicted_positions)
# resume generation with fully repaired cache
outputs = model(query_tokens, past_key_values=repaired_cache)
```

This ignores compute budget entirely. It's the theoretical upper bound: "what if repair were free and perfect?"

### Three Possible Outcomes

**Outcome 1 — Full recovery (oracle ≈ Condition A):**
Degradation is entirely due to lost KV content. Your repair algorithm has a real ceiling to approach. Oracle recovery rate ≥ 90%. **→ Proceed to P6.**

**Outcome 2 — Partial recovery (oracle recovers 40–80% of gap):**
KV content loss explains most but not all degradation. Positional encoding shift or reprefill path dependence also contributes. Design P6 repair to target the recoverable portion; discuss non-recoverable portion as a limitation. **→ Modify P6 scope accordingly.**

**Outcome 3 — No recovery (oracle ≈ Condition B):**
Degradation is not primarily from KV loss. The repair hypothesis needs revision. **→ Do not proceed to P6 until root cause is understood.**

### Diagnostic Tests if Oracle Fails (Outcome 3)

Run these in sequence to identify the true source of degradation:

**Positional encoding test:**
After gap + oracle restore, shift all position IDs in the restored cache to match the original absolute positions from the single-pass run. Does recovery improve? Tests whether RoPE position shift (caused by the two-call structure) is the culprit.

**Reprefill path test:**
Instead of eviction + restore, use exact KV serialization — save the complete KV cache with `save_kv`, sleep, reload with `load_kv`, continue. Does this produce Condition A scores? If yes: your eviction + restore pipeline has a bug. If no: the two-call structure itself degrades quality for a model-intrinsic reason.

**Attention pattern comparison:**
Compare attention weight heatmaps (from P2's `output_attentions=True` hook) between Condition A and oracle-restored condition at the same context length. Do the attention patterns differ? In which layers? Which query positions?

### Oracle Results Table

Report oracle recovery rate = `(Oracle − B) / (A − B)` broken out by:
- Task type: S-NIAH, cross-turn MK-NIAH, VT-3hop, FWE
- Context length: 4K, 16K, 32K, 64K
- Eviction method: SnapKV, StreamingLLM, H2O
- k_budget at eviction: 256, 512, 1024

Key per-task hypothesis: oracle may perfectly recover retrieval tasks (S-NIAH: single needle lookup) but only partially recover aggregation tasks (FWE: global frequency counting requires distributed attention across all context). If so, this is a meaningful finding that motivates different repair strategies for different task types.

### Deliverable

Oracle recovery rate table across all conditions. Root-cause diagnosis if recovery is partial (positional encoding vs. attention pattern analysis). This is the paper's key theoretical contribution: it proves (or bounds) the maximum benefit achievable by KV repair. Even if P6 delivers partial repair, the oracle table justifies why closing the full gap is hard.

---

## Phase 6: Repair Algorithm Implementation and Ablation Study

**Duration:** ~3 weeks  
**Goal:** Budget-constrained repair using recent Q vectors to score evicted tokens; full ablation study isolating each component's contribution.

### Algorithm: Budget-Constrained KV Repair

Given idle budget T seconds and N evicted tokens in the CPU buffer:

**Step 1 — Scoring (CPU-side):**
For each evicted token i in the buffer, compute relevance to the current context:

```
relevance(i) = max over layers of: softmax(Q_recent · K_i^T / sqrt(head_dim))
```

where `Q_recent` is the query vector matrix from the last M tokens of the active (non-evicted) cache. Use approximate nearest-neighbor search (e.g., FAISS on CPU) if N > 5000.

**Step 2 — Selection:**
Take top-K evicted tokens by relevance score. K is determined by the time budget:

```python
K = min(
    floor(T_repair / t_per_token),  # time-limited
    floor(gpu_free_memory / kv_per_token),  # memory-limited
    len(eviction_buffer)  # buffer-limited
)
```

where `t_per_token` comes from P4 profiling.

**Step 3 — Restoration:**
Move selected KV pairs to GPU. Inject into active cache at their original sequence positions using `inject_kv`.

**Step 4 — Continue:**
The next LLM generation turn uses the augmented cache. No additional latency from the model's perspective — repair happened during the idle window.

### Ablation Axes

Each ablation changes exactly one component while holding all others at their default:

| Ablation | Variable | Values tested | Default |
|----------|----------|---------------|---------|
| Selection strategy | Algorithm used for scoring evicted tokens | L2-norm, dot-product, random, recency-inverse | L2-norm Q vectors |
| Repair budget K | Tokens promoted from buffer to active cache | 50, 100, 250, 500, 1000 | 250 |
| Query window M | Number of recent tokens used to score evicted tokens | 32, 64, 128, 256 | 64 |
| Eviction base | Which eviction algorithm was used | SnapKV, StreamingLLM, H2O | SnapKV |
| Layer selection | Whether to repair all layers or only high-attention layers | All layers, top-8 layers, top-4 layers | All layers |
| Buffer strategy | What gets stored in eviction buffer | L2-norm Q, original attn score, random, recency-inverse | L2-norm Q |

### Compute-Efficiency Frontier — The Main System Result

For each ablation configuration, compute:
- **X axis:** Repair compute cost in FLOPs = `K × M × head_dim × n_layers × 2`
- **Y axis:** RULER recovery rate = `(C − B) / (A − B)`

Plot the Pareto frontier. Compare against:
- Oracle (theoretical ceiling)
- Random selection repair (selection-agnostic baseline)
- No repair (Condition B)

Each eviction method (SnapKV, StreamingLLM, H2O) gets its own frontier curve.

**Secondary plot:** Recovery rate vs. tool call duration T (x-axis in seconds). Connect the system result directly to the real tool call duration distribution from SWE-bench traces — pytest calls at 15s give far more repair budget than `cat` calls at 0.1s.

### Per-Task Analysis

Report recovery rate separately for each RULER-KVR task:

- **S-NIAH:** Expected high recovery — single needle is a discrete token, easy to locate in buffer
- **Cross-turn MK-NIAH:** Tests whether repair prevents recency-bias distractor errors specifically
- **VT-3hop (split-chain):** Hardest — requires restoring distributed chain state, not just discrete token
- **Streaming FWE (mid-stream gap):** Tests whether aggregation state (frequency counts) is restorable

Key paper claim: recovery rate should be task-type-dependent in a mechanistically predictable way.

### Depth × Repair Interaction

Cross-tabulate: recovery rate vs. needle depth (5%, 25%, 50%, 75%, 95%). Hypothesis: repair should most benefit needles at depth 5–25% because these are the tokens most likely to be evicted by SnapKV (far from the observation window, not recent, potentially not in the attention sink). Needles at depth 90–95% likely survive eviction regardless (recency window) and show minimal repair benefit.

### Multi-Gap Robustness

Test sequences with multiple tool calls: 3, 5, 10 gaps. Key question: does each repair introduce accumulated noise that compounds over subsequent gaps, or does recovery rate stay stable? Run at 32K context with 10 gaps at T_idle = 5s each.

### Deliverable

`src/repair/` — full repair algorithm with all ablation hooks. Main results tables (Tables 2–4 in paper): recovery rate × task × eviction method × budget. Compute-efficiency frontier plots (Figures 3–5). Multi-gap robustness curves.

---

## Phase 7: End-to-End Evaluation on Real Agentic Workloads

**Duration:** ~2 weeks  
**Goal:** Validate the system on real SWE-bench traces with real tool calls and real idle windows.

### SWE-bench Integration

Run Qwen-7B on mini-swe-agent (100-task subset of SWE-bench Verified — use the same 100 tasks consistently for reproducibility). Instrument the agent loop as follows:

```python
for turn in agent_loop:
    # LLM generates action
    response = llm.generate(context, kv_cache=active_cache)
    action = parse_tool_call(response)
    
    # Tool call starts: evict KV cache, start repair in background thread
    evicted_tokens = eviction_policy.evict(active_cache, budget=k_budget)
    eviction_buffer.push_all(evicted_tokens)
    
    tool_start = time.time()
    repair_thread = Thread(target=repair_worker, args=(eviction_buffer, active_cache))
    repair_thread.start()
    
    # Execute actual tool call (real latency)
    tool_result = execute_tool(action)
    tool_duration = time.time() - tool_start
    
    # Repair completes within tool call window
    repair_thread.join(timeout=tool_duration * 0.9)
    
    # Resume with repaired cache
    active_cache = repaired_cache
    context = append(context, tool_result)
```

**Measure:** Resolve rate (% of SWE-bench tasks correctly patched) with vs. without repair. Even 1–2% improvement on SWE-bench represents a meaningful applied result.

### Real Tool Call Duration Distribution

From Continuum (arXiv:2511.02230) and our analysis: SWE-bench tool calls are bimodal. Use actual measured durations from SWE-bench traces to set realistic repair budgets:

| Tool type | Median duration | p90 duration | Typical repair budget |
|-----------|----------------|--------------|----------------------|
| cat, ls, head | 0.08–0.12s | 0.3s | ~40ms (token transfer only) |
| grep, find | 0.4–0.5s | 2.0s | ~200ms, K≈50 tokens |
| git | 0.7s | 2.5s | ~500ms, K≈100 tokens |
| python (script) | 2.5s | 12s | ~2.3s, K≈400 tokens |
| pytest | 14s | 60s | ~13.8s, K≈2000 tokens |
| pip install | 18s | 90s | full oracle feasible |

Report: expected recovery rate weighted by the real SWE-bench tool call duration distribution. This answers "what recovery do you actually get in practice?" not just best-case.

### Latency Overhead Measurement

Verify that repair does not add wall-clock latency to the agent from the user's perspective. Measure:
- P50/P90/P99 of additional latency at tool call resume (repair thread must finish before model resumes)
- Cases where repair thread did not finish in time: how often, and what was the fallback behavior?
- GPU memory overhead of maintaining the eviction buffer alongside the active model

### MCP Tool Call Extension

Run a small experiment with real MCP tool calls (Gmail search, Google Drive fetch) using the MCP infrastructure from earlier discussion. These have 200ms–2s latency. With the P4 profiling table, determine what K is feasible at 200ms budget and test whether repair at that K helps on RULER-KVR cross-turn MK-NIAH (the task most analogous to real assistant workflows).

### Failure Case Analysis

For SWE-bench tasks where repair did not improve the outcome, collect:
- What was the actual tool call duration (repair budget)?
- What was the eviction k_budget?
- Were the relevant tokens present in the eviction buffer?
- Was the failure a repair miss (tokens were there but not selected) or a buffer miss (tokens were never stored)?

This produces the paper's limitations section and motivates future work.

### Deliverable

SWE-bench resolve rate table: baseline (no eviction) vs. SnapKV alone vs. SnapKV + repair. Latency overhead profile (p50/p90/p99). Real-world compute-efficiency curve weighted by actual tool call duration distribution. Failure case taxonomy. This is the paper's Section 5 (Applied Evaluation).

---

## Summary: Deliverables and Decision Points

| Phase | Duration | Key deliverable | Gate condition |
|-------|----------|-----------------|----------------|
| P0 | ~1 week | `results/baseline_ruler.json` | Scores within ±2% of paper before proceeding |
| P1 | ~1 week | Degradation curves (Condition B) | Degradation > 3% at 32K confirms problem exists |
| P2 | ~1 week | `src/kv_utils.py` + attention heatmaps | Round-trip identity test must pass |
| P3 | ~2 weeks | `src/eviction/` + Pitfalls reproductions | Must reproduce at least 2 Pitfalls failure modes |
| P4 | ~1 week | `src/eviction_buffer.py` + latency table | Profiling shows K≥100 feasible in 2s budget |
| P5 | ~1 week | Oracle recovery rate table | Oracle recovery ≥ 40% to justify P6 |
| P6 | ~3 weeks | `src/repair/` + full ablation tables | Recovery rate > random baseline on ≥3 task types |
| P7 | ~2 weeks | SWE-bench resolve rate + latency overhead | End-to-end improvement ≥ 0.5% resolve rate |

**Total estimated duration:** ~13 weeks

**Critical path:** P0 → P1 → P3 → P4 → P5 → P6 → P7 (P2 can overlap with P1)

**Highest-risk phase:** P5 (oracle gate). If oracle does not recover performance, the core hypothesis requires revision and P6–P7 cannot proceed as planned.

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

**Repair recovery rate:** `(C − B) / (A − B)`
- A = Condition A score (monolithic pass, P0 baseline)
- B = Condition B score (gap, no repair)
- C = Condition C score (gap with repair)
- Interpretation: 1.0 = full recovery to monolithic; 0.0 = repair provides no benefit; negative = repair hurts

**Oracle recovery rate:** `(Oracle − B) / (A − B)`
- Oracle = score when ALL evicted tokens are restored with no budget constraint
- Sets the theoretical ceiling for any repair algorithm

**Compute efficiency:** `recovery_rate / repair_FLOPs`
- Used to compare selection strategies at equal compute budgets

**Repair budget K:** Number of evicted tokens promoted from CPU buffer to active GPU cache during the idle window. Determined by: `K = min(floor(T_repair / t_per_token), gpu_memory_headroom / kv_per_token)`