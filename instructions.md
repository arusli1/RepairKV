# KV Cache Repair via GPU Idle Time: Research Development Plan

## Overview

**Motivation:** GPU idle time during tool calls in batch-1 edge-compute LLM agentic workloads is entirely unused. This plan develops a system that uses this idle time to repair the KV cache — restoring evicted tokens that have become newly relevant — thereby improving agent response quality without adding latency.

**Problem:** Token eviction algorithms (SnapKV, StreamingLLM, H2O) compress the KV cache by dropping tokens deemed unimportant by attention scores at eviction time. As shown in *The Pitfalls of KV Cache Compression* (arXiv:2510.00231), this causes important details to be forgotten when context needs shift mid-conversation. Agentic tool calls (script executions, MCP calls, pytest runs: often 2–60 seconds) create idle windows where recomputation is free in wall-clock terms.

**Proposed solution:** During the tool call idle window, recompute attention scores between the most recent query vectors and a CPU-resident buffer of evicted token KV pairs. Promote the highest-scoring evicted tokens back to the active GPU KV cache before the next LLM turn begins.

**Target venue:** 3–4 page workshop paper (e.g. NeurIPS Efficient Reasoning, ICLR Sparsity in LLMs, or similar). The bar for accepted workshop papers in this space is: one novel mechanism, one clean results table across 2–3 conditions on 1–2 models, one figure showing the budget tradeoff. Degradation from KV eviction on reasoning-heavy tasks is already established in the published literature (NeurIPS 2025: "token eviction methods struggle with tasks that rely heavily on in-context learning, passkey retrieval, and long-context reasoning because they tend to eject critical tokens") — cite this and focus experimental effort on the repair contribution.

**Evaluation framework:** A modified RULER benchmark (RULER-KVR) with three conditions:
- **Condition A** — monolithic pass (standard RULER, single forward pass, no eviction)
- **Condition B** — eviction during prefill, query on compressed cache (no repair)
- **Condition C** — eviction during prefill, repair during idle window, query on repaired cache

Primary metric: **repair recovery rate** = `(C − B) / (A − B)`. A value of 1.0 means repair fully restores monolithic performance; 0.0 means no benefit over eviction alone.

---

## Research Arc

```
P0          P1              P2            P3              P4              P5              P6            P7
Baseline → Eviction      → KV access  → Eviction     → CPU eviction → [ORACLE     → Repair      → End-to-end
RULER       degradation    layer         implementation  buffer          GATE]         algorithm     evaluation
            measurement    (infra)       + validation    + profiling     go/no-go      + ablation    (SWE-bench)
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

## Phase 5: Oracle Experiment — The Go/No-Go Gate

**Goal:** Determine whether theoretically perfect KV repair recovers performance. This is the most important experiment in the paper and likely its core novel contribution regardless of how well the practical repair algorithm performs.

> Do not proceed to P6 until this is complete and interpreted.

### What the Oracle Does

During the idle window, restore ALL evicted tokens — no selection, no budget limit:

```python
all_evicted_positions = list(eviction_buffer.buffer.keys())
all_evicted_kv = eviction_buffer.to_gpu(all_evicted_positions)
repaired_cache = inject_kv(active_cache, all_evicted_kv, all_evicted_positions)
outputs = model(query_tokens, past_key_values=repaired_cache)
```

This ignores compute budget entirely. It answers: "what if repair were free and perfect?"

### Three Possible Outcomes

**Outcome 1 — Full recovery (oracle ≈ condition A, ≥ 90%):**
Degradation is entirely due to lost KV content. Your repair algorithm has a real ceiling to approach. **→ Proceed to P6.**

**Outcome 2 — Partial recovery (40–80%):**
KV content loss explains most but not all degradation. Positional encoding shift or reprefill path dependence also contributes. Design P6 to target the recoverable portion. **→ Modify P6 scope.**

**Outcome 3 — No recovery (oracle ≈ condition B):**
Degradation is not primarily from KV loss. The repair hypothesis needs revision. **→ Diagnose before P6.**

### Diagnostic Tests if Oracle Fails

**Positional encoding test:** After oracle restore, shift all position IDs to match original absolute positions. Does recovery improve? Tests whether RoPE position shift is the culprit.

**Reprefill path test:** Use exact KV serialization (`save_kv` / `load_kv`, no eviction). Does this produce condition A? If yes: eviction + restore pipeline has a bug. If no: the two-call structure itself degrades quality for a model-intrinsic reason.

**Attention pattern comparison:** Compare attention heatmaps between condition A and oracle-restored condition via `output_attentions=True`. Do patterns differ? Which layers?

### Oracle Results Table

Report oracle recovery rate = `(Oracle − B) / (A − B)` broken out by:
- Task type: VT-4hop, MQ-NIAH-4q, S-NIAH
- Context length: 16K, 32K
- k_budget at eviction: the onset budgets selected in P1 (default reduced-harness sweep: 4096, 8192, 16384)

Per-task hypothesis: oracle may perfectly recover S-NIAH (single discrete needle) but only partially recover VT (requires restoring distributed chain state). If so, this shapes P6.

### Why the Oracle Table Is the Paper's Core Contribution

Even if P6 delivers only partial repair, the oracle table is independently publishable at a workshop: it proves (or bounds) the maximum benefit achievable by any KV repair scheme, and quantifies how much of eviction-induced degradation is theoretically recoverable vs. structural. This result does not exist in the published literature.

### Deliverable

Oracle recovery rate table across conditions. Root-cause diagnosis if recovery is partial. This becomes a central table or figure in the paper regardless of P6 results.

---

## Phase 6: Repair Algorithm Implementation and Ablation Study

**Goal:** Budget-constrained repair using recent Q vectors to score evicted tokens; ablation study isolating each component's contribution.

### Algorithm: Budget-Constrained KV Repair

Given idle budget T seconds and N evicted tokens in the CPU buffer:

**Step 1 — Scoring (CPU-side):**
```
relevance(i) = max over layers of: softmax(Q_recent · K_i^T / sqrt(head_dim))
```
where `Q_recent` is the query vector matrix from the last M tokens of the active cache.

**Step 2 — Selection:**
```python
K = min(
    floor(T_repair / t_per_token),         # time-limited (from P4 profiling)
    floor(gpu_free_memory / kv_per_token),  # memory-limited
    len(eviction_buffer)                    # buffer-limited
)
```

**Step 3 — Restoration:** Move top-K selected KV pairs to GPU. Inject at original sequence positions via `inject_kv`.

**Step 4 — Continue:** Next LLM turn uses the augmented cache. No additional latency — repair happened during the idle window.

### Ablation Axes

Each ablation changes exactly one component while holding others at default:

| Ablation | Variable | Values | Default |
|----------|----------|--------|---------|
| Selection strategy | Scoring algorithm | L2-norm Q, dot-product, random, recency-inverse | L2-norm Q |
| Repair budget K | Tokens promoted to active cache | 50, 100, 250, 500, 1000 | 250 |
| Query window M | Recent tokens used to score evicted tokens | 32, 64, 128, 256 | 64 |
| Eviction base | Eviction algorithm | SnapKV, StreamingLLM, H2O | SnapKV |
| Layer selection | Which layers to repair | All layers, top-8, top-4 | All layers |

### Compute-Efficiency Frontier — The Main System Result

For each configuration:
- **X:** Repair compute cost in FLOPs = `K × M × head_dim × n_layers × 2`
- **Y:** Recovery rate = `(C − B) / (A − B)`

Plot the Pareto frontier. Compare against oracle (ceiling), random selection repair (selection-agnostic baseline), and no repair (condition B). Each eviction method gets its own curve.

**Secondary plot:** Recovery rate vs. tool call duration T (x-axis in seconds). Connects the system result to the real SWE-bench tool call distribution — pytest calls at 15s give far more repair budget than `cat` calls at 0.1s.

### Negative Controls

Run FWE and CWE under repair at the same budgets used for VT and MQ-NIAH. Expected: near-zero recovery on aggregation tasks. This proves repair is targeted (restoring specific localized tokens) rather than a global context enhancement. If repair significantly helps FWE, investigate before claiming the mechanism is understood.

### Per-Task Analysis

Report recovery rate separately for each task type. Key paper claim: recovery rate is task-type-dependent in a mechanistically predictable way — high for VT and MQ-NIAH (localized signal), low for FWE/CWE (distributed signal).

### Multi-Gap Robustness

Test sequences with 3, 5, 10 gaps. Does repair compound well, or does each repair introduce accumulated noise? Run at 32K context with 5-gap sequences at T_idle = 5s each.

### Deliverable

`src/repair/` — full repair algorithm with all ablation hooks. Main results tables: recovery rate × task × eviction method × budget. Compute-efficiency frontier plots. Multi-gap robustness curves.

---

## Phase 7: End-to-End Evaluation on Real Agentic Workloads

**Goal:** Validate the system on real SWE-bench traces with real tool calls and real idle windows. This is a stretch goal that strengthens the applied claim for the workshop paper.

### SWE-bench Integration

Run Qwen-7B on mini-swe-agent (100-task subset of SWE-bench Verified — use same 100 tasks consistently). Instrument the agent loop:

```python
for turn in agent_loop:
    response = llm.generate(context, kv_cache=active_cache)
    action = parse_tool_call(response)

    tool_start = time.time()
    evicted = eviction_policy.evict(active_cache, budget=k_budget)
    eviction_buffer.push_all(evicted)

    repair_thread = Thread(target=repair_worker, args=(eviction_buffer, active_cache))
    repair_thread.start()

    tool_result = execute_tool(action)
    tool_duration = time.time() - tool_start

    repair_thread.join(timeout=tool_duration * 0.9)
    active_cache = repaired_cache
    context = append(context, tool_result)
```

**Measure:** Resolve rate with vs. without repair. Even 1–2% improvement on SWE-bench is a strong applied result.

### Real Tool Call Duration Budget

From Continuum (arXiv:2511.02230), SWE-bench tool calls are bimodal:

| Tool type | Median | p90 | Repair budget |
|-----------|--------|-----|---------------|
| cat, ls, head | ~0.1s | 0.3s | ~40ms, token transfer only |
| grep, find | ~0.45s | 2.0s | ~200ms, K≈50 |
| git | ~0.7s | 2.5s | ~500ms, K≈100 |
| python script | ~2.5s | 12s | ~2.3s, K≈400 |
| pytest | ~14s | 60s | ~13.8s, K≈2000 |
| pip install | ~18s | 90s | full oracle feasible |

Report expected recovery rate weighted by the real SWE-bench tool call duration distribution — this answers "what recovery do you actually get in practice?"

### Latency Overhead Measurement

Verify repair does not add wall-clock latency from the user's perspective. Measure P50/P90/P99 of additional latency at tool call resume, and frequency of cases where repair thread did not finish in time.

### Failure Case Analysis

For tasks where repair did not help: what was the tool call duration? Were relevant tokens in the buffer? Was the failure a repair miss (tokens present but not selected) or a buffer miss (tokens never stored)? This produces the paper's limitations section.

### Deliverable

SWE-bench resolve rate table: baseline vs. SnapKV alone vs. SnapKV + repair. Latency overhead profile. Real-world compute-efficiency curve weighted by actual tool call duration distribution. Failure case taxonomy.

---

## Summary: Deliverables and Decision Points

| Phase | Key deliverable | Gate condition |
|-------|-----------------|----------------|
| P0 | `results/baseline_ruler.json` | Scores within ±2% of paper |
| P1 | Condition B degradation curves | VT-4hop degrades > 3% at k_budget=512 on 32K |
| P2 | `src/kv_utils.py` + attention heatmaps | Round-trip identity test passes |
| P3 | `src/eviction/` + Pitfalls reproductions | Two failure modes reproducible |
| P4 | `src/eviction_buffer.py` + latency table | K≥100 feasible in 2s budget |
| P5 | Oracle recovery rate table | Oracle recovery ≥ 40% to justify P6 |
| P6 | `src/repair/` + full ablation tables | Recovery > random baseline on ≥2 task types |
| P7 | SWE-bench resolve rate + latency overhead | End-to-end improvement ≥ 0.5% resolve rate |

**Critical path:** P0 → P1 → P3 → P4 → P5 → P6 → P7 (P2 overlaps with P1)

**Highest-risk phase:** P5. If oracle does not recover performance, the core hypothesis requires revision.

**For the workshop paper:** Oracle table (P5) + condition B/C comparison on VT-4hop and MQ-NIAH (P6) + compute-efficiency frontier (P6) is the full paper. P7 strengthens the applied claim but is a stretch goal.

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

**Repair recovery rate:** `(C − B) / (A − B)`
- A = condition A score (monolithic pass, P0 baseline)
- B = condition B score (eviction during prefill, no repair)
- C = condition C score (eviction during prefill, repair during idle window)
- 1.0 = full recovery; 0.0 = repair provides no benefit; negative = repair hurts

**Oracle recovery rate:** `(Oracle − B) / (A − B)`
- Oracle = score when ALL evicted tokens are restored with no budget constraint
- Sets the theoretical ceiling for any repair algorithm

**Compute efficiency:** `recovery_rate / repair_FLOPs`
- Compares selection strategies at equal compute budgets

**Repair budget K:** `min(floor(T_repair / t_per_token), gpu_memory_headroom / kv_per_token)`
- `t_per_token` from P4 profiling
- Determines maximum tokens promotable from CPU buffer per idle window
