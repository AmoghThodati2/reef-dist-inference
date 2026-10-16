# reef-dist: config-driven distributed inference for GPT Neo on Reef

Reef is 1 host CPU + N NPUs (N = 1, 2 or 4). Until real NPU kernels and the interconnect
exist, each NPU is simulated by a separate CPU process; kernels, interconnect, sharding
strategy and compute model are all pluggable, so real ones can be swapped in without
touching the sharding logic.

The one question this framework answers is **how each sharding strategy and NPU count
would perform on Reef**. Laptop timings can't answer that, so every run reports three
separate kinds of result:

| section | what | source | deterministic |
|---|---|---|---|
| `exact` | correctness vs HF, FLOPs, bytes, messages, memory | counted from tensor shapes and calls | yes, on any machine (except the float logit diff) |
| `modeled` | Reef step time, TTFT, tokens/s, MFU | `exact` counts × `hardware_profile` | yes, no timers |
| `measured_laptop` | wall-clock median/p90 | `time.perf_counter` | no; only for sanity checks and same-machine comparisons |

**`measured_laptop` is never a Reef estimate.** Reef numbers come only from `modeled`.

## Quick start

```bash
uv venv -p 3.11 .venv && uv pip install -p .venv/bin/python -r requirements.txt
.venv/bin/python run.py configs/tp4_decode.yaml            # -> results/tp4_decode.json + console summary
.venv/bin/python scripts/compare_results.py results/        # strategy x npus x workload table
.venv/bin/python scripts/sweep_hardware.py results/tp1_decode.json results/tp4_decode.json results/pp4_decode.json
.venv/bin/python scripts/export_kernel_shapes.py results/ -o kernel_shapes.csv
.venv/bin/python -m pytest                                  # ~7 minutes on an M1
```

`configs/` has tp1, tp2, tp4, pp2, pp4 and hybrid_2x2, each × prefill / decode / spec_verify.
The first run downloads `EleutherAI/gpt-neo-125m` (~500 MB).

## Layout

```
reefdist/
  config.py        defaults, YAML loading, validation (ConfigError with the field name)
  registry.py      string -> class registries; auto-imports every module in the plugin packages
  model/           ModelSpec (shapes from the HF config); gpt_neo.py = single-device reference forward
  weights/         HF checkpoint -> canonical weight dict; shard.py executes a plan's weight split
  kernels/         KernelBackend, TorchBackend, cpp stub, CountingKernels, TimingKernels
  comm/            CommBackend, LocalGlooBackend, reef_link stub, CountingComm, TimingComm
  strategies/      ShardingStrategy, megatron.py (shared tp x pp engine), TensorParallel,
                   PipelineParallel, Hybrid
  runtime/         launcher (processes), worker, generation loop, KV cache, prompts, HF reference
  perfmodel/       FLOP conventions, ComputeModel (ConstantEfficiency, KernelTable stub), modeled outputs
  metrics/         tracer + records, exact aggregation, laptop timing, machine info, JSON/console report
scripts/           compare_results.py, sweep_hardware.py, export_kernel_shapes.py
tests/             one file per milestone (+ architecture rules)
run.py             python run.py configs/tp4_prefill.yaml
```

## GPT Neo, re-implemented

The forward pass (`reefdist/model/gpt_neo.py` and `reefdist/strategies/megatron.py`) is
written from HF transformers **v4.26.1** `modeling_gpt_neo.py`; `transformers` is used only
to load weights/config and as the correctness reference. Verified against that source:

* attention scores are **not** scaled by 1/sqrt(head_dim); Q·Kᵀ and softmax are in FP32; masked
  entries are set to `finfo(fp32).min` with `torch.where` (as HF), then softmax
* Q/K/V projections have no bias; `out_proj` has one
* learned absolute position embeddings (`wte(ids) + wpe(pos)`), no embedding scaling
* LM head tied to `wte`, no bias
* layers alternate global / local; local = causal sliding window: query *i* sees keys *j* with
  `i - j < 256` (256 keys including itself; HF builds it as `tril ^ tril(-window)`)
* MLP `c_fc -> gelu_new -> c_proj` (both biased), `gelu_new` = tanh approximation, d_ff = 4·d_model
* pre-LN blocks, LayerNorm eps 1e-5, final `ln_f`
* like HF, local layers keep their full KV history; the mask enforces the window

The single-device reference matches HF logits **bit-for-bit** (max diff 0.0) for 40-, 300-
and 700-token prompts.

## Architecture

Four components, each chosen by name in the YAML and resolved through `reefdist/registry.py`:

| config key | interface | implemented | stub |
|---|---|---|---|
| `kernels` | `KernelBackend` (`kernels/base.py`) | `torch` | `cpp` (pybind11 plan in `kernels/cpp_backend.py`) |
| `comm` | `CommBackend` (`comm/base.py`) | `local_gloo` | `reef_link` |
| `strategy` | `ShardingStrategy` (`strategies/base.py`) | `tensor_parallel`, `pipeline_parallel`, `hybrid` | |
| `hardware_profile.compute_model` | `ComputeModel` (`perfmodel/compute_model.py`) | `constant` | `table` |

**Hard rule:** a strategy's `forward` only calls `kernels.*` and `comm.*`; `tests/test_architecture.py`
fails if any strategy module mentions `torch`. Layout ops (split/merge heads, take last token,
position ids) are `KernelBackend` methods too, so an NPU backend controls its own memory layout.

### Sharding strategies

All three share `strategies/megatron.py`: `pp` pipeline stages of `tp` NPUs each
(`npu = stage * tp + tp_rank`). `TensorParallel` is `tp = N, pp = 1`; `PipelineParallel` is
`tp = 1, pp = N`; `Hybrid` takes `strategy_args: {tp, pp}`.

Within a stage (Megatron-style):
* W_Q, W_K, W_V and W_in are split by **columns** (output features), grouped by head; each
  NPU computes whole heads and caches K/V **only for its own heads**.
* W_O and W_out are split by **rows**; one **all-reduce after attention and one after the
  MLP**. Their biases live on tp_rank 0 and are added before the all-reduce (same sum, no extra op).
* LayerNorms and residual adds run on every NPU of the group (marked `kernels.replicated()`).
* Embedding/LM head are **vocab-parallel** (each NPU holds 1/tp of `wte`'s rows): embedding is a
  masked lookup + **one all-reduce per pass** (tp_rank 0 also adds `wpe` before it); logits are
  a local matmul + an all-gather. This keeps the 154 MB tied table from being replicated.
* `fused_qkv: true` stores per NPU `[Q heads | K heads | V heads]` of its own heads as one
  matrix; one matmul replaces three.

Across stages: contiguous layer blocks (12 layers / 4 stages = 3 each); tp_rank r of stage s
sends its hidden state to tp_rank r of stage s+1. The last stage keeps a replica of its `wte`
shard for the tied LM head. In decode, the last stage sends each sampled token back to stage 0
(counted under layer `"sample"`).

`plan()` returns plain data (it is echoed as `exact.plan`): stages, per-NPU heads/columns/vocab
ranges, every weight tensor with its global/local shape, split dim, range and owner flag, comm
groups, and every collective per layer with `bytes_per_token`, `messages_per_call` and
`bytes_sent_per_token_per_npu`.

## Counting (the source of all exact and modeled numbers)

`CountingKernels` / `CountingComm` wrap the backends and record every call from shapes
(never from timers), tagged with pass, NPU and layer (`embed`, 0..L-1, `head`, `sample`).
Byte counts use the configured `dtype` (fp16 halves them; the laptop still computes in FP32).
Conventions live in `perfmodel/flops.py` and the wrapper docstrings:

* **matmul** `[M×K]·[K×N]`: `2·M·K·N` FLOPs; reads input activations + weight, writes output.
  A bias is a separate `bias_add` record (M·N FLOPs, reads the bias).
* **attention** (one fused kernel, three records): `attn_scores` q·Kᵀ (`2·Tq·ctx·hd` per head,
  reads q and the K context), `softmax` (1 FLOP/element), `attn_pv` probs·V (`2·Tq·ctx·hd`,
  reads V, writes output and the appended K/V). Records keep heads, query tokens, context length
  and head_dim. Context = all cached tokens for global layers, `min(ctx, Tq + 255)` for local
  ones; the score rectangle is dense (masked entries are not skipped).
* **elementwise** (layernorm, gelu, add, softmax, bias_add): 1 FLOP per element.
* **embedding**: 0 FLOPs; reads and writes the gathered rows.
* **comm** (ring algorithm, S bytes over n NPUs, per NPU): all-reduce = 2(n−1) messages of S/n;
  all-gather / reduce-scatter / all-to-all = (n−1) of S/n; send = 1 of S; recv sends nothing.
  Size-1 groups send nothing.

Replicated work (LayerNorms/residuals in a TP group) is counted where it executes (`flops`) and
once in `unique_flops` (owned by tp_rank 0). Replicated weights (LayerNorm params, the last
stage's `wte` replica) are in `weight_bytes`; `unique_weight_bytes` counts each tensor once.

## Modeled Reef performance

From the exact counts and `hardware_profile` (`perfmodel/modeled.py`), per NPU:

```
compute = Σ compute_model.price(op, M, K, N, dtype, batch) seconds × count
memory  = bytes_read / (npu_mem_bandwidth_GBps × 1e9)
comm    = messages × link_alpha_us × 1e-6 + bytes_sent / (link_beta_GBps × 1e9)
step    = max(compute, memory) + comm        (overlap_comm: false)
        = max(compute, memory, comm)         (overlap_comm: true)
```

A pass's step = Σ over pipeline stages of the slowest NPU's step in that stage (pure TP: the
slowest NPU; PP: the sum of stage times, with stage-to-stage sends in the sender's comm time).
`bound_by` is the largest of compute / memory / comm.

* `per_pass`: prefill = all prefill chunks summed (= TTFT); decode = the **average** decode step;
  spec_verify = the verify pass. `per_layer` uses the same phase, reporting the critical NPU.
* `time_to_first_token_ms` = Σ prefill chunk steps; `per_token_ms` = decode step / verify pass
  (null for prefill); `tokens_per_second` = prompt tokens/s (prefill), batch/step (decode),
  batch·draft_tokens/step (spec_verify, all drafts accepted).
* `mfu` = useful FLOPs of the phase / (phase time × num_npus × peak).
* `modeled.pricing`: how many kernel records (all passes, all NPUs) each rule priced:
  `exact`, `interpolated`, `scaled`, `fallback` (`table`, see [Compute model](#compute-model)) or
  `constant`. With `compute_model: constant` every record is `constant`.
* `modeled.passes` lists every pass with its per-NPU breakdown.

## Compute model

`compute` in the modeled step time comes from a pluggable `ComputeModel`
(`perfmodel/compute_model.py`), chosen by `hardware_profile.compute_model`. It answers one
question: *how long does one kernel call of this shape take on one Reef NPU, ignoring memory
and communication?* Memory and link time are modeled separately (see above), so the compute
model must not include them.

```python
class ComputeModel:
    def price(self, op, M, K, N, dtype, batch=1) -> tuple[float, str] | None: ...  # seconds, rule
```

`price` returns the time for one call and the rule that priced it (`constant` for
`ConstantEfficiency`; see the lookup rules below for `table`). `op="attention"` asks for a whole
fused attention call (`K = head_dim`, `N = ctx`, the same convention as the table's `attention`
rows); it is the only op that may return None, meaning "price `attn_scores`, `softmax` and
`attn_pv` separately". `ConstantEfficiency` always returns None for it.

It receives the **per-NPU shapes that were actually executed**, after sharding. It never splits
anything itself: on 4-NPU TP, W_in is priced as `[M×768]·[768×768]`, not `[M×768]·[768×3072] / 4`.
That is what lets measured data capture the effect of smaller per-NPU kernels.

| `compute_model` | status | what it assumes |
|---|---|---|
| `constant` | implemented | every op runs at a fixed fraction of peak, regardless of shape |
| `table` | stub (`NotImplementedError`) | per-shape times measured by the kernel team |

### `constant`: fixed efficiency

```
time = FLOPs / (npu_peak_gflops × 1e9 × matmul_efficiency)
```

```yaml
hardware_profile:
  compute_model: constant
  matmul_efficiency: 0.5    # 0.5 = every op runs at half of peak
```

Simple and deterministic, but a placeholder. It is wrong in predictable ways:

* **Ignores shape.** A decode matmul with M = 1 can't fill a matrix unit built for many rows, so
  real efficiency is far below 0.5; a 512-token prefill matmul can approach peak. Both get 0.5.
* **Ignores per-call overhead.** Kernel launch/setup/sync is a fixed cost per call; at batch 1 it
  can dominate small ops (layernorm, bias add, residual add).
* **Applies to elementwise ops too**, which on real hardware are rarely limited by FLOPs.

How much this matters depends on the regime: compute only sets the step time when an NPU is
compute-bound. Batch-1 decode is memory-bound, so `constant` rarely changes its answer (with the
placeholder profile, the TP4-vs-1-NPU decode crossover is the same for every efficiency from 0.1
to 0.8); prefill and spec_verify are where it matters. Use `scripts/sweep_hardware.py` across a
range of `matmul_efficiency` to check whether a conclusion depends on it:

```bash
python scripts/sweep_hardware.py results/tp1_prefill.json results/tp4_prefill.json \
    --crossover-vs matmul_efficiency
```

### `table`: measured kernel times (planned)

```yaml
hardware_profile:
  compute_model: table
  kernel_table: profiles/reef_fp16.csv
```

The kernel team times their kernels on the closest thing to real Reef (RTL/cycle simulator,
FPGA, then silicon) and delivers two files.

**`<name>.csv`**: one row per measured kernel shape:

| column | meaning |
|---|---|
| `op` | `matmul`, `attn_scores`, `attn_pv`, `softmax`, `layernorm`, `gelu`, `add`, `bias_add`, `embedding`, and optionally `attention` (one fused scores + softmax + probs·V call, keyed like `attn_scores`: `K = head_dim`, `N = ctx`) |
| `batch` | 1 for matmuls and elementwise ops; `batch_size × heads on this NPU` for attention ops |
| `M` | rows: `batch_size × tokens in the pass` (for attention, query tokens per head) |
| `K` | inner dimension of a GEMM; **1** for elementwise ops |
| `N` | output dimension of a GEMM; feature width (or `ctx` for softmax) for elementwise ops |
| `dtype` | `fp16`, `int8`, … |
| `time_us_warm` | median time with all operands already in on-chip SRAM: **pure compute, used by the model** |
| `time_us_cold` | median time with weights starting in DRAM: **used to validate the model** |
| `time_us_p90` | 90th percentile of the warm time |
| `cycles` | cycle count, if the platform reports one |
| `runs` | measurements behind the median |

**`<name>_meta.yaml`**: what the numbers were measured on, so tables are never mixed across
hardware or kernel versions:

```yaml
platform: rtl_sim          # rtl_sim | cycle_sim | fpga | silicon
platform_version: "..."
clock_mhz: 500
kernel_version: "<git sha>"
date: 2026-10-20
```

**Why warm, not cold.** The modeled step is `max(compute, memory) + comm`, with memory already
computed from bytes ÷ DRAM bandwidth. A cold time already includes streaming weights from
DRAM, so using it as `compute` would count memory twice. Cold is still worth collecting: for
each shape, compare `max(time_us_warm, bytes_read / bandwidth)` with `time_us_cold`, where
`bytes_read` is the same per-call figure the counting wrapper records. If they agree, the
roofline assumption holds; if cold is consistently higher, compute and memory don't overlap as
well as the model assumes.

**dtype must match the run.** Lookups key on `dtype`, and every record carries the config's
`dtype` (default `fp32`). Price against an fp16 table with `dtype: fp16` in the config (it only
changes byte counts; the laptop still computes in FP32). `int8` is reserved and rejected by
config validation today, so int8 rows can't be used until that dtype is enabled.

#### How each op maps to (batch, M, K, N)

The table must use the same shape convention as the counting records (`perfmodel/flops.py`,
`kernels/counting.py`): GEMMs are `batch` independent `[M×K]·[K×N]` products; elementwise ops
are `batch × M × N` elements with `K = 1`. For GPT Neo 125M (d_model 768, d_ff 3072, 12 heads,
head_dim 64), batch_size 1, with `M` = tokens in the pass and `ctx` = context the kernel reads
(all cached tokens on global layers, `min(ctx, M + 255)` on local layers, so 256 in decode but
up to `M + 255` in a prefill chunk):

| op | batch | M | K | N | example on 4-NPU TP, decode |
|---|---|---|---|---|---|
| `matmul` (Q, K or V) | 1 | tokens | 768 | 768 / tp | `1, 1, 768, 192` |
| `matmul` (fused QKV, `fused_qkv: true`) | 1 | tokens | 768 | 3·768 / tp | `1, 1, 768, 576` |
| `matmul` (W_O) | 1 | tokens | 768 / tp | 768 | `1, 1, 192, 768` |
| `matmul` (W_in) | 1 | tokens | 768 | 3072 / tp | `1, 1, 768, 768` |
| `matmul` (W_out) | 1 | tokens | 3072 / tp | 768 | `1, 1, 768, 768` |
| `matmul` (LM head) | 1 | tokens | 768 | vocab shard | `1, 1, 768, 12565` (`exact.plan.vocab_shard_sizes`) |
| `attn_scores` (q·Kᵀ) | heads / tp | query tokens | 64 | ctx | `3, 1, 64, ctx` |
| `attn_pv` (probs·V) | heads / tp | query tokens | ctx | 64 | `3, 1, ctx, 64` |
| `softmax` | heads / tp | query tokens | 1 | ctx | `3, 1, 1, ctx` |
| `attention` (fused, optional) | heads / tp | query tokens | 64 | ctx | `3, 1, 64, ctx` |
| `layernorm`, `add` | 1 | tokens | 1 | 768 | `1, 1, 1, 768` |
| `gelu` | 1 | tokens | 1 | 3072 / tp | `1, 1, 1, 768` |
| `bias_add` | 1 | tokens | 1 | output width | `1, 1, 1, 768` |
| `embedding` | 1 | tokens | 1 | 768 | 0 FLOPs; memory only |

With `batch_size > 1`, multiply `M` by it for matmuls and elementwise ops, and `batch` by it for
attention ops.

#### Which shapes to measure

`scripts/export_kernel_shapes.py results/ -o kernel_shapes.csv` lists every per-NPU
shape the runs used, GEMMs and elementwise ops alike, with columns `op, batch, M, K, N, dtype` in
the convention above. Measure a **grid around it** so new prompt lengths, chunk sizes and draft counts can be interpolated without re-profiling:

* **M**: 1, 2, 4, 8, 16 (decode, spec_verify) and 32, 64, 128, 256, 512, 1024, 2048 (prefill, chunks)
* **(K, N)** for tp = 1, 2 and 4, i.e. the full shapes and their halves and quarters:

  | weight | tp = 1 | tp = 2 | tp = 4 |
  |---|---|---|---|
  | Q, K or V | 768 × 768 | 768 × 384 | 768 × 192 |
  | fused QKV | 768 × 2304 | 768 × 1152 | 768 × 576 |
  | W_O | 768 × 768 | 384 × 768 | 192 × 768 |
  | W_in | 768 × 3072 | 768 × 1536 | 768 × 768 |
  | W_out | 3072 × 768 | 1536 × 768 | 768 × 768 |
  | LM head | 768 × 50257 | 768 × 25129 and 768 × 25128 | 768 × 12565 and 768 × 12564 |

  Vocab (50257) doesn't divide evenly, so the LM head shards differ by one column
  (`exact.plan.vocab_shard_sizes`). If only one of each pair is measured, lookup rule 3
  prices the other.

* **Attention**: batch = 12, 6, 3; head_dim 64; ctx = 64, 128, 256, 512, 1024, 2048 (the three
  separate ops, plus fused `attention` rows at the same shapes if the kernel team has one)
* **Elementwise**: widths 768, 1536, 3072 (and ctx values for softmax)
* **dtype**: whatever Reef supports (likely fp16 and int8)

The sharded shapes and small M values matter most: they are where efficiency is lowest and
hardest to predict, and they decide whether splitting across NPUs pays off.

#### Lookup rules (to implement in `KernelTable`)

1. **Exact match** on `(op, batch, M, K, N, dtype)`: use `time_us_warm`.
2. **Interpolate on the dimensions that vary per run.**
   - Matmuls and elementwise ops: fixed key `(op, batch, K, N, dtype)`, interpolate on M
     (log scale).
   - Attention ops (`attn_scores`, `attn_pv`, `softmax`, and fused `attention`): fixed key
     `(op, batch, head_dim, dtype)`, interpolate on M and ctx together (bilinear, log scales).
     ctx is N for `attn_scores`, `softmax` and `attention`, K for `attn_pv`. ctx grows by one
     every decode step (the runs use 513, 514, 515, …), so it almost never matches exactly.
3. **Near match on a fixed dimension (matmuls only):** if no rows exist for the exact `K` or `N`
   but a measured value is within ±1%, use the nearest one and scale the time linearly
   (`time × N_actual / N_measured`, same for K). Mark the op `scaled`. This covers uneven
   vocab shards (e.g. 12,565 vs 12,564 on 4-NPU TP).
4. **No usable rows:** fall back to `constant`, print a warning, and mark the op `fallback`.

**Fused attention.** Each attention call is counted as three records (`attn_scores`,
`softmax`, `attn_pv`). If a fused `attention` row matches the call, exactly or by
interpolation (rules 1–2), the call is priced from it and that time is split across the three
records in proportion to their FLOPs; otherwise the three records are priced separately.

Every modeled op records which rule priced it (`exact`, `interpolated`, `scaled`, `fallback`;
a fused split gives all three records the fused row's rule), and `modeled.pricing` counts
them per run, so it is obvious which estimates rest on real data.

## Workloads

* `prefill`: the prompt in one pass, or `ceil(prompt_len / prefill_chunk)` chunks.
* `decode`: prefill, then `gen_tokens − 1` single-token passes (token 1 comes from prefill), greedy.
* `spec_verify`: prefill, then one pass over `draft_tokens` tokens against the prompt's KV cache.
  The draft is HF's own greedy continuation (a perfect draft model), so every draft is accepted.

Prompts are a fixed English passage cycled to `prompt_len` (`runtime/prompts.py`; `seed` picks
the offsets for batch rows). Real text keeps greedy comparisons from hinging on near-tied logits.

## Output JSON

`config` (echo), `machine`, and:

* `exact.correctness`: `max_abs_logit_diff_vs_hf`, `tokens_match_hf` (prefill logits of the
  last token, every decode step, the verify pass).
* `exact.phase`: which phase `per_layer`, `per_npu` and `communication_totals` cover
  (decode: summed over all decode steps). `exact.by_phase` totals every phase.
* `exact.per_layer`, `exact.per_npu` (`weight_bytes`, `unique_weight_bytes`, `kv_cache_bytes` at the
  end of the run, `peak_activation_bytes` = largest single-op activation footprint, `flops`,
  `unique_flops`, `redundant_flops`, bytes, messages, `kernel_calls`).
* `exact.kernel_shapes`: **the list to hand to the kernel team.** Every distinct per-NPU kernel
  shape (matmuls, attention ops, softmax, layernorm, gelu, add, bias_add, embedding) as
  `op, batch, M, K, N, dtype` in the [Compute model](#compute-model) convention, with `count`
  (calls, all passes and NPUs) and `names` (the layer ops that used it). Attention is one record
  with `batch = batch_size × heads on this NPU`, not single-head GEMMs.
  `scripts/export_kernel_shapes.py` writes it to CSV.
* `exact.communication_totals.by_collective`: `{calls, num_messages, bytes_sent}` per collective type.
* `exact.plan`, `exact.weights` (model vs summed weight bytes), `exact.pass_profiles` (the
  per-pass, per-NPU shape counts that `sweep_hardware.py` re-prices).
* `modeled.*` as above; `measured_laptop.*`: warmup/timed runs, median/p90 total, per-layer
  kernel and comm wall time (median over runs, slowest NPU).

## Extending: one file each

New modules in the plugin packages are imported automatically; registering a name is enough.

**Strategy**, `reefdist/strategies/my_strategy.py`:
```python
@STRATEGIES.register("my_strategy")
class MyStrategy(ShardingStrategy):
    @classmethod
    def validate_args(cls, strategy_args, num_npus, spec): ...   # raise ConfigError
    def plan(self, model_spec, num_npus): ...                    # plain dict, see strategies/base.py
    def shard_weights(self, full_weights, plan): ...             # reefdist.weights.shard.shard_by_plan helps
    def forward(self, npu_id, x, kv_cache, kernels, comm): ...   # kernels.* and comm.* only
```
Wrap work in `kernels.scope(layer)` so it is attributed per layer, and replicated work in
`kernels.replicated()`. Plans need `stages`, `npus`, `comm_groups`, `input_npus`, `logits_npus`,
`token_routes` and `weights` (see `strategies/base.py`).

**Kernel backend**, `reefdist/kernels/my_backend.py`: subclass `KernelBackend`, implement its
methods, `@KERNELS.register("my_backend")`, set `kernels: my_backend`. Validate with
`tests/test_m0_reference.py`. Counting doesn't change: it comes from shapes.

**Comm backend**, `reefdist/comm/my_link.py`: subclass `CommBackend` with constructor
`(rank, world_size, groups, port, timeout_s)`, `@COMMS.register("my_link")`. No artificial delays.

**Compute model**, `reefdist/perfmodel/my_model.py`: subclass `ComputeModel`, implement
`price(op, M, K, N, dtype, batch=1) -> (seconds, rule)` (None allowed only for `op="attention"`), `@COMPUTE_MODELS.register("my_model")`,
set `hardware_profile.compute_model: my_model`.

## When real Reef numbers arrive

* **Chip specs**: edit `hardware_profile` (peak GFLOP/s, DRAM GB/s, link α/β, `overlap_comm`).
  Old runs can be re-priced without re-running: `scripts/sweep_hardware.py` reads
  `exact.pass_profiles`.
* **Kernel-team profiling**: `scripts/export_kernel_shapes.py results/ -o kernel_shapes.csv`
  gives every kernel shape the runs used; the kernel team measures a grid around it (see
  [Compute model](#compute-model) for the CSV format, shape convention and lookup rules). Then
  implement `KernelTable` in `perfmodel/compute_model.py` and set `compute_model: table`,
  `kernel_table: <csv>`. Until then, selecting `table` raises `NotImplementedError`.
* **Real kernels / link**: implement `CppBackend` / `ReefLinkBackend` (stubs document the plan).

## Tests (per milestone)

| file | milestone | checks |
|---|---|---|
| `test_m0_reference.py` | 0 | GPT Neo details vs HF source; unscaled FP32 attention and 256-key window; logits ≤ 1e-4 vs HF (actually 0.0) and 32 greedy tokens identical, for prompts of 40 and 300 tokens; batch of 2 |
| `test_m1_tensor_parallel.py` | 1 | TP n = 1, 2, 4 × both prompts: tokens identical to milestone 0; n = 1 logits bit-identical (prefill and every decode step); heads and KV cache split per NPU |
| `test_m2_counting.py` | 2 | unique FLOPs summed over NPUs = 1-NPU FLOPs; per pass 2×layers all-reduces of B·T·d_model·4 bytes (+1 for the embedding); unique weight bytes sum = model bytes; matmul K or N divided by n; byte-identical `exact`/`modeled` on rerun; hand-calculated 1-NPU decode step; `table` raises `NotImplementedError`; config validation |
| `test_m3_workloads.py` | 3 | chunked prefill = unchunked logits (reference and TP2); decode and spec_verify pass structure, tokens vs HF |
| `test_m4_pipeline_hybrid.py` | 4 | PP2, PP4, hybrid 2×2 against milestone 0 (PP bit-identical); FLOPs/weights/all-reduce/send counts; pass step = Σ stages; token feedback |
| `test_m5_fused_qkv.py` | 5 | fused = unfused tokens and logits (≤ 1e-4), identical FLOPs, 2 fewer kernel calls per layer per pass; fused column order keeps each NPU's heads |
| `test_m6_measurement_scripts.py` | 6 | measured/machine sections, all comm ops and ring counts on 2 processes, the three scripts |
| `test_architecture.py` | – | strategies never use torch; registries; stubs raise |

## Known deviations and limits

* **Distributed logits are not bit-identical to one device.** Row-parallel all-reduces sum partial
  products in a different order. With TP the logits drift by up to ~6e-4 (|logit| ≤ ~65, so ~1e-5
  relative); greedy tokens are identical in every test. Distributed tests use `atol = 1e-3`
  (`tests/conftest.py: DIST_ATOL`); n = 1 TP and pure PP are bit-identical.
* **TP has 2×layers + 1 all-reduces per pass.** The extra one is the vocab-parallel embedding;
  the alternative is replicating the 154 MB `wte` on every NPU.
* Sharding does change *executed* FLOPs slightly (replicated LayerNorms and residual adds, <0.5%),
  so the "sum of FLOPs = 1-NPU FLOPs" check uses `unique_flops`. Weight sums likewise use
  `unique_weight_bytes` (LayerNorm params are replicated in TP; PP's last stage holds a `wte` replica).
* `LocalGlooBackend` splits large messages into ≤1 MB pieces because gloo's libuv transport on
  macOS aborts intermittently ("Unexpected opcode") on multi-MB all-gathers. This only affects
  laptop transport; counts are recorded above the backend.
* Thread pinning to cores uses `sched_setaffinity` and so only works on Linux; macOS gets
  `torch.set_num_threads` only (`machine.threads_pinned_to_cores` says which).
* `dtype: fp16` affects byte counts only; math runs in FP32 on the laptop. `parallel_block` is
  reserved and rejected if true. `int8`/`int4` are rejected as not yet supported.
