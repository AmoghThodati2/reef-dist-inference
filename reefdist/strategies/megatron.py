"""Shared engine for TensorParallel, PipelineParallel and Hybrid: tp x pp Megatron-style layout.

NPUs are arranged as ``pp`` pipeline stages of ``tp`` NPUs each (``npu = stage * tp + tp_rank``).
Each stage owns a contiguous block of layers; inside a stage the layers are split
Megatron-style across the stage's tensor-parallel group:

* W_Q, W_K, W_V and W_in are split by columns (output features), grouped by head, so each
  NPU computes whole heads and keeps the KV cache only for its own heads.
* W_O and W_out are split by rows (input features); their partial outputs are summed by an
  all-reduce after attention and after the MLP. Their biases live on tp_rank 0 and are added
  before the all-reduce (same sum, no extra op).
* LayerNorms and residual adds are replicated in the group (marked ``kernels.replicated()``).
* The token embedding / tied LM head is vocab-parallel: each NPU holds a block of vocab rows.
  Embedding = masked lookup + all-reduce (tp_rank 0 also adds the position embedding before
  it); logits = local matmul + all-gather.
* Between stages, tp_rank r of stage s sends its full hidden state to tp_rank r of stage s+1.
  The last stage holds a replica of its wte shard for the tied LM head.

Weights use nn.Linear layout [out, in]: "columns" = dim 0, "rows" = dim 1.
"""
from ..errors import ConfigError
from ..weights.shard import shard_by_plan
from .base import ShardingStrategy, StepInput


def _partition(num_layers, pp):
    base, extra = divmod(num_layers, pp)
    out, start = [], 0
    for s in range(pp):
        n = base + (1 if s < extra else 0)
        out.append(list(range(start, start + n)))
        start += n
    return out


def _ring(op, n):
    """(messages per call, fraction of S sent per call) for one NPU."""
    if n == 1:
        return 0, 0.0
    if op == "all_reduce":
        return 2 * (n - 1), 2 * (n - 1) / n
    if op in ("all_gather", "reduce_scatter", "all_to_all"):
        return n - 1, (n - 1) / n
    return 1, 1.0  # send


class MegatronStages(ShardingStrategy):
    def degrees(self, num_npus):
        """Return (tp, pp)."""
        raise NotImplementedError

    @staticmethod
    def check(spec, tp, pp, num_npus):
        if tp * pp != num_npus:
            raise ConfigError(f"tp ({tp}) x pp ({pp}) must equal num_npus ({num_npus})")
        if spec.num_heads % tp:
            raise ConfigError(f"tensor-parallel degree {tp} must divide num_heads ({spec.num_heads})")
        if spec.d_ff % tp:
            raise ConfigError(f"tensor-parallel degree {tp} must divide d_ff ({spec.d_ff})")
        if pp > spec.num_layers:
            raise ConfigError(f"pipeline degree {pp} exceeds num_layers ({spec.num_layers})")

    # ---- plan ---------------------------------------------------------------------------
    def plan(self, model_spec, num_npus):
        spec = model_spec
        tp, pp = self.degrees(num_npus)
        self.check(spec, tp, pp, num_npus)
        D, F, H, hd, V = spec.d_model, spec.d_ff, spec.num_heads, spec.head_dim, spec.vocab_size
        db = self.dtype_bytes
        hp, fp = H // tp, F // tp
        cols = hp * hd
        vs = -(-V // tp)
        vocab = [[r * vs, min(V, (r + 1) * vs)] for r in range(tp)]
        layer_split = _partition(spec.num_layers, pp)

        stages, npus = [], []
        for s in range(pp):
            ids = [s * tp + r for r in range(tp)]
            stages.append({"stage": s, "npus": ids, "layers": layer_split[s],
                           "has_embedding": s == 0, "has_lm_head": s == pp - 1})
            for r in range(tp):
                npus.append({
                    "npu_id": s * tp + r, "stage": s, "tp_rank": r, "tp_group": ids,
                    "layers": layer_split[s],
                    "heads": [r * hp, (r + 1) * hp],
                    "qkv_cols": [r * cols, (r + 1) * cols],
                    "ff_cols": [r * fp, (r + 1) * fp],
                    "vocab_range": vocab[r],
                    "prev_npu": (s - 1) * tp + r if s > 0 else None,
                    "next_npu": (s + 1) * tp + r if s < pp - 1 else None,
                    "is_first": s == 0, "is_last": s == pp - 1,
                    "kv_cache_bytes_per_token": len(layer_split[s]) * 2 * cols * db,
                })

        def coll(layer, op, after, elements, group=None, src=None, dst=None):
            n = len(group) if group else 2
            msgs, frac = _ring(op, n)
            e = {"layer": layer, "op": op, "after": after, "group_size": n,
                 "elements_per_token": elements, "bytes_per_token": elements * db,
                 "messages_per_call": msgs, "bytes_sent_per_token_per_npu": elements * db * frac}
            if group:
                e["group"] = group
            if src is not None:
                e.update(src=src, dst=dst)
            return e

        collectives = []
        for s, st in enumerate(stages):
            g = st["npus"]
            if s == 0:
                collectives.append(coll("embed", "all_reduce", "vocab-parallel token embedding (+ position embedding on tp_rank 0)", D, g))
            for i in st["layers"]:
                collectives.append(coll(i, "all_reduce", "attn.out_proj (row-parallel)", D, g))
                collectives.append(coll(i, "all_reduce", "mlp.proj (row-parallel)", D, g))
            if s < pp - 1:
                for r in range(tp):
                    collectives.append(coll(st["layers"][-1], "send", "stage output -> next stage", D,
                                            src=g[r], dst=stages[s + 1]["npus"][r]))
            else:
                collectives.append(coll("head", "all_gather", "vocab-parallel lm_head logits (per logit row)", V, g))

        token_routes = []
        if pp > 1:
            token_routes = [[stages[-1]["npus"][r], stages[0]["npus"][r]] for r in range(tp)]

        plan = {
            "strategy": self.registry_name, "num_npus": num_npus, "tp": tp, "pp": pp,
            "fused_qkv": self.fused_qkv,
            "weight_layout": "[out_features, in_features]; 'columns' = dim 0 (output features), 'rows' = dim 1 (input features)",
            "stages": stages,
            "npus": npus,
            "comm_groups": [st["npus"] for st in stages],
            "input_npus": stages[0]["npus"],
            "logits_npus": stages[-1]["npus"],
            "token_routes": token_routes,
            "vocab_shard_sizes": [b - a for a, b in vocab],
            "collectives": collectives,
            "token_feedback": [coll("sample", "send", "next token ids -> first stage (between passes)", 1, src=a, dst=b)
                               for a, b in token_routes],
        }
        plan["weights"] = [{"npu_id": n["npu_id"], "tensors": self._weight_entries(spec, n, tp)} for n in npus]
        return plan

    def _weight_entries(self, spec, me, tp):
        D, F, V, P = spec.d_model, spec.d_ff, spec.vocab_size, spec.max_positions
        r, first, last = me["tp_rank"], me["is_first"], me["is_last"]
        rep = "replicated" if tp > 1 else "none"
        out = []

        def add(name, sources, global_shape, dim=None, rng=None, split="none", owner=True):
            shape = list(global_shape)
            if dim is not None:
                shape[dim] = (rng[1] - rng[0]) * len(sources)
            out.append({"name": name, "sources": sources, "global_shape": list(global_shape), "shape": shape,
                        "split": split, "dim": dim, "range": rng, "owner": owner})

        vsplit = "vocab rows" if tp > 1 else "none"
        if first:
            add("wte", ["wte"], [V, D], 0, me["vocab_range"], vsplit)
            if r == 0:
                add("wpe", ["wpe"], [P, D], split="tp_rank 0 only")
        for i in me["layers"]:
            p = f"h.{i}."
            for ln in ("ln_1", "ln_2"):
                add(p + ln + ".w", [p + ln + ".w"], [D], split=rep, owner=r == 0)
                add(p + ln + ".b", [p + ln + ".b"], [D], split=rep, owner=r == 0)
            qkv = me["qkv_cols"]
            if self.fused_qkv:
                add(p + "attn.qkv", [p + "attn.q", p + "attn.k", p + "attn.v"], [3 * D, D], 0, qkv,
                    "fused columns: [q | k | v] for this NPU's heads")
            else:
                for x in ("q", "k", "v"):
                    add(p + "attn." + x, [p + "attn." + x], [D, D], 0, qkv, "columns (by head)")
            add(p + "attn.o", [p + "attn.o"], [D, D], 1, qkv, "rows (by head)")
            if r == 0:
                add(p + "attn.o_b", [p + "attn.o_b"], [D], split="tp_rank 0 only")
            ff = me["ff_cols"]
            add(p + "mlp.fc", [p + "mlp.fc"], [F, D], 0, ff, "columns")
            add(p + "mlp.fc_b", [p + "mlp.fc_b"], [F], 0, ff, "columns")
            add(p + "mlp.proj", [p + "mlp.proj"], [D, F], 1, ff, "rows")
            if r == 0:
                add(p + "mlp.proj_b", [p + "mlp.proj_b"], [D], split="tp_rank 0 only")
        if last:
            add("ln_f.w", ["ln_f.w"], [D], split=rep, owner=r == 0)
            add("ln_f.b", ["ln_f.b"], [D], split=rep, owner=r == 0)
            if not first:
                add("wte", ["wte"], [V, D], 0, me["vocab_range"],
                    "vocab rows (replica of the stage-0 shard for the tied lm_head)", owner=False)
        return out

    # ---- weights ------------------------------------------------------------------------
    def shard_weights(self, full_weights, plan):
        return shard_by_plan(full_weights, plan)

    # ---- forward ------------------------------------------------------------------------
    def forward(self, npu_id, x: StepInput, kv_cache, kernels, comm):
        plan, spec, W = self.plan_data, self.spec, self.weights
        me = plan["npus"][npu_id]
        group = tuple(me["tp_group"])

        if me["is_first"]:
            with kernels.scope("embed"):
                h = kernels.embedding(x.token_ids, W["wte"], vocab_start=me["vocab_range"][0])
                if me["tp_rank"] == 0:
                    pos = kernels.embedding(kernels.positions(kv_cache.seq_len, x.num_tokens, x.batch), W["wpe"])
                    h = kernels.add(h, pos)
                h = comm.all_reduce(h, group)
        else:
            with kernels.scope(me["layers"][0]):
                h = comm.recv((x.batch, x.num_tokens, spec.d_model), kernels.activation_dtype, me["prev_npu"])

        for i in me["layers"]:
            with kernels.scope(i):
                h = self._block(i, h, kv_cache, kernels, comm, group, me["tp_rank"] == 0)

        if not me["is_last"]:
            with kernels.scope(me["layers"][-1]):
                comm.send(h, me["next_npu"])
            return None

        with kernels.scope("head"):
            with kernels.replicated():
                if not x.all_logits:
                    h = kernels.last_token(h)
                h = kernels.layernorm(h, W["ln_f.w"], W["ln_f.b"], spec.layer_norm_eps)
            logits = kernels.matmul(h, W["wte"], name="lm_head")  # tied to the token embedding
            return comm.all_gather(logits, group, dim=-1, sizes=plan["vocab_shard_sizes"])

    def _block(self, i, h, kv_cache, kernels, comm, group, rank0):
        spec, W = self.spec, self.weights
        p = f"h.{i}."
        eps = spec.layer_norm_eps
        heads = spec.num_heads // len(group)

        with kernels.replicated():
            a = kernels.layernorm(h, W[p + "ln_1.w"], W[p + "ln_1.b"], eps)
        if self.fused_qkv:
            qkv = kernels.matmul(a, W[p + "attn.qkv"], name="qkv_proj")
            q, k, v = kernels.split_last(qkv, [heads * spec.head_dim] * 3)
        else:
            q = kernels.matmul(a, W[p + "attn.q"], name="q_proj")
            k = kernels.matmul(a, W[p + "attn.k"], name="k_proj")
            v = kernels.matmul(a, W[p + "attn.v"], name="v_proj")
        q, k, v = (kernels.split_heads(t, heads) for t in (q, k, v))
        o = kernels.attention(q, k, v, kv_cache, i, spec.window(i))
        o = kernels.merge_heads(o)
        o = kernels.matmul(o, W[p + "attn.o"], W[p + "attn.o_b"] if rank0 else None, name="out_proj")
        o = comm.all_reduce(o, group)
        with kernels.replicated():
            h = kernels.add(o, h)
            m = kernels.layernorm(h, W[p + "ln_2.w"], W[p + "ln_2.b"], eps)
        m = kernels.matmul(m, W[p + "mlp.fc"], W[p + "mlp.fc_b"], name="fc_in")
        m = kernels.gelu(m)
        m = kernels.matmul(m, W[p + "mlp.proj"], W[p + "mlp.proj_b"] if rank0 else None, name="fc_out")
        m = comm.all_reduce(m, group)
        with kernels.replicated():
            h = kernels.add(h, m)
        return h
