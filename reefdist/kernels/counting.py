"""CountingKernels: wraps a KernelBackend and records every math call from tensor shapes.

Nothing here reads a clock. Byte counts use the configured dtype (``dtype: fp16``
halves them even though the laptop computes in FP32).

Memory traffic model (per call):
  * matmul: reads input activations + weight, writes output; bias add is a separate
    ``bias_add`` record that reads the bias (fused epilogue, no extra activation traffic).
  * attention is modeled as one fused kernel split into three records:
    ``attn_scores`` (reads q and the K context), ``softmax`` (no traffic) and
    ``attn_pv`` (reads the V context, writes the output and appends the new K/V).
    The context length is what the kernel must read: all cached tokens for global
    layers, ``min(ctx, q_tokens + window - 1)`` for local layers. Score FLOPs are the
    dense ``q_tokens x context`` rectangle (masked entries are not skipped).
  * elementwise ops read their inputs and write their output.
  * embedding reads the gathered rows and writes them.
"""
from ..perfmodel.flops import op_flops
from ..metrics.tracer import KernelRecord


class CountingKernels:
    def __init__(self, inner, tracer, dtype_name: str, dtype_bytes: int):
        self.inner = inner
        self.tracer = tracer
        self.dtype = dtype_name
        self.db = dtype_bytes

    def __getattr__(self, item):  # layout ops and properties pass straight through
        return getattr(self.inner, item)

    def scope(self, layer):
        return self.tracer.scope(layer)

    def replicated(self):
        return self.tracer.replicated()

    def _rec(self, op, name, M, K, N, batch=1, read=0, written=0, act=0, extra=None):
        t = self.tracer
        t.kernel_records.append(KernelRecord(
            t.pass_id, t.layer, op, name, M, K, N, batch, self.dtype,
            op_flops(op, M, K, N, batch), read, written, act, t.is_replicated, extra,
        ))

    # ---- math -------------------------------------------------------------------------
    def matmul(self, x, w, bias=None, name="matmul"):
        K = x.shape[-1]
        M = x.numel() // K
        N = w.shape[0]
        db = self.db
        self._rec("matmul", name, M, K, N, read=(M * K + N * K) * db, written=M * N * db,
                  act=(M * K + M * N) * db)
        if bias is not None:
            self._rec("bias_add", name + "_bias", M, 1, N, read=N * db)
        return self.inner.matmul(x, w, bias, name)

    def attention(self, q, k, v, kv_cache, layer, window=None):
        B, H, Tq, hd = q.shape
        ctx = kv_cache.layer_len(layer) + Tq
        c = ctx if window is None else min(ctx, Tq + window - 1)
        db = self.db
        q_bytes = B * H * Tq * hd * db
        kv_bytes = B * H * c * hd * db  # K (or V) context read
        extra = {"heads": H, "q_tokens": Tq, "context_len": c, "head_dim": hd, "cached_len": ctx}
        self._rec("attn_scores", "attn_scores", Tq, hd, c, B * H, read=q_bytes + kv_bytes,
                  act=q_bytes, extra=extra)
        self._rec("softmax", "attn_softmax", Tq, 1, c, B * H)
        self._rec("attn_pv", "attn_pv", Tq, c, hd, B * H, read=kv_bytes,
                  written=q_bytes + 2 * q_bytes, act=q_bytes)  # output + new K and V
        return self.inner.attention(q, k, v, kv_cache, layer, window)

    def layernorm(self, x, weight, bias, eps):
        N = x.shape[-1]
        M = x.numel() // N
        db = self.db
        self._rec("layernorm", "layernorm", M, 1, N, read=(M * N + 2 * N) * db,
                  written=M * N * db, act=2 * M * N * db)
        return self.inner.layernorm(x, weight, bias, eps)

    def gelu(self, x):
        N = x.shape[-1]
        M = x.numel() // N
        b = M * N * self.db
        self._rec("gelu", "gelu", M, 1, N, read=b, written=b, act=2 * b)
        return self.inner.gelu(x)

    def softmax(self, x):
        N = x.shape[-1]
        M = x.numel() // N
        b = M * N * self.db
        self._rec("softmax", "softmax", M, 1, N, read=b, written=b, act=2 * b)
        return self.inner.softmax(x)

    def embedding(self, ids, table, vocab_start=0):
        M, N = ids.numel(), table.shape[-1]
        b = M * N * self.db
        self._rec("embedding", "embedding", M, 1, N, read=b, written=b, act=b)
        return self.inner.embedding(ids, table, vocab_start)

    def add(self, a, b):
        N = a.shape[-1]
        M = a.numel() // N
        n = M * N * self.db
        self._rec("add", "add", M, 1, N, read=2 * n, written=n, act=3 * n)
        return self.inner.add(a, b)
