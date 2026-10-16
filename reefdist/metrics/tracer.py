"""Tracer: shared state for the counting/timing wrappers of one NPU.

It knows which pass and layer is currently executing, so kernel and comm records can
be attributed per pass, per layer. Records are plain tuples (cheap to create and pickle).
"""
from collections import defaultdict
from contextlib import contextmanager
from typing import NamedTuple, Optional


class KernelRecord(NamedTuple):
    pass_id: int
    layer: object  # int layer index, or "embed" / "head" / "sample"
    op: str  # matmul, attn_scores, softmax, attn_pv, bias_add, layernorm, gelu, add, embedding
    name: str  # e.g. q_proj, lm_head
    M: int
    K: int
    N: int
    batch: int  # independent GEMMs of this shape (attention: batch * heads)
    dtype: str
    flops: int
    bytes_read: int
    bytes_written: int
    act_bytes: int  # activation bytes touched (in + out), for peak activation memory
    replicated: bool  # identical work on every NPU of the TP group
    extra: Optional[dict]  # attention: heads, q_tokens, context_len, head_dim


class CommRecord(NamedTuple):
    pass_id: int
    layer: object
    op: str  # all_reduce, all_gather, reduce_scatter, all_to_all, send, recv, barrier
    tensor_bytes: int  # S: size of the logical tensor
    group_size: int
    messages: int  # point-to-point messages sent by this NPU (ring algorithm)
    bytes_sent: int
    bytes_received: int
    peer: Optional[int]


class Tracer:
    def __init__(self):
        self.pass_id = -1
        self.layer = "other"
        self.is_replicated = False
        self.kernel_records = []
        self.comm_records = []
        # timing: (pass, layer) -> [kernel seconds, comm seconds]
        self.times = defaultdict(lambda: [0.0, 0.0])

    def begin_pass(self, pass_id: int):
        self.pass_id = pass_id
        self.layer = "other"

    @contextmanager
    def scope(self, layer):
        prev, self.layer = self.layer, layer
        try:
            yield
        finally:
            self.layer = prev

    @contextmanager
    def replicated(self):
        prev, self.is_replicated = self.is_replicated, True
        try:
            yield
        finally:
            self.is_replicated = prev


class NullTracer(Tracer):
    """Used when running without wrappers."""
