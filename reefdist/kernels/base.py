"""KernelBackend: every math operation a strategy may perform.

Strategies never call ``torch.*``; they call these methods, so a real NPU kernel
library can replace :class:`TorchBackend` without touching sharding logic.

Conventions
-----------
* Linear weights use nn.Linear layout ``w: [N, K]``; ``matmul(x, w)`` computes
  ``x[..., K] @ w.T -> [..., N]`` (the ``[M x K] . [K x N]`` GEMM with ``M = prod(x.shape[:-1])``).
* Attention tensors are ``[batch, heads, tokens, head_dim]``.
* "Layout" methods (split_heads, merge_heads, split_last, last_token, positions) move
  or reinterpret data but do no math; they are not counted.
* ``scope``/``replicated`` are tracing hooks; plain backends ignore them, the counting
  and timing wrappers use them to attribute work to a layer.
"""
from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Optional, Sequence


class KernelBackend(ABC):
    # ---- math -------------------------------------------------------------------------
    @abstractmethod
    def matmul(self, x, w, bias=None, name: str = "matmul"):
        """``x[..., K] @ w[N, K].T (+ bias[N])``."""

    @abstractmethod
    def attention(self, q, k, v, kv_cache, layer: int, window: Optional[int] = None):
        """Append ``k, v`` (new tokens) to ``kv_cache`` for ``layer`` and attend.

        q/k/v: ``[B, H, T, hd]``. Queries sit at absolute positions
        ``[ctx - T, ctx)`` where ``ctx`` is the cache length after appending.
        Causal; if ``window`` is set, query i only sees keys j with ``i - j < window``.
        GPT Neo semantics: scores are NOT scaled by 1/sqrt(hd); scores/softmax in FP32.
        Returns ``[B, H, T, hd]``.
        """

    @abstractmethod
    def layernorm(self, x, weight, bias, eps: float):
        """LayerNorm over the last dimension."""

    @abstractmethod
    def gelu(self, x):
        """GELU, tanh approximation ("gelu_new")."""

    @abstractmethod
    def softmax(self, x):
        """Softmax over the last dimension."""

    @abstractmethod
    def embedding(self, ids, table, vocab_start: int = 0):
        """Row lookup ``table[ids - vocab_start]``; ids outside this shard's rows give zeros."""

    @abstractmethod
    def add(self, a, b):
        """Elementwise add (residual connections)."""

    # ---- layout (no math) ----------------------------------------------------------------
    @abstractmethod
    def split_heads(self, x, num_heads: int):
        """``[B, T, H*hd] -> [B, H, T, hd]``."""

    @abstractmethod
    def merge_heads(self, x):
        """``[B, H, T, hd] -> [B, T, H*hd]``."""

    @abstractmethod
    def split_last(self, x, sizes: Sequence[int]):
        """Split the last dimension into chunks of the given sizes."""

    @abstractmethod
    def last_token(self, x):
        """``[B, T, D] -> [B, 1, D]``."""

    @abstractmethod
    def positions(self, start: int, length: int, batch: int):
        """Position ids ``[batch, length]`` = ``start .. start+length-1``."""

    @abstractmethod
    def create_kv_cache(self):
        """A fresh, empty KV cache living in this backend's memory."""

    @property
    @abstractmethod
    def activation_dtype(self):
        """Dtype of activations exchanged between NPUs (used to size recv buffers)."""

    # ---- tracing hooks (no-ops unless wrapped) -------------------------------------------
    def scope(self, layer):
        """Attribute the enclosed calls (kernels and comm) to ``layer``."""
        return nullcontext()

    def replicated(self):
        """Mark the enclosed calls as identical on every NPU of the tensor-parallel group."""
        return nullcontext()
