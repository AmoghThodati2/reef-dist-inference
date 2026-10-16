"""TorchBackend: GPT Neo math in PyTorch, following HF transformers v4.26.1 modeling_gpt_neo.py."""
import math

import torch
import torch.nn.functional as F

from ..registry import KERNELS
from ..runtime.kv_cache import KVCache
from .base import KernelBackend


@KERNELS.register("torch")
class TorchBackend(KernelBackend):
    def __init__(self):
        self._masks = {}

    # ---- math -------------------------------------------------------------------------
    def matmul(self, x, w, bias=None, name="matmul"):
        return F.linear(x, w, bias)  # same call as nn.Linear in HF

    def attention(self, q, k, v, kv_cache, layer, window=None):
        k, v = kv_cache.append(layer, k, v)
        # HF GPTNeoSelfAttention._attn: FP32 scores, no 1/sqrt(head_dim) scaling.
        query = q.to(torch.float32)
        key = k.to(torch.float32)
        scores = torch.matmul(query, key.transpose(-1, -2))
        mask = self._mask(query.size(-2), key.size(-2), window)
        mask_value = torch.tensor(torch.finfo(scores.dtype).min, dtype=scores.dtype)
        scores = torch.where(mask, scores, mask_value)
        probs = torch.softmax(scores, dim=-1).to(v.dtype)
        return torch.matmul(probs, v)

    def _mask(self, q_len, k_len, window):
        key = (q_len, k_len, window)
        if key not in self._masks:
            i = torch.arange(k_len - q_len, k_len).unsqueeze(1)  # absolute query positions
            j = torch.arange(k_len).unsqueeze(0)
            allowed = j <= i
            if window is not None:  # HF: bias ^ tril(bias, -window) -> keep i - j < window
                allowed &= (i - j) < window
            self._masks[key] = allowed
        return self._masks[key]

    def layernorm(self, x, weight, bias, eps):
        return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)

    def gelu(self, x):
        # transformers.activations.NewGELUActivation ("gelu_new"), verbatim.
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

    def softmax(self, x):
        return torch.softmax(x, dim=-1)

    def embedding(self, ids, table, vocab_start=0):
        if vocab_start == 0 and table.shape[0] >= _max_id(ids) + 1:
            return F.embedding(ids, table)
        local = ids - vocab_start
        inside = (local >= 0) & (local < table.shape[0])
        out = F.embedding(local.clamp(0, table.shape[0] - 1), table)
        return out * inside.unsqueeze(-1).to(out.dtype)

    def add(self, a, b):
        return a + b

    # ---- layout -------------------------------------------------------------------------
    def split_heads(self, x, num_heads):
        b, t, d = x.shape
        return x.view(b, t, num_heads, d // num_heads).permute(0, 2, 1, 3)

    def merge_heads(self, x):
        b, h, t, hd = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(b, t, h * hd)

    def split_last(self, x, sizes):
        return torch.split(x, list(sizes), dim=-1)

    def last_token(self, x):
        return x[:, -1:, :]

    def positions(self, start, length, batch):
        return torch.arange(start, start + length, dtype=torch.long).unsqueeze(0).expand(batch, length)

    def create_kv_cache(self):
        return KVCache()

    @property
    def activation_dtype(self):
        return torch.float32


def _max_id(ids):
    return int(ids.max()) if ids.numel() else 0
