"""KV cache: per-layer key/value tensors for the heads and layers this NPU owns."""
import torch


class KVCache:
    def __init__(self):
        self.k = {}
        self.v = {}
        self.seq_len = 0  # tokens processed so far (absolute position of the next token)

    def layer_len(self, layer: int) -> int:
        return self.k[layer].shape[-2] if layer in self.k else 0

    def append(self, layer: int, k, v):
        """Append new keys/values ``[B, H, T, hd]``; return the full cached K and V.

        Like HF GPT Neo, local layers keep the whole history; the attention mask
        enforces the window.
        """
        if layer in self.k:
            k = torch.cat((self.k[layer], k), dim=-2)
            v = torch.cat((self.v[layer], v), dim=-2)
        self.k[layer], self.v[layer] = k, v
        return k, v

    def advance(self, num_tokens: int):
        self.seq_len += num_tokens

    def num_elements(self) -> int:
        return sum(t.numel() for t in self.k.values()) + sum(t.numel() for t in self.v.values())
