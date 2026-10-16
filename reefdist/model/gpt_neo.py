"""Single-device GPT Neo forward (milestone 0), re-implemented from HF v4.26.1 modeling_gpt_neo.py.

Uses only KernelBackend calls, in the same order as the strategies use them, so
TensorParallel with num_npus=1 reproduces it bit-for-bit.

GPT Neo details (verified against the HF source):
  * hidden = wte(ids) + wpe(positions); learned absolute positions; no embedding scaling
  * block: x + attn(ln_1(x)), then x + mlp(ln_2(x))
  * q/k/v projections have no bias, out_proj has one; scores are NOT scaled by 1/sqrt(hd)
    and are computed in FP32; masked entries get finfo.min before softmax
  * attention_layers alternate global/local; local = causal window of ``window_size`` keys
    (query i sees keys j with i - j < 256, including itself)
  * MLP: c_fc -> gelu_new -> c_proj, both with bias; final ln_f; lm_head tied to wte, no bias
"""


class GPTNeoReference:
    def __init__(self, spec, weights, kernels):
        self.spec, self.W, self.k = spec, weights, kernels

    def forward(self, ids, kv_cache, all_logits=False):
        spec, W, K = self.spec, self.W, self.k
        B, T = ids.shape
        h = K.embedding(ids, W["wte"])
        h = K.add(h, K.embedding(K.positions(kv_cache.seq_len, T, B), W["wpe"]))
        heads = spec.num_heads
        for i in range(spec.num_layers):
            p = f"h.{i}."
            a = K.layernorm(h, W[p + "ln_1.w"], W[p + "ln_1.b"], spec.layer_norm_eps)
            q = K.split_heads(K.matmul(a, W[p + "attn.q"]), heads)
            k = K.split_heads(K.matmul(a, W[p + "attn.k"]), heads)
            v = K.split_heads(K.matmul(a, W[p + "attn.v"]), heads)
            o = K.merge_heads(K.attention(q, k, v, kv_cache, i, spec.window(i)))
            h = K.add(K.matmul(o, W[p + "attn.o"], W[p + "attn.o_b"]), h)
            m = K.layernorm(h, W[p + "ln_2.w"], W[p + "ln_2.b"], spec.layer_norm_eps)
            m = K.matmul(K.gelu(K.matmul(m, W[p + "mlp.fc"], W[p + "mlp.fc_b"])), W[p + "mlp.proj"], W[p + "mlp.proj_b"])
            h = K.add(h, m)
        if not all_logits:
            h = K.last_token(h)
        h = K.layernorm(h, W["ln_f.w"], W["ln_f.b"], spec.layer_norm_eps)
        kv_cache.advance(T)
        return K.matmul(h, W["wte"])

    def generate_greedy(self, prompt_ids, num_tokens, prefill_chunk=None):
        """Greedy decode; returns (tokens [B, num_tokens], per-step last-token logits list)."""
        import torch

        cache = self.k.create_kv_cache()
        P = prompt_ids.shape[1]
        chunk = prefill_chunk or P
        for s in range(0, P, chunk):
            logits = self.forward(prompt_ids[:, s:s + chunk], cache)
        steps = [logits[:, -1]]
        tokens = [logits[:, -1].argmax(-1)]
        for _ in range(num_tokens - 1):
            logits = self.forward(tokens[-1][:, None], cache)
            steps.append(logits[:, -1])
            tokens.append(logits[:, -1].argmax(-1))
        return torch.stack(tokens, 1), steps
