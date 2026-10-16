"""Load a Hugging Face GPT Neo checkpoint into reefdist's canonical weight dict.

Canonical names (all linear weights in nn.Linear layout ``[out_features, in_features]``,
i.e. ``[N, K]`` for the matmul ``[M x K] . [K x N]``):

    wte [V, D]  (also the tied LM head)      wpe [P, D]
    h.{i}.ln_1.w / ln_1.b                    h.{i}.ln_2.w / ln_2.b
    h.{i}.attn.q / attn.k / attn.v [D, D]    (no bias)
    h.{i}.attn.o [D, D], h.{i}.attn.o_b [D]
    h.{i}.mlp.fc [F, D], h.{i}.mlp.fc_b [F]
    h.{i}.mlp.proj [D, F], h.{i}.mlp.proj_b [D]
    ln_f.w / ln_f.b
"""
from functools import lru_cache
from typing import Dict

import torch

DTYPE_BYTES = {"fp32": 4, "fp16": 2, "int8": 1}


@lru_cache(maxsize=2)
def load_hf_model(name: str):
    """HF model, used only to read weights and as the correctness reference."""
    from transformers import GPTNeoForCausalLM

    model = GPTNeoForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
    model.eval()
    return model


@lru_cache(maxsize=2)
def load_tokenizer(name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def full_weights_from_hf(model, spec) -> Dict[str, torch.Tensor]:
    sd = {k: v.detach() for k, v in model.state_dict().items()}
    t = "transformer."
    w = {
        "wte": sd[t + "wte.weight"],
        "wpe": sd[t + "wpe.weight"],
        "ln_f.w": sd[t + "ln_f.weight"],
        "ln_f.b": sd[t + "ln_f.bias"],
    }
    # The LM head is tied to the token embedding; check rather than assume.
    assert torch.equal(sd["lm_head.weight"], w["wte"]), "expected lm_head tied to wte"
    for i in range(spec.num_layers):
        p = f"{t}h.{i}."
        a = p + "attn.attention."
        for name in ("q", "k", "v"):
            assert a + f"{name}_proj.bias" not in sd, "GPT Neo q/k/v projections have no bias"
        w.update({
            f"h.{i}.ln_1.w": sd[p + "ln_1.weight"],
            f"h.{i}.ln_1.b": sd[p + "ln_1.bias"],
            f"h.{i}.attn.q": sd[a + "q_proj.weight"],
            f"h.{i}.attn.k": sd[a + "k_proj.weight"],
            f"h.{i}.attn.v": sd[a + "v_proj.weight"],
            f"h.{i}.attn.o": sd[a + "out_proj.weight"],
            f"h.{i}.attn.o_b": sd[a + "out_proj.bias"],
            f"h.{i}.ln_2.w": sd[p + "ln_2.weight"],
            f"h.{i}.ln_2.b": sd[p + "ln_2.bias"],
            f"h.{i}.mlp.fc": sd[p + "mlp.c_fc.weight"],
            f"h.{i}.mlp.fc_b": sd[p + "mlp.c_fc.bias"],
            f"h.{i}.mlp.proj": sd[p + "mlp.c_proj.weight"],
            f"h.{i}.mlp.proj_b": sd[p + "mlp.c_proj.bias"],
        })
    return w


def tensor_bytes(tensors: Dict[str, torch.Tensor], dtype_bytes: int) -> int:
    """Bytes of a weight dict when stored in the configured dtype."""
    return sum(t.numel() for t in tensors.values()) * dtype_bytes
