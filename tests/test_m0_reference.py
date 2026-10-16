"""Milestone 0: hand-written single-device GPT Neo matches Hugging Face."""
import pytest
import torch

from reefdist.kernels.torch_backend import TorchBackend
from reefdist.runtime.kv_cache import KVCache
from reefdist.runtime.prompts import build_prompt
from reefdist.runtime.reference import hf_reference

PROMPTS = [40, 300]  # shorter and longer than the 256-token local window


def test_gpt_neo_details_match_hf_source(spec, hf):
    assert spec.attention_types == ("global", "local") * 6
    assert spec.window_size == 256 and spec.activation == "gelu_new"
    assert spec.d_ff == 4 * spec.d_model and spec.head_dim == 64
    attn = hf.transformer.h[0].attn.attention
    assert attn.q_proj.bias is None and attn.k_proj.bias is None and attn.v_proj.bias is None
    assert attn.out_proj.bias is not None
    assert hf.lm_head.weight.data_ptr() == hf.transformer.wte.weight.data_ptr()  # tied
    assert hf.lm_head.bias is None


def test_attention_unscaled_fp32_and_local_window():
    torch.manual_seed(0)
    B, H, T, hd, W = 1, 2, 300, 8, 256
    q, k, v = (torch.randn(B, H, T, hd) for _ in range(3))
    for window in (None, W):
        out = TorchBackend().attention(q, k, v, KVCache(), 0, window)
        scores = q @ k.transpose(-1, -2)  # no 1/sqrt(hd)
        i, j = torch.arange(T)[:, None], torch.arange(T)[None, :]
        allowed = (j <= i) & ((i - j < W) if window else True)
        expect = torch.softmax(scores.masked_fill(~allowed, torch.finfo(torch.float32).min), -1) @ v
        torch.testing.assert_close(out, expect, rtol=0, atol=1e-6)
    assert int(allowed[-1].sum()) == 256  # last query sees exactly 256 keys, itself included


@pytest.mark.parametrize("P", PROMPTS)
def test_logits_match_hf(P, reference, hf, tok):
    ids = build_prompt(tok, 1, P, 0)
    ours = reference.forward(ids, KVCache(), all_logits=True)
    theirs = hf(ids).logits
    assert (ours - theirs).abs().max().item() <= 1e-4


@pytest.mark.parametrize("P", PROMPTS)
def test_greedy_32_tokens_match_hf(P, reference, hf, tok):
    ids = build_prompt(tok, 1, P, 0)
    ref = hf_reference(hf, ids, {"mode": "decode", "gen_tokens": 32})
    tokens, steps = reference.generate_greedy(ids, 32)
    assert torch.equal(tokens, ref["decode_tokens"])
    for ours, theirs in zip(steps[1:], ref["decode_logits"]):
        assert (ours - theirs).abs().max().item() <= 1e-4


def test_batched_prompt_matches_hf(reference, hf, tok):
    ids = build_prompt(tok, 2, 64, 3)
    assert not torch.equal(ids[0], ids[1])
    ours = reference.forward(ids, KVCache(), all_logits=True)
    assert (ours - hf(ids).logits).abs().max().item() <= 1e-4
