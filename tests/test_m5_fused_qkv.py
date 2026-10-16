"""Milestone 5: fused QKV keeps each NPU's own Q, K, V heads together."""
import pytest
import torch

from reefdist.strategies.tensor_parallel import TensorParallel

WL = dict(mode="decode", prompt_len=300, gen_tokens=16)


@pytest.mark.parametrize("n", [1, 2, 4])
def test_fused_matches_unfused(runner, spec, n):
    plain = runner(num_npus=n, workload=WL, all_logits=True)
    fused = runner(num_npus=n, workload=WL, all_logits=True, fused_qkv=True)
    a, b = plain["_artifacts"]["reporter"], fused["_artifacts"]["reporter"]
    assert torch.equal(a["tokens"]["decode"], b["tokens"]["decode"])
    for x, y in zip([a["logits"]["prefill"]] + a["logits"]["decode"], [b["logits"]["prefill"]] + b["logits"]["decode"]):
        assert (x - y).abs().max().item() <= 1e-4
    assert fused["exact"]["correctness"]["tokens_match_hf"]

    pa, pb = plain["exact"]["per_npu"], fused["exact"]["per_npu"]
    assert [x["flops"] for x in pa] == [x["flops"] for x in pb]
    assert [x["bytes_read"] for x in pa] != [x["bytes_read"] for x in pb]  # input read once, not 3x
    passes = fused["exact"]["phase"]["passes"]
    for x, y in zip(pa, pb):  # one matmul instead of three, per layer per pass
        assert x["kernel_calls"] - y["kernel_calls"] == 2 * spec.num_layers * passes
    names = {nm for s in fused["exact"]["kernel_shapes"] for nm in s["names"]}
    assert "qkv_proj" in names and "q_proj" not in names


def test_fused_columns_keep_each_npus_heads_together(spec, full_weights):
    s = TensorParallel(fused_qkv=True)
    plan = s.plan(spec, 4)
    shards = s.shard_weights(full_weights, plan)
    D = spec.d_model
    for npu, shard in zip(plan["npus"], shards):
        a, b = npu["qkv_cols"]
        expect = torch.cat([full_weights[f"h.0.attn.{x}"][a:b] for x in "qkv"], 0)
        assert torch.equal(shard["h.0.attn.qkv"], expect)
        assert shard["h.0.attn.qkv"].shape == (3 * D // 4, D)
        assert (b - a) // spec.head_dim == 3  # three whole heads of Q, then of K, then of V
