"""Milestone 1: tensor parallel over 1, 2, 4 simulated NPUs reproduces milestone 0."""
import pytest
import torch

from conftest import DIST_ATOL
from reefdist.runtime.kv_cache import KVCache


def check_against_reference(r, reference, exact=False):
    a = r["_artifacts"]
    ours, prompt = a["reporter"], a["prompt"]
    ref_logits = reference.forward(prompt, KVCache(), all_logits=True)
    ref_tokens, ref_steps = reference.generate_greedy(prompt, ours["tokens"]["decode"].shape[1])
    assert torch.equal(ours["tokens"]["decode"], ref_tokens)
    pairs = [(ours["logits"]["prefill"], ref_logits)] + list(zip(ours["logits"]["decode"], ref_steps[1:]))
    for got, want in pairs:
        if exact:
            assert torch.equal(got, want)
        else:
            assert (got - want).abs().max().item() <= DIST_ATOL
    assert r["exact"]["correctness"]["tokens_match_hf"]


@pytest.mark.parametrize("P", [40, 300])
@pytest.mark.parametrize("n", [1, 2, 4])
def test_tp_matches_milestone0(n, P, runner, reference):
    r = runner(num_npus=n, workload=dict(mode="decode", prompt_len=P, gen_tokens=32), all_logits=True)
    check_against_reference(r, reference, exact=(n == 1))


def test_tp_plan_splits_heads_and_kv_cache(runner, spec):
    r = runner(num_npus=4, workload=dict(mode="decode", prompt_len=300, gen_tokens=32), all_logits=True)
    plan = r["exact"]["plan"]
    assert [n["heads"] for n in plan["npus"]] == [[0, 3], [3, 6], [6, 9], [9, 12]]
    # each NPU caches K and V for its 3 heads of every layer only
    ctx = 300 + 31
    for n in r["exact"]["per_npu"]:
        assert n["kv_cache_bytes"] == spec.num_layers * 2 * 3 * spec.head_dim * ctx * 4
