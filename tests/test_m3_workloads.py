"""Milestone 3: prefill (chunked or not), decode, spec_verify."""
import torch

from conftest import DIST_ATOL
from reefdist.runtime.kv_cache import KVCache
from reefdist.runtime.prompts import build_prompt

P = 300


def test_chunked_prefill_reference_same_logits(reference, tok):
    ids = build_prompt(tok, 1, P, 0)
    whole = reference.forward(ids, KVCache(), all_logits=True)
    cache = KVCache()
    chunks = [reference.forward(ids[:, s:s + 128], cache, all_logits=True) for s in range(0, P, 128)]
    assert (torch.cat(chunks, 1) - whole).abs().max().item() <= 1e-4


def test_chunked_prefill_distributed_same_logits(runner):
    wl = dict(mode="prefill", prompt_len=P)
    whole = runner(num_npus=2, workload=wl, all_logits=True)
    chunked = runner(num_npus=2, workload=dict(wl, prefill_chunk=128), all_logits=True)
    a, b = whole["_artifacts"]["reporter"]["logits"]["prefill"], chunked["_artifacts"]["reporter"]["logits"]["prefill"]
    assert a.shape == b.shape == (1, P, 50257)
    assert (a - b).abs().max().item() <= 1e-4
    passes = chunked["modeled"]["passes"]
    assert [(p["tokens"], p["context"]) for p in passes] == [(128, 128), (128, 256), (44, 300)]
    assert chunked["modeled"]["time_to_first_token_ms"] == sum(p["step_ms"] for p in passes)
    assert chunked["exact"]["correctness"]["tokens_match_hf"]
    # same math either way, apart from attention, whose context grows chunk by chunk
    flops = lambda r, op: sum(s["count"] * 2 * s["M"] * s["K"] * s["N"]  # noqa: E731
                              for s in r["exact"]["matmul_shapes"] if s["op"] == op)
    assert flops(whole, "matmul") == flops(chunked, "matmul")


def test_decode_workload(runner):
    r = runner(num_npus=2, workload=dict(mode="decode", prompt_len=P, gen_tokens=32), all_logits=True)
    phases = [p["phase"] for p in r["modeled"]["passes"]]
    assert phases == ["prefill"] + ["decode"] * 31
    assert r["exact"]["phase"] == {**r["exact"]["phase"], "name": "decode", "passes": 31}
    assert r["exact"]["correctness"]["tokens_match_hf"]
    assert r["exact"]["correctness"]["max_abs_logit_diff_vs_hf"] <= DIST_ATOL
    m = r["modeled"]
    decode_steps = [p["step_ms"] for p in m["passes"] if p["phase"] == "decode"]
    assert m["per_token_ms"] == sum(decode_steps) / 31
    assert abs(m["tokens_per_second"] - 1000 / m["per_token_ms"]) < 1e-9


def test_spec_verify_workload(runner, hf, tok):
    k = 4
    r = runner(num_npus=2, workload=dict(mode="spec_verify", prompt_len=P, draft_tokens=k))
    passes = r["modeled"]["passes"]
    assert [(p["phase"], p["tokens"], p["context"]) for p in passes] == [("prefill", P, P), ("verify", k, P + k)]
    ours = r["_artifacts"]["reporter"]
    assert ours["logits"]["verify"].shape == (1, k, 50257)
    assert r["exact"]["correctness"]["tokens_match_hf"]
    assert r["exact"]["correctness"]["max_abs_logit_diff_vs_hf"] <= DIST_ATOL
    # verifying a perfect draft: token i+1 is the model's own prediction at position i
    draft = r["_artifacts"]["reference"]["draft_tokens"]
    assert torch.equal(ours["tokens"]["verify"][:, :-1], draft[:, 1:])
    # attention sees the cached prompt plus the draft: K context = P + k for global layers
    scores = [s for s in r["exact"]["matmul_shapes"] if s["op"] == "attn_scores" and s["M"] == k]
    assert {s["N"] for s in scores} == {P + k, 256 + k - 1}
    assert r["modeled"]["tokens_per_second"] == k / (r["modeled"]["per_token_ms"] * 1e-3)
