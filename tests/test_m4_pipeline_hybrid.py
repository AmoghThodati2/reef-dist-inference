"""Milestone 4: pipeline parallel and hybrid (2-way TP x 2-way PP)."""
from collections import defaultdict

import pytest

from test_m1_tensor_parallel import check_against_reference

CASES = {
    "pp2": dict(strategy="pipeline_parallel", num_npus=2),
    "pp4": dict(strategy="pipeline_parallel", num_npus=4),
    "hybrid_2x2": dict(strategy="hybrid", num_npus=4, strategy_args={"tp": 2, "pp": 2}),
}
B, P = 2, 96


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("prompt", [40, 300])
def test_matches_milestone0(runner, reference, case, prompt):
    r = runner(**CASES[case], workload=dict(mode="decode", prompt_len=prompt, gen_tokens=32), all_logits=True)
    check_against_reference(r, reference, exact=(case != "hybrid_2x2"))  # pure PP does the same math


@pytest.mark.parametrize("case", CASES)
def test_counts(runner, spec, case):
    single = runner(num_npus=1, workload=dict(mode="prefill", batch_size=B, prompt_len=P))["exact"]
    r = runner(**CASES[case], workload=dict(mode="prefill", batch_size=B, prompt_len=P))
    ex, plan = r["exact"], r["exact"]["plan"]
    tp, pp = plan["tp"], plan["pp"]
    assert sum(n["unique_flops"] for n in ex["per_npu"]) == single["per_npu"][0]["flops"]
    if tp == 1:  # nothing is replicated across a pipeline
        assert sum(n["flops"] for n in ex["per_npu"]) == single["per_npu"][0]["flops"]

    model = single["weights"]["model_weight_bytes"]
    assert sum(n["unique_weight_bytes"] for n in ex["per_npu"]) == model
    wte = spec.vocab_size * spec.d_model * 4  # tied LM head: last stage keeps a copy of its wte shard
    ln = (4 * spec.num_layers + 2) * spec.d_model * 4
    assert sum(n["weight_bytes"] for n in ex["per_npu"]) == model + wte + (tp - 1) * ln

    # stage-to-stage: tp sends of the full hidden state per boundary, per pass
    sends = defaultdict(list)
    for res in r["_artifacts"]["results"]:
        for c in res["comm_records"]:
            if c.op == "send":
                sends[c.pass_id].append(c)
    S = B * P * spec.d_model * 4
    assert len(sends[0]) == (pp - 1) * tp and all(c.tensor_bytes == S and c.messages == 1 for c in sends[0])
    # every TP group: 2 all-reduces per layer + 1 for the embedding (size-1 groups send nothing)
    ar = ex["communication_totals"]["by_collective"]["all_reduce"]
    assert ar["calls"] == tp * (2 * spec.num_layers + 1)
    assert ar["num_messages"] == ar["calls"] * 2 * (tp - 1)

    # stages run one after another: pass step = sum over stages of the slowest NPU in each
    p = r["modeled"]["passes"][0]
    rows = {x["npu_id"]: x for x in p["per_npu"]}
    expect = sum(max(rows[n]["step_ms"] for n in st["npus"]) for st in plan["stages"])
    assert p["step_ms"] == pytest.approx(expect, rel=1e-12)


def test_pipeline_decode_feeds_tokens_back(runner, spec):
    r = runner(**CASES["pp4"], workload=dict(mode="decode", prompt_len=300, gen_tokens=32), all_logits=True)
    last = r["_artifacts"]["results"][3]
    fb = [c for c in last["comm_records"] if c.layer == "sample"]
    assert len(fb) == 31 and all(c.op == "send" and c.peer == 0 and c.tensor_bytes == 8 for c in fb)
    stages = r["exact"]["plan"]["stages"]
    assert [s["layers"] for s in stages] == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
