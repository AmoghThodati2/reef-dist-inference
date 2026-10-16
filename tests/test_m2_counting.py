"""Milestone 2: exact counting, modeled outputs and JSON."""
import json
from collections import defaultdict

import pytest

from reefdist.config import make_config, validate
from reefdist.errors import ConfigError
from reefdist.metrics.report import public
from reefdist.runtime.launcher import run

B, P = 2, 96
PREFILL = dict(mode="prefill", batch_size=B, prompt_len=P)


def prefill_run(runner, n):
    return runner(num_npus=n, workload=PREFILL)


@pytest.mark.parametrize("n", [2, 4])
def test_total_flops_equal_single_npu(runner, n):
    single = prefill_run(runner, 1)["exact"]["per_npu"]
    multi = prefill_run(runner, n)["exact"]["per_npu"]
    assert len(single) == 1 and single[0]["redundant_flops"] == 0
    # Unique math (each replicated LayerNorm/residual counted once) is unchanged by sharding.
    assert sum(x["unique_flops"] for x in multi) == single[0]["flops"]
    assert sum(x["flops"] - x["redundant_flops"] for x in multi) == single[0]["flops"]
    # Executed math also includes the replicated LayerNorms/residual adds on tp_rank > 0: small.
    assert multi[0]["redundant_flops"] == 0
    overhead = sum(x["flops"] for x in multi) / single[0]["flops"] - 1
    assert 0 < overhead < 0.005


@pytest.mark.parametrize("n", [2, 4])
def test_tp_all_reduce_count_and_size(runner, spec, n):
    r = prefill_run(runner, n)
    for res in r["_artifacts"]["results"]:
        by_pass = defaultdict(list)
        for c in res["comm_records"]:
            if c.op == "all_reduce":
                by_pass[c.pass_id].append(c)
        for p in res["passes"]:
            ars = by_pass[p["pass"]]
            in_layers = [c for c in ars if isinstance(c.layer, int)]
            assert len(in_layers) == 2 * spec.num_layers
            assert [c.layer for c in ars if not isinstance(c.layer, int)] == ["embed"]  # vocab-parallel embedding
            S = B * p["tokens"] * spec.d_model * 4
            for c in ars:
                assert c.tensor_bytes == S and c.group_size == n
                assert c.messages == 2 * (n - 1) and c.bytes_sent == 2 * (n - 1) * S // n


@pytest.mark.parametrize("n", [1, 2, 4])
def test_weight_bytes_sum_to_model(runner, spec, n):
    ex = prefill_run(runner, n)["exact"]
    model = 125_198_592 * 4
    assert ex["weights"]["model_weight_bytes"] == model
    assert sum(x["unique_weight_bytes"] for x in ex["per_npu"]) == model
    ln_bytes = (4 * spec.num_layers + 2) * spec.d_model * 4  # LayerNorm params, replicated per NPU
    assert sum(x["weight_bytes"] for x in ex["per_npu"]) == model + (n - 1) * ln_bytes


def _shapes(r):
    out = defaultdict(set)
    for s in r["exact"]["matmul_shapes"]:
        for name in s["names"]:
            out[name].add((s["M"], s["K"], s["N"]))
    return out


@pytest.mark.parametrize("n", [2, 4])
def test_matmul_shapes_split_by_n(runner, spec, n):
    one, many = _shapes(prefill_run(runner, 1)), _shapes(prefill_run(runner, n))
    for name in ("q_proj", "k_proj", "v_proj", "fc_in"):  # column-parallel: N / n
        (M, K, N), = one[name]
        assert many[name] == {(M, K, N // n)}
    for name in ("out_proj", "fc_out"):  # row-parallel: K / n
        (M, K, N), = one[name]
        assert many[name] == {(M, K // n, N)}
    (M, K, V), = one["lm_head"]  # vocab-parallel: N ~ V / n (uneven split of 50257)
    assert {s[2] for s in many["lm_head"]} == {-(-V // n), V - (n - 1) * -(-V // n)}
    (M, K, N), = one["attn_scores"]
    assert many["attn_scores"] == {(M, K, N)}  # per-head shape unchanged, fewer heads per NPU
    counts = {s["op"]: 0 for s in prefill_run(runner, n)["exact"]["matmul_shapes"]}
    assert set(counts) == {"matmul", "attn_scores", "attn_pv"}


def test_same_config_twice_is_byte_identical():
    cfg = dict(num_npus=2, workload=dict(mode="decode", prompt_len=64, gen_tokens=4),
               measurement=dict(warmup_runs=0, timed_runs=1))
    a, b = (public(run(cfg, write=False, verbose=False)) for _ in range(2))
    for key in ("exact", "modeled"):
        assert json.dumps(a[key]) == json.dumps(b[key])


def test_hand_calculated_decode_step(runner, spec):
    hw = make_config()["hardware_profile"]
    r = runner(num_npus=1, workload=dict(mode="decode", prompt_len=300, gen_tokens=3))
    D, F, H, hd, V, db = spec.d_model, spec.d_ff, spec.num_heads, spec.head_dim, spec.vocab_size, 4

    def step(ctx):
        flops, elems = D, 4 * D  # embedding: two row reads + add (reads 2D, writes D)
        for i in range(spec.num_layers):
            c = ctx if spec.attention_types[i] == "global" else min(ctx, spec.window_size)
            flops += D + 3 * 2 * D * D + 2 * H * c * hd + H * c + 2 * H * c * hd
            flops += 2 * D * D + D + D + D + 2 * D * F + F + F + 2 * F * D + D + D
            elems += 3 * D + 3 * (D + D * D)  # ln_1 reads x, gamma, beta; q/k/v read x and W
            elems += H * hd + 2 * H * c * hd  # q plus K and V context
            elems += (D + D * D) + D + 2 * D  # out_proj x and W, its bias, residual add
            elems += 3 * D + (D + F * D) + F  # ln_2, fc_in x and W, its bias
            elems += F + (F + D * F) + D + 2 * D  # gelu, fc_out x and W, its bias, residual add
        flops += D + 2 * D * V  # ln_f on the last token, lm_head
        elems += 3 * D + (D + V * D)
        compute = flops / (hw["npu_peak_gflops"] * 1e9 * hw["matmul_efficiency"])
        memory = elems * db / (hw["npu_mem_bandwidth_GBps"] * 1e9)
        return max(compute, memory) * 1e3, flops

    passes = r["modeled"]["passes"]
    assert [p["phase"] for p in passes] == ["prefill", "decode", "decode"]
    for p in passes[1:]:
        ms, flops = step(p["context"])
        assert p["comm_ms"] == 0
        assert p["step_ms"] == pytest.approx(ms, rel=1e-12)
        assert sum(n["flops"] for n in r["exact"]["pass_profiles"][p["pass"]]["npus"]) == flops
    assert r["modeled"]["per_token_ms"] == pytest.approx((step(301)[0] + step(302)[0]) / 2, rel=1e-12)
    assert r["modeled"]["per_pass"]["bound_by"] == "memory"


def test_kernel_table_is_a_clear_stub(tmp_path):
    with pytest.raises(ConfigError, match="kernel_table"):
        validate(make_config({"hardware_profile": {"compute_model": "table"}}))
    cfg = {"hardware_profile": {"compute_model": "table", "kernel_table": str(tmp_path / "k.csv")}}
    with pytest.raises(NotImplementedError, match="op, M, K, N, dtype, time_us"):
        run(cfg, write=False, verbose=False)


@pytest.mark.parametrize("override, match", [
    ({"num_npus": 3}, "num_npus"),
    ({"workload": {"batch_size": 0}}, "batch_size"),
    ({"hardware_profile": {"link_alpha_us": 0}}, "link_alpha_us"),
    ({"hardware_profile": {"npu_mem_bandwidth_GBps": -1}}, "npu_mem_bandwidth_GBps"),
    ({"strategy": "hybrid", "num_npus": 4, "strategy_args": {"tp": 2, "pp": 1}}, "must equal num_npus"),
    ({"strategy": "hybrid", "num_npus": 4}, "strategy_args.tp"),
    ({"strategy": "nope"}, "strategy must be one of"),
    ({"workload": {"mode": "train"}}, "workload.mode"),
    ({"workload": {"prompt_len": 2100}}, "positions"),
    ({"dtype": "int8"}, "reserved"),
    ({"parallel_block": True}, "parallel_block"),
    ({"workload": {"mode": "decode", "gen_tokens": 1}}, "gen_tokens"),
    ({"pipeline_parallel": 1}, "unknown config key"),
])
def test_config_validation(override, match):
    with pytest.raises(ConfigError, match=match):
        validate(make_config(override))


def test_tp_must_divide_heads(spec):
    from dataclasses import replace

    odd = replace(spec, num_heads=10)
    with pytest.raises(ConfigError, match="must divide num_heads"):
        validate(make_config({"num_npus": 4}), spec=odd)


def test_json_has_required_sections(runner):
    r = public(prefill_run(runner, 2))
    assert set(r) == {"config", "machine", "exact", "modeled", "measured_laptop"}
    for k in ("correctness", "per_layer", "per_npu", "matmul_shapes", "communication_totals", "plan"):
        assert k in r["exact"]
    for k in ("hardware_profile", "per_layer", "per_pass", "time_to_first_token_ms", "per_token_ms",
              "tokens_per_second", "mfu"):
        assert k in r["modeled"]
    json.dumps(r)  # serialisable
    assert r["exact"]["per_layer"][1]["layer"] == 0 and r["exact"]["per_layer"][1]["attention_type"] == "global"
