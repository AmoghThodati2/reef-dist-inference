"""Milestone 6: laptop measurement, machine info, comm backend ops, and the scripts."""
import csv
import json
import multiprocessing as mp
import os
import subprocess
import sys

import pytest
import torch

from conftest import ROOT
from reefdist.runtime.launcher import _free_port, run

WL = dict(mode="decode", prompt_len=128, gen_tokens=8)
CASES = {
    "tp1_decode.json": dict(num_npus=1),
    "tp4_decode.json": dict(num_npus=4),
    "pp4_decode.json": dict(num_npus=4, strategy="pipeline_parallel"),
}


@pytest.fixture(scope="module")
def results_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("results")
    for name, cfg in CASES.items():
        run(dict(cfg, workload=WL, output_dir=str(d), measurement=dict(warmup_runs=1, timed_runs=3)),
            name=name, verbose=False)
    return d


def script(name, *args):
    out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", name), *map(str, args)],
                         capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_measured_and_machine_sections(results_dir):
    r = json.load(open(results_dir / "tp4_decode.json"))
    m = r["measured_laptop"]
    assert m["warmup_runs"] == 1 and m["timed_runs"] == 3 and m["threads_per_process"] == 1
    assert 0 < m["median_total_ms"] <= m["p90_total_ms"]
    layers = {x["layer"] for x in m["per_layer"]}
    assert set(range(12)) <= layers and "embed" in layers and "head" in layers
    assert all(x["comm_ms"] > 0 for x in m["per_layer"] if isinstance(x["layer"], int))  # TP all-reduces
    assert {"cpu", "cores", "os", "torch_version"} <= set(r["machine"])
    assert r["machine"]["torch_version"] == torch.__version__


def test_compare_results(results_dir):
    out = script("compare_results.py", results_dir)
    assert "tp1" in out and "tp4" in out and "pp4" in out and "laptop" not in out.split("\n", 1)[1]
    out = script("compare_results.py", results_dir, "--measured")
    assert "laptop med ms" in out


def test_sweep_hardware_finds_crossover(results_dir):
    out = script("sweep_hardware.py", results_dir / "tp1_decode.json", results_dir / "tp4_decode.json",
                 results_dir / "pp4_decode.json")
    lines = out.splitlines()
    assert any(l.strip().startswith("200") for l in lines)  # alpha rows 1..200
    assert "tp4 vs tp1: stops beating it at link_alpha_us" in out
    out = script("sweep_hardware.py", results_dir / "tp1_decode.json", results_dir / "tp4_decode.json",
                 "--crossover-vs", "matmul_efficiency")
    assert "0.8" in out


def test_sweep_matches_saved_modeled_values(results_dir):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from sweep_hardware import metric_of, with_param

    r = json.load(open(results_dir / "tp4_decode.json"))
    hw = with_param(r, "link_alpha_us", r["modeled"]["hardware_profile"]["link_alpha_us"])
    assert metric_of(r, hw, "per_token_ms") == r["modeled"]["per_token_ms"]
    # comm cost grows linearly with alpha: +10 us per message sent by the (symmetric) TP NPUs
    a = metric_of(r, with_param(r, "link_alpha_us", 10.0), "per_token_ms")
    b = metric_of(r, with_param(r, "link_alpha_us", 20.0), "per_token_ms")
    decode = [p for p in r["exact"]["pass_profiles"] if p["phase"] == "decode"]
    msgs = decode[0]["npus"][0]["num_messages"]
    assert msgs == (2 * 12 + 1) * 2 * 3 + 3  # all-reduces (layers + embedding) + logits all-gather
    assert b - a == pytest.approx(msgs * 10e-3, rel=1e-9)


def test_export_kernel_shapes(results_dir, tmp_path):
    out_csv = tmp_path / "shapes.csv"
    script("export_kernel_shapes.py", results_dir, "-o", out_csv)
    rows = list(csv.reader(open(out_csv)))
    assert rows[0] == ["op", "batch", "M", "K", "N", "dtype"]
    body = [tuple(r) for r in rows[1:]]
    assert len(body) == len(set(body)) > 0
    assert ("matmul", "1", "1", "768", "192", "fp32") in body  # TP4 decode q/k/v per NPU
    assert ("layernorm", "1", "1", "1", "768", "fp32") in body  # elementwise: K = 1
    batches = {(r[1], r[3]) for r in body if r[0] == "attn_scores" and r[2] == "1"}
    assert ("3", "64") in batches and ("12", "64") in batches  # heads per NPU: tp4 vs tp1/pp4


def test_exported_csv_has_every_executed_op(results_dir, tmp_path):
    out_csv = tmp_path / "shapes.csv"
    script("export_kernel_shapes.py", results_dir, "-o", out_csv)
    exported = {(r["op"], int(r["batch"]), int(r["M"]), int(r["K"]), int(r["N"]), r["dtype"])
                for r in csv.DictReader(open(out_csv))}
    executed = set()
    for p in results_dir.glob("*.json"):
        for pp in json.load(open(p))["exact"]["pass_profiles"]:
            for npu in pp["npus"]:
                executed |= {(op, b, M, K, N, dt) for op, M, K, N, b, dt, _ in npu["kernels"]}
    assert exported == executed
    assert {e[0] for e in exported} == {"matmul", "bias_add", "attn_scores", "softmax", "attn_pv",
                                        "layernorm", "gelu", "add", "embedding"}


def _comm_worker(rank, port, q):
    from reefdist.comm.counting import CountingComm
    from reefdist.comm.local_gloo import LocalGlooBackend
    from reefdist.metrics.tracer import Tracer

    g = (0, 1)
    tr = Tracer()
    c = CountingComm(LocalGlooBackend(rank, 2, groups=[g], port=port), tr, 4)
    x = torch.arange(4, dtype=torch.float32) + 10 * rank
    out = {
        "all_reduce": c.all_reduce(x.clone(), g),
        "all_gather": c.all_gather(torch.full((1, 2 + rank), float(rank)), g, dim=-1, sizes=[2, 3]),
        "reduce_scatter": c.reduce_scatter(x.clone(), g, dim=0),
        "all_to_all": c.all_to_all(x.clone(), g, dim=0),
    }
    if rank == 0:
        c.send(torch.ones(3), 1)
    else:
        out["recv"] = c.recv((3,), torch.float32, 0)
    c.barrier()
    q.put((rank, {k: v.tolist() for k, v in out.items()}, [r._asdict() for r in tr.comm_records]))
    c.close()


def test_local_gloo_ops_and_ring_counts():
    ctx = mp.get_context("spawn")
    q, port = ctx.Queue(), _free_port()
    ps = [ctx.Process(target=_comm_worker, args=(r, port, q)) for r in range(2)]
    for p in ps:
        p.start()
    got = dict((r, (o, c)) for r, o, c in (q.get(timeout=120) for _ in ps))
    for p in ps:
        p.join()
    o0, c0 = got[0]
    o1, _ = got[1]
    assert o0["all_reduce"] == [10.0, 12.0, 14.0, 16.0]
    assert o0["all_gather"] == [[0.0, 0.0, 1.0, 1.0, 1.0]]
    assert o0["reduce_scatter"] == [10.0, 12.0] and o1["reduce_scatter"] == [14.0, 16.0]
    assert o0["all_to_all"] == [0.0, 1.0, 10.0, 11.0] and o1["all_to_all"] == [2.0, 3.0, 12.0, 13.0]
    assert o1["recv"] == [1.0, 1.0, 1.0]
    rec = {r["op"]: r for r in c0}
    assert (rec["all_reduce"]["messages"], rec["all_reduce"]["bytes_sent"]) == (2, 16)  # 2(n-1) x S/n
    assert (rec["all_gather"]["tensor_bytes"], rec["all_gather"]["messages"]) == (20, 1)
    assert (rec["reduce_scatter"]["messages"], rec["reduce_scatter"]["bytes_sent"]) == (1, 8)
    assert (rec["send"]["messages"], rec["send"]["bytes_sent"]) == (1, 12)
