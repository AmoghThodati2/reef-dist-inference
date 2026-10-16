"""Run one config end to end: reference, plan, shard, launch NPU processes, build the result JSON."""
import multiprocessing as mp
import os
import shutil
import socket
import tempfile
import time

import torch

from ..config import SUPPORTED_DTYPES, make_config, validate
from ..metrics.exact import build_exact, layer_order, weight_info_from_plan
from ..metrics.machine import machine_info
from ..metrics.measured import build_measured
from ..metrics.report import console_summary, write_json
from ..model import ModelSpec
from ..perfmodel.modeled import PHASE_OF_MODE, build_compute_model, model_layers, model_run
from ..registry import STRATEGIES
from ..weights import full_weights_from_hf, load_hf_model, load_tokenizer, tensor_bytes
from .prompts import build_prompt
from .reference import hf_reference
from .worker import worker_main


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def launch(job, shards, timeout_s=1800):
    """Start one process per NPU and wait; any failure stops all of them."""
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker_main, args=(rank, job, shards[rank]), daemon=True)
             for rank in range(len(shards))]
    for p in procs:
        p.start()
    deadline = time.time() + timeout_s
    try:
        while any(p.is_alive() for p in procs):
            failed = [i for i, p in enumerate(procs) if p.exitcode not in (None, 0)]
            if failed or time.time() > deadline:
                break
            time.sleep(0.05)
        failed = [i for i, p in enumerate(procs) if p.exitcode not in (None, 0)]
        if failed or any(p.is_alive() for p in procs):
            errs = []
            for i in failed:
                err = os.path.join(job["out_dir"], f"npu{i}.pt.err")
                errs.append(open(err).read() if os.path.exists(err) else f"npu {i} exit code {procs[i].exitcode}")
            raise RuntimeError("NPU process failed:\n" + ("\n".join(errs) or "timeout"))
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
            p.join()
    return [torch.load(os.path.join(job["out_dir"], f"npu{r}.pt"), weights_only=False) for r in range(len(shards))]


def check_correctness(ours, ref, workload):
    diffs, match = [], True

    def cmp(a, b):
        diffs.append(float((a - b).abs().max()))

    cmp(ours["logits"]["prefill"], ref["prefill_logits"])
    match &= torch.equal(ours["tokens"]["first"], ref["first_token"])
    if workload["mode"] == "decode":
        mine, theirs = ours["tokens"]["decode"], ref["decode_tokens"]
        match &= torch.equal(mine, theirs)
        for i, (a, b) in enumerate(zip(ours["logits"]["decode"], ref["decode_logits"])):
            if not torch.equal(mine[:, :i + 1], theirs[:, :i + 1]):
                break  # inputs diverged; later logits are not comparable
            cmp(a, b)
    elif workload["mode"] == "spec_verify":
        cmp(ours["logits"]["verify"], ref["verify_logits"])
        match &= torch.equal(ours["tokens"]["verify"], ref["verify_tokens"])
    return {
        "max_abs_logit_diff_vs_hf": max(diffs),
        "tokens_match_hf": bool(match),
        "reference": "transformers GPTNeoForCausalLM (FP32), same prompt, greedy",
        "logit_sets_compared": len(diffs),
    }


def default_output_name(cfg, plan):
    wl = cfg["workload"]
    s = {"tensor_parallel": "tp", "pipeline_parallel": "pp"}.get(cfg["strategy"], cfg["strategy"])
    shape = f"{plan['tp']}x{plan['pp']}" if cfg["strategy"] == "hybrid" else str(cfg["num_npus"])
    extra = "_fused" if cfg["fused_qkv"] else ""
    return f"{s}{shape}_{wl['mode']}_b{wl['batch_size']}_p{wl['prompt_len']}{extra}.json"


def run(cfg, *, name=None, all_logits=False, write=True, verbose=True, return_artifacts=False):
    """Run a config (dict, defaults applied) and return the result dict.

    all_logits: return logits for every prompt position (tests); the default computes
    the LM head for the last token only, as a real prefill would.
    """
    cfg = validate(make_config(cfg))
    build_compute_model(cfg["hardware_profile"])  # fail fast (e.g. the KernelTable stub)
    torch.set_grad_enabled(False)
    spec = ModelSpec.from_pretrained(cfg["model"])
    db = SUPPORTED_DTYPES[cfg["dtype"]]
    wl = cfg["workload"]

    hf = load_hf_model(cfg["model"])
    prompt = build_prompt(load_tokenizer(cfg["model"]), wl["batch_size"], wl["prompt_len"], wl["seed"])
    ref = hf_reference(hf, prompt, wl, all_logits=all_logits)

    strategy = STRATEGIES.build(cfg["strategy"], fused_qkv=cfg["fused_qkv"], dtype_bytes=db, **cfg["strategy_args"])
    plan = strategy.plan(spec, cfg["num_npus"])
    full = full_weights_from_hf(hf, spec)
    shards = strategy.shard_weights(full, plan)
    weight_info = weight_info_from_plan(plan, db, tensor_bytes(full, db))

    out_dir = tempfile.mkdtemp(prefix="reefdist_")
    job = {"config": cfg, "plan": plan, "spec": spec, "dtype_bytes": db, "workload": wl,
           "prompt_ids": prompt, "draft_ids": ref.get("draft_tokens"), "all_logits": all_logits,
           "port": _free_port(), "out_dir": out_dir}
    try:
        results = launch(job, shards)
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)

    reporter = results[min(plan["logits_npus"])]
    exact, pass_profiles, layer_summaries = build_exact(results, plan, spec, cfg, weight_info)
    exact = {"correctness": check_correctness(reporter, ref, wl), **exact}

    hw = cfg["hardware_profile"]
    modeled = model_run(pass_profiles, plan, hw, wl, cfg["num_npus"])
    divisor = exact["phase"]["passes"] if PHASE_OF_MODE[wl["mode"]] == "decode" else 1
    modeled = {
        "hardware_profile": dict(hw),
        "per_layer": model_layers(layer_summaries, hw, divisor, spec.attention_types),
        **modeled,
        "notes": {
            "per_pass": "prefill: all chunks summed (= TTFT); decode: average decode step; spec_verify: the verify pass",
            "per_token_ms": "decode: average decode step; spec_verify: one verify pass (draft_tokens tokens); prefill: n/a",
            "tokens_per_second": "prefill: prompt tokens/s; decode: batch/step; spec_verify: batch*draft_tokens/step (all accepted)",
            "mfu": "useful (non-replicated) FLOPs of the phase / (phase time x num_npus x peak)",
        },
    }
    key = lambda x: layer_order(x, spec.num_layers)  # noqa: E731
    result = {
        "config": cfg,
        "machine": machine_info(any(r["pinned"] for r in results)),
        "exact": exact,
        "modeled": modeled,
        "measured_laptop": build_measured(results, cfg["measurement"], key),
    }
    path = None
    if write:
        path = write_json(result, os.path.join(cfg["output_dir"], name or default_output_name(cfg, plan)))
    if verbose:
        console_summary(result, path)
    if return_artifacts:
        result["_artifacts"] = {"reporter": reporter, "reference": ref, "results": results, "prompt": prompt}
    return result
