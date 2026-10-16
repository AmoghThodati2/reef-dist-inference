"""Modeled Reef performance: exact counts x hardware profile. Deterministic, no timers.

Input is a *work summary* (built from exact counts, see metrics/exact.py)::

    {"kernels": [[op, M, K, N, batch, dtype, count], ...],
     "flops": int, "unique_flops": int, "bytes_read": int,
     "num_messages": int, "bytes_sent": int}

Per NPU:
    compute = sum compute_model.compute_time(op, M, K, N, dtype, batch) * count
    memory  = bytes_read / (npu_mem_bandwidth_GBps * 1e9)
    comm    = num_messages * link_alpha_us * 1e-6 + bytes_sent / (link_beta_GBps * 1e9)
    step    = max(compute, memory) + comm          (overlap_comm: false)
            = max(compute, memory, comm)           (overlap_comm: true)
Per pass: stages run one after another, NPUs within a stage in parallel, so
    pass step = sum over stages of (slowest NPU's step in that stage).
Tensor parallel is one stage (step = slowest NPU); pipeline parallel sums the stages,
and the stage-to-stage sends are part of the sending NPU's comm time.
"""
from ..registry import COMPUTE_MODELS

PHASE_OF_MODE = {"prefill": "prefill", "decode": "decode", "spec_verify": "verify"}
_COMPONENTS = ("compute", "memory", "comm")


def build_compute_model(hw):
    return COMPUTE_MODELS.build(hw["compute_model"], hw)


def summary_times(s, hw, cm):
    compute = sum(cm.compute_time(op, M, K, N, dt, b) * cnt for op, M, K, N, b, dt, cnt in s["kernels"])
    memory = s["bytes_read"] / (hw["npu_mem_bandwidth_GBps"] * 1e9)
    comm = s["num_messages"] * hw["link_alpha_us"] * 1e-6 + s["bytes_sent"] / (hw["link_beta_GBps"] * 1e9)
    return compute, memory, comm


def combine(compute, memory, comm, overlap):
    step = max(compute, memory, comm) if overlap else max(compute, memory) + comm
    vals = {"compute": compute, "memory": memory, "comm": comm}
    return step, max(_COMPONENTS, key=lambda k: vals[k])  # ties: compute > memory > comm


def _row(c, m, k, overlap, **extra):
    step, bound = combine(c, m, k, overlap)
    return dict(extra, compute_ms=c * 1e3, memory_ms=m * 1e3, comm_ms=k * 1e3, step_ms=step * 1e3, bound_by=bound)


def model_pass(npu_summaries, stages, hw, cm):
    """npu_summaries: {npu_id: summary}. Returns per-NPU rows and the pass total."""
    overlap = hw["overlap_comm"]
    per_npu = {}
    for npu, s in npu_summaries.items():
        per_npu[npu] = _row(*summary_times(s, hw, cm), overlap, npu_id=npu)
    total = {"compute_ms": 0.0, "memory_ms": 0.0, "comm_ms": 0.0, "step_ms": 0.0}
    critical = []
    for st in stages:
        rows = [per_npu[n] for n in st["npus"] if n in per_npu]
        if not rows:
            continue
        worst = max(rows, key=lambda r: r["step_ms"])
        critical.append(worst["npu_id"])
        for k in total:
            total[k] += worst[k]
    total["bound_by"] = max(_COMPONENTS, key=lambda k: total[k + "_ms"])
    total["critical_npus"] = critical
    return total, [per_npu[n] for n in sorted(per_npu)]


def model_run(pass_profiles, plan, hw, workload, num_npus):
    """Model every pass, then roll up into TTFT, per-token time, tokens/s and MFU."""
    cm = build_compute_model(hw)
    stages = plan["stages"]
    passes = []
    for p in pass_profiles:
        summaries = {n["npu_id"]: n for n in p["npus"]}
        total, per_npu = model_pass(summaries, stages, hw, cm)
        passes.append({"pass": p["pass"], "phase": p["phase"], "tokens": p["tokens"], "context": p["context"],
                       **total, "per_npu": per_npu,
                       "unique_flops": sum(n["unique_flops"] for n in p["npus"])})

    phase = PHASE_OF_MODE[workload["mode"]]
    rep = [p for p in passes if p["phase"] == phase]
    div = len(rep) if phase == "decode" else 1  # decode: average step; prefill: all chunks
    per_pass = {k: sum(p[k] for p in rep) / div for k in ("compute_ms", "memory_ms", "comm_ms", "step_ms")}
    per_pass["bound_by"] = max(_COMPONENTS, key=lambda k: per_pass[k + "_ms"])
    per_pass["phase"] = phase
    per_pass["passes_averaged" if div > 1 else "passes_summed"] = len(rep)
    if len(rep) == 1:
        per_pass["per_npu"] = rep[0]["per_npu"]

    B = workload["batch_size"]
    ttft = sum(p["step_ms"] for p in passes if p["phase"] == "prefill")
    if phase == "prefill":
        per_token, tps = None, B * workload["prompt_len"] / (ttft * 1e-3)
    elif phase == "decode":
        per_token = per_pass["step_ms"]
        tps = B / (per_token * 1e-3)
    else:
        per_token = per_pass["step_ms"]
        tps = B * workload["draft_tokens"] / (per_token * 1e-3)
    phase_time_s = sum(p["step_ms"] for p in rep) * 1e-3
    phase_flops = sum(p["unique_flops"] for p in rep)
    mfu = phase_flops / (phase_time_s * num_npus * hw["npu_peak_gflops"] * 1e9)
    for p in passes:
        del p["unique_flops"]
    return {
        "per_pass": per_pass,
        "time_to_first_token_ms": ttft,
        "per_token_ms": per_token,
        "tokens_per_second": tps,
        "mfu": mfu,
        "passes": passes,
    }


def model_layers(layer_summaries, hw, divisor, attention_types):
    """layer_summaries: {label: {npu_id: summary}} for the representative phase.

    Each row reports the NPU with the largest step for that layer (the critical one).
    """
    cm = build_compute_model(hw)
    rows = []
    for label, by_npu in layer_summaries.items():
        best = None
        for npu, s in sorted(by_npu.items()):
            c, m, k = (t / divisor for t in summary_times(s, hw, cm))
            row = _row(c, m, k, hw["overlap_comm"], layer=label,
                       attention_type=attention_types[label] if isinstance(label, int) else None, npu_id=npu)
            if best is None or row["step_ms"] > best["step_ms"]:
                best = row
        rows.append(best)
    return rows
