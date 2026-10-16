"""Aggregate raw kernel/comm records into the ``exact`` section and the work summaries
that the performance model consumes. Counts only; no timing anywhere in this module."""
from collections import Counter, defaultdict

from ..perfmodel.modeled import PHASE_OF_MODE

COLLECTIVES = ("all_reduce", "all_gather", "reduce_scatter", "all_to_all", "send", "recv", "barrier")


def is_call(r):
    """One record per backend call (attention and biased matmuls emit sub-records)."""
    return r.op != "bias_add" and r.name not in ("attn_softmax", "attn_pv")


def layer_order(label, num_layers):
    if isinstance(label, int):
        return (1, label)
    return {"embed": (0, 0), "head": (2, 0), "sample": (3, 0)}.get(label, (4, str(label)))


class Summary:
    __slots__ = ("kernels", "flops", "unique_flops", "bytes_read", "bytes_written",
                 "num_messages", "bytes_sent", "kernel_calls", "comm_calls")

    def __init__(self):
        self.kernels = Counter()
        self.flops = self.unique_flops = self.bytes_read = self.bytes_written = 0
        self.num_messages = self.bytes_sent = self.kernel_calls = self.comm_calls = 0

    def add_kernel(self, r, redundant):
        self.kernels[(r.op, r.M, r.K, r.N, r.batch, r.dtype)] += 1
        self.flops += r.flops
        if not redundant:
            self.unique_flops += r.flops
        self.bytes_read += r.bytes_read
        self.bytes_written += r.bytes_written
        self.kernel_calls += is_call(r)

    def add_comm(self, r):
        self.num_messages += r.messages
        self.bytes_sent += r.bytes_sent
        self.comm_calls += 1

    def to_dict(self, with_kernels=True):
        d = {k: getattr(self, k) for k in self.__slots__ if k != "kernels"}
        if with_kernels:
            d["kernels"] = [[*k, c] for k, c in sorted(self.kernels.items())]
        return d


def build_exact(npu_results, plan, spec, cfg, weight_info):
    """Returns (exact_section, pass_profiles, layer_summaries for the representative phase)."""
    mode = cfg["workload"]["mode"]
    phase = PHASE_OF_MODE[mode]
    passes = npu_results[0]["passes"]
    phase_of = {p["pass"]: p["phase"] for p in passes}
    rep = {p["pass"] for p in passes if p["phase"] == phase}
    npu_info = {n["npu_id"]: n for n in plan["npus"]}
    L = spec.num_layers

    per_pass = defaultdict(Summary)  # (pass, npu)
    per_layer_npu = defaultdict(Summary)  # (layer, npu), representative phase
    per_npu = defaultdict(Summary)  # npu, representative phase
    per_phase = defaultdict(Summary)  # phase, all NPUs
    shapes = {}
    peak_act = Counter()
    collectives = defaultdict(lambda: Counter())

    for res in npu_results:
        npu = res["npu_id"]
        tp_rank = npu_info[npu]["tp_rank"]
        for r in res["kernel_records"]:
            redundant = r.replicated and tp_rank != 0
            per_pass[(r.pass_id, npu)].add_kernel(r, redundant)
            per_phase[phase_of[r.pass_id]].add_kernel(r, redundant)
            peak_act[npu] = max(peak_act[npu], r.act_bytes)
            entry = shapes.setdefault((r.op, r.batch, r.M, r.K, r.N, r.dtype), [0, set()])
            entry[0] += 1
            entry[1].add(r.name)
            if r.pass_id in rep:
                per_layer_npu[(r.layer, npu)].add_kernel(r, redundant)
                per_npu[npu].add_kernel(r, redundant)
        for r in res["comm_records"]:
            per_pass[(r.pass_id, npu)].add_comm(r)
            per_phase[phase_of[r.pass_id]].add_comm(r)
            if r.pass_id in rep:
                per_layer_npu[(r.layer, npu)].add_comm(r)
                per_npu[npu].add_comm(r)
                c = collectives[r.op]
                c["calls"] += 1
                c["num_messages"] += r.messages
                c["bytes_sent"] += r.bytes_sent

    # per layer (summed over NPUs and over the representative phase's passes)
    labels = sorted({lab for lab, _ in per_layer_npu}, key=lambda x: layer_order(x, L))
    per_layer, layer_summaries = [], {}
    for lab in labels:
        by_npu = {n: per_layer_npu[(lab, n)] for (l2, n) in per_layer_npu if l2 == lab}
        tot = Summary()
        for s in by_npu.values():
            for k in ("flops", "unique_flops", "bytes_read", "bytes_written", "num_messages",
                      "bytes_sent", "kernel_calls", "comm_calls"):
                setattr(tot, k, getattr(tot, k) + getattr(s, k))
        per_layer.append({"layer": lab,
                          "attention_type": spec.attention_types[lab] if isinstance(lab, int) else None,
                          **tot.to_dict(with_kernels=False)})
        layer_summaries[lab] = {n: s.to_dict() for n, s in sorted(by_npu.items())}

    per_npu_out = []
    for res in sorted(npu_results, key=lambda r: r["npu_id"]):
        n = res["npu_id"]
        s = per_npu[n]
        per_npu_out.append({
            "npu_id": n, "stage": npu_info[n]["stage"], "tp_rank": npu_info[n]["tp_rank"],
            "weight_bytes": weight_info["resident"][n],
            "unique_weight_bytes": weight_info["unique"][n],
            "kv_cache_bytes": res["kv_cache_elements"] * weight_info["dtype_bytes"],
            "peak_activation_bytes": peak_act[n],
            **s.to_dict(with_kernels=False),
            "redundant_flops": s.flops - s.unique_flops,
        })

    kernel_shapes = [{"op": k[0], "batch": k[1], "M": k[2], "K": k[3], "N": k[4], "dtype": k[5], "count": v[0],
                      "names": sorted(v[1])} for k, v in sorted(shapes.items())]

    by_collective = {op: dict(calls=0, num_messages=0, bytes_sent=0) for op in COLLECTIVES}
    for op, c in collectives.items():
        by_collective[op] = {"calls": c["calls"], "num_messages": c["num_messages"], "bytes_sent": c["bytes_sent"]}
    comm_totals = {
        "num_messages": sum(c["num_messages"] for c in by_collective.values()),
        "bytes_sent": sum(c["bytes_sent"] for c in by_collective.values()),
        "by_collective": by_collective,
    }

    by_phase = {}
    for ph, s in per_phase.items():
        ids = [p for p in passes if p["phase"] == ph]
        by_phase[ph] = {"passes": len(ids), "tokens_per_pass": [p["tokens"] for p in ids],
                        **s.to_dict(with_kernels=False)}

    pass_profiles = []
    for p in passes:
        pass_profiles.append({**p, "npus": [{"npu_id": n, **per_pass[(p["pass"], n)].to_dict()}
                                            for n in sorted(npu_info)]})

    exact = {
        "phase": {"name": phase, "passes": len(rep),
                  "note": "per_layer, per_npu (except memory) and communication_totals cover this phase; "
                          "decode sums all decode steps; see by_phase for the rest"},
        "per_layer": per_layer,
        "per_npu": per_npu_out,
        "kernel_shapes": kernel_shapes,
        "communication_totals": comm_totals,
        "by_phase": by_phase,
        "weights": {"model_weight_bytes": weight_info["model"],
                    "sum_resident_weight_bytes": sum(weight_info["resident"].values()),
                    "sum_unique_weight_bytes": sum(weight_info["unique"].values())},
        "plan": plan,
        "pass_profiles": pass_profiles,
    }
    return exact, pass_profiles, layer_summaries


def weight_info_from_plan(plan, dtype_bytes, model_weight_bytes):
    resident, unique = {}, {}
    for npu in plan["weights"]:
        n = npu["npu_id"]
        resident[n] = sum(_numel(e["shape"]) for e in npu["tensors"]) * dtype_bytes
        unique[n] = sum(_numel(e["shape"]) for e in npu["tensors"] if e["owner"]) * dtype_bytes
    return {"resident": resident, "unique": unique, "model": model_weight_bytes, "dtype_bytes": dtype_bytes}


def _numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n
