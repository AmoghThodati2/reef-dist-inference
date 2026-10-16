"""Laptop wall-clock aggregation. For sanity checks and same-machine comparisons only."""
import statistics


def _p90(xs):
    xs = sorted(xs)
    k = (len(xs) - 1) * 0.9
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def build_measured(npu_results, measurement, layer_key):
    runs = len(npu_results[0]["timed"])
    totals, layers = [], {}
    for i in range(runs):
        totals.append(max(r["timed"][i]["total_ms"] for r in npu_results))
        labels = {lab for r in npu_results for lab in r["timed"][i]["layers"]}
        for lab in labels:
            vals = [r["timed"][i]["layers"].get(lab, (0.0, 0.0)) for r in npu_results]
            layers.setdefault(lab, []).append((max(v[0] for v in vals), max(v[1] for v in vals)))
    per_layer = [{"layer": lab, "compute_ms": statistics.median(v[0] for v in layers[lab]),
                  "comm_ms": statistics.median(v[1] for v in layers[lab])}
                 for lab in sorted(layers, key=layer_key)]
    return {
        "note": "laptop wall-clock; NOT Reef performance (see 'modeled')",
        "warmup_runs": measurement["warmup_runs"], "timed_runs": measurement["timed_runs"],
        "threads_per_process": measurement["threads_per_process"],
        "median_total_ms": statistics.median(totals), "p90_total_ms": _p90(totals),
        "per_layer": per_layer,
    }
