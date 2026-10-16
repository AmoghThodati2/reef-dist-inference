"""Re-compute modeled outputs over a range of hardware values, from saved exact counts.

No model is re-run: each result file's exact.pass_profiles are re-priced with a modified
hardware_profile. The first file (or --baseline) is the reference, typically 1 NPU.

    # at what link latency do 4 NPUs stop beating 1?
    python scripts/sweep_hardware.py results/tp1_decode.json results/tp4_decode.json results/pp4_decode.json
    # how does that crossover move with compute efficiency?
    python scripts/sweep_hardware.py results/tp1_decode.json results/tp4_decode.json --crossover-vs matmul_efficiency
    # any parameter / values / metric
    python scripts/sweep_hardware.py a.json b.json --param link_beta_GBps --values 1,5,10,50 --metric tokens_per_second
"""
import argparse
import copy

from _common import label, load, table
from reefdist.perfmodel.modeled import model_run

DEFAULT_VALUES = {
    "link_alpha_us": [1, 2, 5, 10, 20, 50, 100, 200],
    "matmul_efficiency": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "link_beta_GBps": [1, 2, 5, 10, 25, 50, 100],
    "npu_mem_bandwidth_GBps": [5, 10, 20, 50, 100, 200],
    "npu_peak_gflops": [25, 50, 100, 200, 400, 1000],
}
HIGHER_IS_BETTER = {"tokens_per_second", "mfu"}


def metric_of(r, hw, metric):
    ex = r["exact"]
    m = model_run(ex["pass_profiles"], ex["plan"], hw, r["config"]["workload"], r["config"]["num_npus"])
    if metric == "step_ms":
        return m["per_pass"]["step_ms"]
    return m[metric]


def default_metric(r):
    return "time_to_first_token_ms" if r["config"]["workload"]["mode"] == "prefill" else "per_token_ms"


def with_param(r, param, value, extra=None):
    hw = copy.deepcopy(r["modeled"]["hardware_profile"])
    hw[param] = value
    hw.update(extra or {})
    return hw


def better(a, b, metric):
    return a > b if metric in HIGHER_IS_BETTER else a < b


def crossover(base, other, param, metric, extra=None, lo=1e-3, hi=1e4):
    """Smallest value of ``param`` in [lo, hi] at which ``other`` stops beating ``base`` (bisection)."""
    f = lambda v: better(metric_of(other, with_param(other, param, v, extra), metric),  # noqa: E731
                         metric_of(base, with_param(base, param, v, extra), metric), metric)
    if not f(lo):
        return lo, "never better"
    if f(hi):
        return hi, "always better"
    for _ in range(60):
        mid = (lo * hi) ** 0.5
        lo, hi = (mid, hi) if f(mid) else (lo, mid)
    return hi, ""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--param", default="link_alpha_us", choices=sorted(DEFAULT_VALUES))
    ap.add_argument("--values", help="comma-separated values (default: a preset range per param)")
    ap.add_argument("--metric", choices=["per_token_ms", "time_to_first_token_ms", "tokens_per_second", "step_ms", "mfu"])
    ap.add_argument("--baseline", help="file to compare against (default: first file)")
    ap.add_argument("--crossover-vs", choices=sorted(DEFAULT_VALUES),
                    help="for each value of this second parameter, find the --param value where each "
                         "file stops beating the baseline")
    args = ap.parse_args(argv)

    results = [load(p) for p in args.files]
    base = load(args.baseline) if args.baseline else results[0]
    others = [r for r in results if r["_path"] != base["_path"]]
    metric = args.metric or default_metric(base)
    values = [float(v) for v in args.values.split(",")] if args.values else DEFAULT_VALUES[args.param]
    print(f"Modeled {metric} (from exact counts; no model re-run). Baseline: {label(base)}. "
          f"Other hardware values as in each file.")

    headers = [args.param, label(base) + " (base)"] + [label(r) for r in others] + ["best"]
    rows = []
    for v in values:
        vals = [metric_of(r, with_param(r, args.param, v), metric) for r in [base] + others]
        best = [base] + others
        idx = max(range(len(vals)), key=lambda i: vals[i]) if metric in HIGHER_IS_BETTER else \
            min(range(len(vals)), key=lambda i: vals[i])
        rows.append([v] + vals + [label(best[idx])])
    print(table(headers, rows))

    if others:
        print()
        if args.crossover_vs:
            second = DEFAULT_VALUES[args.crossover_vs]
            headers = [args.crossover_vs] + [f"{label(r)}: {args.param} where it stops beating {label(base)}" for r in others]
            rows = []
            for s in second:
                row = [s]
                for r in others:
                    x, note = crossover(base, r, args.param, metric, {args.crossover_vs: s})
                    row.append(note or f"{x:.4g}")
                rows.append(row)
            print(table(headers, rows))
        else:
            for r in others:
                x, note = crossover(base, r, args.param, metric)
                print(f"{label(r)} vs {label(base)}: " + (note if note else f"stops beating it at {args.param} = {x:.4g}"))


if __name__ == "__main__":
    main()
