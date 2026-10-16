"""Compare result files: strategy x num_npus x workload, from MODELED and EXACT values.

    python scripts/compare_results.py results/
    python scripts/compare_results.py results/*decode*.json --measured

Laptop wall-clock columns appear only with --measured, and are never Reef estimates.
"""
import argparse

from _common import expand, label, load, table


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="result JSON files or directories")
    ap.add_argument("--measured", action="store_true", help="also show measured_laptop columns")
    args = ap.parse_args(argv)

    results = [load(p) for p in expand(args.files)]
    results.sort(key=lambda r: (r["config"]["workload"]["mode"], r["config"]["workload"]["prompt_len"],
                                r["config"]["workload"]["batch_size"], r["config"]["num_npus"], label(r)))
    headers = ["strategy", "npus", "mode", "B", "prompt", "TTFT ms", "token ms", "tok/s", "step ms", "bound",
               "MFU", "msgs/step", "MB sent/step", "max W MB/npu", "max KV MB/npu", "max |dlogit|", "tokens ok"]
    if args.measured:
        headers += ["laptop med ms", "laptop p90 ms"]
    rows = []
    for r in results:
        wl, ex, mo = r["config"]["workload"], r["exact"], r["modeled"]
        passes = ex["phase"]["passes"] if ex["phase"]["name"] == "decode" else 1
        row = [label(r), r["config"]["num_npus"], wl["mode"], wl["batch_size"], wl["prompt_len"],
               mo["time_to_first_token_ms"], mo["per_token_ms"], mo["tokens_per_second"],
               mo["per_pass"]["step_ms"], mo["per_pass"]["bound_by"], mo["mfu"],
               ex["communication_totals"]["num_messages"] // passes,
               ex["communication_totals"]["bytes_sent"] / passes / 2**20,
               max(n["weight_bytes"] for n in ex["per_npu"]) / 2**20,
               max(n["kv_cache_bytes"] for n in ex["per_npu"]) / 2**20,
               ex["correctness"]["max_abs_logit_diff_vs_hf"], ex["correctness"]["tokens_match_hf"]]
        if args.measured:
            row += [r["measured_laptop"]["median_total_ms"], r["measured_laptop"]["p90_total_ms"]]
        rows.append(row)
    hw = {tuple(sorted((k, str(v)) for k, v in r["modeled"]["hardware_profile"].items())) for r in results}
    print("Modeled Reef estimates (exact counts x hardware_profile)"
          + ("; laptop columns are wall-clock sanity checks only" if args.measured else ""))
    if len(hw) > 1:
        print("WARNING: files use different hardware profiles; modeled columns are not directly comparable")
    print(table(headers, rows))


if __name__ == "__main__":
    main()
