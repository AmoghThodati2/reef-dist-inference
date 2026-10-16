"""JSON writer and console summary."""
import json
import os


def public(result):
    return {k: v for k, v in result.items() if not k.startswith("_")}


def write_json(result, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(public(result), f, indent=1)
        f.write("\n")
    return path


def _mb(b):
    return f"{b / 2**20:.1f} MB"


def console_summary(result, path=None):
    cfg, ex, mo, me = result["config"], result["exact"], result["modeled"], result["measured_laptop"]
    plan = ex["plan"]
    wl = cfg["workload"]
    c = ex["correctness"]
    pp = mo["per_pass"]
    lines = [
        f"== {cfg['strategy']} (tp={plan['tp']}, pp={plan['pp']}) on {cfg['num_npus']} NPU(s) | "
        f"{wl['mode']} B={wl['batch_size']} prompt={wl['prompt_len']} | {cfg['model']} {cfg['dtype']}",
        f"   correctness vs HF: max |dlogit| = {c['max_abs_logit_diff_vs_hf']:.3g}, tokens match = {c['tokens_match_hf']}",
        f"   exact ({ex['phase']['name']}, {ex['phase']['passes']} pass(es)): "
        f"FLOPs={sum(n['flops'] for n in ex['per_npu']):,}  msgs={ex['communication_totals']['num_messages']:,}  "
        f"sent={_mb(ex['communication_totals']['bytes_sent'])}  "
        f"max weights/NPU={_mb(max(n['weight_bytes'] for n in ex['per_npu']))}",
        f"   MODELED REEF ({cfg['hardware_profile']['compute_model']}): step={pp['step_ms']:.3f} ms "
        f"[compute {pp['compute_ms']:.3f} | memory {pp['memory_ms']:.3f} | comm {pp['comm_ms']:.3f}] "
        f"bound by {pp['bound_by']}",
        f"   TTFT={mo['time_to_first_token_ms']:.3f} ms  per-token="
        + (f"{mo['per_token_ms']:.3f} ms" if mo["per_token_ms"] is not None else "n/a")
        + f"  tokens/s={mo['tokens_per_second']:.1f}  MFU={mo['mfu']:.1%}",
        f"   laptop wall-clock (not Reef): median {me['median_total_ms']:.1f} ms, p90 {me['p90_total_ms']:.1f} ms "
        f"over {me['timed_runs']} runs",
    ]
    if path:
        lines.append(f"   -> {path}")
    print("\n".join(lines))
