import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def expand(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += sorted(glob.glob(os.path.join(p, "*.json")))
        else:
            out += sorted(glob.glob(p)) or [p]
    return out


def load(path):
    with open(path) as f:
        r = json.load(f)
    r["_path"] = path
    return r


def label(r):
    plan = r["exact"]["plan"]
    s = {"tensor_parallel": "tp", "pipeline_parallel": "pp", "hybrid": "hybrid"}.get(plan["strategy"], plan["strategy"])
    shape = f"{plan['tp']}x{plan['pp']}" if plan["strategy"] == "hybrid" else str(plan["num_npus"])
    return s + shape + ("+fusedqkv" if plan["fused_qkv"] else "")


def table(headers, rows):
    rows = [["" if v is None else v for v in row] for row in rows]
    cells = [[_fmt(v) for v in row] for row in rows]
    widths = [max(len(str(h)), *(len(r[i]) for r in cells)) for i, h in enumerate(headers)]
    line = lambda vals: "  ".join(str(v).rjust(w) for v, w in zip(vals, widths))  # noqa: E731
    return "\n".join([line(headers), line("-" * w for w in widths)] + [line(r) for r in cells])


def _fmt(v):
    if isinstance(v, bool):
        return "yes" if v else "NO"
    if isinstance(v, float):
        return f"{v:.4g}" if abs(v) < 1e4 else f"{v:,.0f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)
