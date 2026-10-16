"""Collect every GEMM shape used across result files into a CSV for the kernel team.

    python scripts/export_kernel_shapes.py results/ -o kernel_shapes.csv

Columns: op, M, K, N, dtype (one row per distinct shape; [M x K] . [K x N]). Ops are
``matmul`` (projections, MLP, lm_head) and the per-head attention GEMMs ``attn_scores``
(q . K^T) and ``attn_pv`` (probs . V). The kernel team adds a ``time_us`` column; the file
then feeds ``hardware_profile.compute_model: table`` / ``kernel_table``.
"""
import argparse
import csv

from _common import expand, load


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="result JSON files or directories")
    ap.add_argument("-o", "--output", default="kernel_shapes.csv")
    args = ap.parse_args(argv)

    shapes = set()
    files = expand(args.files)
    for p in files:
        for s in load(p)["exact"]["matmul_shapes"]:
            shapes.add((s["op"], s["M"], s["K"], s["N"], s["dtype"]))
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["op", "M", "K", "N", "dtype"])
        w.writerows(sorted(shapes))
    print(f"wrote {len(shapes)} shapes from {len(files)} result file(s) to {args.output}")


if __name__ == "__main__":
    main()
