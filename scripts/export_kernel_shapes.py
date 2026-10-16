"""Collect every kernel shape used across result files into a CSV for the kernel team.

    python scripts/export_kernel_shapes.py results/ -o kernel_shapes.csv

Columns: op, batch, M, K, N, dtype (one row per distinct per-NPU shape, in the convention
of perfmodel/flops.py: GEMMs are ``batch`` x [M x K] . [K x N]; elementwise ops have K = 1).
Reads ``exact.kernel_shapes``, which covers every op the counters record: GEMMs (``matmul``,
``attn_scores``, ``attn_pv``; attention with ``batch = batch_size x heads on this NPU``) and
elementwise / memory ops. The kernel team measures a grid around these shapes and adds
``time_us_warm`` (and friends, see README "Compute model"); the file then feeds
``hardware_profile.compute_model: table`` / ``kernel_table``.
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
        for k in load(p)["exact"]["kernel_shapes"]:
            shapes.add((k["op"], k["batch"], k["M"], k["K"], k["N"], k["dtype"]))
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["op", "batch", "M", "K", "N", "dtype"])
        w.writerows(sorted(shapes))
    print(f"wrote {len(shapes)} shapes from {len(files)} result file(s) to {args.output}")


if __name__ == "__main__":
    main()
