"""Run one config:  python run.py configs/tp4_prefill.yaml"""
import argparse
import os

from reefdist.config import load_config
from reefdist.runtime.launcher import run


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", help="YAML config file")
    ap.add_argument("--output-dir", help="override output_dir")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    name = os.path.splitext(os.path.basename(args.config))[0] + ".json"
    run(cfg, name=name)


if __name__ == "__main__":
    main()
