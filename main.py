"""Single CLI entry point for every TA-FU experiment (Section C: "Exact
Experiment Commands").

Examples
--------
    python main.py --experiment ablation --dataset both --seeds 5
    python main.py --experiment noise_sweep --dataset bank --seeds 3
    python main.py --experiment scalability --dataset both --seeds 3
    python main.py --experiment participation --dataset both --seeds 3
    python main.py --experiment all --seeds 5
    python main.py --experiment report

`--experiment all` runs ablation at --seeds, then noise_sweep/
scalability/participation at min(--seeds, 3) (documented reduced
default -- see docs/METHODOLOGY.md, "Seed counts by experiment"), then
generates every table/figure. Equivalent to running
scripts/run_all_experiments.sh followed by scripts/generate_report.py.
"""
from __future__ import annotations
import argparse

from ta_fu import config as cfg_module
from ta_fu.experiments.run_ablation import run_ablation
from ta_fu.experiments.run_noise_sweep import run_noise_sweep
from ta_fu.experiments.run_scalability import run_scalability
from ta_fu.experiments.run_participation import run_participation
from ta_fu.utils import save_json


def main():
    ap = argparse.ArgumentParser(description="TA-FU experiment runner.")
    ap.add_argument("--experiment",
                     choices=["ablation", "noise_sweep", "scalability", "participation",
                              "all", "report"],
                     default="all")
    ap.add_argument("--dataset", choices=["bank", "breast_cancer", "both"], default="both")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    datasets = ["bank", "breast_cancer"] if args.dataset == "both" else [args.dataset]
    seeds_main = cfg_module.RANDOM_SEEDS[:args.seeds]
    seeds_sweep = cfg_module.RANDOM_SEEDS[:min(args.seeds, 3)]

    if args.experiment in ("ablation", "all"):
        for ds in datasets:
            save_json(run_ablation(ds, seeds_main), f"{args.out_dir}/ablation_{ds}.json")
    if args.experiment in ("noise_sweep", "all"):
        for ds in datasets:
            save_json(run_noise_sweep(ds, seeds_sweep), f"{args.out_dir}/noise_sweep_{ds}.json")
    if args.experiment in ("scalability", "all"):
        combined = {"seeds_used": seeds_sweep, "by_dataset": {}}
        for ds in datasets:
            combined["by_dataset"][ds] = run_scalability(ds, seeds_sweep)
        save_json(combined, f"{args.out_dir}/scalability_results.json")
    if args.experiment in ("participation", "all"):
        for ds in datasets:
            save_json(run_participation(ds, seeds_sweep), f"{args.out_dir}/participation_{ds}.json")
    if args.experiment in ("report", "all"):
        import subprocess
        import sys
        subprocess.run([sys.executable, "scripts/generate_report.py"], check=True)


if __name__ == "__main__":
    main()
