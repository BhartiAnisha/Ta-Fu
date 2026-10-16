# TA-FU: Threat-Aware Federated Unlearning

Research codebase for *"Adaptive Threat-Aware Federated Unlearning for
Mitigating Gradient Leakage and Reconstruction Attacks"*.

This is a **research-oriented** codebase: the goal is scientifically
defensible findings, not a deployable system. There is no serving layer,
no packaging, no infra — just data, a from-scratch MLP, an attack, a
defense, unlearning, and the experiments/tables/figures needed to support
a journal manuscript.

## Layout

```
threat_aware_federated_unlearning.py   original single-file implementation (UNCHANGED, kept for reference)
banking_management.csv, breast_cancer.csv   datasets (unchanged)
results_bank.json, results_breast_cancer.json, results.json   original reference results (unchanged, Section 31)

ta_fu/                  strengthened, modular package
  config.py             all tunables in one place
  data.py                dataset loading + generalized non-IID client partitioning
  model.py               manual MLP forward/backward + metrics
  attack.py              DLG-style reconstruction attack
  defense.py             clipping + fixed/adaptive noise (Baselines A/B/C/E)
  federated.py           FedAvg + client-participation subsampling + comm accounting
  unlearning.py          gradient-ascent unlearning + gold-standard retraining (Baselines D/F)
  metrics.py             forgetting gap, gold gap, mean+/-std, paired significance tests
  pipeline.py            the single unified train->attack->unlearn->evaluate engine
  reporting.py           builds derived JSON + markdown tables from results/*.json
  figures.py             builds all journal figures from results/*.json
  experiments/
    run_ablation.py       Sections 7 + 12 (ablation + multi-seed)
    run_noise_sweep.py    Sections 8 + 23 (fixed vs adaptive noise sweep)
    run_scalability.py    Section 15 (client-count scalability)
    run_participation.py  Section 16 (partial client participation)

main.py                 single CLI entry point (see below)
scripts/
  run_all_experiments.sh  runs every experiment sequentially (set PYTHON=... to choose the interpreter)
  generate_report.py      builds every table/figure from results/*.json

results/   raw + aggregated experiment JSON (git-trackable, not huge)
tables/    markdown tables for the manuscript
figures/   PNG figures for the manuscript
docs/      threat model, DP positioning, methodology, architecture review, paper draft
```

## Setup

```bash
python -m venv .venv                  # Python 3.11+
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Datasets (`banking_management.csv`, `breast_cancer.csv`) are read from the repository root,
regardless of the directory you run from. Tests: `python -m unittest discover tests`.

## Running experiments

```bash
python main.py --experiment ablation --dataset both --seeds 5
python main.py --experiment noise_sweep --dataset both --seeds 3
python main.py --experiment scalability --dataset both --seeds 3
python main.py --experiment participation --dataset both --seeds 3
python main.py --experiment report          # builds tables/ and figures/ from results/
python main.py --experiment all --seeds 5   # everything, end to end
```

Or directly:

```bash
python -m ta_fu.experiments.run_ablation --dataset both --seeds 5
python scripts/generate_report.py
```


## Reproducibility

Every result in `results/*.json` records exactly which seeds were used.
Nothing in `tables/` or `figures/` is generated from a number that isn't
traceable back to a `results/*.json` file; anything not yet executed is
labelled "requires execution" rather than fabricated.

## Reproducibility note

The files in `results/` were produced on Windows with Python 3.11, numpy 2.4.6, pandas 3.0.5 and
scipy 1.17.1. Re-running on the same machine gives identical output (checked: ablation, breast_cancer,
5 seeds). A re-run on Linux (numpy 2.4.4, pandas 3.0.2) reproduced the no-defense numbers exactly but
gave different values for the noise-based configurations (for example the fixed-noise
reconstruction-MSE increase), so noise-dependent results are not bit-reproducible across platforms or
library versions. Treat small differences there as expected and compare means over seeds.
