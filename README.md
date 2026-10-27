# Federated Learning for Privacy-Preserving Multi-Institutional Cardiovascular Disease Prediction

MSc thesis project (MIVA Open University). A federated learning (FedAvg) framework in which four
institutions from the UCI Heart Disease dataset (Cleveland, Hungary, Switzerland, VA Long Beach)
collaboratively train a heart disease classifier without sharing raw patient records.

The federated model is compared against two baselines:

| Setup | Description | Role |
|---|---|---|
| Siloed | One model per institution, trained on local data only | Lower bound |
| Centralized | One model trained on all pooled data | Upper bound |
| Federated | FedAvg across the four institutions (Flower simulation) | Proposed |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python scripts/download_data.py        # fetches the 4 UCI files into data/raw/
python scripts/audit_data.py           # per-site rows, missing values, label balance
make all                               # siloed + centralized + federated + comparison
```

## Repository layout

```
configs/            experiment settings (YAML)
data/raw/           original UCI files (not committed)
data/processed/     cleaned per-site datasets (not committed)
docs/figures/       thesis figures (architecture diagrams)
notebooks/          exploration only; no logic lives here
results/            metrics, figures, logs (generated, not committed)
scripts/            command-line entry points for each experiment
src/fl_heart/       the reusable package
  data/             loading and per-site preprocessing
  models/           feed-forward network
  training/         shared train/eval loops, siloed and centralized runs
  federated/        Flower client, FedAvg strategy, simulation
  evaluation/       metrics and result aggregation
  utils/            seeding, logging
tests/              unit tests
```

## Reproducibility

All runs read `configs/default.yaml` and are repeated over the seeds listed there.
Results are reported as mean +/- standard deviation across seeds.

## Data

UCI Heart Disease dataset (Detrano et al., 1989), public and de-identified.
Source: https://archive.ics.uci.edu/dataset/45/heart+disease
