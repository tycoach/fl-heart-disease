"""Run the siloed experiment over all seeds in the config."""
import json
from pathlib import Path

from fl_heart.config import load_config
from fl_heart.training.siloed import run_siloed
from fl_heart.utils.seed import set_seed

cfg = load_config()
out_dir = Path(cfg["output"]["metrics_dir"])
out_dir.mkdir(parents=True, exist_ok=True)

results = []
for seed in cfg["seed_list"]:
    set_seed(seed)
    results.append({"seed": seed, "metrics": run_siloed(cfg, seed)})

with open(out_dir / "siloed.json", "w") as fh:
    json.dump(results, fh, indent=2, default=float)
print("saved", out_dir / "siloed.json")
