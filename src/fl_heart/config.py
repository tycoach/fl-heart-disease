"""Load experiment configuration from YAML."""
from pathlib import Path
import yaml


def load_config(path: str | Path = "configs/default.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
