"""Read the four UCI Heart Disease site files."""
from pathlib import Path
import pandas as pd

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal", "num",
]
FEATURES = COLUMNS[:-1]
TARGET = "num"
SITES = ["cleveland", "hungarian", "switzerland", "va"]


def load_site(site: str, raw_dir: str | Path = "data/raw") -> pd.DataFrame:
    """Return one institution's raw records ('?' parsed as missing)."""
    path = Path(raw_dir) / f"processed.{site}.data"
    return pd.read_csv(path, header=None, names=COLUMNS, na_values="?")


def load_all_sites(raw_dir: str | Path = "data/raw") -> dict[str, pd.DataFrame]:
    return {site: load_site(site, raw_dir) for site in SITES}
