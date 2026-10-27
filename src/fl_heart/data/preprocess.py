"""Per-site preprocessing (zero-as-missing, imputation, encoding, scaling, splitting).

Everything is fitted on a site's own TRAINING split only, to mirror a real federated
setting where statistics cannot be shared across institutions or leak from test data.

Design notes
------------
* No missingness-indicator features are added. At Switzerland (94% positive) a
  "value was missing" flag would let the model learn site identity as a proxy for the
  label, which inflates results without learning anything clinical.
* Categorical features use FIXED category sets from the config, so every site produces
  exactly the same columns even if a category never occurs locally. This keeps the
  feature space identical across clients, which FedAvg requires.
* A kept feature that is entirely missing in a site's training split raises an error
  instead of being silently imputed from nothing.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fl_heart.data.loader import TARGET, load_all_sites


def _safe_split(X: pd.DataFrame, y: pd.Series, test_size: float, seed: int, site: str, what: str):
    """Stratified split; falls back to a plain split if a class is too small to stratify."""
    try:
        return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    except ValueError:
        warnings.warn(f"[{site}] could not stratify the {what} split (class too small); "
                      "using a non-stratified split.")
        return train_test_split(X, y, test_size=test_size, random_state=seed)


def _fit_fill_values(train: pd.DataFrame, continuous, binary, categorical, site: str) -> dict:
    """Median for continuous, mode for binary/categorical, computed on TRAIN only."""
    fill = {}
    for col in list(continuous) + list(binary) + list(categorical):
        s = train[col].dropna()
        if s.empty:
            raise ValueError(f"[{site}] feature '{col}' is entirely missing in the training "
                             "split; remove it from the config feature set.")
        fill[col] = s.median() if col in continuous else s.mode().iloc[0]
    return fill


def _encode(df: pd.DataFrame, fill: dict, scaler: dict, continuous, binary, categorical):
    """Impute, scale continuous, keep binary, one-hot categorical (fixed categories)."""
    df = df.fillna(value=fill)
    cols, names = [], []
    for c in continuous:
        mean, std = scaler[c]
        cols.append(((df[c] - mean) / std).to_numpy(dtype=np.float32))
        names.append(c)
    for c in binary:
        cols.append(df[c].to_numpy(dtype=np.float32))
        names.append(c)
    for c, cats in categorical.items():
        for k in cats:
            cols.append((df[c] == k).to_numpy(dtype=np.float32))
            names.append(f"{c}={k}")
    return np.column_stack(cols).astype(np.float32), names


def preprocess_site(df: pd.DataFrame, cfg: dict, seed: int, site: str = "site") -> dict:
    """Clean, split, impute, scale and encode one institution's records.

    Returns {"train": (X, y), "val": (X, y), "test": (X, y), "feature_names": [...],
             "stats": {...}} with float32 X and int64 y.
    """
    d = cfg["data"]
    continuous, binary = list(d["continuous"]), list(d["binary"])
    categorical = {k: list(v) for k, v in d["categorical"].items()}
    features = continuous + binary + list(categorical)

    df = df.copy()
    # 0 means "not recorded" for these columns; only matters if they are kept as features.
    for col in d.get("zero_is_missing", []):
        if col in df.columns:
            df.loc[df[col] == 0, col] = np.nan

    y = (df[TARGET] > 0).astype(np.int64)
    X = df[features].astype(float)

    X_tv, X_test, y_tv, y_test = _safe_split(X, y, d["test_size"], seed, site, "test")
    X_train, X_val, y_train, y_val = _safe_split(X_tv, y_tv, d["val_size"], seed, site, "validation")

    fill = _fit_fill_values(X_train, continuous, binary, categorical, site)
    X_train_f = X_train.fillna(value=fill)
    scaler = {}
    for c in continuous:
        std = X_train_f[c].std(ddof=0)
        scaler[c] = (X_train_f[c].mean(), std if std > 0 else 1.0)

    out = {}
    for name, (Xs, ys) in {"train": (X_train, y_train), "val": (X_val, y_val),
                           "test": (X_test, y_test)}.items():
        arr, names = _encode(Xs, fill, scaler, continuous, binary, categorical)
        out[name] = (arr, ys.to_numpy(dtype=np.int64))
    out["feature_names"] = names
    out["stats"] = {
        "fill": {k: float(v) for k, v in fill.items()},
        "scaler": {k: (float(m), float(s)) for k, (m, s) in scaler.items()},
        "n": {k: len(out[k][1]) for k in ("train", "val", "test")},
        "positive_rate": {k: float(out[k][1].mean()) for k in ("train", "val", "test")},
    }
    return out


def preprocess_all(cfg: dict, seed: int) -> dict:
    """Preprocess every institution independently. Returns {site: preprocess_site(...)}."""
    raw = load_all_sites(cfg["data"]["raw_dir"])
    return {site: preprocess_site(raw[site], cfg, seed, site=site) for site in cfg["data"]["sites"]}


def pool_sites(site_data: dict) -> dict:
    """Concatenate per-site splits for the centralized baseline.

    Uses the already per-site-preprocessed arrays, so centralized, siloed and federated
    models see identical inputs and identical train/val/test rows. Differences in
    results are then attributable to the training paradigm alone (thesis Section 3.6).
    """
    pooled = {}
    for split in ("train", "val", "test"):
        pooled[split] = (
            np.concatenate([s[split][0] for s in site_data.values()]),
            np.concatenate([s[split][1] for s in site_data.values()]),
        )
    return pooled
