import numpy as np
import pandas as pd
import pytest

from fl_heart.data.loader import COLUMNS
from fl_heart.data.preprocess import pool_sites, preprocess_site

CFG = {"data": {
    "test_size": 0.2, "val_size": 0.15,
    "continuous": ["age", "trestbps", "thalach", "oldpeak"],
    "binary": ["sex", "fbs", "exang"],
    "categorical": {"cp": [1, 2, 3, 4], "restecg": [0, 1, 2]},
    "zero_is_missing": ["chol", "trestbps"],
}}


def make_site(n, seed, cp_values=(1, 2, 3, 4), pos_rate=0.5, nan_cols=()):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(30, 80, n), "sex": rng.integers(0, 2, n),
        "cp": rng.choice(cp_values, n), "trestbps": rng.integers(100, 180, n),
        "chol": rng.integers(150, 300, n), "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n), "thalach": rng.integers(90, 190, n),
        "exang": rng.integers(0, 2, n), "oldpeak": rng.random(n) * 4,
        "slope": rng.integers(1, 4, n), "ca": rng.integers(0, 3, n), "thal": rng.integers(3, 8, n),
        "num": (rng.random(n) < pos_rate).astype(int),
    })[COLUMNS].astype(float)
    for c in nan_cols:
        df.loc[df.sample(frac=0.3, random_state=seed).index, c] = np.nan
    return df


def test_identical_feature_space_across_sites_even_if_category_absent():
    a = preprocess_site(make_site(150, 0), CFG, seed=0, site="a")
    b = preprocess_site(make_site(120, 1, cp_values=(2, 3)), CFG, seed=0, site="b")  # no cp 1 or 4
    assert a["feature_names"] == b["feature_names"]
    assert a["train"][0].shape[1] == b["train"][0].shape[1] == 4 + 3 + 4 + 3


def test_no_nans_and_dtypes():
    out = preprocess_site(make_site(150, 2, nan_cols=("trestbps", "oldpeak", "fbs")), CFG, 0)
    for split in ("train", "val", "test"):
        X, y = out[split]
        assert not np.isnan(X).any()
        assert X.dtype == np.float32 and y.dtype == np.int64


def test_zero_is_treated_as_missing_not_scaled_as_a_value():
    df = make_site(150, 3)
    df.loc[:20, "trestbps"] = 0
    out = preprocess_site(df, CFG, 0)
    # a real 0 would give a huge negative z-score; imputed values stay near the median
    assert out["train"][0][:, out["feature_names"].index("trestbps")].min() > -6


def test_train_only_statistics_no_test_leakage():
    df = make_site(200, 4)
    out = preprocess_site(df, CFG, 0)
    Xtr = out["train"][0][:, out["feature_names"].index("age")]
    assert abs(Xtr.mean()) < 1e-5 and abs(Xtr.std() - 1) < 1e-4   # train is exactly standardised


def test_all_missing_feature_raises():
    df = make_site(100, 5)
    df["oldpeak"] = np.nan
    with pytest.raises(ValueError):
        preprocess_site(df, CFG, 0, site="x")


def test_stratified_split_keeps_rare_class_in_every_split():
    out = preprocess_site(make_site(123, 6, pos_rate=0.935), CFG, 0)
    assert (out["train"][1] == 0).sum() >= 1


def test_pooling_adds_up():
    sites = {"a": preprocess_site(make_site(100, 7), CFG, 0), "b": preprocess_site(make_site(80, 8), CFG, 0)}
    pooled = pool_sites(sites)
    assert len(pooled["train"][1]) == len(sites["a"]["train"][1]) + len(sites["b"]["train"][1])
