"""Per-site preprocessing (imputation, encoding, scaling, splitting).

Everything is fitted on a site's own TRAINING split only, to mirror a real
federated setting where statistics cannot be shared across institutions.
"""


def preprocess_site(df, cfg: dict, seed: int):
    """TODO: binarise target, impute, encode categoricals, scale, split.

    Returns train/val/test arrays (X, y) for the site.
    """
    raise NotImplementedError


def pool_sites(site_splits: dict):
    """TODO: concatenate per-site splits for the centralized baseline."""
    raise NotImplementedError
