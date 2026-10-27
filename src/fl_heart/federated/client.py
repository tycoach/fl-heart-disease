"""Flower NumPyClient for one institution."""


class HospitalClient:
    """TODO: wraps a site's DataLoaders; fit() trains locally for cfg.federated.local_epochs
    and returns weights + n_samples; evaluate() returns local validation loss."""
