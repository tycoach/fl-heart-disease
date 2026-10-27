"""Shared training and evaluation loops (used by siloed, centralized and FL clients)."""


def train_one_epoch(model, loader, optimizer, device):
    """TODO"""
    raise NotImplementedError


def evaluate(model, loader, device) -> dict:
    """TODO: return loss and probability/label arrays for metric computation."""
    raise NotImplementedError


def fit(model, train_loader, val_loader, cfg: dict, device):
    """TODO: epochs loop with early stopping; returns best model state."""
    raise NotImplementedError
