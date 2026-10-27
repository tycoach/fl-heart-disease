"""FedAvg strategy with per-round logging of loss and communication volume."""


def build_strategy(cfg: dict, log):
    """TODO: weighted FedAvg (by n_samples) + on-evaluate hooks that record
    validation loss per round and bytes exchanged per round."""
    raise NotImplementedError
