"""Aggregate results across seeds and build the federated-vs-centralized-vs-siloed table."""


def aggregate_runs(run_results: list[dict]) -> dict:
    """TODO: mean +/- std per metric across seeds."""
    raise NotImplementedError


def build_comparison_table(siloed: dict, centralized: dict, federated: dict):
    """TODO: return a DataFrame ready for the thesis results chapter."""
    raise NotImplementedError
