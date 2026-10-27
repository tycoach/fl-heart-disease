import numpy as np
from fl_heart.evaluation.metrics import compute_metrics


def test_perfect_predictions():
    m = compute_metrics([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2])
    assert m["accuracy"] == 1.0 and m["f1"] == 1.0 and m["auc_roc"] == 1.0


def test_single_class_auc_is_nan():
    m = compute_metrics([1, 1, 1], [0.6, 0.7, 0.8])
    assert np.isnan(m["auc_roc"])
