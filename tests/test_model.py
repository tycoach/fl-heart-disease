import pytest

torch = pytest.importorskip("torch")
from fl_heart.models.mlp import HeartMLP


def test_output_shape():
    model = HeartMLP(in_features=13, hidden_sizes=(64, 32), dropout=0.3)
    out = model(torch.randn(8, 13))
    assert out.shape == (8,)
