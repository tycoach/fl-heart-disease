"""Feed-forward network from thesis Section 3.5 (Figure 3.1)."""
import torch.nn as nn


class HeartMLP(nn.Module):
    """Input -> [Linear-ReLU-Dropout] x N -> Linear -> (sigmoid in the loss)."""

    def __init__(self, in_features: int, hidden_sizes=(64, 32), dropout: float = 0.3):
        super().__init__()
        layers, prev = [], in_features
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Returns logits; use BCEWithLogitsLoss for training and sigmoid for probabilities.
        return self.net(x).squeeze(-1)
