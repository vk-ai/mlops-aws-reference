"""PyTorch MLP for default risk. Standardisation is stored in the module so it takes raw features."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class CreditRiskMLP(nn.Module):
    def __init__(self, n_features: int, hidden: list[int], mean: np.ndarray, std: np.ndarray):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.1)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(np.where(std == 0, 1.0, std), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # returns P(default)
        return torch.sigmoid(self.net((x - self.mean) / self.std)).squeeze(-1)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / self.std).squeeze(-1)
