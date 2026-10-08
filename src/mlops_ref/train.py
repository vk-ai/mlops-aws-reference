"""Training loop (CPU, seeded) and evaluation."""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data import TARGET
from .features import MODEL_FEATURES
from .metrics import classification_metrics
from .model import CreditRiskMLP


@dataclass
class TrainResult:
    model: CreditRiskMLP
    train_metrics: dict
    history: list[float]
    threshold: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def to_xy(df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.tensor(df[MODEL_FEATURES].to_numpy(np.float32)), torch.tensor(df[TARGET].to_numpy(np.float32))


def predict_proba(model: CreditRiskMLP, df: pd.DataFrame) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(df[MODEL_FEATURES].to_numpy(np.float32))).numpy()


def train(train_df: pd.DataFrame, hidden: list[int], epochs: int, lr: float, weight_decay: float, batch_size: int, seed: int) -> TrainResult:
    seed_everything(seed)
    torch.set_num_threads(1)  # determinism on CPU
    x, y = to_xy(train_df)
    model = CreditRiskMLP(x.shape[1], hidden, x.mean(0).numpy(), x.std(0).numpy())
    pos_weight = ((len(y) - y.sum()) / y.sum()).detach()  # class imbalance (70/30)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(seed)
    history = []
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(x), generator=gen)
        total = 0.0
        for i in range(0, len(x), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model.logits(x[idx]), y[idx])
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        history.append(total / len(x))
    p = predict_proba(model, train_df)
    return TrainResult(model, classification_metrics(train_df[TARGET].to_numpy(), p), history, 0.5)


def evaluate(model: CreditRiskMLP, df: pd.DataFrame, threshold: float = 0.5) -> dict:
    return classification_metrics(df[TARGET].to_numpy(), predict_proba(model, df), threshold)
