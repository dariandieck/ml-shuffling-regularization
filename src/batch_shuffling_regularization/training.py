import os
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from batch_shuffling_regularization.data import MnistData, Split

BatchOrder = Callable[[int, int, torch.Generator], torch.Tensor]


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 100
    lr: float = 0.1


def make_deterministic() -> None:
    # Required by cuBLAS for deterministic matrix multiplications; must be set before the first CUDA matmul.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)


@torch.no_grad()
def evaluate(model: nn.Module, split: Split) -> tuple[float, float]:
    model.eval()
    logits = model(split.x)
    loss = F.cross_entropy(logits, split.y).item()
    acc = (logits.argmax(dim=1) == split.y).float().mean().item()
    return loss, acc


def epoch_metrics(model: nn.Module, data: MnistData, epoch: int) -> dict[str, float]:
    row: dict[str, float] = {"epoch": epoch}
    for name, split in [("train", data.train), ("val", data.val), ("test", data.test)]:
        row[f"{name}_loss"], row[f"{name}_acc"] = evaluate(model, split)
    return row


def train(model: nn.Module, data: MnistData, batch_order: BatchOrder, cfg: TrainConfig, order_seed: int) -> pd.DataFrame:
    if len(data.train) % cfg.batch_size != 0:
        raise ValueError("batch_size must divide the training set size")

    device = data.train.x.device
    gen = torch.Generator().manual_seed(order_seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)

    # Metrics are computed on the full splits after each epoch with fixed weights. The running
    # average of the batch losses during an epoch would not be comparable to the validation loss.
    history = [epoch_metrics(model, data, epoch=0)]
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for idx in batch_order(len(data.train), cfg.batch_size, gen).to(device):
            loss = F.cross_entropy(model(data.train.x[idx]), data.train.y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        history.append(epoch_metrics(model, data, epoch))

    return pd.DataFrame(history).set_index("epoch")
