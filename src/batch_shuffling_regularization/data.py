from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torchvision.datasets import MNIST, FashionMNIST

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Both datasets share the MNIST format: 60k train / 10k test grayscale 28x28 images, 10 classes.
DATASETS = {"mnist": MNIST, "fashion_mnist": FashionMNIST}


@dataclass
class Split:
    x: torch.Tensor
    y: torch.Tensor

    def __len__(self) -> int:
        return len(self.y)


@dataclass
class DataSplits:
    train: Split
    val: Split
    test: Split


def load_dataset(device: torch.device, name: str = "mnist", n_val: int = 10_000, split_seed: int = 0) -> DataSplits:
    dataset = DATASETS[name]
    train_raw = dataset(DATA_DIR, train=True, download=True)
    test_raw = dataset(DATA_DIR, train=False, download=True)

    x_all = train_raw.data.reshape(-1, 28 * 28).float() / 255
    y_all = train_raw.targets

    # Validation samples are drawn at random, but the training set keeps the original
    # order of the dataset so that "no shuffling" really trains on the raw sequence.
    perm = torch.randperm(len(y_all), generator=torch.Generator().manual_seed(split_seed))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:].sort().values

    # Normalisation statistics come from the training split only.
    mean = x_all[train_idx].mean()
    std = x_all[train_idx].std()

    def make_split(x: torch.Tensor, y: torch.Tensor) -> Split:
        return Split(((x - mean) / std).to(device), y.to(device))

    x_test = test_raw.data.reshape(-1, 28 * 28).float() / 255
    return DataSplits(
        train=make_split(x_all[train_idx], y_all[train_idx]),
        val=make_split(x_all[val_idx], y_all[val_idx]),
        test=make_split(x_test, test_raw.targets),
    )


def load_mnist(device: torch.device) -> DataSplits:
    return load_dataset(device, "mnist")


def sort_train_by_class(data: DataSplits) -> DataSplits:
    # Stable sort: within a class the original order is kept.
    order = data.train.y.argsort(stable=True)
    return replace(data, train=Split(data.train.x[order], data.train.y[order]))
