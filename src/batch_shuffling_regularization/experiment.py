import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from batch_shuffling_regularization.data import MnistData, load_mnist, sort_train_by_class
from batch_shuffling_regularization.model import build_mlp
from batch_shuffling_regularization.shuffling import VARIANTS
from batch_shuffling_regularization.training import BatchOrder, TrainConfig, make_deterministic, train

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
SEEDS = range(10)


@dataclass(frozen=True)
class Variant:
    name: str
    batch_order: BatchOrder
    sort_by_class: bool = False


EXPERIMENT_VARIANTS = [Variant(name, batch_order) for name, batch_order in VARIANTS.items()] + [
    Variant("E", VARIANTS["A"], sort_by_class=True),
]


def run_path(variant: str, seed: int) -> Path:
    return RESULTS_DIR / "runs" / f"{variant}_seed{seed}.csv"


def run_experiment(data: MnistData, cfg: TrainConfig) -> None:
    sorted_data = sort_train_by_class(data)
    for seed in SEEDS:
        for variant in EXPERIMENT_VARIANTS:
            path = run_path(variant.name, seed)
            if path.exists():
                continue
            start = time.perf_counter()
            # Same seed -> same initial weights for every variant, so runs can be compared pairwise.
            model = build_mlp(seed=seed, device=data.train.x.device)
            history = train(model, sorted_data if variant.sort_by_class else data, variant.batch_order, cfg, order_seed=seed)
            path.parent.mkdir(parents=True, exist_ok=True)
            history.to_csv(path)
            print(f"{variant.name} seed {seed}: best val_acc {history['val_acc'].max():.4f} "
                  f"({time.perf_counter() - start:.0f} s)", flush=True)


def load_results() -> pd.DataFrame:
    frames = [
        pd.read_csv(run_path(variant.name, seed)).assign(variant=variant.name, seed=seed)
        for variant in EXPERIMENT_VARIANTS
        for seed in SEEDS
    ]
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    make_deterministic()
    run_experiment(load_mnist(torch.device("cuda")), TrainConfig())
