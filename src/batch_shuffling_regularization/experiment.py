import argparse
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from batch_shuffling_regularization.data import load_dataset, sort_train_by_class
from batch_shuffling_regularization.model import build_mlp
from batch_shuffling_regularization.shuffling import VARIANTS
from batch_shuffling_regularization.training import BatchOrder, TrainConfig, make_deterministic, train

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
SEEDS = range(20)


@dataclass(frozen=True)
class Variant:
    name: str
    batch_order: BatchOrder
    sort_by_class: bool = False


EXPERIMENT_VARIANTS = [Variant(name, batch_order) for name, batch_order in VARIANTS.items()] + [
    Variant("E", VARIANTS["A"], sort_by_class=True),
]


@dataclass(frozen=True)
class Experiment:
    name: str
    dataset: str
    cfg: TrainConfig


# Experiments 2 and 3 each change one thing compared to experiment 1. With batch size 10 the
# learning rate is scaled down by the same factor, so lr / batch_size stays constant.
EXPERIMENTS = {
    experiment.name: experiment
    for experiment in [
        Experiment("mnist_bs100", "mnist", TrainConfig(epochs=30, batch_size=100, lr=0.1)),
        Experiment("mnist_bs10", "mnist", TrainConfig(epochs=30, batch_size=10, lr=0.01)),
        Experiment("fashion_bs100", "fashion_mnist", TrainConfig(epochs=30, batch_size=100, lr=0.1)),
    ]
}


def run_path(experiment: Experiment, variant: str, seed: int) -> Path:
    return RESULTS_DIR / experiment.name / "runs" / f"{variant}_seed{seed}.csv"


def run_experiment(experiment: Experiment, device: torch.device, seeds: Iterable[int] = SEEDS) -> None:
    data = load_dataset(device, experiment.dataset)
    sorted_data = sort_train_by_class(data)
    for seed in seeds:
        for variant in EXPERIMENT_VARIANTS:
            path = run_path(experiment, variant.name, seed)
            if path.exists():
                continue
            start = time.perf_counter()
            # Same seed -> same initial weights for every variant, so runs can be compared pairwise.
            model = build_mlp(seed=seed, device=device)
            run_data = sorted_data if variant.sort_by_class else data
            history = train(model, run_data, variant.batch_order, experiment.cfg, order_seed=seed)
            path.parent.mkdir(parents=True, exist_ok=True)
            history.to_csv(path)
            print(f"{experiment.name} {variant.name} seed {seed}: best val_acc {history['val_acc'].max():.4f} "
                  f"({time.perf_counter() - start:.0f} s)", flush=True)


def load_results(experiment: Experiment) -> pd.DataFrame:
    # Only finished runs are loaded; check completeness before analysing a partial experiment.
    frames = [
        pd.read_csv(path).assign(variant=variant.name, seed=seed)
        for variant in EXPERIMENT_VARIANTS
        for seed in SEEDS
        if (path := run_path(experiment, variant.name, seed)).exists()
    ]
    return pd.concat(frames, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the shuffling experiments.")
    parser.add_argument("experiments", nargs="*", help=f"experiments to run, any of {', '.join(EXPERIMENTS)} (default: all)")
    parser.add_argument("--shard", default="0/1", help="i/n: only run seeds with seed %% n == i, to run several processes in parallel")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    index, count = map(int, args.shard.split("/"))
    seeds = [seed for seed in SEEDS if seed % count == index]
    make_deterministic()
    for name in args.experiments or EXPERIMENTS:
        run_experiment(EXPERIMENTS[name], torch.device("cuda"), seeds)
