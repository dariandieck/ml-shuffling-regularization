import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from batch_shuffling_regularization.data import load_dataset, sort_train_by_class
from batch_shuffling_regularization.model import build_mlp
from batch_shuffling_regularization.shuffling import VARIANTS
from batch_shuffling_regularization.training import BatchOrder, TrainConfig, train

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


def format_duration(seconds: float) -> str:
    minutes, secs = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min" if hours else f"{minutes} min {secs:02d} s" if minutes else f"{secs} s"


def run_experiment(experiment: Experiment, device: torch.device, seeds: Iterable[int] = SEEDS) -> int:
    seeds = list(seeds)
    todo = [
        (seed, variant)
        for seed in seeds
        for variant in EXPERIMENT_VARIANTS
        if not run_path(experiment, variant.name, seed).exists()
    ]
    total = len(seeds) * len(EXPERIMENT_VARIANTS)
    cfg = experiment.cfg
    print(f"\n{experiment.name}: {experiment.dataset}, batch size {cfg.batch_size}, lr {cfg.lr}, {cfg.epochs} epochs")
    if not todo:
        print(f"  All {total} runs are already in results/{experiment.name}/runs, nothing to train", flush=True)
        return 0
    print(f"  {total - len(todo)}/{total} runs already done, training {len(todo)}", flush=True)

    data = load_dataset(device, experiment.dataset)
    sorted_data = sort_train_by_class(data)
    start = time.perf_counter()
    for i, (seed, variant) in enumerate(todo, start=1):
        run_start = time.perf_counter()
        # Same seed -> same initial weights for every variant, so runs can be compared pairwise.
        model = build_mlp(seed=seed, device=device)
        run_data = sorted_data if variant.sort_by_class else data
        history = train(model, run_data, variant.batch_order, cfg, order_seed=seed)
        path = run_path(experiment, variant.name, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        history.to_csv(path)
        # Extrapolated from the runs so far on this machine.
        remaining = (time.perf_counter() - start) / i * (len(todo) - i)
        print(f"  [{i:>{len(str(len(todo)))}}/{len(todo)}] {variant.name} seed {seed}: best val_acc {history['val_acc'].max():.4f} "
              f"({time.perf_counter() - run_start:.0f} s" + (f", ~{format_duration(remaining)} left)" if i < len(todo) else ")"), flush=True)
    return len(todo)


def load_results(experiment: Experiment) -> pd.DataFrame:
    # Only finished runs are loaded; check completeness before analysing a partial experiment.
    frames = [
        pd.read_csv(path).assign(variant=variant.name, seed=seed)
        for variant in EXPERIMENT_VARIANTS
        for seed in SEEDS
        if (path := run_path(experiment, variant.name, seed)).exists()
    ]
    return pd.concat(frames, ignore_index=True)
