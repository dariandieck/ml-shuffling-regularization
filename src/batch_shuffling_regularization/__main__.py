import argparse
import asyncio
import sys
import time
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

from batch_shuffling_regularization.experiment import EXPERIMENTS, SEEDS, format_duration, run_experiment
from batch_shuffling_regularization.training import device_name, get_device, make_deterministic

NOTEBOOK_DIR = Path(__file__).resolve().parents[2] / "notebooks"
ANALYSIS_NOTEBOOK = NOTEBOOK_DIR / "04_analysis.ipynb"
FIGURE_DIR = NOTEBOOK_DIR.parent / "figures"


class ProgressExecutor(ExecutePreprocessor):
    def preprocess_cell(self, cell, resources, index):
        if cell.cell_type == "markdown" and cell.source.startswith("## "):
            print(f"  {cell.source.splitlines()[0][3:]}", flush=True)
        return super().preprocess_cell(cell, resources, index)


def run(experiments: list[str], shard: str) -> int:
    index, count = map(int, shard.split("/"))
    seeds = [seed for seed in SEEDS if seed % count == index]
    names = experiments or list(EXPERIMENTS)
    make_deterministic()
    device = get_device()
    print(f"Device: {device_name(device)}" + (" (no CUDA GPU found, training is much slower)" if device.type == "cpu" else ""))
    print(f"Experiments: {', '.join(names)}")
    print(f"Seeds: {seeds}" + (f" (shard {shard})" if count > 1 else ""), flush=True)
    start = time.perf_counter()
    trained = sum(run_experiment(EXPERIMENTS[name], device, seeds) for name in names)
    if trained:
        print(f"\nTrained {trained} runs in {format_duration(time.perf_counter() - start)}, results are in results/", flush=True)
    else:
        print("\nNothing to train, all runs are already in results/", flush=True)
    return trained


def analyse() -> None:
    print(f"\nExecuting notebooks/{ANALYSIS_NOTEBOOK.name} (reads results/, writes figures/)", flush=True)
    start, start_wall = time.perf_counter(), time.time()
    if sys.platform == "win32":
        # zmq needs add_reader, which the default Proactor loop lacks; without this it warns and starts a helper thread.
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    nb = nbformat.read(ANALYSIS_NOTEBOOK, as_version=4)
    # The kernel warns that it talks over unencrypted TCP. It only listens on localhost, so the warning is just noise here.
    executor = ProgressExecutor(timeout=None, kernel_name="python3", extra_arguments=["--IPKernelApp.log_level=ERROR"])
    executor.preprocess(nb, {"metadata": {"path": str(NOTEBOOK_DIR)}})
    nbformat.write(nb, ANALYSIS_NOTEBOOK)
    figures = sorted(path.name for path in FIGURE_DIR.glob("*.png") if path.stat().st_mtime >= start_wall)
    print(f"\nDone in {format_duration(time.perf_counter() - start)}: updated {ANALYSIS_NOTEBOOK.name} and {len(figures)} figures in figures/")
    for name in figures:
        print(f"  {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="shuffling", description="Does shuffling the training data act as a regularizer?")
    commands = parser.add_subparsers(dest="command", required=True)

    experiments_help = f"experiments to run, any of {', '.join(EXPERIMENTS)} (default: all)"
    shard_help = "i/n: only run seeds with seed %% n == i, to run several processes in parallel"
    for name, help_text in [
        ("run", "train all runs that are not in results/ yet"),
        ("all", "run, then analyse"),
    ]:
        command = commands.add_parser(name, help=help_text)
        command.add_argument("experiments", nargs="*", help=experiments_help)
        command.add_argument("--shard", default="0/1", help=shard_help)
    commands.add_parser("analyse", help="execute notebooks/04_analysis.ipynb and write the figures")
    args = parser.parse_args()
    if unknown := [name for name in getattr(args, "experiments", []) if name not in EXPERIMENTS]:
        parser.error(f"unknown experiment {', '.join(unknown)}, choose from {', '.join(EXPERIMENTS)}")
    return args


def main() -> None:
    args = parse_args()
    if args.command in ("run", "all"):
        trained = run(args.experiments, args.shard)
        if args.command == "run" and trained:
            print("Run `shuffling analyse` to update the figures.")
    if args.command in ("analyse", "all"):
        analyse()


if __name__ == "__main__":
    main()
