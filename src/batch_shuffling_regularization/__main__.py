import argparse
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

from batch_shuffling_regularization.experiment import EXPERIMENTS, SEEDS, run_experiment
from batch_shuffling_regularization.training import device_name, get_device, make_deterministic

NOTEBOOK_DIR = Path(__file__).resolve().parents[2] / "notebooks"
ANALYSIS_NOTEBOOK = NOTEBOOK_DIR / "04_analysis.ipynb"


def run(experiments: list[str], shard: str) -> None:
    index, count = map(int, shard.split("/"))
    seeds = [seed for seed in SEEDS if seed % count == index]
    make_deterministic()
    device = get_device()
    print(f"Device: {device_name(device)}, seeds: {seeds}", flush=True)
    for name in experiments or EXPERIMENTS:
        run_experiment(EXPERIMENTS[name], device, seeds)


def analyse() -> None:
    # Executes the analysis notebook in place, which also rewrites the figures in figures/.
    nb = nbformat.read(ANALYSIS_NOTEBOOK, as_version=4)
    ExecutePreprocessor(timeout=None, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(NOTEBOOK_DIR)}})
    nbformat.write(nb, ANALYSIS_NOTEBOOK)
    print(f"Executed {ANALYSIS_NOTEBOOK.name}, figures written to figures/")


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command in ("run", "all"):
        run(args.experiments, args.shard)
    if args.command in ("analyse", "all"):
        analyse()


if __name__ == "__main__":
    main()
