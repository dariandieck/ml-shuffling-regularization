import numpy as np
import pandas as pd
from scipy import stats

from batch_shuffling_regularization.experiment import EXPERIMENT_VARIANTS, Experiment, load_results

VARIANT_NAMES = [variant.name for variant in EXPERIMENT_VARIANTS]
SHUFFLE_VARIANTS = ["A", "B", "C", "D"]


def load_complete(experiment: Experiment) -> pd.DataFrame:
    # Only seeds for which every variant has finished are kept, so all comparisons stay paired.
    df = load_results(experiment)
    df["gap"] = df["val_loss"] - df["train_loss"]
    variants_per_seed = df.groupby("seed")["variant"].nunique()
    complete_seeds = variants_per_seed[variants_per_seed == len(VARIANT_NAMES)].index
    return df[df["seed"].isin(complete_seeds)].reset_index(drop=True)


def common_train_loss(df: pd.DataFrame) -> float:
    # The largest final train loss among the runs of A-D: every run gets at least this far,
    # so the validation loss of all runs can be compared at this level of fit.
    final_train_loss = df[df["variant"].isin(SHUFFLE_VARIANTS)].groupby(["variant", "seed"])["train_loss"].min()
    return float(final_train_loss.max())


def val_loss_at_train_loss(run: pd.DataFrame, level: float) -> float:
    train_loss = run["train_loss"].to_numpy()
    val_loss = run["val_loss"].to_numpy()
    reached = np.flatnonzero(train_loss <= level)
    if len(reached) == 0:
        raise ValueError("run never reaches the train loss level")
    i = reached[0]
    if i == 0:
        return float(val_loss[0])
    # Linear interpolation on a log scale of the train loss between the two neighbouring epochs.
    w = (np.log(train_loss[i - 1]) - np.log(level)) / (np.log(train_loss[i - 1]) - np.log(train_loss[i]))
    return float(val_loss[i - 1] + w * (val_loss[i] - val_loss[i - 1]))


def run_metrics(df: pd.DataFrame, train_loss_level: float) -> pd.DataFrame:
    rows = []
    for (variant, seed), run in df.groupby(["variant", "seed"]):
        run = run.set_index("epoch").sort_index()
        best = run["val_acc"].idxmax()
        last = run.index.max()
        rows.append({
            "variant": variant,
            "seed": seed,
            "best_val_acc": run.loc[best, "val_acc"],
            "best_val_acc_epoch": best,
            "test_acc_at_best": run.loc[best, "test_acc"],
            "min_val_loss": run["val_loss"].min(),
            "min_val_loss_epoch": run["val_loss"].idxmin(),
            "gap_last_epoch": run.loc[last, "gap"],
            "acc_gap_last_epoch": run.loc[last, "train_acc"] - run.loc[last, "val_acc"],
            "val_loss_at_common_train_loss": (
                val_loss_at_train_loss(run, train_loss_level) if variant in SHUFFLE_VARIANTS else np.nan
            ),
        })
    return pd.DataFrame(rows).set_index(["variant", "seed"])


def paired_comparison(metrics: pd.DataFrame, pairs: list[tuple[str, str]], columns: list[str]) -> pd.DataFrame:
    rows = []
    for a, b in pairs:
        for column in columns:
            # Both series are indexed by seed, so the subtraction pairs runs with identical initial weights.
            diff = metrics.loc[a, column] - metrics.loc[b, column]
            low, high = stats.t.interval(0.95, len(diff) - 1, loc=diff.mean(), scale=stats.sem(diff))
            rows.append({
                "Vergleich": f"{a} - {b}",
                "Kennzahl": column,
                "Differenz": diff.mean(),
                "95%-KI unten": low,
                "95%-KI oben": high,
                "Seeds mit Differenz > 0": f"{(diff > 0).sum()}/{len(diff)}",
                "p (Wilcoxon)": stats.wilcoxon(diff).pvalue if (diff != 0).any() else 1.0,
            })
    return pd.DataFrame(rows).set_index(["Vergleich", "Kennzahl"])
