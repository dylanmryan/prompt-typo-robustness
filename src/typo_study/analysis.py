"""Statistics and figures from results JSONL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import yaml

FIGURES = Path("figures")
RESULTS = Path("results")


def load_results(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    main = df[df.phase == "main"]
    rows = []
    for (model, severity), g in main.groupby(["model", "severity"]):
        vals = g.correct.astype(int).to_numpy()
        lo, hi = bootstrap_ci(vals)
        rows.append({"model": model, "severity": severity,
                     "accuracy": vals.mean(), "ci_lo": lo, "ci_hi": hi, "n": len(vals)})
    return pd.DataFrame(rows).sort_values(["model", "severity"]).reset_index(drop=True)


def fit_logit(df: pd.DataFrame) -> str:
    main = df[df.phase == "main"].assign(correct=lambda d: d.correct.astype(int))
    model = smf.logit("correct ~ severity * C(model)", data=main).fit(disp=False)
    return str(model.summary())


def fig_degradation(df: pd.DataFrame, out: Path) -> None:
    table = accuracy_table(df)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for model, g in table.groupby("model"):
        g = g.sort_values("severity")
        ax.plot(g.severity * 100, g.accuracy, marker="o", label=model)
        ax.fill_between(g.severity * 100, g.ci_lo, g.ci_hi, alpha=0.15)
    ax.set_xlabel("Typo severity (% of words corrupted)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs. typo severity (95% bootstrap CI)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def fig_typo_types(df: pd.DataFrame, out: Path) -> None:
    bd = df[df.phase == "breakdown"]
    acc = bd.groupby("typo_type").correct.mean().sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(acc.index, acc.values)
    ax.set_ylabel("Accuracy (all models pooled)")
    ax.set_title(f"Accuracy by typo type at {bd.severity.iloc[0]:.0%} severity")
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def fig_heatmap(df: pd.DataFrame, out: Path) -> None:
    main = df[df.phase == "main"]
    lo = main[main.severity == main.severity.min()].groupby(["task", "model"]).correct.mean()
    hi = main[main.severity == main.severity.max()].groupby(["task", "model"]).correct.mean()
    drop = (lo - hi).unstack()
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(drop.values, cmap="Reds", aspect="auto")
    ax.set_xticks(range(len(drop.columns)), drop.columns, rotation=20)
    ax.set_yticks(range(len(drop.index)), drop.index)
    for i in range(drop.shape[0]):
        for j in range(drop.shape[1]):
            ax.text(j, i, f"{drop.values[i, j]:+.2f}", ha="center", va="center")
    ax.set_title("Accuracy drop, 0% → 20% severity (higher = more fragile)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    df = load_results(Path(config["results_path"]))
    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    accuracy_table(df).to_csv(RESULTS / "summary.csv", index=False)
    (RESULTS / "logit_summary.txt").write_text(fit_logit(df))
    fig_degradation(df, FIGURES / "degradation_curves.png")
    fig_typo_types(df, FIGURES / "typo_types.png")
    fig_heatmap(df, FIGURES / "task_model_heatmap.png")
    print("wrote results/summary.csv, results/logit_summary.txt, figures/*.png")


if __name__ == "__main__":
    main()
