from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot horizon sweep results.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory containing sweep CSVs.")
    parser.add_argument(
        "--summary",
        type=Path,
        help="Path to horizon_sweep_summary.csv. Defaults to <output-dir>/horizon_sweep_summary.csv.",
    )
    parser.add_argument(
        "--best",
        type=Path,
        help="Path to horizon_sweep_best.csv. Defaults to <output-dir>/horizon_sweep_best.csv.",
    )
    return parser.parse_args()


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_path = args.summary or args.output_dir / "horizon_sweep_summary.csv"
    best_path = args.best or args.output_dir / "horizon_sweep_best.csv"
    summary = pd.read_csv(summary_path)
    best = pd.read_csv(best_path)
    return summary.sort_values(["code", "horizon"]), best.sort_values("code")


def save_overview(summary: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    panels = [
        ("rmse_improvement_vs_best_baseline", "RMSE Improvement vs Best Baseline", lambda s: s * 100, "%"),
        ("rmse_improvement_vs_zero", "RMSE Improvement vs Zero Baseline", lambda s: s * 100, "%"),
        ("model_corr", "Correlation", lambda s: s, ""),
        ("prediction_std_to_actual_std", "Prediction Std / Actual Std", lambda s: s, "x"),
    ]

    for ax, (column, title, transform, suffix) in zip(axes.ravel(), panels):
        for code, group in summary.groupby("code"):
            ax.plot(group["horizon"], transform(group[column]), marker="o", linewidth=1.8, label=code)
        ax.axhline(0, color="black", linewidth=1, alpha=0.45)
        if column == "prediction_std_to_actual_std":
            ax.axhline(1, color="black", linewidth=1, linestyle="--", alpha=0.35)
        ax.set_title(title)
        ax.set_xlabel("Horizon (trading days)")
        ax.set_ylabel(suffix)
        ax.grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)
    fig.suptitle("Horizon Sweep Overview", y=0.99, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = output_dir / "horizon_sweep_overview.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def save_rmse_by_code(summary: pd.DataFrame, output_dir: Path) -> Path:
    codes = list(summary["code"].drop_duplicates())
    fig, axes = plt.subplots(len(codes), 1, figsize=(14, 3.2 * len(codes)), sharex=True)
    if len(codes) == 1:
        axes = [axes]

    rmse_cols = [
        ("model_rmse", "Model"),
        ("zero_rmse", "Zero"),
        ("train_mean_rmse", "Train Mean"),
        ("prev_horizon_rmse", "Prev Horizon"),
        ("rolling_mean_rmse", "Rolling Mean"),
    ]

    for ax, code in zip(axes, codes):
        group = summary[summary["code"] == code]
        for column, label in rmse_cols:
            ax.plot(group["horizon"], group[column], marker="o", linewidth=1.5, label=label)
        ax.set_title(f"{code} RMSE by Horizon")
        ax.set_ylabel("RMSE")
        ax.grid(alpha=0.25)
        ax.legend(ncol=5, fontsize=9, loc="upper left")

    axes[-1].set_xlabel("Horizon (trading days)")
    fig.tight_layout()
    path = output_dir / "horizon_sweep_rmse_by_code.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def save_best_prediction_paths(best: pd.DataFrame, output_dir: Path) -> Path:
    rows = len(best)
    fig, axes = plt.subplots(rows, 1, figsize=(14, 3.2 * rows), sharex=False)
    if rows == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, best.iterrows()):
        code = str(row["code"])
        horizon = int(row["horizon"])
        pred_path = output_dir / f"{code.lower()}_h{horizon}_best_predictions.csv"
        pred = pd.read_csv(pred_path, parse_dates=["trade_date"])
        ax.plot(pred["trade_date"], pred["actual_return"], label="Actual", linewidth=1.4)
        ax.plot(pred["trade_date"], pred["predicted_return"], label="Model", linewidth=1.2)
        ax.plot(pred["trade_date"], pred["baseline_train_mean_return"], label="Train Mean", linewidth=1.0, alpha=0.75)
        ax.axhline(0, color="black", linewidth=1, alpha=0.4)
        ax.set_title(
            f"{code} Best Horizon={horizon}D | "
            f"improve_best={row['rmse_improvement_vs_best_baseline']:.2%} "
            f"corr={row['model_corr']:.3f} hit={row['model_direction_hit_rate']:.1%}"
        )
        ax.set_ylabel("Return")
        ax.grid(alpha=0.25)
        ax.legend(ncol=3, fontsize=9, loc="upper left")

    fig.tight_layout()
    path = output_dir / "horizon_sweep_best_predictions.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    summary, best = load_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        save_overview(summary, args.output_dir),
        save_rmse_by_code(summary, args.output_dir),
        save_best_prediction_paths(best, args.output_dir),
    ]
    for path in paths:
        print(f"[plot] {path}", flush=True)


if __name__ == "__main__":
    main()
