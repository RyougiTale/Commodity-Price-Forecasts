from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_qlib_forecast as forecast


DEFAULT_HORIZONS = "1,3,5,10,20,40,60"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search forecast horizons without changing run_qlib_forecast.py.")
    parser.add_argument(
        "--code",
        default="ALL",
        choices=sorted(forecast.PRICE_FILES) + ["ALL"],
        help="Commodity code, or ALL to run every configured commodity.",
    )
    parser.add_argument("--codes", help="Comma-separated commodity codes, for example CU,AU,SC.")
    parser.add_argument(
        "--horizons",
        default=DEFAULT_HORIZONS,
        help="Comma-separated forecast horizons in trading days.",
    )
    parser.add_argument("--train-start", default=forecast.DEFAULT_TRAIN_START, help="Train start date in YYYY-MM-DD.")
    parser.add_argument("--train-end", default=forecast.DEFAULT_TRAIN_END, help="Train end date in YYYY-MM-DD.")
    parser.add_argument("--valid-end", default=forecast.DEFAULT_VALID_END, help="Validation end date in YYYY-MM-DD.")
    parser.add_argument(
        "--embargo-days",
        type=int,
        default=forecast.DEFAULT_EMBARGO_DAYS,
        help="Trading days to skip after train/valid boundaries before the next segment starts.",
    )
    parser.add_argument(
        "--monthly-macro-lag-days",
        type=int,
        default=forecast.DEFAULT_MONTHLY_MACRO_LAG_DAYS,
        help="Calendar-day delay applied to monthly macro indicators before as-of merging.",
    )
    parser.add_argument("--macro-path", type=Path, default=forecast.DEFAULT_MACRO_PATH, help="Macro summary CSV path.")
    parser.add_argument("--output-dir", type=Path, default=forecast.DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument(
        "--save-all-predictions",
        action="store_true",
        help="Save prediction CSVs for every code/horizon instead of only the selected best horizons.",
    )
    return parser.parse_args()


def parse_horizons(value: str) -> list[int]:
    horizons = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not horizons:
        raise ValueError("--horizons cannot be empty")
    invalid = [horizon for horizon in horizons if horizon <= 0]
    if invalid:
        raise ValueError(f"horizons must be positive: {invalid}")
    return horizons


def make_split_config(args: argparse.Namespace, horizon: int) -> forecast.SplitConfig:
    return forecast.SplitConfig(
        train_start=pd.Timestamp(args.train_start),
        train_end=pd.Timestamp(args.train_end),
        valid_end=pd.Timestamp(args.valid_end),
        horizon=horizon,
        embargo_days=args.embargo_days,
    )


def build_metric_row(code: str, horizon: int, metrics: dict[str, float], pred_df: pd.DataFrame) -> dict[str, Any]:
    baseline_rmse_values = {
        "zero": metrics["zero_rmse"],
        "train_mean": metrics["train_mean_rmse"],
        "prev_horizon": metrics["prev_horizon_rmse"],
        "rolling_mean": metrics["rolling_mean_rmse"],
    }
    best_baseline_name = min(baseline_rmse_values, key=baseline_rmse_values.get)
    best_baseline_rmse = baseline_rmse_values[best_baseline_name]
    model_rmse = metrics["model_rmse"]
    pred_std = float(pred_df["predicted_return"].std())
    actual_std = float(pred_df["actual_return"].std())

    return {
        "code": code,
        "horizon": horizon,
        "rows": int(metrics["rows"]),
        "model_rmse": model_rmse,
        "zero_rmse": metrics["zero_rmse"],
        "train_mean_rmse": metrics["train_mean_rmse"],
        "prev_horizon_rmse": metrics["prev_horizon_rmse"],
        "rolling_mean_rmse": metrics["rolling_mean_rmse"],
        "best_baseline": best_baseline_name,
        "best_baseline_rmse": best_baseline_rmse,
        "rmse_improvement_vs_best_baseline": 1 - model_rmse / best_baseline_rmse,
        "rmse_improvement_vs_zero": 1 - model_rmse / metrics["zero_rmse"],
        "model_corr": metrics["model_corr"],
        "model_direction_hit_rate": metrics["model_direction_hit_rate"],
        "predicted_return_mean": float(pred_df["predicted_return"].mean()),
        "predicted_return_std": pred_std,
        "actual_return_mean": float(pred_df["actual_return"].mean()),
        "actual_return_std": actual_std,
        "prediction_std_to_actual_std": pred_std / actual_std if actual_std else np.nan,
    }


def run_one(code: str, horizon: int, args: argparse.Namespace) -> tuple[dict[str, Any], pd.DataFrame]:
    split_config = make_split_config(args, horizon)
    samples, feature_cols = forecast.build_samples(
        code=code,
        macro_path=args.macro_path,
        horizon=horizon,
        monthly_macro_lag_days=args.monthly_macro_lag_days,
    )
    prepared = forecast.prepare_data(samples, feature_cols, split_config)
    dataset = forecast.build_dataset(prepared, code)
    pred = forecast.fit_predict(dataset)
    pred_df = forecast.build_prediction_frame(prepared, pred, code, horizon)
    metrics = forecast.evaluate(pred_df)
    return build_metric_row(code, horizon, metrics, pred_df), pred_df


def select_best(summary: pd.DataFrame) -> pd.DataFrame:
    ordered = summary.sort_values(
        by=["code", "rmse_improvement_vs_best_baseline", "model_corr", "model_direction_hit_rate"],
        ascending=[True, False, False, False],
    )
    return ordered.groupby("code", as_index=False).head(1).reset_index(drop=True)


def prediction_path(output_dir: Path, code: str, horizon: int, suffix: str) -> Path:
    return output_dir / f"{code.lower()}_h{horizon}_{suffix}_predictions.csv"


def main() -> None:
    args = parse_args()
    codes = forecast.resolve_codes(args)
    horizons = parse_horizons(args.horizons)

    if args.embargo_days < 0:
        raise ValueError("--embargo-days cannot be negative")
    if args.monthly_macro_lag_days < 0:
        raise ValueError("--monthly-macro-lag-days cannot be negative")

    forecast.init_qlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    predictions: dict[tuple[str, int], pd.DataFrame] = {}
    for code in codes:
        for horizon in horizons:
            row, pred_df = run_one(code, horizon, args)
            rows.append(row)
            predictions[(code, horizon)] = pred_df
            if args.save_all_predictions:
                pred_df.to_csv(
                    prediction_path(args.output_dir, code, horizon, "sweep"),
                    index=False,
                    encoding="utf-8-sig",
                )
            print(
                "[sweep] "
                f"code={code} horizon={horizon} "
                f"rmse={row['model_rmse']:.6f} "
                f"best_baseline={row['best_baseline']} "
                f"improve_best={row['rmse_improvement_vs_best_baseline']:.2%} "
                f"corr={row['model_corr']:.4f} "
                f"hit={row['model_direction_hit_rate']:.2%}",
                flush=True,
            )

    summary = pd.DataFrame(rows)
    best = select_best(summary)

    summary_path = args.output_dir / "horizon_sweep_summary.csv"
    best_path = args.output_dir / "horizon_sweep_best.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    best.to_csv(best_path, index=False, encoding="utf-8-sig")

    for _, row in best.iterrows():
        code = str(row["code"])
        horizon = int(row["horizon"])
        predictions[(code, horizon)].to_csv(
            prediction_path(args.output_dir, code, horizon, "best"),
            index=False,
            encoding="utf-8-sig",
        )

    print(f"[sweep] summary_path={summary_path}", flush=True)
    print(f"[sweep] best_path={best_path}", flush=True)


if __name__ == "__main__":
    main()
