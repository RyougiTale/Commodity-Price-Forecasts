from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qlib
from qlib.contrib.model.gbdt import LGBModel
from qlib.workflow import R
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.loader import StaticDataLoader


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_MACRO_PATH = DATA_DIR / "macro_daily.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "output"
DEFAULT_MLRUNS_DIR = DEFAULT_OUTPUT_DIR / "mlruns"
DEFAULT_CODE = "CU"
DEFAULT_HORIZON = 20
DEFAULT_TRAIN_END = "2022-12-31"
DEFAULT_VALID_END = "2024-12-31"

PRICE_FILES = {
    "CU": DATA_DIR / "cu_continuous_daily.csv",
    "AU": DATA_DIR / "au_continuous_daily.csv",
    "SC": DATA_DIR / "sc_continuous_daily.csv",
    "RB": DATA_DIR / "rb_continuous_daily.csv",
    "C": DATA_DIR / "c_continuous_daily.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple commodity forecast with qlib.")
    parser.add_argument("--code", default=DEFAULT_CODE, choices=sorted(PRICE_FILES), help="Commodity code.")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help="Forecast horizon in trading days.")
    parser.add_argument("--train-end", default=DEFAULT_TRAIN_END, help="Train segment end date in YYYY-MM-DD.")
    parser.add_argument("--valid-end", default=DEFAULT_VALID_END, help="Validation segment end date in YYYY-MM-DD.")
    parser.add_argument("--macro-path", type=Path, default=DEFAULT_MACRO_PATH, help="Macro summary CSV path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    return parser.parse_args()


def load_price(code: str) -> pd.DataFrame:
    path = PRICE_FILES[code]
    df = pd.read_csv(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.sort_values("trade_date").reset_index(drop=True)


def load_macro(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    out = df.pivot(index="trade_date", columns="indicator_code", values="value")
    return out.sort_index().reset_index()


def build_samples(code: str, macro_path: Path, horizon: int) -> tuple[pd.DataFrame, list[str]]:
    price = load_price(code)
    macro = load_macro(macro_path)
    df = price.merge(macro, on="trade_date", how="left").sort_values("trade_date").reset_index(drop=True)

    df["ret_1"] = df["close"].pct_change(1)
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_10"] = df["close"].pct_change(10)
    df["ret_20"] = df["close"].pct_change(20)
    df["ret_60"] = df["close"].pct_change(60)
    df["ret_horizon"] = df["close"].pct_change(horizon)
    df["volatility_5"] = df["close"].pct_change().rolling(5).std()
    df["volatility_10"] = df["close"].pct_change().rolling(10).std()
    df["volatility_20"] = df["close"].pct_change().rolling(20).std()
    df["volatility_60"] = df["close"].pct_change().rolling(60).std()
    df["ma_5_ratio"] = df["close"] / df["close"].rolling(5).mean() - 1
    df["ma_10_ratio"] = df["close"] / df["close"].rolling(10).mean() - 1
    df["ma_20_ratio"] = df["close"] / df["close"].rolling(20).mean() - 1
    df["ma_60_ratio"] = df["close"] / df["close"].rolling(60).mean() - 1
    df["vol_change_1"] = df["vol"].pct_change(1)
    df["vol_5_ratio"] = df["vol"] / df["vol"].rolling(5).mean() - 1
    df["vol_20_ratio"] = df["vol"] / df["vol"].rolling(20).mean() - 1
    df["oi_change_1"] = df["oi"].pct_change(1)
    df["oi_5_ratio"] = df["oi"] / df["oi"].rolling(5).mean() - 1
    df["oi_20_ratio"] = df["oi"] / df["oi"].rolling(20).mean() - 1
    df["dist_to_high_20"] = df["close"] / df["high"].rolling(20).max() - 1
    df["dist_to_low_20"] = df["close"] / df["low"].rolling(20).min() - 1
    df["dist_to_high_60"] = df["close"] / df["high"].rolling(60).max() - 1
    df["dist_to_low_60"] = df["close"] / df["low"].rolling(60).min() - 1
    df["close_zscore_20"] = (df["close"] - df["close"].rolling(20).mean()) / df["close"].rolling(20).std()
    df["close_zscore_60"] = (df["close"] - df["close"].rolling(60).mean()) / df["close"].rolling(60).std()
    df["vol_zscore_20"] = (df["vol"] - df["vol"].rolling(20).mean()) / df["vol"].rolling(20).std()
    df["oi_zscore_20"] = (df["oi"] - df["oi"].rolling(20).mean()) / df["oi"].rolling(20).std()
    df["label"] = df["close"].shift(-horizon) / df["close"] - 1
    df["future_close"] = df["close"].shift(-horizon)

    feature_cols = [
        "open",
        "high",
        "low",
        "close",
        "vol",
        "oi",
        "roll_flag",
        "ret_1",
        "ret_5",
        "ret_10",
        "ret_20",
        "ret_60",
        "volatility_5",
        "volatility_10",
        "volatility_20",
        "volatility_60",
        "ma_5_ratio",
        "ma_10_ratio",
        "ma_20_ratio",
        "ma_60_ratio",
        "vol_change_1",
        "vol_5_ratio",
        "vol_20_ratio",
        "oi_change_1",
        "oi_5_ratio",
        "oi_20_ratio",
        "dist_to_high_20",
        "dist_to_low_20",
        "dist_to_high_60",
        "dist_to_low_60",
        "close_zscore_20",
        "close_zscore_60",
        "vol_zscore_20",
        "oi_zscore_20",
        "USDCNY",
        "CN10Y",
        "US10Y",
        "SHIBOR_3M",
        "DR007",
        "CPI_YOY",
        "PPI_YOY",
        "PMI_MANUFACTURE",
        "M1_YOY",
        "M2_YOY",
        "SOCIAL_FINANCING_YOY",
        "000300.SH",
        "000985.CSI",
    ]
    df[feature_cols] = df[feature_cols].ffill()
    df = df.dropna(subset=feature_cols + ["label", "future_close"]).copy()
    return df.reset_index(drop=True), feature_cols


def build_dataset(df: pd.DataFrame, feature_cols: list[str], code: str, train_end: str, valid_end: str) -> DatasetH:
    df = df.copy()
    df["instrument"] = code
    qlib_df = (
        pd.concat(
            {
                "feature": df.set_index(["trade_date", "instrument"])[feature_cols],
                "label": df.set_index(["trade_date", "instrument"])[["label"]],
            },
            axis=1,
        )
        .sort_index()
    )
    handler = DataHandlerLP(
        data_loader=StaticDataLoader(qlib_df),
        infer_processors=[],
        learn_processors=[],
    )
    return DatasetH(
        handler=handler,
        segments={
            "train": ("2019-01-01", train_end),
            "valid": (pd.Timestamp(train_end) + pd.Timedelta(days=1), valid_end),
            "test": (pd.Timestamp(valid_end) + pd.Timedelta(days=1), "2099-12-31"),
        },
    )


def fit_predict(dataset: DatasetH) -> pd.Series:
    R.log_metrics = lambda *args, **kwargs: None
    model = LGBModel(
        loss="mse",
        num_boost_round=200,
        early_stopping_rounds=50,
        learning_rate=0.05,
        num_leaves=31,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        min_data_in_leaf=20,
        lambda_l2=1.0,
        verbose=-1,
    )
    model.fit(dataset)
    return model.predict(dataset, segment="test")


def build_prediction_frame(df: pd.DataFrame, pred: pd.Series, code: str, horizon: int) -> pd.DataFrame:
    pred_df = pred.rename("predicted_return").reset_index()
    pred_df = pred_df.rename(columns={"datetime": "trade_date"})
    pred_df["trade_date"] = pd.to_datetime(pred_df["trade_date"])

    out = df.merge(pred_df.loc[:, ["trade_date", "predicted_return"]], on="trade_date", how="inner")
    out["actual_return"] = out["label"]
    out["baseline_zero_return"] = 0.0
    out["baseline_mean_return"] = out["ret_horizon"].fillna(0.0)
    out["predicted_future_close"] = out["close"] * (1 + out["predicted_return"])
    out["actual_future_close"] = out["future_close"]
    out["abs_error"] = (out["predicted_return"] - out["actual_return"]).abs()
    out["direction_hit"] = (
        np.sign(out["predicted_return"]).astype(int) == np.sign(out["actual_return"]).astype(int)
    ).astype(int)
    out["instrument"] = code
    out["horizon"] = horizon
    return out.loc[
        :,
        [
            "trade_date",
            "instrument",
            "horizon",
            "close",
            "actual_future_close",
            "predicted_future_close",
            "actual_return",
            "predicted_return",
            "baseline_zero_return",
            "baseline_mean_return",
            "abs_error",
            "direction_hit",
        ],
    ].sort_values("trade_date")


def evaluate(pred_df: pd.DataFrame) -> dict[str, float]:
    rmse = float(np.sqrt(np.mean((pred_df["predicted_return"] - pred_df["actual_return"]) ** 2)))
    mae = float(np.mean(np.abs(pred_df["predicted_return"] - pred_df["actual_return"])))
    corr = float(pred_df["predicted_return"].corr(pred_df["actual_return"]))
    hit_rate = float(pred_df["direction_hit"].mean())
    zero_rmse = float(np.sqrt(np.mean((pred_df["baseline_zero_return"] - pred_df["actual_return"]) ** 2)))
    mean_rmse = float(np.sqrt(np.mean((pred_df["baseline_mean_return"] - pred_df["actual_return"]) ** 2)))
    zero_mae = float(np.mean(np.abs(pred_df["baseline_zero_return"] - pred_df["actual_return"])))
    mean_mae = float(np.mean(np.abs(pred_df["baseline_mean_return"] - pred_df["actual_return"])))
    return {
        "rows": float(len(pred_df)),
        "rmse": rmse,
        "mae": mae,
        "corr": corr,
        "direction_hit_rate": hit_rate,
        "baseline_zero_rmse": zero_rmse,
        "baseline_mean_rmse": mean_rmse,
        "baseline_zero_mae": zero_mae,
        "baseline_mean_mae": mean_mae,
    }


def save_plot(pred_df: pd.DataFrame, metrics: dict[str, float], output_path: Path, code: str, horizon: int) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    axes[0].plot(pred_df["trade_date"], pred_df["actual_return"], label="Actual Return", linewidth=1.5)
    axes[0].plot(pred_df["trade_date"], pred_df["predicted_return"], label="Model Return", linewidth=1.2)
    axes[0].plot(pred_df["trade_date"], pred_df["baseline_zero_return"], label="Zero Baseline", linewidth=1.0, alpha=0.8)
    axes[0].plot(pred_df["trade_date"], pred_df["baseline_mean_return"], label="Mean Baseline", linewidth=1.0, alpha=0.8)
    axes[0].axhline(0, color="black", linewidth=1, alpha=0.5)
    axes[0].set_title(f"{code} {horizon}D Return Forecast")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].scatter(pred_df["actual_return"], pred_df["predicted_return"], s=18, alpha=0.65, label="Model")
    low = min(pred_df["actual_return"].min(), pred_df["predicted_return"].min())
    high = max(pred_df["actual_return"].max(), pred_df["predicted_return"].max())
    axes[1].plot([low, high], [low, high], color="black", linewidth=1, alpha=0.6, label="Ideal")
    axes[1].set_xlabel("Actual Return")
    axes[1].set_ylabel("Predicted Return")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    metric_text = (
        f"rows={int(metrics['rows'])}  "
        f"model_rmse={metrics['rmse']:.4f}  "
        f"zero_rmse={metrics['baseline_zero_rmse']:.4f}  "
        f"mean_rmse={metrics['baseline_mean_rmse']:.4f}  "
        f"mae={metrics['mae']:.4f}  "
        f"corr={metrics['corr']:.4f}  "
        f"hit={metrics['direction_hit_rate']:.2%}"
    )
    fig.suptitle(metric_text, fontsize=10, y=0.98)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_metrics(metrics: dict[str, float], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(metrics).to_csv(output_path, header=False)
    return output_path


def main() -> None:
    args = parse_args()
    qlib.init(
        provider_uri=str(BASE_DIR),
        region="cn",
        exp_manager={
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {
                "uri": f"file:{DEFAULT_MLRUNS_DIR.resolve()}",
                "default_exp_name": "commodity_forecast",
            }
        },
    )

    samples, feature_cols = build_samples(args.code, args.macro_path, args.horizon)
    dataset = build_dataset(samples, feature_cols, args.code, args.train_end, args.valid_end)
    pred = fit_predict(dataset)
    pred_df = build_prediction_frame(samples, pred, args.code, args.horizon)
    metrics = evaluate(pred_df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / f"{args.code.lower()}_qlib_predictions.csv"
    metrics_path = args.output_dir / f"{args.code.lower()}_qlib_metrics.csv"
    plot_path = args.output_dir / f"{args.code.lower()}_qlib_forecast.png"

    pred_df.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    save_metrics(metrics, metrics_path)
    save_plot(pred_df, metrics, plot_path, args.code, args.horizon)

    print(f"[qlib] code={args.code} rows={len(pred_df)}", flush=True)
    print(f"[qlib] prediction_path={prediction_path}", flush=True)
    print(f"[qlib] metrics_path={metrics_path}", flush=True)
    print(f"[qlib] plot_path={plot_path}", flush=True)
    print(
        "[qlib] "
        f"rmse={metrics['rmse']:.6f} "
        f"baseline_zero_rmse={metrics['baseline_zero_rmse']:.6f} "
        f"baseline_mean_rmse={metrics['baseline_mean_rmse']:.6f} "
        f"mae={metrics['mae']:.6f} "
        f"corr={metrics['corr']:.6f} "
        f"direction_hit_rate={metrics['direction_hit_rate']:.2%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
