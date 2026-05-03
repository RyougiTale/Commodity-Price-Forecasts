from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qlib
from qlib.contrib.model.gbdt import LGBModel
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.loader import StaticDataLoader
from qlib.workflow import R


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_MACRO_PATH = DATA_DIR / "macro_daily.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "output"
DEFAULT_MLRUNS_DIR = DEFAULT_OUTPUT_DIR / "mlruns"
DEFAULT_CODE = "CU"
DEFAULT_HORIZON = 20
DEFAULT_TRAIN_START = "2019-01-01"
DEFAULT_TRAIN_END = "2022-12-31"
DEFAULT_VALID_END = "2024-12-31"
DEFAULT_EMBARGO_DAYS = 0
DEFAULT_MONTHLY_MACRO_LAG_DAYS = 0

PRICE_FILES = {
    "CU": DATA_DIR / "cu_continuous_daily.csv",
    "AU": DATA_DIR / "au_continuous_daily.csv",
    "SC": DATA_DIR / "sc_continuous_daily.csv",
    "RB": DATA_DIR / "rb_continuous_daily.csv",
    "C": DATA_DIR / "c_continuous_daily.csv",
}

MONTHLY_MACRO_INDICATORS = {
    "CPI_YOY",
    "PPI_YOY",
    "PMI_MANUFACTURE",
    "M1_YOY",
    "M2_YOY",
    "SOCIAL_FINANCING_YOY",
}

RAW_PRICE_FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "vol",
    "oi",
    "roll_flag",
]

TECHNICAL_FEATURES = [
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
    "roll_gap_1",
    "days_since_roll",
]

MACRO_FEATURES = [
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

FEATURE_COLUMNS = RAW_PRICE_FEATURES + TECHNICAL_FEATURES + MACRO_FEATURES

PREDICTION_COLUMNS = {
    "model": "predicted_return",
    "zero": "baseline_zero_return",
    "train_mean": "baseline_train_mean_return",
    "prev_horizon": "baseline_prev_horizon_return",
    "rolling_mean": "baseline_rolling_mean_return",
}


@dataclass(frozen=True)
class SplitConfig:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_end: pd.Timestamp
    horizon: int
    embargo_days: int


@dataclass(frozen=True)
class PreparedData:
    samples: pd.DataFrame
    feature_cols: list[str]
    segments: dict[str, tuple[pd.Timestamp, pd.Timestamp | str]]
    split_summary: pd.DataFrame
    train_mean_return: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a commodity return forecast with qlib.")
    parser.add_argument(
        "--code",
        default=DEFAULT_CODE,
        choices=sorted(PRICE_FILES) + ["ALL"],
        help="Commodity code, or ALL to run every configured commodity.",
    )
    parser.add_argument(
        "--codes",
        help="Comma-separated commodity codes, for example CU,AU,SC. Overrides --code when provided.",
    )
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help="Forecast horizon in trading days.")
    parser.add_argument("--train-start", default=DEFAULT_TRAIN_START, help="Train segment start date in YYYY-MM-DD.")
    parser.add_argument("--train-end", default=DEFAULT_TRAIN_END, help="Train segment end date in YYYY-MM-DD.")
    parser.add_argument("--valid-end", default=DEFAULT_VALID_END, help="Validation segment end date in YYYY-MM-DD.")
    parser.add_argument(
        "--embargo-days",
        type=int,
        default=DEFAULT_EMBARGO_DAYS,
        help="Trading days to skip after train/valid boundaries before the next segment starts.",
    )
    parser.add_argument(
        "--monthly-macro-lag-days",
        type=int,
        default=DEFAULT_MONTHLY_MACRO_LAG_DAYS,
        help="Calendar-day delay applied to monthly macro indicators before as-of merging.",
    )
    parser.add_argument("--macro-path", type=Path, default=DEFAULT_MACRO_PATH, help="Macro summary CSV path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--skip-plot", action="store_true", help="Skip PNG plot generation.")
    return parser.parse_args()


def resolve_codes(args: argparse.Namespace) -> list[str]:
    if args.codes:
        codes = [item.strip().upper() for item in args.codes.split(",") if item.strip()]
    elif args.code == "ALL":
        codes = sorted(PRICE_FILES)
    else:
        codes = [args.code]

    unknown = sorted(set(codes) - set(PRICE_FILES))
    if unknown:
        raise ValueError(f"unknown commodity codes: {unknown}; supported codes: {sorted(PRICE_FILES)}")
    return codes


def parse_split_config(args: argparse.Namespace) -> SplitConfig:
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive")
    if args.embargo_days < 0:
        raise ValueError("--embargo-days cannot be negative")
    if args.monthly_macro_lag_days < 0:
        raise ValueError("--monthly-macro-lag-days cannot be negative")

    config = SplitConfig(
        train_start=pd.Timestamp(args.train_start),
        train_end=pd.Timestamp(args.train_end),
        valid_end=pd.Timestamp(args.valid_end),
        horizon=args.horizon,
        embargo_days=args.embargo_days,
    )
    if not config.train_start < config.train_end < config.valid_end:
        raise ValueError("--train-start, --train-end, and --valid-end must be strictly ordered")
    return config


def load_price(code: str) -> pd.DataFrame:
    path = PRICE_FILES[code]
    df = pd.read_csv(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.sort_values("trade_date").reset_index(drop=True)


def load_macro(path: Path, monthly_lag_days: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    if monthly_lag_days:
        monthly_mask = df["indicator_code"].isin(MONTHLY_MACRO_INDICATORS)
        df.loc[monthly_mask, "trade_date"] = df.loc[monthly_mask, "trade_date"] + pd.Timedelta(
            days=monthly_lag_days
        )

    out = df.pivot_table(index="trade_date", columns="indicator_code", values="value", aggfunc="last")
    return out.sort_index().reset_index()


def merge_price_macro(price: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    return pd.merge_asof(
        price.sort_values("trade_date"),
        macro.sort_values("trade_date"),
        on="trade_date",
        direction="backward",
    ).reset_index(drop=True)


def add_price_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    out = df.copy()
    close_ret = out["close"].pct_change()

    out["ret_1"] = out["close"].pct_change(1)
    out["ret_5"] = out["close"].pct_change(5)
    out["ret_10"] = out["close"].pct_change(10)
    out["ret_20"] = out["close"].pct_change(20)
    out["ret_60"] = out["close"].pct_change(60)
    out["known_horizon_return"] = out["close"].pct_change(horizon)
    out["volatility_5"] = close_ret.rolling(5).std()
    out["volatility_10"] = close_ret.rolling(10).std()
    out["volatility_20"] = close_ret.rolling(20).std()
    out["volatility_60"] = close_ret.rolling(60).std()
    out["ma_5_ratio"] = out["close"] / out["close"].rolling(5).mean() - 1
    out["ma_10_ratio"] = out["close"] / out["close"].rolling(10).mean() - 1
    out["ma_20_ratio"] = out["close"] / out["close"].rolling(20).mean() - 1
    out["ma_60_ratio"] = out["close"] / out["close"].rolling(60).mean() - 1
    out["vol_change_1"] = out["vol"].pct_change(1)
    out["vol_5_ratio"] = out["vol"] / out["vol"].rolling(5).mean() - 1
    out["vol_20_ratio"] = out["vol"] / out["vol"].rolling(20).mean() - 1
    out["oi_change_1"] = out["oi"].pct_change(1)
    out["oi_5_ratio"] = out["oi"] / out["oi"].rolling(5).mean() - 1
    out["oi_20_ratio"] = out["oi"] / out["oi"].rolling(20).mean() - 1
    out["dist_to_high_20"] = out["close"] / out["high"].rolling(20).max() - 1
    out["dist_to_low_20"] = out["close"] / out["low"].rolling(20).min() - 1
    out["dist_to_high_60"] = out["close"] / out["high"].rolling(60).max() - 1
    out["dist_to_low_60"] = out["close"] / out["low"].rolling(60).min() - 1
    out["close_zscore_20"] = (out["close"] - out["close"].rolling(20).mean()) / out["close"].rolling(20).std()
    out["close_zscore_60"] = (out["close"] - out["close"].rolling(60).mean()) / out["close"].rolling(60).std()
    out["vol_zscore_20"] = (out["vol"] - out["vol"].rolling(20).mean()) / out["vol"].rolling(20).std()
    out["oi_zscore_20"] = (out["oi"] - out["oi"].rolling(20).mean()) / out["oi"].rolling(20).std()
    out["roll_gap_1"] = np.where(out["roll_flag"].astype(bool), close_ret, 0.0)
    out["days_since_roll"] = out.groupby(out["roll_flag"].astype(bool).cumsum()).cumcount()

    out["label"] = out["close"].shift(-horizon) / out["close"] - 1
    out["future_close"] = out["close"].shift(-horizon)
    out["label_end_date"] = out["trade_date"].shift(-horizon)
    out["baseline_prev_horizon_return"] = out["known_horizon_return"]
    out["baseline_rolling_mean_return"] = out["known_horizon_return"].rolling(252, min_periods=20).mean()
    return out.replace([np.inf, -np.inf], np.nan)


def build_samples(code: str, macro_path: Path, horizon: int, monthly_macro_lag_days: int) -> tuple[pd.DataFrame, list[str]]:
    price = load_price(code)
    macro = load_macro(macro_path, monthly_macro_lag_days)
    df = merge_price_macro(price, macro)
    df = add_price_features(df, horizon)

    missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"missing feature columns: {missing_cols}")

    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].ffill()
    required_cols = FEATURE_COLUMNS + ["label", "future_close", "label_end_date"]
    df = df.dropna(subset=required_cols).copy()
    return df.reset_index(drop=True), list(FEATURE_COLUMNS)


def next_trading_date_after(dates: pd.Series, boundary: pd.Timestamp, embargo_days: int) -> pd.Timestamp:
    candidates = dates[dates > boundary].drop_duplicates().reset_index(drop=True)
    if len(candidates) <= embargo_days:
        raise ValueError(f"not enough rows after {boundary.date()} for embargo_days={embargo_days}")
    return pd.Timestamp(candidates.iloc[embargo_days])


def prepare_data(
    samples: pd.DataFrame,
    feature_cols: list[str],
    split_config: SplitConfig,
) -> PreparedData:
    dates = samples["trade_date"]
    valid_start = next_trading_date_after(dates, split_config.train_end, split_config.embargo_days)
    test_start = next_trading_date_after(dates, split_config.valid_end, split_config.embargo_days)

    train_mask = (
        (samples["trade_date"] >= split_config.train_start)
        & (samples["trade_date"] <= split_config.train_end)
        & (samples["label_end_date"] <= split_config.train_end)
    )
    valid_mask = (
        (samples["trade_date"] >= valid_start)
        & (samples["trade_date"] <= split_config.valid_end)
        & (samples["label_end_date"] <= split_config.valid_end)
    )
    test_mask = samples["trade_date"] >= test_start

    out = samples.copy()
    out["segment"] = np.select(
        [train_mask, valid_mask, test_mask],
        ["train", "valid", "test"],
        default=None,
    )
    out = out[out["segment"].notna()].reset_index(drop=True)
    if out.empty:
        raise ValueError("no samples remain after applying purged split")

    train_rows = out[out["segment"] == "train"]
    valid_rows = out[out["segment"] == "valid"]
    test_rows = out[out["segment"] == "test"]
    if train_rows.empty or valid_rows.empty or test_rows.empty:
        raise ValueError(
            "train, valid, and test segments must all be non-empty after applying purged split "
            f"(train={len(train_rows)}, valid={len(valid_rows)}, test={len(test_rows)})"
        )

    split_summary = (
        out.groupby("segment", sort=False)
        .agg(
            rows=("trade_date", "size"),
            feature_start=("trade_date", "min"),
            feature_end=("trade_date", "max"),
            label_end=("label_end_date", "max"),
        )
        .reset_index()
    )
    train_mean_return = float(train_rows["label"].mean())
    out["baseline_train_mean_return"] = train_mean_return
    out["baseline_zero_return"] = 0.0
    out["baseline_prev_horizon_return"] = out["baseline_prev_horizon_return"].fillna(0.0)
    out["baseline_rolling_mean_return"] = out["baseline_rolling_mean_return"].fillna(train_mean_return)

    segments: dict[str, tuple[pd.Timestamp, pd.Timestamp | str]] = {
        "train": (split_config.train_start, split_config.train_end),
        "valid": (valid_start, split_config.valid_end),
        "test": (test_start, "2099-12-31"),
    }
    return PreparedData(
        samples=out,
        feature_cols=feature_cols,
        segments=segments,
        split_summary=split_summary,
        train_mean_return=train_mean_return,
    )


def build_dataset(prepared: PreparedData, code: str) -> DatasetH:
    df = prepared.samples.copy()
    df["instrument"] = code
    qlib_df = (
        pd.concat(
            {
                "feature": df.set_index(["trade_date", "instrument"])[prepared.feature_cols],
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
    return DatasetH(handler=handler, segments=prepared.segments)


def fit_predict(dataset: DatasetH) -> pd.Series:
    R.log_metrics = lambda *args, **kwargs: None
    model = LGBModel(
        loss="mse",
        num_boost_round=300,
        early_stopping_rounds=50,
        learning_rate=0.03,
        num_leaves=31,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        min_data_in_leaf=20,
        lambda_l2=1.0,
        seed=42,
        feature_fraction_seed=42,
        bagging_seed=42,
        verbose=-1,
    )
    model.fit(dataset)
    return model.predict(dataset, segment="test")


def build_prediction_frame(prepared: PreparedData, pred: pd.Series, code: str, horizon: int) -> pd.DataFrame:
    pred_df = pred.rename("predicted_return").reset_index()
    pred_df = pred_df.rename(columns={"datetime": "trade_date"})
    pred_df["trade_date"] = pd.to_datetime(pred_df["trade_date"])

    baseline_cols = [
        "trade_date",
        "label",
        "future_close",
        "baseline_zero_return",
        "baseline_train_mean_return",
        "baseline_prev_horizon_return",
        "baseline_rolling_mean_return",
    ]
    out = prepared.samples.merge(pred_df.loc[:, ["trade_date", "predicted_return"]], on="trade_date", how="inner")
    out = out.merge(prepared.samples.loc[:, baseline_cols], on="trade_date", how="left", suffixes=("", "_base"))
    if "label_base" in out.columns:
        out["label"] = out["label_base"]
        out["future_close"] = out["future_close_base"]

    out["actual_return"] = out["label"]
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
            "baseline_train_mean_return",
            "baseline_prev_horizon_return",
            "baseline_rolling_mean_return",
            "abs_error",
            "direction_hit",
        ],
    ].sort_values("trade_date")


def prediction_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    error = predicted - actual
    corr = predicted.corr(actual)
    direction_hit = (np.sign(predicted).astype(int) == np.sign(actual).astype(int)).mean()
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "corr": float(corr) if pd.notna(corr) else np.nan,
        "direction_hit_rate": float(direction_hit),
    }


def evaluate(pred_df: pd.DataFrame) -> dict[str, float]:
    metrics: dict[str, float] = {"rows": float(len(pred_df))}
    for name, col in PREDICTION_COLUMNS.items():
        values = prediction_metrics(pred_df["actual_return"], pred_df[col])
        for metric_name, value in values.items():
            metrics[f"{name}_{metric_name}"] = value

    metrics["rmse"] = metrics["model_rmse"]
    metrics["mae"] = metrics["model_mae"]
    metrics["corr"] = metrics["model_corr"]
    metrics["direction_hit_rate"] = metrics["model_direction_hit_rate"]
    metrics["baseline_zero_rmse"] = metrics["zero_rmse"]
    metrics["baseline_zero_mae"] = metrics["zero_mae"]
    metrics["baseline_train_mean_rmse"] = metrics["train_mean_rmse"]
    metrics["baseline_train_mean_mae"] = metrics["train_mean_mae"]
    metrics["baseline_prev_horizon_rmse"] = metrics["prev_horizon_rmse"]
    metrics["baseline_prev_horizon_mae"] = metrics["prev_horizon_mae"]
    metrics["baseline_rolling_mean_rmse"] = metrics["rolling_mean_rmse"]
    metrics["baseline_rolling_mean_mae"] = metrics["rolling_mean_mae"]
    return metrics


def evaluate_by_year(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for year, group in pred_df.groupby(pred_df["trade_date"].dt.year):
        item: dict[str, float | int] = {"year": int(year), "rows": int(len(group))}
        for name, col in PREDICTION_COLUMNS.items():
            values = prediction_metrics(group["actual_return"], group[col])
            for metric_name, value in values.items():
                item[f"{name}_{metric_name}"] = value
        rows.append(item)
    return pd.DataFrame(rows)


def save_plot(pred_df: pd.DataFrame, metrics: dict[str, float], output_path: Path, code: str, horizon: int) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    axes[0].plot(pred_df["trade_date"], pred_df["actual_return"], label="Actual Return", linewidth=1.5)
    axes[0].plot(pred_df["trade_date"], pred_df["predicted_return"], label="Model Return", linewidth=1.2)
    axes[0].plot(pred_df["trade_date"], pred_df["baseline_zero_return"], label="Zero Baseline", linewidth=1.0, alpha=0.8)
    axes[0].plot(
        pred_df["trade_date"],
        pred_df["baseline_prev_horizon_return"],
        label="Prev Horizon Baseline",
        linewidth=1.0,
        alpha=0.8,
    )
    axes[0].plot(
        pred_df["trade_date"],
        pred_df["baseline_train_mean_return"],
        label="Train Mean Baseline",
        linewidth=1.0,
        alpha=0.8,
    )
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
        f"model_rmse={metrics['model_rmse']:.4f}  "
        f"zero_rmse={metrics['zero_rmse']:.4f}  "
        f"prev_horizon_rmse={metrics['prev_horizon_rmse']:.4f}  "
        f"train_mean_rmse={metrics['train_mean_rmse']:.4f}  "
        f"corr={metrics['model_corr']:.4f}  "
        f"hit={metrics['model_direction_hit_rate']:.2%}"
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


def save_split_summary(split_summary: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = split_summary.copy()
    for col in ["feature_start", "feature_end", "label_end"]:
        out[col] = pd.to_datetime(out[col]).dt.strftime("%Y-%m-%d")
    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def init_qlib() -> None:
    qlib.init(
        provider_uri=str(BASE_DIR),
        region="cn",
        exp_manager={
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {
                "uri": f"file:{DEFAULT_MLRUNS_DIR.resolve()}",
                "default_exp_name": "commodity_forecast",
            },
        },
    )


def run_code(code: str, args: argparse.Namespace, split_config: SplitConfig) -> dict[str, float | str]:
    samples, feature_cols = build_samples(code, args.macro_path, args.horizon, args.monthly_macro_lag_days)
    prepared = prepare_data(samples, feature_cols, split_config)
    dataset = build_dataset(prepared, code)
    pred = fit_predict(dataset)
    pred_df = build_prediction_frame(prepared, pred, code, args.horizon)
    metrics = evaluate(pred_df)
    yearly_metrics = evaluate_by_year(pred_df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / f"{code.lower()}_qlib"
    prediction_path = prefix.with_name(f"{prefix.name}_predictions.csv")
    metrics_path = prefix.with_name(f"{prefix.name}_metrics.csv")
    yearly_metrics_path = prefix.with_name(f"{prefix.name}_metrics_by_year.csv")
    split_summary_path = prefix.with_name(f"{prefix.name}_split_summary.csv")
    plot_path = prefix.with_name(f"{prefix.name}_forecast.png")

    pred_df.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    save_metrics(metrics, metrics_path)
    yearly_metrics.to_csv(yearly_metrics_path, index=False, encoding="utf-8-sig")
    save_split_summary(prepared.split_summary, split_summary_path)
    if not args.skip_plot:
        save_plot(pred_df, metrics, plot_path, code, args.horizon)

    print(f"[qlib] code={code} rows={len(pred_df)}", flush=True)
    print(f"[qlib] prediction_path={prediction_path}", flush=True)
    print(f"[qlib] metrics_path={metrics_path}", flush=True)
    print(f"[qlib] yearly_metrics_path={yearly_metrics_path}", flush=True)
    print(f"[qlib] split_summary_path={split_summary_path}", flush=True)
    if not args.skip_plot:
        print(f"[qlib] plot_path={plot_path}", flush=True)
    print(
        "[qlib] "
        f"rmse={metrics['model_rmse']:.6f} "
        f"zero_rmse={metrics['zero_rmse']:.6f} "
        f"train_mean_rmse={metrics['train_mean_rmse']:.6f} "
        f"prev_horizon_rmse={metrics['prev_horizon_rmse']:.6f} "
        f"rolling_mean_rmse={metrics['rolling_mean_rmse']:.6f} "
        f"mae={metrics['model_mae']:.6f} "
        f"corr={metrics['model_corr']:.6f} "
        f"direction_hit_rate={metrics['model_direction_hit_rate']:.2%}",
        flush=True,
    )
    return {
        "code": code,
        "rows": metrics["rows"],
        "model_rmse": metrics["model_rmse"],
        "zero_rmse": metrics["zero_rmse"],
        "train_mean_rmse": metrics["train_mean_rmse"],
        "prev_horizon_rmse": metrics["prev_horizon_rmse"],
        "rolling_mean_rmse": metrics["rolling_mean_rmse"],
        "model_corr": metrics["model_corr"],
        "model_direction_hit_rate": metrics["model_direction_hit_rate"],
    }


def main() -> None:
    args = parse_args()
    codes = resolve_codes(args)
    split_config = parse_split_config(args)
    init_qlib()

    summary_rows = []
    for code in codes:
        summary_rows.append(run_code(code, args, split_config))

    if len(summary_rows) > 1:
        summary_path = args.output_dir / "qlib_metrics_summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"[qlib] summary_path={summary_path}", flush=True)


if __name__ == "__main__":
    main()
