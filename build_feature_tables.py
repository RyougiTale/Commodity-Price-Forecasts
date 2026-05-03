from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_DIR = BASE_DIR / "data" / "raw" / "tushare"
DEFAULT_FEATURES_DIR = BASE_DIR / "data" / "features"


@dataclass(frozen=True)
class CommodityConfig:
    code: str
    exchange: str
    file_name: str
    name: str


TARGETS = [
    CommodityConfig(code="CU", exchange="SHFE", file_name="cu_features_daily.csv", name="铜"),
    CommodityConfig(code="AU", exchange="SHFE", file_name="au_features_daily.csv", name="黄金"),
    CommodityConfig(code="SC", exchange="INE", file_name="sc_features_daily.csv", name="原油"),
    CommodityConfig(code="RB", exchange="SHFE", file_name="rb_features_daily.csv", name="螺纹钢"),
    CommodityConfig(code="C", exchange="DCE", file_name="c_features_daily.csv", name="玉米"),
]


WSR_FEATURE_COLUMNS = [
    "warehouse_receipt_total",
    "warehouse_receipt_change",
    "warehouse_count",
]
HOLDING_FEATURE_COLUMNS = [
    "top20_long_sum",
    "top20_short_sum",
    "top20_long_chg",
    "top20_short_chg",
    "top5_long_sum",
    "top5_short_sum",
    "broker_count",
    "long_short_imbalance",
    "long_concentration",
    "short_concentration",
]


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_raw(raw_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "basic": read_csv_safe(raw_dir / "fut_basic.csv"),
        "daily": read_csv_safe(raw_dir / "fut_daily_all_contracts.csv"),
        "mapping": read_csv_safe(raw_dir / "fut_mapping.csv"),
        "holding": read_csv_safe(raw_dir / "fut_holding.csv"),
        "wsr": read_csv_safe(raw_dir / "fut_wsr.csv"),
    }


def to_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype(str), format="%Y%m%d", errors="coerce")


def build_term_structure(daily_with_meta: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for trade_date, group in daily_with_meta.groupby("trade_date", sort=True):
        active = group[group["oi"].fillna(0) > 0].copy()
        if active.empty:
            continue
        by_oi = active.sort_values("oi", ascending=False).reset_index(drop=True)
        main = by_oi.iloc[0]
        second = by_oi.iloc[1] if len(by_oi) > 1 else None

        forward = active.dropna(subset=["days_to_expiry"])
        forward = forward[forward["days_to_expiry"] > 0].sort_values("days_to_expiry")
        near = forward.iloc[0] if not forward.empty else None
        far = forward.iloc[-1] if not forward.empty else None

        total_vol = active["vol"].sum()
        total_oi = active["oi"].sum()

        row: dict[str, object] = {
            "trade_date": trade_date,
            "main_ts_code": main["ts_code"],
            "main_open": main["open"],
            "main_high": main["high"],
            "main_low": main["low"],
            "main_close": main["close"],
            "main_settle": main.get("settle"),
            "main_pre_close": main.get("pre_close"),
            "main_pre_settle": main.get("pre_settle"),
            "main_change1": main.get("change1"),
            "main_change2": main.get("change2"),
            "main_vol": main["vol"],
            "main_amount": main.get("amount"),
            "main_oi": main["oi"],
            "main_oi_chg": main.get("oi_chg"),
            "main_days_to_expiry": main["days_to_expiry"],
            "main_volume_share": main["vol"] / total_vol if total_vol > 0 else np.nan,
            "main_oi_share": main["oi"] / total_oi if total_oi > 0 else np.nan,
            "active_contract_count": int(len(active)),
        }

        if second is not None:
            row["second_main_ts_code"] = second["ts_code"]
            row["second_main_close"] = second["close"]
            row["second_main_settle"] = second.get("settle")
            row["second_main_oi"] = second["oi"]
            row["second_main_main_spread"] = second["close"] - main["close"]

        if near is not None:
            row["near_ts_code"] = near["ts_code"]
            row["near_close"] = near["close"]
            row["near_settle"] = near.get("settle")
            row["near_days_to_expiry"] = near["days_to_expiry"]
            row["main_near_spread"] = main["close"] - near["close"]

        if far is not None:
            row["far_ts_code"] = far["ts_code"]
            row["far_close"] = far["close"]
            row["far_settle"] = far.get("settle")
            row["far_days_to_expiry"] = far["days_to_expiry"]

        if near is not None and far is not None:
            day_diff = far["days_to_expiry"] - near["days_to_expiry"]
            if near["close"] and near["close"] > 0 and day_diff > 0:
                ratio = far["close"] / near["close"]
                row["far_near_spread"] = far["close"] - near["close"]
                row["annualized_carry"] = (ratio - 1) * 365 / day_diff
                row["term_structure_slope"] = float(np.log(ratio)) * 365 / day_diff

        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_holdings(holding_df: pd.DataFrame, contracts: pd.DataFrame) -> pd.DataFrame:
    if holding_df.empty or contracts.empty:
        return pd.DataFrame(columns=["trade_date", "ts_code", *HOLDING_FEATURE_COLUMNS])

    sym_to_code = dict(zip(contracts["symbol"].astype(str), contracts["ts_code"].astype(str)))
    df = holding_df.copy()
    df["trade_date"] = to_dt(df["trade_date"])
    df["ts_code"] = df["symbol"].astype(str).map(sym_to_code)
    df = df.dropna(subset=["ts_code", "trade_date"])
    if df.empty:
        return pd.DataFrame(columns=["trade_date", "ts_code", *HOLDING_FEATURE_COLUMNS])

    for col in ("long_hld", "short_hld", "long_chg", "short_chg"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    grouped = df.groupby(["trade_date", "ts_code"], sort=False)
    agg = grouped.agg(
        top20_long_sum=("long_hld", "sum"),
        top20_short_sum=("short_hld", "sum"),
        top20_long_chg=("long_chg", "sum"),
        top20_short_chg=("short_chg", "sum"),
        broker_count=("broker", "nunique"),
    ).reset_index()

    top5_long = (
        df.sort_values(["trade_date", "ts_code", "long_hld"], ascending=[True, True, False])
        .groupby(["trade_date", "ts_code"], sort=False)
        .head(5)
        .groupby(["trade_date", "ts_code"], sort=False)["long_hld"]
        .sum()
        .reset_index(name="top5_long_sum")
    )
    top5_short = (
        df.sort_values(["trade_date", "ts_code", "short_hld"], ascending=[True, True, False])
        .groupby(["trade_date", "ts_code"], sort=False)
        .head(5)
        .groupby(["trade_date", "ts_code"], sort=False)["short_hld"]
        .sum()
        .reset_index(name="top5_short_sum")
    )
    agg = agg.merge(top5_long, on=["trade_date", "ts_code"], how="left")
    agg = agg.merge(top5_short, on=["trade_date", "ts_code"], how="left")

    sum_pos = agg["top20_long_sum"].fillna(0) + agg["top20_short_sum"].fillna(0)
    agg["long_short_imbalance"] = np.where(
        sum_pos > 0,
        (agg["top20_long_sum"].fillna(0) - agg["top20_short_sum"].fillna(0)) / sum_pos,
        np.nan,
    )
    agg["long_concentration"] = np.where(
        agg["top20_long_sum"] > 0, agg["top5_long_sum"] / agg["top20_long_sum"], np.nan
    )
    agg["short_concentration"] = np.where(
        agg["top20_short_sum"] > 0, agg["top5_short_sum"] / agg["top20_short_sum"], np.nan
    )
    return agg


def aggregate_wsr(wsr_df: pd.DataFrame, fut_code: str) -> pd.DataFrame:
    if wsr_df.empty or "fut_code" not in wsr_df.columns:
        return pd.DataFrame(columns=["trade_date", *WSR_FEATURE_COLUMNS])
    df = wsr_df[wsr_df["fut_code"].astype(str) == fut_code].copy()
    if df.empty:
        return pd.DataFrame(columns=["trade_date", *WSR_FEATURE_COLUMNS])
    df["trade_date"] = to_dt(df["trade_date"])
    df["vol"] = pd.to_numeric(df.get("vol"), errors="coerce")
    df["vol_chg"] = pd.to_numeric(df.get("vol_chg"), errors="coerce")
    agg = df.groupby("trade_date", sort=True).agg(
        warehouse_receipt_total=("vol", "sum"),
        warehouse_receipt_change=("vol_chg", "sum"),
        warehouse_count=("warehouse", "nunique"),
    ).reset_index()
    return agg


def build_features_for_commodity(config: CommodityConfig, raws: dict[str, pd.DataFrame]) -> pd.DataFrame:
    basic = raws["basic"]
    daily = raws["daily"]
    if basic.empty or daily.empty:
        return pd.DataFrame()

    contracts = basic[basic["fut_code"].astype(str) == config.code].copy()
    if contracts.empty:
        return pd.DataFrame()

    contracts["delist_dt"] = to_dt(contracts["delist_date"])
    last_dt = to_dt(contracts.get("last_ddate", pd.Series(index=contracts.index, dtype=object)))
    contracts["delist_dt"] = contracts["delist_dt"].fillna(last_dt)

    daily_for = daily[daily["ts_code"].isin(contracts["ts_code"])].copy()
    if daily_for.empty:
        return pd.DataFrame()

    daily_for["trade_date"] = to_dt(daily_for["trade_date"])
    for col in ("open", "high", "low", "close", "settle", "vol", "oi", "amount", "oi_chg"):
        if col in daily_for.columns:
            daily_for[col] = pd.to_numeric(daily_for[col], errors="coerce")
    daily_for = daily_for.merge(contracts[["ts_code", "delist_dt"]], on="ts_code", how="left")
    daily_for["days_to_expiry"] = (daily_for["delist_dt"] - daily_for["trade_date"]).dt.days

    features = build_term_structure(daily_for)
    if features.empty:
        return features

    holdings = aggregate_holdings(raws["holding"], contracts)
    if not holdings.empty:
        features = features.merge(
            holdings,
            left_on=["trade_date", "main_ts_code"],
            right_on=["trade_date", "ts_code"],
            how="left",
        ).drop(columns=["ts_code"])

    wsr_agg = aggregate_wsr(raws["wsr"], config.code)
    if not wsr_agg.empty:
        features = features.merge(wsr_agg, on="trade_date", how="left")

    features["trade_date"] = features["trade_date"].dt.strftime("%Y-%m-%d")
    return features.sort_values("trade_date").reset_index(drop=True)


def parse_codes(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip().upper() for item in value.split(",") if item.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate raw Tushare CSVs into per-commodity feature tables."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Raw CSV directory.")
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR, help="Output features directory.")
    parser.add_argument("--codes", help="Comma-separated commodity codes, for example RB,C.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raws = load_raw(args.raw_dir)
    code_filter = parse_codes(args.codes)
    targets = [cfg for cfg in TARGETS if code_filter is None or cfg.code in code_filter]
    args.features_dir.mkdir(parents=True, exist_ok=True)

    for cfg in targets:
        df = build_features_for_commodity(cfg, raws)
        path = args.features_dir / cfg.file_name
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[features] {cfg.code} rows={len(df)} cols={len(df.columns)} -> {path}", flush=True)


if __name__ == "__main__":
    main()
