from __future__ import annotations
from typing import Tuple
import pandas as pd
from loadData import (
    load_training_consumption,
    load_training_prices,
    load_example_hourly,
    load_example_monthly,
)

def get_48h_forecast_index(consumption_df: pd.DataFrame | None = None) -> pd.DatetimeIndex:
    if consumption_df is None:
        consumption_df = load_training_consumption()

    train_end = consumption_df.index.max()
    forecast_index = pd.date_range(
        start=train_end + pd.Timedelta(hours=1),
        periods=48,
        freq="H",
        tz=consumption_df.index.tz, 
    )
    return forecast_index

def _build_hourly_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame(index=index)
    df["hour"] = index.hour
    df["day_of_week"] = index.dayofweek  
    df["month"] = index.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df

def _build_monthly_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame(index=index)
    df["month"] = index.month
    df["year"] = index.year
    df["t"] = range(1, len(index) + 1)
    return df

def prepare_hourly_training(
    consumption_df: pd.DataFrame | None = None,
    prices_df: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if consumption_df is None:
        consumption_df = load_training_consumption()
    if prices_df is None:
        prices_df = load_training_prices()
    common_index = consumption_df.index.intersection(prices_df.index)
    cons_aligned = consumption_df.loc[common_index].sort_index()
    price = prices_df.loc[common_index, "eur_per_mwh"].sort_index()
    price = price.interpolate(method="time").ffill().bfill()
    exog = _build_hourly_calendar_features(common_index)
    exog["price"] = price
    return cons_aligned, exog


def prepare_monthly_training(
    consumption_df: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if consumption_df is None:
        consumption_df = load_training_consumption()
    cons_monthly = consumption_df.resample("MS").sum().sort_index()
    exog_monthly = _build_monthly_calendar_features(cons_monthly.index)

    return cons_monthly, exog_monthly

def build_future_exog_48h(
    forecast_index: pd.DatetimeIndex | None = None,
    prices_df: pd.DataFrame | None = None,
    consumption_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if consumption_df is None:
        consumption_df = load_training_consumption()
    if prices_df is None:
        prices_df = load_training_prices()
    if forecast_index is None:
        forecast_index = get_48h_forecast_index(consumption_df)
    price_series = prices_df["eur_per_mwh"].copy()
    price_series.index = prices_df.index
    price_series = price_series.sort_index()
    price_series = price_series.interpolate(method="time").ffill().bfill()
    train_end = consumption_df.index.max()
    last_price_time = price_series.index.max()
    window_start = train_end - pd.Timedelta(days=27)
    window = price_series.loc[window_start:train_end]
    hourly_pattern = window.groupby(window.index.hour).mean()
    price_future = pd.Series(index=forecast_index, dtype=float)
    for t in forecast_index:
        if t <= last_price_time and t in price_series.index:
            price_future[t] = price_series.loc[t]
        else:
            price_future[t] = hourly_pattern.loc[t.hour]
    exog_future = _build_hourly_calendar_features(forecast_index)
    exog_future["price"] = price_future
    return exog_future

def build_future_exog_12m(
    example_monthly_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if example_monthly_df is None:
        example_monthly_df = load_example_monthly()
    forecast_index = example_monthly_df.index.sort_values()
    exog_future = _build_monthly_calendar_features(forecast_index)
    return exog_future

def build_weekly_baseline_48h(
    consumption_df: pd.DataFrame,
    forecast_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Naive baseline for 48h horizon: same hour one week earlier.
    Returns a wide DataFrame with the same columns as consumption_df.
    """
    baseline_index = forecast_index - pd.Timedelta(days=7)
    baseline = consumption_df.reindex(baseline_index)
    baseline.index = forecast_index
    return baseline

"""
import pandas as pd

def prepare_groups_with_metadata(groups: pd.DataFrame) -> pd.DataFrame:
    g = groups.copy()
    g["group_id"] = g["group_id"].astype(int)
    parts = g["group_label"].str.split(" | ", expand=True, regex=False)
    col_names = [
        "macro_region",
        "county",
        "municipality",
        "segment",
        "product_type",
        "consumption_bucket",
    ]
    for i, col in enumerate(col_names):
        if i < parts.shape[1]:
            g[col] = parts[i].str.strip()
        else:
            g[col] = None
    return g

def add_calendar_features(
    df: pd.DataFrame,
    ts_col: str = "measured_at",
) -> pd.DataFrame:
    df = df.copy()
    dt = df[ts_col]
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6])
    df["day"] = dt.dt.day
    df["month"] = dt.dt.month
    df["year"] = dt.dt.year
    df["hour_of_week"] = df["day_of_week"] * 24 + df["hour"]
    return df

def build_hourly_feature_table(
    consumption_long: pd.DataFrame,
    prices: pd.DataFrame,
    groups : pd.DataFrame,
) -> pd.DataFrame:
    df = consumption_long.merge(
        prices,
        on="measured_at",
        how="left",
    )
    df = df.sort_values(["group_id", "measured_at"]).reset_index(drop=True)
    groups_meta = prepare_groups_with_metadata(groups)
    df = df.merge(
        groups_meta,
        on="group_id",
        how="left",
    )
    df = add_calendar_features(df, ts_col="measured_at")
    return df

def build_monthly_feature_table(
    consumption_long: pd.DataFrame,
    groups: pd.DataFrame,
) -> pd.DataFrame:
    df = consumption_long.copy()
    df["year"] = df["measured_at"].dt.year
    df["month"] = df["measured_at"].dt.month
    monthly = (
        df.groupby(["group_id", "year", "month"], as_index=False)["fwh"]
          .sum()
          .rename(columns={"fwh": "monthly_fwh"})
    )
    monthly["month_start"] = pd.to_datetime(
        monthly["year"].astype(str) + "-" +
        monthly["month"].astype(str).str.zfill(2) + "-01",
        utc=True,
    )
    monthly["month_of_year"] = monthly["month"]
    monthly = monthly.sort_values(["group_id", "year", "month"]).reset_index(drop=True)
    monthly["time_index"] = (
        monthly.groupby("group_id").cumcount()
    )
    groups_meta = prepare_groups_with_metadata(groups)
    monthly = monthly.merge(groups_meta, on="group_id", how="left")
    return monthly
"""