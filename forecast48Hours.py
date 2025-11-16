from __future__ import annotations
import pickle
from pathlib import Path
from typing import Optional
import pandas as pd
from loadData import load_training_consumption
from dataProcessing import get_48h_forecast_index, build_future_exog_48h
from dataProcessing import build_weekly_baseline_48h

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data"
MODEL_DIR_48H = DATA_DIR / "models" / "sarimax_48h"

def forecast_48h(
    models_dir: Path | None = None,
    verbose: bool = True,
    max_groups: Optional[int] = None,
) -> pd.DataFrame:
    if models_dir is None:
        models_dir = MODEL_DIR_48H
    if verbose:
        print(f"Using models from: {models_dir}")
    cons_hourly = load_training_consumption()
    group_ids = list(cons_hourly.columns)
    if max_groups is not None:
        original_n = len(group_ids)
        group_ids = group_ids[:max_groups]
        if verbose:
            print(f"Restricting to first {len(group_ids)} of {original_n} groups")
    forecast_index = get_48h_forecast_index(consumption_df=cons_hourly)
    exog_future = build_future_exog_48h(
        forecast_index=forecast_index,
        consumption_df=cons_hourly,
    )
    if verbose:
        print(f"Forecast horizon length: {len(forecast_index)} hours")
        print(f"First forecast timestamp: {forecast_index[0]}")
        print(f"Last  forecast timestamp: {forecast_index[-1]}")
        print(f"Number of groups to forecast: {len(group_ids)}")
    forecast_df = pd.DataFrame(index=forecast_index, columns=group_ids, dtype=float)
    total_groups = len(group_ids)
    for idx, gid in enumerate(group_ids, start=1):
        model_path = models_dir / f"group_{gid}.pkl"
        if not model_path.exists():
            if verbose:
                print(f"[{idx}/{total_groups}] No model file for group {gid}, "
                      f"leaving forecast as NaN")
            continue

        # --- Baseline: same hour one week earlier
        baseline = build_weekly_baseline_48h(consumption_df=cons_hourly,
                                         forecast_index=forecast_index)


        if verbose:
            print(f"[{idx}/{total_groups}] Forecasting group {gid} using {model_path.name}...")
        with open(model_path, "rb") as f:
            results = pickle.load(f)
        steps = len(forecast_index)
        fc = results.get_forecast(steps=steps, exog=exog_future)
        yhat = fc.predicted_mean
        residual_bias = getattr(results, "residual_bias", 0.0)
        yhat_corrected = yhat + residual_bias
        yhat_corrected.index = forecast_index

        # Final forecast: baseline + predicted residual
        y_baseline = baseline[gid]
        yhat = y_baseline + yhat_corrected
        yhat.index = forecast_index

        forecast_df[gid] = yhat_corrected
    return forecast_df

"""
if __name__ == "__main__":
    fc_48h = forecast_48h(verbose=True, max_groups=3)
    print("\nForecast 48h DataFrame shape:", fc_48h.shape)
    print(fc_48h.head())
"""    