from __future__ import annotations
import pickle
from pathlib import Path
from typing import Optional
import pandas as pd
from dataProcessing import prepare_monthly_training, build_future_exog_12m

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR_12M = PROJECT_ROOT / "Data" / "models" / "sarimax_12m"

def forecast_12m(
    models_dir: Path | None = None,
    verbose: bool = True,
    max_groups: Optional[int] = None,
) -> pd.DataFrame:
    if models_dir is None:
        models_dir = MODEL_DIR_12M
    models_dir = Path(models_dir)
    if verbose:
        print(f"Using monthly models from: {models_dir}")
    cons_monthly, _ = prepare_monthly_training()
    group_ids = list(cons_monthly.columns)
    if max_groups is not None:
        original_n = len(group_ids)
        group_ids = group_ids[:max_groups]
        if verbose:
            print(f"Restricting to first {len(group_ids)} of {original_n} groups")
    exog_future = build_future_exog_12m()
    forecast_index = exog_future.index
    if verbose:
        print(f"Forecast horizon length: {len(forecast_index)} months")
        print(f"First forecast timestamp: {forecast_index[0]}")
        print(f"Last  forecast timestamp: {forecast_index[-1]}")
        print(f"Number of groups to forecast: {len(group_ids)}")
    forecast_df = pd.DataFrame(index=forecast_index, columns=group_ids, dtype=float)
    total_groups = len(group_ids)
    for idx, gid in enumerate(group_ids, start=1):
        model_path = models_dir / f"group_{gid}.pkl"
        if not model_path.exists():
            if verbose:
                print(
                    f"[{idx}/{total_groups}] No model file for group {gid}, "
                    "leaving forecast as NaN"
                )
            continue
        if verbose:
            print(
                f"[{idx}/{total_groups}] Forecasting group {gid} "
                f"using {model_path.name}..."
            )
        with open(model_path, "rb") as f:
            results = pickle.load(f)
        steps = len(forecast_index)
        fc = results.get_forecast(steps=steps, exog=exog_future)
        yhat = fc.predicted_mean
        residual_bias = getattr(results, "residual_bias", 0.0)
        yhat_corrected = yhat + residual_bias
        yhat_corrected.index = forecast_index
        forecast_df[gid] = yhat_corrected
    return forecast_df

"""
if __name__ == "__main__":
    fc_12m = forecast_12m(verbose=True, max_groups=3)
    print("\nForecast 12m DataFrame shape:", fc_12m.shape)
    print(fc_12m.head())
"""