from __future__ import annotations
import argparse
from pathlib import Path
from train48Hours import train_sarimax_48h
from train12Months import train_sarimax_12m
from forecast48Hours import forecast_48h
from forecast12Months import forecast_12m
from converter import (
    build_submission_48h,
    build_submission_12m,
    save_submission_csv,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data"
OUTPUT_DIR = DATA_DIR / "forecasts"

def run_pipeline(
    do_train: bool = True,
    train_days_48h: int = 365,
    train_months_12m: int | None = None,
    max_groups: int | None = None,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if do_train:
        print("=== Training 48-hour SARIMAX models ===")
        train_sarimax_48h(train_days=train_days_48h, max_groups=max_groups)

        print("\n=== Training 12-month SARIMAX models ===")
        train_sarimax_12m(train_months=train_months_12m, max_groups=max_groups)
    else:
        print("Skipping training; using existing models on disk.")
    print("\n=== Forecasting 48 hours ===")
    fc_48 = forecast_48h(verbose=True, max_groups=max_groups)
    sub_48 = build_submission_48h(fc_48)
    out_48 = OUTPUT_DIR / "forecast_48h_submission.csv"
    save_submission_csv(sub_48, out_48)
    print(f"48-hour submission saved to: {out_48}")
    print("\n=== Forecasting 12 months ===")
    fc_12 = forecast_12m(verbose=True, max_groups=max_groups)
    sub_12 = build_submission_12m(fc_12)
    out_12 = OUTPUT_DIR / "forecast_12m_submission.csv"
    save_submission_csv(sub_12, out_12)
    print(f"12-month submission saved to: {out_12}")
    print("\nPipeline completed.")

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SARIMAX training and forecasting pipeline."
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Use existing models and skip training.",
    )
    parser.add_argument(
        "--train-days-48h",
        type=int,
        default=365,
        help="Number of days of hourly history to use for 48h models.",
    )
    parser.add_argument(
        "--train-months-12m",
        type=int,
        default=None,
        help="Number of months of history to use for 12m models "
             "(default: use all available).",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=None,
        help="If set, train/forecast only the first N groups (for quick tests).",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        do_train= args.skip_training,
        train_days_48h=30, #args.train_days_48h,
        train_months_12m=24, #args.train_months_12m,
        max_groups=args.max_groups,
    )
