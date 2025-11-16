from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd

from loadData import (
    load_example_hourly,
    load_example_monthly,
    EXAMPLE_HOURLY_CSV,
    EXAMPLE_MONTHLY_CSV,
)

def _align_forecast_to_template(
    forecast_df: pd.DataFrame, template_df: pd.DataFrame
) -> pd.DataFrame:
    fc = forecast_df.copy()
    fc.columns = [str(c) for c in fc.columns]
    template_cols = [str(c) for c in template_df.columns]
    aligned = fc.reindex(index=template_df.index, columns=template_cols)
    missing_groups = [col for col in aligned.columns if aligned[col].isna().all()]
    if missing_groups:
        print(
            "WARNING: no forecasts for these groups; filling with 0.0:\n"
            f"  {', '.join(missing_groups)}"
        )
    aligned = aligned.fillna(0.0)
    return aligned

def build_submission_48h(
    forecast_hourly: pd.DataFrame,
    template_path: Path | str = EXAMPLE_HOURLY_CSV,
) -> pd.DataFrame:
    template = load_example_hourly(template_path)
    aligned = _align_forecast_to_template(forecast_hourly, template)
    submission = aligned.copy()
    measured_at_str = template.index.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    submission.insert(0, "measured_at", measured_at_str)
    return submission

def build_submission_12m(
    forecast_monthly: pd.DataFrame,
    template_path: Path | str = EXAMPLE_MONTHLY_CSV,
) -> pd.DataFrame:
    template = load_example_monthly(template_path)
    aligned = _align_forecast_to_template(forecast_monthly, template)
    submission = aligned.copy()
    measured_at_str = template.index.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    submission.insert(0, "measured_at", measured_at_str)
    return submission

def save_submission_csv(
    submission_df: pd.DataFrame,
    output_path: Path | str,
    verbose: bool = True,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Writing submission file to: {output_path}")
    submission_df.to_csv(
        output_path,
        sep=";",
        index=False,
        decimal=",",
    )
