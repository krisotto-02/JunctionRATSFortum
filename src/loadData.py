from pathlib import Path
from typing import Tuple
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data"
TRAINING_EXCEL = DATA_DIR / "20251111_JUNCTION_training.xlsx"
EXAMPLE_HOURLY_CSV = DATA_DIR / "20251111_JUNCTION_example_hourly.csv"
EXAMPLE_MONTHLY_CSV = DATA_DIR / "20251111_JUNCTION_example_monthly.csv"

def load_groups(path: Path | str = TRAINING_EXCEL) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="groups")
    df["group_id"] = df["group_id"].astype(int)
    return df

def load_training_consumption(path: Path | str = TRAINING_EXCEL) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="training_consumption")
    df["measured_at"] = pd.to_datetime(df["measured_at"], utc=True)
    df = df.set_index("measured_at").sort_index()
    return df


def load_training_prices(path: Path | str = TRAINING_EXCEL) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="training_prices")
    df["measured_at"] = pd.to_datetime(df["measured_at"], utc=True)
    df = df.set_index("measured_at").sort_index()
    return df

def load_all_training_data(
    path: Path | str = TRAINING_EXCEL,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups_df = load_groups(path)
    consumption_df = load_training_consumption(path)
    prices_df = load_training_prices(path)
    return groups_df, consumption_df, prices_df

def load_example_hourly(path: Path | str = EXAMPLE_HOURLY_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", decimal=",")
    df["measured_at"] = pd.to_datetime(df["measured_at"], utc=True)
    df = df.set_index("measured_at").sort_index()
    return df


def load_example_monthly(path: Path | str = EXAMPLE_MONTHLY_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", decimal=",")
    df["measured_at"] = pd.to_datetime(df["measured_at"], utc=True)
    df = df.set_index("measured_at").sort_index()
    return df

def load_all_templates(
    hourly_path: Path | str = EXAMPLE_HOURLY_CSV,
    monthly_path: Path | str = EXAMPLE_MONTHLY_CSV,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    hourly_df = load_example_hourly(hourly_path)
    monthly_df = load_example_monthly(monthly_path)
    return hourly_df, monthly_df

"""
import pandas as pd
from pathlib import Path

def load_groups(xlsx_path: str | Path) -> pd.DataFrame:
    xlsx_path = Path(xlsx_path)
    groups = pd.read_excel(xlsx_path, sheet_name="groups")
    groups["group_id"] = groups["group_id"].astype(int)
    return groups

def load_consumption_long(xlsx_path: str | Path) -> pd.DataFrame:
    xlsx_path = Path(xlsx_path)
    cons_wide = pd.read_excel(xlsx_path, sheet_name="training_consumption")
    cons_long = cons_wide.melt(
        id_vars="measured_at",
        var_name="group_id",
        value_name="fwh",
    )
    cons_long["group_id"] = cons_long["group_id"].astype(int)
    cons_long["measured_at"] = pd.to_datetime(cons_long["measured_at"], utc=True)
    cons_long = cons_long.sort_values(["group_id", "measured_at"]).reset_index(drop=True)
    return cons_long

def load_prices(xlsx_path: str | Path) -> pd.DataFrame:
    xlsx_path = Path(xlsx_path)
    prices = pd.read_excel(xlsx_path, sheet_name="training_prices")
    prices["measured_at"] = pd.to_datetime(prices["measured_at"], utc=True)
    return prices

def load_all(xlsx_path: str | Path):
    groups = load_groups(xlsx_path)
    consumption = load_consumption_long(xlsx_path)
    prices = load_prices(xlsx_path)
    return groups, consumption, prices
"""