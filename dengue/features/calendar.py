"""Calendar features (stub)."""

"""Calendar features: week-of-year, sin/cos seasonality, monsoon flag."""
import numpy as np
import pandas as pd


def add_calendar(df: pd.DataFrame, monsoon=("06-01", "10-15")) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out["week_start_date"])
    out["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    # sin/cos seasonality
    w = out["week_of_year"].values
    out["woy_sin"] = np.sin(2 * np.pi * w / 52.0)
    out["woy_cos"] = np.cos(2 * np.pi * w / 52.0)
    # monsoon flag
    start_mmdd, end_mmdd = monsoon
    mmdd = dt.dt.strftime("%m-%d")
    out["monsoon_flag"] = ((mmdd >= start_mmdd) & (mmdd <= end_mmdd)).astype(int)
    return out
