"""AR lags (stub)."""

"""Create autoregressive case lags."""
import pandas as pd


def add_case_lags(df: pd.DataFrame, max_lag: int = 8) -> pd.DataFrame:
    out = df.sort_values(["geo_id", "week_start_date"]).copy()
    for k in range(1, max_lag + 1):
        out[f"cases_lag_{k}"] = out.groupby("geo_id")["cases"].shift(k)
    return out
