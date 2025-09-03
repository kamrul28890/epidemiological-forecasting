"""Rolling-origin split generator for weekly forecasting.

Creates a tidy table of (geo_id, week_start_date, target_week, horizon, role, fold_id)
where:
- week_start_date: reference week (t), Monday-anchored
- target_week: week being predicted (t + h), Monday-anchored
- role: 'train' if target_week <= train_end_monday,
       'val'   if train_end_monday < target_week <= val_end_monday
"""

import pandas as pd


def _to_monday(ts) -> pd.Timestamp:
    ts = pd.to_datetime(ts)
    return ts.to_period("W-SUN").start_time  # week ends Sunday → start is Monday


def _add_weeks(dt_series: pd.Series, h: int) -> pd.Series:
    return pd.to_datetime(dt_series) + pd.to_timedelta(h * 7, unit="D")


def build_split_pairs(
    design: pd.DataFrame,
    horizons: list[int],
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
    fold_id: str,
) -> pd.DataFrame:
    # base weekly index per geo, Monday-anchored
    base = (
        design[["geo_id", "week_start_date"]]
        .drop_duplicates()
        .assign(
            week_start_date=lambda d: pd.to_datetime(d["week_start_date"])
            .dt.to_period("W-SUN")
            .dt.start_time
        )
        .sort_values(["geo_id", "week_start_date"])
    )

    # anchor boundaries to Monday-start weeks
    train_end_m = _to_monday(train_end)
    val_end_m = _to_monday(val_end)

    frames = []
    for h in horizons:
        tmp = base.copy()
        tmp["horizon"] = h
        tmp["target_week"] = _add_weeks(tmp["week_start_date"], h)
        tmp["role"] = pd.NA
        tmp.loc[tmp["target_week"] <= train_end_m, "role"] = "train"
        tmp.loc[
            (tmp["target_week"] > train_end_m) & (tmp["target_week"] <= val_end_m),
            "role",
        ] = "val"
        tmp = tmp.dropna(subset=["role"])
        frames.append(tmp)

    out = pd.concat(frames, ignore_index=True)
    out["fold_id"] = fold_id
    return out.sort_values(["fold_id", "geo_id", "week_start_date", "horizon"])
