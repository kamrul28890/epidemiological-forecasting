from __future__ import annotations

import pandas as pd
from typing import List


def bottom_level_units(
    df: pd.DataFrame, geo_col: str, level_list: List[str] | None = None
) -> List[str]:
    """Return the list of bottom-level geo_ids to aggregate."""
    if level_list is not None:
        return level_list
    return sorted(df[geo_col].unique().tolist())



def bottom_up_sum(
    df_pred: pd.DataFrame,
    geo_col: str = "geo_id",
    target_col: str = "y_pred",
    time_col: str = "target_week",
    country_geo: str = "COUNTRY_TOTAL",
    bottom_list: List[str] | None = None,
) -> pd.DataFrame:
    """Build coherent country forecasts by summing non-overlapping bottom-level units."""
    bottom_units = bottom_level_units(df_pred, geo_col, level_list=bottom_list)
    df_bottom = df_pred[df_pred[geo_col].isin(bottom_units)].copy()
    aggregate = (
        df_bottom.groupby([time_col, "horizon"], as_index=False)[target_col]
        .sum()
        .assign(**{geo_col: country_geo})
    )
    return pd.concat([df_pred, aggregate], ignore_index=True)



def with_residual(
    df_input: pd.DataFrame,
    big_name: str,
    sub_name: str,
    geo_col: str = "geo_id",
    target_col: str = "y_pred",
    time_col: str = "target_week",
) -> pd.DataFrame:
    """Replace an overlapping aggregate node with its residual component."""
    df = df_input.copy()
    big = df[df[geo_col] == big_name]
    sub = df[df[geo_col] == sub_name]
    key = [time_col, "horizon"]

    merged = pd.merge(
        big[key + [target_col]],
        sub[key + [target_col]],
        on=key,
        how="inner",
        suffixes=("_big", "_sub"),
    )
    merged["residual"] = (
        merged[f"{target_col}_big"] - merged[f"{target_col}_sub"]
    ).clip(lower=0.0)

    big_residual = (
        merged[key + ["residual"]]
        .rename(columns={"residual": target_col})
        .assign(**{geo_col: f"{big_name}_ex_{sub_name}"})
    )

    df = df[df[geo_col] != big_name]
    return pd.concat([df, big_residual], ignore_index=True)
