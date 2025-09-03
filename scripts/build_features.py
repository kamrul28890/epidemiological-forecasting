"""
Build feature matrix from panel_raw:
 - AR lags (1..8)
 - climate lags/rollings (2/4/8)
 - calendar features
Writes:
  data/processed/design_matrix.parquet
  data/processed/design_sample.csv
"""

from pathlib import Path

import pandas as pd

from dengue.features.ar_terms import add_case_lags
from dengue.features.calendar import add_calendar
from dengue.features.climate_lags import add_climate_lags_rolls

PANEL = Path("data/interim/panel_raw.parquet")
OUT_PARQ = Path("data/processed/design_matrix.parquet")
OUT_CSV = Path("data/processed/design_sample.csv")


def main():
    df = pd.read_parquet(PANEL)
    print(f"[features] loaded panel: {len(df):,} rows, {df['geo_id'].nunique()} geos")
    # add features
    df = add_case_lags(df, max_lag=8)
    df = add_climate_lags_rolls(
        df,
        climate_vars=("rain", "tmean", "humidity", "dewpoint", "soil_moist_0_7cm"),
        lags=range(1, 9),
        roll_means=(2, 4, 8),
        roll_sums=(2, 4, 8),
    )
    df = add_calendar(df, monsoon=("06-01", "10-15"))

    # drop rows with NA due to lagging (first 8 weeks per geo)
    keep = df.copy()
    lag_cols = [c for c in keep.columns if c.startswith("cases_lag_")]
    keep = keep.dropna(subset=lag_cols)

    # save
    OUT_PARQ.parent.mkdir(parents=True, exist_ok=True)
    keep.to_parquet(OUT_PARQ, index=False)
    keep.head(200).to_csv(OUT_CSV, index=False)
    print(f"[features] saved {OUT_PARQ} ({len(keep):,} rows, {keep.shape[1]} cols)")
    print(f"[features] sample → {OUT_CSV}")


if __name__ == "__main__":
    main()
