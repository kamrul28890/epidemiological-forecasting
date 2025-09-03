"""Climate lags/rollings (stub)."""

"""Create climate lags and rolling windows (index-safe)."""
import pandas as pd

def add_climate_lags_rolls(
    df: pd.DataFrame,
    climate_vars=("rain", "tmean", "humidity", "dewpoint", "soil_moist_0_7cm"),
    lags=range(1, 9),
    roll_means=(2, 4, 8),
    roll_sums=(2, 4, 8),
) -> pd.DataFrame:
    out = df.sort_values(["geo_id", "week_start_date"]).copy()

    # Ensure numeric dtypes for climate vars (safe if columns are absent)
    for v in climate_vars:
        if v in out.columns:
            out[v] = pd.to_numeric(out[v], errors="coerce")

    # Lags (aligned to original index)
    for v in climate_vars:
        if v not in out.columns:
            continue
        for k in lags:
            out[f"{v}_lag_{k}"] = out.groupby("geo_id")[v].shift(k)

    # Rolling stats via transform (preserves index; no MultiIndex issues)
    for v in climate_vars:
        if v not in out.columns:
            continue
        if v == "rain":
            for w in roll_sums:
                out[f"{v}_rollsum_{w}"] = (
                    out.groupby("geo_id")[v]
                       .transform(lambda s: s.rolling(w, min_periods=1).sum())
                )
        else:
            for w in roll_means:
                out[f"{v}_rollmean_{w}"] = (
                    out.groupby("geo_id")[v]
                       .transform(lambda s: s.rolling(w, min_periods=1).mean())
                )

    return out
