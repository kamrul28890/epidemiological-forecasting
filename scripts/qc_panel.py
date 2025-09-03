"""Quick QC for data/interim/panel_raw.parquet"""
from pathlib import Path
import pandas as pd

PANEL = Path("data/interim/panel_raw.parquet")

def main():
    df = pd.read_parquet(PANEL)
    print(f"[qc] rows={len(df):,} geos={df['geo_id'].nunique()} "
          f"range={df['week_start_date'].min():%Y-%m-%d}→{df['week_start_date'].max():%Y-%m-%d}")

    issues = []

    # core NA checks
    for col in ["week_start_date", "cases"]:
        n = df[col].isna().sum()
        if n:
            issues.append(f"{col} has {n} NA")

    # non-negativity
    if (df["cases"] < 0).any():
        issues.append("negative case counts found")

    # duplicates
    dups = df.duplicated(["geo_id","week_start_date"]).sum()
    if dups:
        issues.append(f"{dups} duplicate rows on (geo_id, week_start_date)")

    # expected geos (Dhaka split present)
    need = {"DHA_METRO","DHA_DIV"}
    miss = need - set(df["geo_id"].unique())
    if miss:
        issues.append(f"missing geos: {sorted(miss)}")

    # climate cutoff per geo (example: rain)
    if "rain" in df.columns:
        tail = (
            df.assign(has_rain=~df["rain"].isna())
              .groupby("geo_id", group_keys=False)
              .apply(lambda g: g.loc[g["has_rain"], "week_start_date"].max())
        )
        print("[qc] last week with non-NA rain per geo:\n", tail)

    print("[qc] OK" if not issues else "[qc] Issues:")
    for m in issues:
        print(" -", m)

if __name__ == "__main__":
    main()
