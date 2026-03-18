from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def to_date(s): return pd.to_datetime(s).normalize()

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--gap-start", required=True)  # e.g., 2025-07-01
    ap.add_argument("--gap-end", required=True)    # e.g., 2025-08-23
    ap.add_argument("--out", required=True)
    return ap.parse_args()

def main():
    a = parse_args()
    dm = pd.read_parquet(a.design)
    sp = pd.read_parquet(a.splits)

    # Normalize
    dm["week_start_date"] = pd.to_datetime(dm["week_start_date"]).dt.normalize()
    sp["week_start_date"] = pd.to_datetime(sp["week_start_date"]).dt.normalize()
    sp["target_week"]     = pd.to_datetime(sp["target_week"]).dt.normalize()

    # Horizons to use come from current splits
    horizons = sorted(sp["horizon"].unique().tolist())
    geo_ids  = sorted(dm["geo_id"].unique().tolist())

    # The design matrixâ€™s available origins per geo
    available = (dm[["geo_id","week_start_date"]]
                 .drop_duplicates()
                 .groupby("geo_id")["week_start_date"].apply(set)
                 .to_dict())

    # Generate Monday-anchored target weeks in the gap
    gap_start = to_date(a.gap_start)
    gap_end   = to_date(a.gap_end)
    # round each date down to Monday (ISO weekday=0)
    def to_monday(ts): return (ts - pd.to_timedelta(ts.weekday(), unit="D")).normalize()
    # Make a weekly (Mon) range covering the gap window
    first_mon = to_monday(gap_start)
    last_mon  = to_monday(gap_end)
    target_weeks = pd.date_range(first_mon, last_mon, freq="7D")

    # New fold id
    # If fold_id is numeric, weâ€™ll bump; otherwise weâ€™ll create a string id "gap"
    if np.issubdtype(sp["fold_id"].dtype, np.number):
        new_fold_id = int(sp["fold_id"].max()) + 1
    else:
        new_fold_id = "gap"

    rows = []
    for geo in geo_ids:
        avail = available.get(geo, set())
        for h in horizons:
            for tw in target_weeks:
                origin = tw - pd.Timedelta(days=7*h)
                # Only add if we have features at that origin for this geo
                if origin in avail:
                    rows.append({
                        "geo_id": geo,
                        "week_start_date": origin,
                        "target_week": tw,
                        "horizon": h,
                        "role": "val",          # scored window
                        "fold_id": new_fold_id
                    })

    if not rows:
        raise SystemExit("No gap rows could be generated. Check that your design_matrix covers the needed origins.")

    sp_gap = pd.DataFrame(rows)
    # Avoid duplicates if any overlap
    sp_all = pd.concat([sp, sp_gap], ignore_index=True).drop_duplicates(
        ["geo_id","week_start_date","target_week","horizon","role","fold_id"]
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    sp_all.to_parquet(a.out, index=False)
    print(f"Wrote {a.out} with {len(sp_all)} rows (+{len(sp_all)-len(sp)} new gap rows).")

if __name__ == "__main__":
    main()

