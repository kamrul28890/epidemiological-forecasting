"""
Generate rolling-origin splits for 1–12 week horizons using configs/split.yaml.
Writes:
  - data/processed/splits.parquet
  - data/processed/splits_summary.csv
"""

from pathlib import Path

import pandas as pd

from dengue.split.rolling_origin import build_split_pairs
from dengue.utils.io import load_yaml

DESIGN = Path("data/processed/design_matrix.parquet")
SPLITS_PARQ = Path("data/processed/splits.parquet")
SPLITS_SUMMARY = Path("data/processed/splits_summary.csv")


def main():
    cfg_split = load_yaml("configs/split.yaml")
    horizons = list(cfg_split["horizons"])
    folds_cfg = cfg_split["folds"]

    df = pd.read_parquet(DESIGN)
    print(
        f"[splits] design: {len(df):,} rows, weeks {df['week_start_date'].min():%Y-%m-%d}→{df['week_start_date'].max():%Y-%m-%d}"
    )

    all_pairs = []
    for i, f in enumerate(folds_cfg, start=1):
        fold_id = f"F{i}"
        train_end = pd.to_datetime(f["train_end"])
        val_end = pd.to_datetime(f["val_end"])
        pairs = build_split_pairs(df, horizons, train_end, val_end, fold_id)
        n_tr = (pairs["role"] == "train").sum()
        n_va = (pairs["role"] == "val").sum()
        print(
            f"[splits] {fold_id}: train_end={train_end:%Y-%m-%d} val_end={val_end:%Y-%m-%d}  "
            f"rows: train={n_tr:,} val={n_va:,}"
        )
        all_pairs.append(pairs)

    out = pd.concat(all_pairs, ignore_index=True)
    out.to_parquet(SPLITS_PARQ, index=False)

    # compact summary by fold × horizon × role
    summary = (
        out.groupby(["fold_id", "horizon", "role"])
        .size()
        .rename("rows")
        .reset_index()
        .sort_values(["fold_id", "horizon", "role"])
    )
    summary.to_csv(SPLITS_SUMMARY, index=False)

    print(f"[splits] saved {SPLITS_PARQ} ({len(out):,} rows)")
    print(f"[splits] summary → {SPLITS_SUMMARY}")


if __name__ == "__main__":
    main()
