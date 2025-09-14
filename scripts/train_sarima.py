#!/usr/bin/env python3
"""
Seasonal-naive baseline per fold using only (geo_id, week_start_date -> cases).
- For each fold & geo and each origin week, forecast horizon h.
- y_hat(t+h) = y(t+h-52) if available, else mean of last 4 weeks (non-negative).
Writes:
  artifacts/forecasts/sarima_<FOLD>.parquet
  artifacts/tables/metrics_sarima_<FOLD>.csv
"""
from __future__ import annotations

import argparse
from typing import List

import numpy as np
import pandas as pd

from dengue.utils.io import ensure_dir, load_yaml, resolve_paths


def seasonal_naive_for(origin: pd.Timestamp, h: int, series: pd.Series) -> float:
    """Return y[target-52] if present, else mean of last 4 obs (>=0)."""
    target = origin + pd.Timedelta(weeks=h)
    ref = target - pd.Timedelta(weeks=52)
    if ref in series.index and pd.notna(series.loc[ref]):
        return float(max(0.0, series.loc[ref]))
    tail = series.dropna().iloc[-4:]
    if len(tail) == 0:
        return 0.0
    return float(max(0.0, tail.mean()))


def metrics_frame(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    def _rmse(a: pd.DataFrame) -> float:
        return float(np.sqrt(np.mean((a["y_true"] - a["y_pred"]) ** 2)))

    def _mae(a: pd.DataFrame) -> float:
        return float(np.mean(np.abs(a["y_true"] - a["y_pred"])))

    rows = []
    for role, g in df.groupby("role", dropna=False):
        rows.append(
            dict(
                source=source_name,
                role=role,
                horizon=np.nan,
                MAE=_mae(g),
                RMSE=_rmse(g),
            )
        )
    for (role, h), g in df.groupby(["role", "horizon"], dropna=False):
        rows.append(
            dict(
                source=source_name,
                role=role,
                horizon=int(h),
                MAE=_mae(g),
                RMSE=_rmse(g),
            )
        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True, choices=[f"F{i}" for i in range(1, 6)])
    args = ap.parse_args()

    cfg = load_yaml("configs/data.yaml")
    paths = resolve_paths(cfg)

    # data
    design = pd.read_parquet(paths["processed"] / "design_matrix.parquet")
    splits = pd.read_parquet(paths["processed"] / "splits.parquet")

    # tidy types
    for c in ("week_start_date", "target_week"):
        if c in design.columns:
            design[c] = pd.to_datetime(design[c], errors="coerce")
        if c in splits.columns:
            splits[c] = pd.to_datetime(splits[c], errors="coerce")

    # compact series per-geo
    design = design.loc[:, ~design.columns.duplicated()].copy()
    design = (
        design[["geo_id", "week_start_date", "cases"]]
        .drop_duplicates(subset=["geo_id", "week_start_date"])
        .sort_values(["geo_id", "week_start_date"])
        .reset_index(drop=True)
    )

    fold_col = "fold_id" if "fold_id" in splits.columns else "fold"
    F = splits[splits[fold_col] == args.fold].copy()
    F = F[["geo_id", "week_start_date", "target_week", "horizon", "role", fold_col]]
    F = F.sort_values(["geo_id", "week_start_date", "horizon"]).reset_index(drop=True)

    # lookup for y_true at target
    y_lookup = design.rename(columns={"week_start_date": "target_week"}).set_index(
        ["geo_id", "target_week"]
    )["cases"]

    out_rows: List[pd.DataFrame] = []
    for geo, gF in F.groupby("geo_id"):
        g = design[design["geo_id"] == geo].copy()
        s = (
            g.set_index("week_start_date")["cases"]
            .asfreq("W-MON")  # align to weekly starts
            .sort_index()
            .clip(lower=0)
        )
        preds = []
        for _, r in gF.iterrows():
            origin = pd.Timestamp(r["week_start_date"])
            h = int(r["horizon"])
            # use data up to origin (inclusive)
            s_train = s.loc[:origin]
            preds.append(seasonal_naive_for(origin, h, s_train))

        gF = gF.copy()
        gF["y_pred"] = preds
        gF["y_true"] = gF.apply(
            lambda r: float(
                y_lookup.get((r["geo_id"], pd.Timestamp(r["target_week"])), np.nan)
            ),
            axis=1,
        )
        out_rows.append(gF)

    if not out_rows:
        raise SystemExit(f"[sarima] No rows for fold {args.fold}")

    fore = pd.concat(out_rows, ignore_index=True)
    fore["model"] = "sarima"  # keep label so your summarizers/stack pick it up

    ensure_dir(paths["forecasts"])
    fpath = paths["forecasts"] / f"sarima_{args.fold}.parquet"
    fore.to_parquet(fpath, index=False)
    print(f"[sarima] forecasts → {fpath} ({len(fore):,} rows)")

    ensure_dir(paths["tables"])
    m = metrics_frame(fore, f"metrics_sarima_{args.fold}.csv")
    mpath = paths["tables"] / f"metrics_sarima_{args.fold}.csv"
    m.to_csv(mpath, index=False)
    print(f"[sarima] metrics   → {mpath}")


if __name__ == "__main__":
    main()
