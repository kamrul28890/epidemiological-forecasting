"""Seasonal-naive baseline: y_pred(geo, t) = cases(geo, t-52w).
Falls back to persistence if the 52w-back value is missing.
Writes:
  artifacts/forecasts/seasonal_naive_<FOLD>.parquet
  artifacts/tables/metrics_seasonal_naive_<FOLD>.csv
"""

import argparse

import numpy as np
import pandas as pd

from dengue.utils.io import ensure_dir, load_yaml, resolve_paths  # type: ignore


def pair_xy(design: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    # map (geo, date) -> cases
    key = design[["geo_id", "week_start_date", "cases"]].copy()
    key["week_start_date"] = pd.to_datetime(key["week_start_date"])
    key = key.set_index(["geo_id", "week_start_date"])["cases"]

    def get_cases(geo, dt):
        try:
            return float(key.loc[(geo, pd.Timestamp(dt))])
        except KeyError:
            return np.nan

    out = splits.copy()
    out["y_true"] = out.apply(
        lambda r: get_cases(r["geo_id"], r["target_week"]), axis=1
    )
    # seasonal naive: t-364 days -> normalize to Sunday-week start to match index
    out["y_pred"] = out.apply(
        lambda r: get_cases(
            r["geo_id"],
            (pd.Timestamp(r["target_week"]) - pd.Timedelta(days=364))
            .to_period("W-SUN")
            .start_time,
        ),
        axis=1,
    )
    # fallback: persistence (last observed)
    mask = out["y_pred"].isna()
    out.loc[mask, "y_pred"] = out.loc[mask].apply(
        lambda r: get_cases(r["geo_id"], r["week_start_date"]), axis=1
    )
    return out


def metrics(df: pd.DataFrame) -> pd.DataFrame:
    def agg(g):
        e = g["y_true"] - g["y_pred"]
        return pd.Series({"MAE": e.abs().mean(), "RMSE": np.sqrt((e**2).mean())})

    # Opt into future behavior to silence pandas warning
    m1 = df.groupby(["role"]).apply(agg, include_groups=False)
    m2 = df.groupby(["role", "horizon"]).apply(agg, include_groups=False)
    m = pd.concat([m1, m2]).reset_index()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True, help="Fold name, e.g., F1…F5")
    args = ap.parse_args()

    cfg = load_yaml("configs/data.yaml")
    paths = resolve_paths(cfg)

    design = pd.read_parquet(paths["processed"] / "design_matrix.parquet")
    splits = pd.read_parquet(paths["processed"] / "splits.parquet")

    # Validate the requested fold against what's actually present
    available = sorted(map(str, splits["fold_id"].unique()))
    if args.fold not in available:
        raise SystemExit(
            f"[ERR] fold={args.fold} not in splits. Available: {available}"
        )

    splits = splits[splits["fold_id"] == args.fold].copy()

    XY = pair_xy(design, splits)
    XY["model"] = "seasonal_naive"

    out_fore = paths["forecasts"] / f"seasonal_naive_{args.fold}.parquet"
    ensure_dir(out_fore.parent)
    XY.to_parquet(out_fore, index=False)

    m = metrics(XY)
    m.insert(0, "source", f"metrics_seasonal_naive_{args.fold}.csv")

    out_tbl = paths["tables"] / f"metrics_seasonal_naive_{args.fold}.csv"
    ensure_dir(out_tbl.parent)
    m.to_csv(out_tbl, index=False)

    print(f"[seasonal-naive] forecasts → {out_fore} ({len(XY):,} rows)")
    print(f"[seasonal-naive] metrics   → {out_tbl}")


if __name__ == "__main__":
    main()
