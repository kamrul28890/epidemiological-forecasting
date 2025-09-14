"""Diagnostics for a forecasts file (val only):
- MAE by geo_id → artifacts/tables/mae_by_geo_<tag>.csv
- Error vs magnitude (y_true deciles) → artifacts/tables/error_by_quantile_<tag>.csv
- Calibration table (pred deciles) → artifacts/tables/calibration_<tag>.csv
- Optional PNG plots for quick figures.

Usage:
  python scripts/diagnostics.py --forecast artifacts/forecasts/gbm_F2.parquet --tag gbm_F2
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from dengue.utils.io import ensure_dir, load_yaml, resolve_paths  # type: ignore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecast", required=True)
    ap.add_argument("--tag", required=True)  # e.g., gbm_F2
    args = ap.parse_args()

    cfg = load_yaml("configs/data.yaml")
    paths = resolve_paths(cfg)

    df = pd.read_parquet(args.forecast)
    df = df[df["role"] == "val"].copy()

    # --- MAE by geo ---
    e = df["y_true"] - df["y_pred"]
    mae_geo = df.assign(err=e.abs()).groupby("geo_id")["err"].mean().sort_values()
    out1 = paths["tables"] / f"mae_by_geo_{args.tag}.csv"
    mae_geo.to_csv(out1, header=True)

    # --- error vs magnitude (y_true deciles) ---
    q = pd.qcut(df["y_true"], q=10, duplicates="drop")
    err_q = (
        df.assign(q=q, ae=e.abs())
        .groupby("q")["ae"]
        .agg(["count", "mean"])
        .rename(columns={"mean": "MAE"})
    )
    out2 = paths["tables"] / f"error_by_quantile_{args.tag}.csv"
    err_q.to_csv(out2)

    # --- calibration (pred deciles) ---
    qp = pd.qcut(df["y_pred"], q=10, duplicates="drop")
    cal = df.assign(q=qp).groupby("q")[["y_true", "y_pred"]].mean()
    out3 = paths["tables"] / f"calibration_{args.tag}.csv"
    cal.to_csv(out3)

    # Optional simple plots
    ensure_dir(paths["figures"])
    fig1 = paths["figures"] / f"calibration_{args.tag}.png"
    cal.plot()
    plt.title(f"Calibration (mean obs vs pred) — {args.tag}")
    plt.savefig(fig1, bbox_inches="tight")
    plt.close()

    print(f"[diag] MAE by geo → {out1}")
    print(f"[diag] Error vs magnitude → {out2}")
    print(f"[diag] Calibration → {out3}")
    print(f"[diag] Calibration plot → {fig1}")


if __name__ == "__main__":
    main()
