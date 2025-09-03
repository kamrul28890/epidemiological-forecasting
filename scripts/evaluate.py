#!/usr/bin/env python
"""
Aggregate metrics and emit summaries.
Reads artifacts/tables/metrics_*.csv and prints by-role toplines.
Handles files that saved 'index' instead of explicit 'role'/'horizon'.
"""
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd

TAB_DIR = Path("artifacts/tables")


def coerce_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure columns: role (str in {train,val}), horizon (int or NaN), MAE, RMSE, source."""
    out = df.copy()

    # If role is missing but 'index' exists, parse it
    if "role" not in out.columns and "index" in out.columns:
        roles, horizons = [], []
        for s in out["index"].astype(str):
            s = s.strip()
            if s in {"train", "val"}:
                roles.append(s)
                horizons.append(np.nan)
                continue
            # Try to parse tuple-like strings: "('val', 7)"
            try:
                t = ast.literal_eval(s)
                if isinstance(t, tuple) and len(t) >= 2:
                    roles.append(str(t[0]))
                    horizons.append(int(t[1]))
                    continue
            except Exception:
                pass
            # Fallback regex
            m = re.search(r"(train|val).*(\d+)", s)
            if m:
                roles.append(m.group(1))
                horizons.append(int(m.group(2)))
            else:
                roles.append(None)
                horizons.append(np.nan)

        out = out.drop(columns=["index"])
        out["role"] = roles
        out["horizon"] = horizons

    if "horizon" not in out.columns:
        out["horizon"] = np.nan

    # Keep only needed columns if present
    keep = [c for c in ["source", "role", "horizon", "MAE", "RMSE"] if c in out.columns]
    return out[keep]


def main():
    files = sorted(
        p for p in TAB_DIR.glob("metrics_*.csv") if p.name != "metrics_summary.csv"
    )
    if not files:
        print("[eval] No metrics files found.")
        return

    print("[eval] found:", [p.name for p in files])

    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["source"] = f.name  # ensure source column exists/is consistent
        frames.append(coerce_metrics(df))

    df = pd.concat(frames, ignore_index=True)

    # Only keep rows with a valid role value we understand
    df = df[df["role"].isin(["train", "val"])].copy()

    # Overall by (source, role)
    overall = df.groupby(["source", "role"])[["MAE", "RMSE"]].mean().round(3)
    print("\n[eval] Overall metrics by source & role:")
    print(overall)

    # By horizon (val only)
    by_h = (
        df[df["role"] == "val"]
        .dropna(subset=["horizon"])
        .assign(horizon=lambda x: x["horizon"].astype(int))
        .groupby(["source", "horizon"])[["MAE", "RMSE"]]
        .mean()
        .round(3)
    )
    print("\n[eval] Validation metrics by horizon:")
    print(by_h)

    out_csv = TAB_DIR / "metrics_summary.csv"
    (
        df.groupby(["source", "role", "horizon"])[["MAE", "RMSE"]]
        .mean()
        .round(3)
        .reset_index()
        .to_csv(out_csv, index=False)
    )
    print(f"\n[eval] Wrote summary → {out_csv}")


if __name__ == "__main__":
    main()
