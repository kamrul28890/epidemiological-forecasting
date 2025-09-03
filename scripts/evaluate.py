"""
Aggregate baseline metrics and emit simple summaries.
Reads artifacts/tables/metrics_*.csv and prints by-role toplines.
"""

from pathlib import Path

import pandas as pd

TAB_DIR = Path("artifacts/tables")


def main():
    files = sorted(TAB_DIR.glob("metrics_*.csv"))
    if not files:
        print("[eval] No metrics files found.")
        return

    frames = [pd.read_csv(f).assign(source=f.name) for f in files]
    df = pd.concat(frames, ignore_index=True)

    # Overall by (source, role)
    overall = df.groupby(["source", "role"])[["MAE", "RMSE"]].mean().round(3)
    print("\n[eval] Overall metrics by source & role:")
    print(overall)

    # By horizon (val only)
    by_h = (
        df[df["role"] == "val"]
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
