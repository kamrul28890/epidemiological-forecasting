import glob
import os

import numpy as np
import pandas as pd


def fold_from_path(p: str) -> str:
    b = os.path.splitext(os.path.basename(p))[0]  # metrics_glm_F1
    return b.split("_")[-1]  # F1


rows = []

for csv in sorted(glob.glob("artifacts/tables/metrics_glm_F*.csv")):
    fold = fold_from_path(csv)
    df = pd.read_csv(csv)
    mae = rmse = None

    # Preferred path: use the "val, horizon=NaN" summary row
    if {"role", "horizon", "MAE", "RMSE"}.issubset(df.columns):
        sub = df[df["role"].astype(str) == "val"]
        overall = sub[sub["horizon"].isna()]
        if not overall.empty:
            mae = float(overall["MAE"].iloc[0])
            rmse = float(overall["RMSE"].iloc[0])

    # Fallback: compute from forecasts parquet if needed
    if mae is None or rmse is None:
        pq = f"artifacts/forecasts/glm_{fold}.parquet"
        if not os.path.exists(pq):
            print(f"[WARN] Missing parquet for {fold}: {pq}")
            continue
        fx = pd.read_parquet(pq)
        v = fx[fx["role"] == "val"].copy()
        err = v["y_true"] - v["y_pred"]
        mae = float(err.abs().mean())
        rmse = float(np.sqrt((err**2).mean()))

    rows.append({"fold": fold, "MAE": mae, "RMSE": rmse})

res = pd.DataFrame(rows).sort_values("fold")
print("\nFold-wise (default alpha):")
print(res.to_string(index=False))

print("\nMean over folds:")
print(res[["MAE", "RMSE"]].mean().to_frame("mean").T)

os.makedirs("reports/metrics", exist_ok=True)
res.to_csv("reports/metrics/metrics_glm_default_alpha_per_fold.csv", index=False)
res[["MAE", "RMSE"]].mean().to_frame("mean").T.to_csv(
    "reports/metrics/metrics_glm_default_alpha_mean.csv", index=False
)
