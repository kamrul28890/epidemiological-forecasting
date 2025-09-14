import glob
import os

import pandas as pd

rows = []
for path in sorted(glob.glob("artifacts/tables/metrics_gbm_F*.csv")):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] cannot read {path}: {e}")
        continue
    fold = os.path.splitext(os.path.basename(path))[0].split("_")[-1]
    mae = rmse = None
    if {"role", "horizon", "MAE", "RMSE"}.issubset(df.columns):
        m = df[(df["role"] == "val") & (df["horizon"].isna())]
        if not m.empty:
            mae = float(m["MAE"].iloc[0])
            rmse = float(m["RMSE"].iloc[0])
        else:
            mv = df[df["role"] == "val"]
            if not mv.empty:
                g = mv.agg({"MAE": "mean", "RMSE": "mean"})
                mae = float(g["MAE"])
                rmse = float(g["RMSE"])
    if mae is None:
        g = df.agg({"MAE": "mean", "RMSE": "mean"})
        mae = float(g["MAE"])
        rmse = float(g["RMSE"])
    rows.append({"fold": fold, "MAE": mae, "RMSE": rmse})

out = pd.DataFrame(rows).sort_values("fold")
print("\nFold-wise (GBM):")
print(out.to_string(index=False))

mean = out[["MAE", "RMSE"]].mean().to_frame().T
mean.index = ["mean"]
print("\nMean over folds:")
print(mean.to_string())

os.makedirs("reports/metrics", exist_ok=True)
out.to_csv("reports/metrics/metrics_gbm_default_per_fold.csv", index=False)
mean.to_csv("reports/metrics/metrics_gbm_default_mean.csv")
