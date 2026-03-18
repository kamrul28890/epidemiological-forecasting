#!/usr/bin/env bash
set -Eeuo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  if command -v micromamba >/dev/null 2>&1; then
    eval "$(micromamba shell hook -s bash)"
    micromamba activate dengue-forecast || true
  elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate dengue-forecast || true
  else
    echo "[WARN] No micromamba/conda found; assuming Python env is already active."
  fi
fi

REPO_ROOT="$(pwd)"
OUT_DIR="${REPO_ROOT}/reports/metrics"
mkdir -p "${OUT_DIR}"

ALPHAS=(0.01 0.008 0.005)
FOLDS=(F1 F2 F3 F4 F5)

for a in "${ALPHAS[@]}"; do
  echo "==== SWEEP alpha=${a} ===="
  export GLM_ALPHA="${a}"
  for f in "${FOLDS[@]}"; do
    python -u -m scripts.training.train_glm --fold "${f}"
  done

  python - << 'PY'
import glob
import os

import numpy as np
import pandas as pd


def summarize_from_metrics():
    rows = []
    for path in sorted(glob.glob("artifacts/tables/metrics_glm_*.csv")):
        fold = os.path.splitext(os.path.basename(path))[0].split("_")[-1]
        metrics = pd.read_csv(path)
        if {"role", "horizon", "MAE", "RMSE"}.issubset(metrics.columns):
            val_rows = metrics[(metrics["role"] == "val") & (metrics["horizon"].isna())]
            if not val_rows.empty:
                rows.append(
                    {
                        "fold": fold,
                        "MAE": float(val_rows["MAE"].iloc[0]),
                        "RMSE": float(val_rows["RMSE"].iloc[0]),
                    }
                )
    return pd.DataFrame(rows)


def summarize_from_forecasts():
    rows = []
    for path in sorted(glob.glob("artifacts/forecasts/glm_*.parquet")):
        fold = os.path.splitext(os.path.basename(path))[0].split("_")[-1]
        forecast = pd.read_parquet(path)
        if {"role", "y_true", "y_pred"}.issubset(forecast.columns):
            val_rows = forecast[forecast["role"] == "val"].copy()
            if not val_rows.empty:
                mae = (val_rows["y_true"] - val_rows["y_pred"]).abs().mean()
                rmse = np.sqrt(((val_rows["y_true"] - val_rows["y_pred"]) ** 2).mean())
                rows.append({"fold": fold, "MAE": float(mae), "RMSE": float(rmse)})
    return pd.DataFrame(rows)


summary = summarize_from_metrics()
if summary.empty:
    print("[INFO] metrics CSVs missing expected columns; summarizing from forecasts parquet...")
    summary = summarize_from_forecasts()

summary = summary.sort_values("fold")
summary.to_csv(os.path.join("reports", "metrics", "_alpha_last_foldwise.csv"), index=False)
print(summary.to_string(index=False))
PY
done

python - << 'PY'
import glob
import os

import numpy as np
import pandas as pd

out_dir = os.path.join("reports", "metrics")
os.makedirs(out_dir, exist_ok=True)

alphas = [0.01, 0.008, 0.005]
all_rows = []
for alpha in alphas:
    rows = []
    for path in sorted(glob.glob("artifacts/forecasts/glm_*.parquet")):
        fold = os.path.splitext(os.path.basename(path))[0].split("_")[-1]
        forecast = pd.read_parquet(path)
        if {"role", "y_true", "y_pred"}.issubset(forecast.columns):
            val_rows = forecast[forecast["role"] == "val"].copy()
            if val_rows.empty:
                continue
            mae = (val_rows["y_true"] - val_rows["y_pred"]).abs().mean()
            rmse = np.sqrt(((val_rows["y_true"] - val_rows["y_pred"]) ** 2).mean())
            rows.append({"alpha": alpha, "fold": fold, "MAE": float(mae), "RMSE": float(rmse)})
    all_rows.extend(rows)

full = pd.DataFrame(all_rows).sort_values(["alpha", "fold"])
full.to_csv(os.path.join(out_dir, "metrics_glm_alpha_foldwise.csv"), index=False)

means = (
    full.groupby("alpha", as_index=False)
    .agg(
        MAE_mean=("MAE", "mean"),
        RMSE_mean=("RMSE", "mean"),
        MAE_median=("MAE", "median"),
        RMSE_median=("RMSE", "median"),
    )
    .sort_values("RMSE_mean")
)
means.to_csv(os.path.join(out_dir, "metrics_glm_alpha_mean_rmse.csv"), index=False)

best = full.sort_values(["fold", "RMSE"]).groupby("fold", as_index=False).first()
best.to_csv(os.path.join(out_dir, "metrics_glm_best_per_fold.csv"), index=False)

print("\n== Mean across folds ==")
print(means.to_string(index=False))
print("\n== Best alpha per fold ==")
print(best.to_string(index=False))
PY

echo
echo "Summaries written to: reports/metrics/"
