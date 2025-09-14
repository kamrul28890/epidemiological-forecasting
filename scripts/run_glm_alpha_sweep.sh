#!/usr/bin/env bash
set -Eeuo pipefail

# --- Optional env activation (skips if already active) ---
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  if command -v micromamba >/dev/null 2>&1; then
    eval "$(micromamba shell hook -s bash)"
    micromamba activate dengue-forecast || true
  elif command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
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
  echo "==== SWEEP α=${a} ===="
  export GLM_ALPHA="${a}"
  for f in "${FOLDS[@]}"; do
    python -u scripts/train_glm.py --fold "${f}"
  done

  # Summarize (robust to missing 'role' in metrics CSV; falls back to forecasts parquet)
  python - << 'PY'
import os, glob, pandas as pd, numpy as np

def summarize_from_metrics():
    rows=[]
    for p in sorted(glob.glob("artifacts/tables/metrics_glm_*.csv")):
        fold = os.path.splitext(os.path.basename(p))[0].split("_")[-1]
        m = pd.read_csv(p)
        if {"role","horizon","MAE","RMSE"}.issubset(m.columns):
            mv = m[(m["role"]=="val") & (m["horizon"].isna())]
            if not mv.empty:
                rows.append({"fold":fold,"MAE":float(mv["MAE"].iloc[0]),"RMSE":float(mv["RMSE"].iloc[0])})
    return pd.DataFrame(rows)

def summarize_from_forecasts():
    rows=[]
    for p in sorted(glob.glob("artifacts/forecasts/glm_*.parquet")):
        fold = os.path.splitext(os.path.basename(p))[0].split("_")[-1]
        df = pd.read_parquet(p)
        if {"role","y_true","y_pred"}.issubset(df.columns):
            v = df[df["role"]=="val"].copy()
            if not v.empty:
                mae = (v["y_true"]-v["y_pred"]).abs().mean()
                rmse = np.sqrt(((v["y_true"]-v["y_pred"])**2).mean())
                rows.append({"fold":fold,"MAE":float(mae),"RMSE":float(rmse)})
    return pd.DataFrame(rows)

df = summarize_from_metrics()
if df.empty:
    print("[INFO] metrics CSVs missing expected columns; summarizing from forecasts parquet…")
    df = summarize_from_forecasts()

df = df.sort_values("fold")
df.to_csv(os.path.join("reports","metrics","_alpha_last_foldwise.csv"), index=False)
print(df.to_string(index=False))
PY
done

# Combine results across alphas (uses latest artifacts; tags rows by alpha)
python - << 'PY'
import os, glob, pandas as pd, numpy as np

out_dir = os.path.join("reports","metrics")
os.makedirs(out_dir, exist_ok=True)

# Build fold-wise table for each alpha by recomputing from forecasts (most reliable)
ALPHAS = [0.01, 0.008, 0.005]
all_rows=[]
for a in ALPHAS:
    rows=[]
    for p in sorted(glob.glob("artifacts/forecasts/glm_*.parquet")):
        fold = os.path.splitext(os.path.basename(p))[0].split("_")[-1]
        df = pd.read_parquet(p)
        if {"role","y_true","y_pred"}.issubset(df.columns):
            v = df[df["role"]=="val"].copy()
            if v.empty: 
                continue
            mae = (v["y_true"]-v["y_pred"]).abs().mean()
            rmse = np.sqrt(((v["y_true"]-v["y_pred"])**2).mean())
            rows.append({"alpha":a,"fold":fold,"MAE":float(mae),"RMSE":float(rmse)})
    all_rows.extend(rows)

full = pd.DataFrame(all_rows).sort_values(["alpha","fold"])
full.to_csv(os.path.join(out_dir,"metrics_glm_alpha_foldwise.csv"), index=False)

g = full.groupby("alpha", as_index=False).agg(MAE_mean=("MAE","mean"),
                                              RMSE_mean=("RMSE","mean"),
                                              MAE_median=("MAE","median"),
                                              RMSE_median=("RMSE","median")).sort_values("RMSE_mean")
g.to_csv(os.path.join(out_dir,"metrics_glm_alpha_mean_rmse.csv"), index=False)

best = full.sort_values(["fold","RMSE"]).groupby("fold", as_index=False).first()
best.to_csv(os.path.join(out_dir,"metrics_glm_best_per_fold.csv"), index=False)

print("\n== Mean across folds ==")
print(g.to_string(index=False))
print("\n== Best alpha per fold ==")
print(best.to_string(index=False))
PY

echo
echo "Summaries written to: reports/metrics/"
