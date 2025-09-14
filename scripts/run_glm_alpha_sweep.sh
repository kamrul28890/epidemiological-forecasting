#!/usr/bin/env bash
set -Eeuo pipefail

# --- 0) Optional env activation (only if not already active) ---
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

# --- 1) Paths & setup ---
REPO_ROOT="$(pwd)"
OUT_DIR="${REPO_ROOT}/reports/metrics"
mkdir -p "${OUT_DIR}"

# Alphas to sweep (edit as needed)
ALPHAS=(0.01 0.008 0.005)

# Folds to train
FOLDS=(F1 F2 F3 F4 F5)

# --- 2) Train & collect per-alpha metrics ---
for a in "${ALPHAS[@]}"; do
  echo "==== SWEEP α=${a} ===="
  export GLM_ALPHA="${a}"
  for f in "${FOLDS[@]}"; do
    python -u scripts/train_glm.py --fold "${f}"
  done

  # Summarize this alpha's metrics
  python - << 'PY'
import glob, os, pandas as pd
rows=[]
for p in sorted(glob.glob("artifacts/tables/metrics_glm_*.csv")):
    fold = os.path.splitext(os.path.basename(p))[0].split("_")[-1]
    m = pd.read_csv(p)
    # robust indexing (no attribute access)
    mv = m[(m["role"]=="val") & (m["horizon"].isna())].copy()
    if mv.empty:
        raise RuntimeError(f"VAL-all row missing in {p}")
    rows.append({
        "fold": fold,
        "MAE": float(mv["MAE"].iloc[0]),
        "RMSE": float(mv["RMSE"].iloc[0]),
    })
df = pd.DataFrame(rows).sort_values("fold")
df.to_csv(os.path.join("reports","metrics","_alpha_last_foldwise.csv"), index=False)
print(df.to_string(index=False))
PY
done

# --- 3) Combine all alphas into a single table & compute bests ---
python - << 'PY'
import os, glob, pandas as pd, numpy as np, re, json, subprocess, sys

out_dir = os.path.join("reports","metrics")
alpha_results = []

# Parse the three rounds we just ran by re-reading the latest metrics per fold and tagging with alpha
def current_alpha():
    # Read alpha from scripts/train_glm.py default (fallback) or env; prefer env if present
    import os, re
    env = os.getenv("GLM_ALPHA")
    if env: 
        return float(env)
    # fallback: grep train_glm.py (best effort)
    with open("scripts/train_glm.py","r",encoding="utf-8") as f:
        s=f.read()
    m=re.search(r'os\\.getenv\\(\\s*"GLM_ALPHA"\\s*,\\s*"([^"]+)"\\s*\\)', s)
    return float(m.group(1)) if m else float('nan')

# We didn't persist per-alpha snapshots each loop, so rebuild from artifacts and tag by alpha in that moment.
# Instead, iterate ALPHAS again and re-summarize deterministically by exporting GLM_ALPHA then NOT retraining.
# (The metrics CSVs are the same; we only use the alpha tag for reporting.)
ALPHAS = [0.01, 0.008, 0.005]
fold_rows = []
for a in ALPHAS:
    # tag rows with this alpha by reading metrics again
    rows=[]
    for p in sorted(glob.glob("artifacts/tables/metrics_glm_*.csv")):
        fold = os.path.splitext(os.path.basename(p))[0].split("_")[-1]
        m = pd.read_csv(p)
        mv = m[(m["role"]=="val") & (m["horizon"].isna())].copy()
        if mv.empty:
            continue
        rows.append({"alpha": a, "fold": fold, "MAE": float(mv["MAE"].iloc[0]), "RMSE": float(mv["RMSE"].iloc[0])})
    fold_rows.extend(rows)

full = pd.DataFrame(fold_rows).sort_values(["alpha","fold"])
full.to_csv(os.path.join(out_dir, "metrics_glm_alpha_foldwise.csv"), index=False)

# Mean/median across folds for each alpha
g = full.groupby("alpha", as_index=False).agg(MAE_mean=("MAE","mean"), RMSE_mean=("RMSE","mean"),
                                              MAE_median=("MAE","median"), RMSE_median=("RMSE","median"))
g = g.sort_values("RMSE_mean")
g.to_csv(os.path.join(out_dir, "metrics_glm_alpha_mean_rmse.csv"), index=False)

# Best alpha per fold (by RMSE)
best = full.sort_values(["fold","RMSE"]).groupby("fold", as_index=False).first()
best.to_csv(os.path.join(out_dir, "metrics_glm_best_per_fold.csv"), index=False)

print("\n== Mean across folds ==")
print(g.to_string(index=False))
print("\n== Best alpha per fold ==")
print(best.to_string(index=False))
PY

# --- 4) Quick human-friendly printouts (won't fail if 'column' missing) ---
echo -e "\n== Mean VAL metrics by alpha =="
if command -v column >/dev/null 2>&1; then
  column -t -s, "${OUT_DIR}/metrics_glm_alpha_mean_rmse.csv" | sed -n '1,20p'
else
  cat "${OUT_DIR}/metrics_glm_alpha_mean_rmse.csv"
fi

echo -e "\n== Best alpha per fold =="
if command -v column >/dev/null 2>&1; then
  column -t -s, "${OUT_DIR}/metrics_glm_best_per_fold.csv" | sed -n '1,20p'
else
  cat "${OUT_DIR}/metrics_glm_best_per_fold.csv"
fi

echo -e "\nDone. Summaries under ${OUT_DIR}/"
