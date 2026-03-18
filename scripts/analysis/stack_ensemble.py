#!/usr/bin/env python
import glob, os, re
import numpy as np
import pandas as pd

os.makedirs("artifacts/forecasts", exist_ok=True)
os.makedirs("artifacts/tables", exist_ok=True)

def load_one(model):
    out=[]
    for p in sorted(glob.glob(f"artifacts/forecasts/{model}_F*.parquet")):
        df = pd.read_parquet(p)
        m = re.search(r"_(F\d)\.parquet$", p)
        fold = m.group(1) if m else "F?"
        df = df.copy()
        df["fold_id"] = df.get("fold_id", fold)
        df["model"] = model
        out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

glm = load_one("glm")
gbm = load_one("gbm")
if glm.empty or gbm.empty:
    raise SystemExit("Need both glm_*.parquet and gbm_*.parquet under artifacts/forecasts")

# merge level-0 predictions
keys = ["geo_id","week_start_date","target_week","horizon","role","fold_id"]
L = pd.merge(
    gbm[keys+["y_true","y_pred"]].rename(columns={"y_pred":"y_gbm"}),
    glm[keys+["y_pred"]].rename(columns={"y_pred":"y_glm"}),
    on=keys, how="inner"
)

def rmse(y, p): return float(np.sqrt(np.mean((y - p)**2)))

rows_metrics=[]
out_parts=[]

for fold, gfold in L.groupby("fold_id"):
    for h, gh in gfold.groupby("horizon"):
        # grid-search weight on VAL only
        val = gh[gh["role"]=="val"]
        if len(val)==0:
            best_w=0.5
        else:
            ys = val["y_true"].to_numpy(float)
            dg = val["y_gbm"].to_numpy(float) - val["y_glm"].to_numpy(float)
            # closed-form w* may go outside [0,1]; clamp with grid as fallback
            num = float(np.dot(ys - val["y_glm"].to_numpy(float), dg))
            den = float(np.dot(dg, dg)) if float(np.dot(dg, dg))>0 else 0.0
            w_cf = num/den if den>0 else 0.5
            candidates = np.unique(np.clip(np.concatenate([np.linspace(0,1,51), [w_cf]]), 0, 1))
            best_w, best_rmse = 0.5, float("inf")
            for w in candidates:
                p = w*val["y_gbm"] + (1-w)*val["y_glm"]
                r = rmse(val["y_true"], p)
                if r < best_rmse:
                    best_rmse, best_w = r, float(w)

        # apply to both TRAIN+VAL rows for reporting
        gh = gh.copy()
        gh["y_pred"] = best_w*gh["y_gbm"] + (1-best_w)*gh["y_glm"]
        gh["model"] = "ensemble"
        out_parts.append(gh[keys+["y_true","y_pred","model"]])

        # metrics (overall per role + by role/horizon)
        for role, gr in gh.groupby("role"):
            rows_metrics.append({
                "source": f"metrics_ensemble_{fold}.csv",
                "role": role,
                "horizon": np.nan,
                "MAE": float((gr["y_true"]-gh.loc[gr.index,"y_pred"]).abs().mean()),
                "RMSE": rmse(gr["y_true"], gh.loc[gr.index,"y_pred"]),
                "fold": fold,
                "h": int(h),
                "w_gbm": best_w,
            })

# write per-fold forecasts + metrics
ens = pd.concat(out_parts, ignore_index=True)
for fold, gfold in ens.groupby("fold_id"):
    fpath = f"artifacts/forecasts/ensemble_{fold}.parquet"
    gfold.to_parquet(fpath, index=False)
    print(f"[ensemble] forecasts → {fpath} ({len(gfold):,} rows)")

metr = pd.DataFrame(rows_metrics)
m_agg = []
# aggregate rows into classic table: per role overall + per role,horizon
for (fold, role), grp in metr.groupby(["fold","role"]):
    m_agg.append(dict(source=f"metrics_ensemble_{fold}.csv", role=role, horizon=np.nan,
                      MAE=float(grp["MAE"].mean()), RMSE=float(grp["RMSE"].mean())))
for (fold, role, h), grp in metr.groupby(["fold","role","h"]):
    m_agg.append(dict(source=f"metrics_ensemble_{fold}.csv", role=role, horizon=int(h),
                      MAE=float(grp["MAE"].mean()), RMSE=float(grp["RMSE"].mean())))
m_agg = pd.DataFrame(m_agg).sort_values(["role","horizon"]).reset_index(drop=True)
for fold in sorted(ens["fold_id"].unique()):
    out = m_agg[m_agg["source"]==f"metrics_ensemble_{fold}.csv"].drop(columns=["source"])
    out.to_csv(f"artifacts/tables/metrics_ensemble_{fold}.csv", index=False)
    print(f"[ensemble] metrics → artifacts/tables/metrics_ensemble_{fold}.csv")
