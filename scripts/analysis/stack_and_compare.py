#!/usr/bin/env python
import argparse, os, glob
import numpy as np
import pandas as pd

MET_DIR = "reports/metrics"
os.makedirs(MET_DIR, exist_ok=True)

def _overall_from_metrics(csv_glob, tag):
    out=[]
    for p in sorted(glob.glob(csv_glob)):
        df = pd.read_csv(p)
        fold = os.path.splitext(os.path.basename(p))[0].split("_")[-1]
        m = df[df["role"]=="val"].copy()
        if m["horizon"].isna().any():
            row = m[m["horizon"].isna()].iloc[0][["MAE","RMSE"]]
        else:
            row = m.groupby("horizon")[["MAE","RMSE"]].mean().mean(axis=0)
        out.append({"fold":fold, "model":tag, "MAE":float(row["MAE"]), "RMSE":float(row["RMSE"])})
    return pd.DataFrame(out)

def _rmse_per_h(csv_path):
    m = pd.read_csv(csv_path)
    m = m[m["role"]=="val"].copy()
    if m["horizon"].isna().any():
        overall = float(m[m["horizon"].isna()]["RMSE"])
        return pd.Series({h:overall for h in range(1,13)})
    return m.dropna(subset=["horizon"]).set_index("horizon")["RMSE"]

def compare_only():
    glm = _overall_from_metrics("artifacts/tables/metrics_glm_F*.csv", "glm")
    gbm = _overall_from_metrics("artifacts/tables/metrics_gbm_F*.csv", "gbm")
    cmp = pd.concat([glm, gbm], ignore_index=True).sort_values(["fold","RMSE"])
    print("\n== GLM vs GBM (VAL overall) ==")
    print(cmp.to_string(index=False))
    cmp.to_csv(f"{MET_DIR}/metrics_compare_glm_gbm.csv", index=False)
    print(f"Wrote {MET_DIR}/metrics_compare_glm_gbm.csv")

def stack_simple():
    rows=[]
    for fold in ["F1","F2","F3","F4","F5"]:
        g_path = f"artifacts/forecasts/glm_{fold}.parquet"
        b_path = f"artifacts/forecasts/gbm_{fold}.parquet"
        if not (os.path.exists(g_path) and os.path.exists(b_path)):
            print(f"[stack] skip {fold}: missing glm/gbm forecasts")
            continue
        g = pd.read_parquet(g_path)
        b = pd.read_parquet(b_path)
        keys = ["geo_id","week_start_date","target_week","horizon","role"]
        df = g.merge(b, on=keys, suffixes=("_glm","_gbm"))

        rmse_glm = _rmse_per_h(f"artifacts/tables/metrics_glm_{fold}.csv")
        rmse_gbm = _rmse_per_h(f"artifacts/tables/metrics_gbm_{fold}.csv")

        def wavg(r):
            h = int(r["horizon"])
            rg = float(rmse_glm.get(h, rmse_glm.mean()))
            rb = float(rmse_gbm.get(h, rmse_gbm.mean()))
            wg = 1.0/max(rg, 1e-9); wb = 1.0/max(rb, 1e-9)
            s = wg+wb; wg/=s; wb/=s
            return wg*r["y_pred_glm"] + wb*r["y_pred_gbm"]

        df["y_pred"] = df.apply(wavg, axis=1)
        out = df[["geo_id","week_start_date","target_week","horizon","role","y_true_glm"]].rename(columns={"y_true_glm":"y_true"})
        out["y_pred"] = df["y_pred"]
        out_path = f"artifacts/forecasts/stack_simple_{fold}.parquet"
        out.to_parquet(out_path, index=False)
        print(f"[stack] wrote {out_path} ({len(out):,} rows)")

        def eval_frame(X, label):
            v = X[X["role"]=="val"]
            mae = float((v["y_true"]-v["y_pred"]).abs().mean())
            rmse = float(np.sqrt(((v["y_true"]-v["y_pred"])**2).mean()))
            return {"fold":fold, "label":label, "MAE":mae, "RMSE":rmse}

        rows.append(eval_frame(g.rename(columns={"y_pred":"y_pred"}), "glm"))
        rows.append(eval_frame(b.rename(columns={"y_pred":"y_pred"}), "gbm"))
        rows.append(eval_frame(out, "stack_simple"))

    if rows:
        res = pd.DataFrame(rows)
        piv = res.pivot(index="fold", columns="label", values=["MAE","RMSE"]).sort_index()
        print("\n== Fold-wise MAE/RMSE (VAL) ==")
        print(piv.round(3))
        piv.to_csv(f"{MET_DIR}/metrics_glm_gbm_stack_compare.csv")
        print(f"Wrote {MET_DIR}/metrics_glm_gbm_stack_compare.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare-only", action="store_true")
    ap.add_argument("--stack", action="store_true")
    args = ap.parse_args()
    if args.compare_only:
        compare_only()
    elif args.stack:
        stack_simple()
    else:
        compare_only(); stack_simple()

