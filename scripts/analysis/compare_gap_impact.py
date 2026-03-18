from __future__ import annotations
import argparse, glob, os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-a", required=True)
    ap.add_argument("--metrics-b", required=True)
    ap.add_argument("--pred-a", required=True)
    ap.add_argument("--pred-b", required=True)
    ap.add_argument("--truth-cases", required=True)  # counts CSV through Aug-23-2025
    ap.add_argument("--outdir", default="reports/metrics")
    ap.add_argument("--pre-start", default="2025-04-01")
    ap.add_argument("--pre-end",   default="2025-06-30")
    ap.add_argument("--gap-start", default="2025-07-01")
    ap.add_argument("--gap-end",   default="2025-08-23")
    return ap.parse_args()

def resolve_latest(p):
    if any(ch in p for ch in "*?[]"):
        m = glob.glob(p)
        if not m: raise FileNotFoundError(f"No files match: {p}")
        m.sort(key=lambda x: os.path.getmtime(x))
        return m[-1]
    if not os.path.exists(p): raise FileNotFoundError(p)
    return p

def to_monday(s):
    s = pd.to_datetime(s)
    return (s - pd.to_timedelta(s.dt.weekday, unit="D")).dt.normalize()

def melt_truth_counts(counts_csv: str) -> pd.DataFrame:
    df = pd.read_csv(counts_csv, parse_dates=["date"])
    cols = {c.lower(): c for c in df.columns}
    want = {
        "DHA_METRO": "dhaka_metro_ew_new_cases",
        "DHA_DIV_ex_DHA_METRO": "dhaka_div_out_metro_ew_new_cases",
        "BAR": "barishal_ew_new_cases",
        "CHA": "chattogram_ew_new_cases",
        "KHU": "khulna_ew_new_cases",
        "MYM": "mymensingh_ew_new_cases",
        "RAJ": "rajshahi_ew_new_cases",
        "RAN": "rangpur_ew_new_cases",
        "SYL": "sylhet_ew_new_cases",
        "COUNTRY_TOTAL": "country_wide_total_ew_new_cases",
    }
    rows = []
    for geo, col_l in want.items():
        if col_l not in cols:
            raise KeyError(f"Missing column for {geo}: like '{col_l}'")
        col = cols[col_l]
        tmp = df[["date", col]].rename(columns={"date":"target_week", col:"y_true"})
        tmp["geo_id"] = geo
        rows.append(tmp)
    truth = pd.concat(rows, ignore_index=True)
    truth["target_week"] = to_monday(truth["target_week"])
    return truth

def summarize_window(pred_path: str, truth: pd.DataFrame, start: str, end: str, label: str):
    pred = pd.read_csv(pred_path, parse_dates=["target_week"])
    # force Monday for predictions too (robust to any drift)
    pred["target_week"] = to_monday(pred["target_week"])
    mask = (pred["target_week"] >= pd.to_datetime(start)) & (pred["target_week"] <= pd.to_datetime(end))
    predw = pred.loc[mask].copy()

    m = predw.merge(truth, on=["geo_id","target_week"], how="inner")
    if m.empty:
        return pd.DataFrame(), pd.DataFrame()

    def agg(g):
        mae  = (g["y_true"] - g["y_pred"]).abs().mean()
        rmse = float(np.sqrt(((g["y_true"] - g["y_pred"])**2).mean()))
        return pd.Series({"MAE": mae, "RMSE": rmse})

    by_h  = m.groupby(["horizon"], dropna=True).apply(agg, include_groups=False).reset_index()
    by_h["window"] = label
    by_gH = m.groupby(["geo_id","horizon"], dropna=True).apply(agg, include_groups=False).reset_index()
    by_gH["window"] = label
    return by_h, by_gH

def main():
    a = parse_args()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)

    met_a = resolve_latest(a.metrics_a); met_b = resolve_latest(a.metrics_b)
    preda = resolve_latest(a.pred_a);    predb = resolve_latest(a.pred_b)
    truth = melt_truth_counts(a.truth_cases)

    A_pre, A_pre_geo = summarize_window(preda, truth, a.pre_start, a.pre_end, "A_pre")
    A_gap, A_gap_geo = summarize_window(preda, truth, a.gap_start, a.gap_end, "A_gap")
    B_pre, B_pre_geo = summarize_window(predb, truth, a.pre_start, a.pre_end, "B_pre")
    B_gap, B_gap_geo = summarize_window(predb, truth, a.gap_start, a.gap_end, "B_gap")

    by_h  = pd.concat([x for x in [A_pre, A_gap, B_pre, B_gap] if not x.empty], ignore_index=True)
    by_gH = pd.concat([x for x in [A_pre_geo, A_gap_geo, B_pre_geo, B_gap_geo] if not x.empty], ignore_index=True)

    def pivot_and_write(df, name):
        ts = datetime.utcnow().strftime('%Y-%m-%dT%H%M%SZ')
        out = Path(a.outdir) / f"{name}_{ts}.csv"
        if df.empty:
            pd.DataFrame().to_csv(out, index=False); return out
        piv = df.pivot_table(index=(["horizon"] if "geo_id" not in df.columns else ["geo_id","horizon"]),
                             columns="window", values=["MAE","RMSE"])
        piv.to_csv(out); return out

    p_h = pivot_and_write(by_h,  "gap_impact_by_horizon")
    p_g = pivot_and_write(by_gH, "gap_impact_by_geo_horizon")

    print("Resolved files:")
    print("  metrics A:", met_a)
    print("  metrics B:", met_b)
    print("  preds   A:", preda)
    print("  preds   B:", predb)
    print("Truth:", a.truth_cases)
    print("Wrote:", p_h, p_g)

if __name__ == "__main__":
    main()
