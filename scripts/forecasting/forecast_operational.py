from __future__ import annotations
import argparse, json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from datetime import datetime
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler


from dengue.forecasting.hierarchy import bottom_up_sum, with_residual

def rmse_numpy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))



DEFAULT_SLOW_PATTERNS = ["temp", "temperature", "rh", "humid", "soil_temp", "soil_moist", "dewpoint"]
FAST_PATTERNS = ["rain", "precip", "wind"]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="data/processed/design_matrix.parquet")
    ap.add_argument("--splits", default="data/processed/splits.parquet")
    ap.add_argument("--outdir", default="reports/forecasts")
    ap.add_argument("--metricsdir", default="reports/metrics")

    ap.add_argument("--mode", choices=["exoglite", "slow_locf"], required=True)
    ap.add_argument("--as-of-ceiling", default="2025-06-30")
    ap.add_argument("--locf-cap-weeks", type=int, default=8)

    ap.add_argument("--alpha", type=float, default=1.0, help="Ridge penalty (PoissonRegressor)")
    ap.add_argument("--standardize", action="store_true", help="Z-score features before GLM fit")

    ap.add_argument("--country-geo", default="COUNTRY_TOTAL")
    ap.add_argument("--dhaka-division", default="Dhaka_Division")
    ap.add_argument("--dhaka-metro", default="Dhaka_Metro")
    ap.add_argument("--apply-residual", action="store_true")
    return ap.parse_args()



def to_date(s) -> pd.Timestamp:
    return pd.to_datetime(s).normalize()

def detect_target_col(dm: pd.DataFrame) -> str:
    candidates = ["y", "cases", "count", "cases_total"]
    for c in candidates:
        if c in dm.columns:
            return c
    # Last resort: try something that looks like a count (integer nonnegative)
    numeric = [c for c in dm.columns if np.issubdtype(dm[c].dtype, np.number)]
    if "y" in numeric:
        return "y"
    raise AssertionError("Could not find target column. Expected one of: y, cases, count, cases_total.")

def select_feature_columns(dm: pd.DataFrame, y_col: str) -> List[str]:
    exclude = {"geo_id","week_start_date","role","fold_id","origin_week", y_col}
    return [c for c in dm.columns if c not in exclude and dm[c].dtype != "O"]

def drop_exogenous(cols: List[str]) -> List[str]:
    drop_keys = DEFAULT_SLOW_PATTERNS + FAST_PATTERNS + ["rain","wind","soil","temp","humid","dew","precip"]
    keep = []
    for c in cols:
        low = c.lower()
        if any(k in low for k in drop_keys):
            continue
        keep.append(c)
    return keep

def split_slow_vs_fast(cols: List[str]) -> Tuple[List[str], List[str]]:
    slow, fast = [], []
    for c in cols:
        low = c.lower()
        if any(k in low for k in DEFAULT_SLOW_PATTERNS):
            slow.append(c)
        elif any(k in low for k in FAST_PATTERNS):
            fast.append(c)
    return slow, fast

def locf_cap(df: pd.DataFrame, group_cols: List[str], fill_cols: List[str],
             time_col: str, ceiling_date: pd.Timestamp, max_weeks: int):
    """
    For rows after ceiling_date, carry forward last <= ceiling value up to max_weeks; beyond that set NaN.
    Leaves rows at/<= ceiling untouched. Works per group (e.g., geo_id).
    """
    df = df.sort_values(group_cols + [time_col]).copy()

    # Last observed per group at/<= ceiling
    before = df[df[time_col] <= ceiling_date]
    if not before.empty:
        last = (before
                .sort_values(group_cols + [time_col])
                .drop_duplicates(subset=group_cols, keep="last"))[group_cols + fill_cols].copy()
        last.columns = group_cols + [f"{c}__last" for c in fill_cols]
        df = df.merge(last, on=group_cols, how="left")
    else:
        # No prior values â†’ all future rows become NaN on these cols
        for c in fill_cols:
            df.loc[df[time_col] > ceiling_date, c] = pd.NA
        return df

    # Week offsets after ceiling
    w = ((df[time_col] - ceiling_date).dt.days // 7).astype("Int64")
    mask_after = df[time_col] > ceiling_date

    for c in fill_cols:
        lastc = f"{c}__last"
        # within cap: fill with last; beyond cap: NaN
        df.loc[mask_after & (w <= max_weeks), c] = df.loc[mask_after & (w <= max_weeks), lastc]
        df.loc[mask_after & (w >  max_weeks), c] = pd.NA
        df.drop(columns=[lastc], inplace=True)

    return df



def train_eval_one_horizon(dm: pd.DataFrame,
                           sp: pd.DataFrame,
                           horizon: int,
                           as_of_ceiling: pd.Timestamp,
                           mode: str,
                           alpha: float,
                           locf_cap_weeks: int,
                           y_col: str,
                           do_standardize: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dm = dm.copy()
    cols_all = select_feature_columns(dm, y_col=y_col)

    if mode == "exoglite":
        feat_cols = drop_exogenous(cols_all)
    else:
        slow_cols, fast_cols = split_slow_vs_fast(cols_all)
        keep_set = set(cols_all) - set(fast_cols)
        feat_cols = [c for c in cols_all if c in keep_set]
        dm = locf_cap(
            df=dm,
            group_cols=["geo_id"],
            fill_cols=[c for c in slow_cols if c in dm.columns],
            time_col="week_start_date",
            ceiling_date=as_of_ceiling,
            max_weeks=locf_cap_weeks
        )

    # Remove rows with any NaNs in selected features or missing target
    dm = dm.dropna(subset=feat_cols + [y_col])

    # Truth table for target_week
    truth = (dm[["geo_id","week_start_date", y_col]]
             .rename(columns={"week_start_date":"target_week", y_col:"y_true"}))

    sp_h = sp[sp["horizon"] == horizon].copy()

    preds, metrics = [], []
    for fold_id, sp_fold in sp_h.groupby("fold_id"):
        tr_idx = sp_fold[sp_fold["role"] == "train"]
        va_idx = sp_fold[sp_fold["role"] == "val"]

        # Join features by (geo_id, week_start_date)
        trX = tr_idx.merge(dm, on=["geo_id","week_start_date"], how="inner")
        vaX = va_idx.merge(dm, on=["geo_id","week_start_date"], how="inner")

        # Enforce ceiling on TRAIN origins
        trX = trX[trX["week_start_date"] <= as_of_ceiling].copy()
        if trX.empty or vaX.empty:
            continue

        # Assemble matrices
        Xtr = trX[feat_cols].values
        Xva = vaX[feat_cols].values
        ytr = tr_idx.merge(truth, on=["geo_id","target_week"], how="inner")["y_true"].values
        yva = va_idx.merge(truth, on=["geo_id","target_week"], how="inner")["y_true"].values

        # Optional standardization (stabilizes optimizer)
        if do_standardize:
            scaler = StandardScaler(with_mean=True, with_std=True)
            Xtr = scaler.fit_transform(Xtr)
            Xva = scaler.transform(Xva)

        model = PoissonRegressor(alpha=alpha, max_iter=1000, fit_intercept=True)
        model.fit(Xtr, ytr)
        yhat = model.predict(Xva).clip(min=0.0)

        out = va_idx[["geo_id","week_start_date","target_week"]].copy()
        out["horizon"] = horizon
        out["y_pred"] = yhat
        out["fold_id"] = fold_id
        preds.append(out)

        mae = mean_absolute_error(yva, yhat)
        rmse = rmse_numpy(yva, yhat)
        metrics.append({"horizon": horizon, "fold_id": fold_id, "mode": mode, "MAE": mae, "RMSE": rmse})

    pred_df = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    met_df  = pd.DataFrame(metrics)
    return pred_df, met_df


def main():
    args = parse_args()
    as_of_ceiling = to_date(args.as_of_ceiling)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    Path(args.metricsdir).mkdir(parents=True, exist_ok=True)

    dm = pd.read_parquet(args.design)
    sp = pd.read_parquet(args.splits)

    # Normalize/required columns
    assert {"geo_id","week_start_date"}.issubset(dm.columns), "design_matrix needs geo_id, week_start_date"
    assert {"geo_id","week_start_date","horizon","role","fold_id","target_week"}.issubset(sp.columns), \
        "splits must have geo_id, week_start_date, horizon, role, fold_id, target_week"

    dm["week_start_date"] = pd.to_datetime(dm["week_start_date"]).dt.normalize()
    sp["week_start_date"] = pd.to_datetime(sp["week_start_date"]).dt.normalize()
    sp["target_week"]     = pd.to_datetime(sp["target_week"]).dt.normalize()

    y_col = detect_target_col(dm)

    horizons = sorted(sp["horizon"].unique().tolist())
    all_pred, all_met = [], []

    for h in horizons:
        p, m = train_eval_one_horizon(
            dm, sp, h, as_of_ceiling, args.mode, args.alpha, args.locf_cap_weeks,
            y_col=y_col, do_standardize=args.standardize
        )
        if not p.empty:
            all_pred.append(p)
        if not m.empty:
            all_met.append(m)

    if not all_pred:
        print("[warn] No predictions produced. Check splits/design alignment.")
        return

    pred = pd.concat(all_pred, ignore_index=True)
    met  = pd.concat(all_met,  ignore_index=True) if all_met else pd.DataFrame()

    if args.apply_residual:
        pred = with_residual(pred, big_name=args.dhaka_division, sub_name=args.dhaka_metro)

    pred_rec = bottom_up_sum(pred, geo_col="geo_id", target_col="y_pred",
                             time_col="target_week", country_geo=args.country_geo)

    stamp = datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ")
    mode_tag = args.mode
    pred_file = Path(args.outdir) / f"forecast_operational_{mode_tag}_{stamp}.csv"
    met_file  = Path(args.metricsdir) / f"metrics_operational_{mode_tag}_{stamp}.csv"
    pred_rec.to_csv(pred_file, index=False)
    met.to_csv(met_file, index=False)

    run_info = {
        "as_of_ceiling": str(as_of_ceiling.date()),
        "mode": mode_tag,
        "alpha": args.alpha,
        "locf_cap_weeks": args.locf_cap_weeks,
        "apply_residual": bool(args.apply_residual),
        "country_geo": args.country_geo,
        "files": {"predictions": str(pred_file), "metrics": str(met_file)}
    }
    (Path(args.outdir) / f"forecast_operational_{mode_tag}_{stamp}.json").write_text(json.dumps(run_info, indent=2))
    print(json.dumps(run_info, indent=2))


if __name__ == "__main__":
    main()

