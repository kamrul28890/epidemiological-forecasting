#!/usr/bin/env python3
"""
Production forecast for 1..12 weeks ahead using our GLM Poisson+ridge baseline.

- Trains per-horizon models using splits (train vs val), reports MAE/RMSE.
- Re-fits on train+val per horizon.
- Scores the latest available origin ("as_of" = max week_start_date in design).
- Writes CSV to reports/forecasts/forecast_glm_<YYYY-MM-DD>.csv
"""

from __future__ import annotations
import argparse, math
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import statsmodels.api as sm

from dengue.utils.io import load_yaml, resolve_paths, ensure_dir  # type: ignore


def _select_feature_columns(dm: pd.DataFrame) -> List[str]:
    drop = {"geo_id", "geo_name", "week_start_date", "cases"}
    cols = [c for c in dm.columns if c not in drop]
    num = [c for c in cols if pd.api.types.is_numeric_dtype(dm[c])]
    return sorted(num)


def _label_lookup(dm: pd.DataFrame) -> pd.Series:
    # map (geo_id, week) -> cases
    s = dm[["geo_id", "week_start_date", "cases"]].copy()
    s["week_start_date"] = pd.to_datetime(s["week_start_date"])
    return s.set_index(["geo_id", "week_start_date"])["cases"]


def _build_pairs(dm: pd.DataFrame, splits: pd.DataFrame, horizon: int, role_filter: List[str]) -> pd.DataFrame:
    feats = _select_feature_columns(dm)
    id_cols = ["geo_id", "week_start_date"]
    X = dm[id_cols + feats].copy()
    X["week_start_date"] = pd.to_datetime(X["week_start_date"])

    S = splits[(splits["horizon"] == horizon) & (splits["role"].isin(role_filter))].copy()
    S["week_start_date"] = pd.to_datetime(S["week_start_date"])
    S["target_week"] = pd.to_datetime(S["target_week"])

    # features at origin
    DF = S.merge(X, on=id_cols, how="left")

    # labels at target
    ymap = _label_lookup(dm)
    DF["y_true"] = DF.apply(lambda r: float(ymap.get((r["geo_id"], r["target_week"]), np.nan)), axis=1)

    # drop rows w/ no label or NA features
    feats_ok = DF[feats].notna().all(axis=1)
    DF = DF[feats_ok & DF["y_true"].notna()].copy()

    return DF, feats


def _zscore_train_apply(train: pd.DataFrame, apply_df: pd.DataFrame, feat_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    mu = train[feat_cols].mean()
    sd = train[feat_cols].std().replace(0.0, np.nan)
    keep = sd.notna()
    kept = [c for c in feat_cols if keep.get(c, False)]
    if not kept:
        return train.assign(**{c: 0.0 for c in feat_cols}), apply_df.assign(**{c: 0.0 for c in feat_cols}), []

    def z(d: pd.DataFrame) -> pd.DataFrame:
        Z = (d[kept] - mu[kept]) / sd[kept]
        return Z.clip(-6, 6).fillna(0.0)

    return z(train), z(apply_df), kept


def _fit_glm_poisson_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    Xc = sm.add_constant(X, has_constant="add")
    fam = sm.families.Poisson()
    mdl = sm.GLM(y, Xc, family=fam)
    res = mdl.fit_regularized(alpha=alpha, L1_wt=0.0, maxiter=1000)
    return res.params  # includes intercept


def _predict_glm(params: np.ndarray, X: np.ndarray) -> np.ndarray:
    fam = sm.families.Poisson()
    Xc = sm.add_constant(X, has_constant="add")
    eta = Xc @ params
    yhat = fam.link.inverse(eta)
    return np.clip(yhat, 0.0, None)


@dataclass
class HorizonModel:
    horizon: int
    feat_cols: List[str]
    z_mu: pd.Series
    z_sd: pd.Series
    params: np.ndarray  # GLM coefficients (with intercept)


def _train_one_horizon(dm: pd.DataFrame, splits: pd.DataFrame, h: int) -> Tuple[HorizonModel, Dict[str, float]]:
    # train/val split from precomputed splits
    TR, feat_cols = _build_pairs(dm, splits, h, role_filter=["train"])
    VA, _ = _build_pairs(dm, splits, h, role_filter=["val"])

    # if train is empty, bail to constant mean
    if TR.empty:
        mu = 0.0
        return HorizonModel(h, [], pd.Series(dtype=float), pd.Series(dtype=float), np.array([mu])), {"MAE": np.nan, "RMSE": np.nan}

    # standardize on train; apply to val
    Ztr, Zva, kept = _zscore_train_apply(TR, VA, feat_cols)
    if not kept:
        mu = float(TR["y_true"].mean())
        # params vector: intercept only (mean in log link space is risky; use identity here via predict wrapper)
        return HorizonModel(h, [], pd.Series(dtype=float), pd.Series(dtype=float), np.array([mu])), {"MAE": float(abs(VA["y_true"] - mu).mean()) if not VA.empty else np.nan,
                                                                                                   "RMSE": float(np.sqrt(((VA["y_true"] - mu) ** 2).mean())) if not VA.empty else np.nan}

    Xtr = Ztr[kept].values.astype("float64")
    ytr = TR["y_true"].values.astype("float64")

    # ridge strength: mild scale with #features
    alpha = max(0.5, 0.1 * Xtr.shape[1])

    try:
        params = _fit_glm_poisson_ridge(Xtr, ytr, alpha=alpha)
        # val metrics
        if not VA.empty:
            yhat = _predict_glm(params, Zva[kept].values.astype("float64"))
            e = VA["y_true"].values - yhat
            MAE = float(np.mean(np.abs(e)))
            RMSE = float(np.sqrt(np.mean(e ** 2)))
        else:
            MAE = RMSE = np.nan
    except Exception:
        # fallback: constant mean
        mu = float(ytr.mean())
        params = np.array([mu])  # intercept-only in our predict wrapper (special cased below)
        if not VA.empty:
            MAE = float(abs(VA["y_true"] - mu).mean())
            RMSE = float(np.sqrt(((VA["y_true"] - mu) ** 2).mean()))
        else:
            MAE = RMSE = np.nan

    # Refit on train+val for production (if val exists)
    ALL = pd.concat([TR, VA], ignore_index=True)
    Zall, _, kept2 = _zscore_train_apply(ALL, ALL, kept)  # recompute on ALL; keeps stability
    if kept2:
        try:
            params = _fit_glm_poisson_ridge(Zall[kept2].values.astype("float64"), ALL["y_true"].values.astype("float64"), alpha=max(0.5, 0.1 * len(kept2)))
            kept = kept2
        except Exception:
            pass  # keep previous params

    # store z-score stats (on ALL, so inference matches final fit)
    z_mu = Zall[kept].mean()*0  # we don't need actual mu/sd for Zall; we’ll recompute from ALL below
    z_sd = Zall[kept].std().replace(0.0, np.nan)

    # But we want train+val mean/sd for live z-scoring on the as_of row:
    MU = ALL[kept].mean()
    SD = ALL[kept].std().replace(0.0, np.nan)

    return HorizonModel(h, kept, MU, SD, params), {"MAE": MAE, "RMSE": RMSE}


def _score_asof(dm: pd.DataFrame, model: HorizonModel, as_of: pd.Timestamp) -> pd.DataFrame:
    # single origin row per geo at as_of
    id_cols = ["geo_id", "week_start_date"]
    base = dm[["geo_id", "week_start_date"] + model.feat_cols].copy()
    base = base[base["week_start_date"] == as_of].copy()
    if base.empty:
        return pd.DataFrame(columns=["geo_id", "origin_week", "target_week", "horizon", "y_pred", "model"])

    # z-score with MU/SD captured in model
    Z = (base[model.feat_cols] - model.z_mu) / model.z_sd
    Z = Z.clip(-6, 6).fillna(0.0)

    if len(model.params) == 1 and (not model.feat_cols):  # intercept-only fallback
        yhat = np.full(len(Z), model.params[0], dtype=float)
    else:
        yhat = _predict_glm(model.params, Z.values.astype("float64"))

    out = base[["geo_id"]].copy()
    out["origin_week"] = as_of
    out["horizon"] = model.horizon
    out["target_week"] = as_of + pd.to_timedelta(7 * model.horizon, unit="D")
    out["y_pred"] = yhat
    out["model"] = "glm_poisson_ridge"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", type=str, default=None, help="Origin week (YYYY-MM-DD). Default: max week_start_date in design.")
    ap.add_argument("--horizons", type=str, default="1-12", help="Range like '1-12' or comma list like '1,2,3,6,12'.")
    args = ap.parse_args()

    cfg = load_yaml("configs/data.yaml")
    paths = resolve_paths(cfg)

    dm = pd.read_parquet(paths["processed"] / "design_matrix.parquet")
    dm["week_start_date"] = pd.to_datetime(dm["week_start_date"])
    splits = pd.read_parquet(paths["processed"] / "splits.parquet")

    as_of = pd.to_datetime(args.as_of) if args.as_of else dm["week_start_date"].max()

    # parse horizons
    if "-" in args.horizons:
        a, b = args.horizons.split("-", 1)
        horizons = list(range(int(a), int(b) + 1))
    else:
        horizons = [int(x) for x in args.horizons.split(",")]

    # train & collect per-horizon models + val metrics
    models: List[HorizonModel] = []
    metrics_rows = []
    for h in horizons:
        hm, mets = _train_one_horizon(dm, splits, h)
        models.append(hm)
        metrics_rows.append({"horizon": h, "MAE_val": mets["MAE"], "RMSE_val": mets["RMSE"]})

    # score the as_of week
    all_fore = []
    for hm in models:
        fore_h = _score_asof(dm, hm, as_of)
        all_fore.append(fore_h)
    fore = pd.concat(all_fore, ignore_index=True) if all_fore else pd.DataFrame(columns=["geo_id","origin_week","target_week","horizon","y_pred","model"])

    # write CSV
    out_dir = (paths["root"] / "reports" / "forecasts")
    ensure_dir(out_dir)
    fname = f"forecast_glm_{as_of.date()}.csv"
    fout = out_dir / fname
    fore.sort_values(["geo_id", "horizon"]).to_csv(fout, index=False)

    # metrics CSV (optional but useful)
    mdir = (paths["root"] / "reports" / "metrics")
    ensure_dir(mdir)
    (pd.DataFrame(metrics_rows).sort_values("horizon")).to_csv(mdir / f"forecast_glm_{as_of.date()}_val_metrics.csv", index=False)

    print(f"[forecast] Wrote {len(fore):,} predictions → {fout}")
    print(f"[forecast] Val metrics per horizon → {mdir / f'forecast_glm_{as_of.date()}_val_metrics.csv'}")


if __name__ == "__main__":
    main()
