#!/usr/bin/env python3
"""
Poisson GLM (ridge) baseline on the design matrix (AR + climate + calendar).
Per fold & horizon, per geo:
  - Fit on train rows, predict on train and val.
Writes:
  artifacts/forecasts/glm_<FOLD>.parquet
  artifacts/tables/metrics_glm_<FOLD>.csv
"""
from __future__ import annotations

import argparse
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

from dengue.utils.io import ensure_dir, load_yaml, resolve_paths  # type: ignore

# --- keep the console clean from non-actionable GLM warnings ---
warnings.filterwarnings("ignore", category=PerfectSeparationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*divide by zero.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*overflow.*")
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*Perfect separation.*"
)


def build_pairs(
    design: pd.DataFrame, splits: pd.DataFrame
) -> Tuple[pd.DataFrame, List[str]]:
    # features at origin week; y at target week
    X = design.copy()
    X["week_start_date"] = pd.to_datetime(X["week_start_date"])
    y = X.set_index(["geo_id", "week_start_date"])["cases"]

    # candidate features: all numeric except identifiers/targets/meta
    drop_cols = {"cases", "geo_name", "target_week", "role", "fold_id"}
    feats = [c for c in X.columns if c not in drop_cols and X[c].dtype != "O"]

    X = X[["geo_id", "week_start_date"] + feats].copy()

    def y_at(geo, dt):
        try:
            return float(y.loc[(geo, pd.Timestamp(dt))])
        except KeyError:
            return np.nan

    out = splits.copy()
    out = out.merge(X, on=["geo_id", "week_start_date"], how="left")
    out["y_true"] = out.apply(lambda r: y_at(r["geo_id"], r["target_week"]), axis=1)

    # simple NA handling
    for c in feats:
        out[c] = out[c].astype("float64").fillna(0.0)

    return out, feats


def _standardize_train_apply(train: pd.DataFrame, val: pd.DataFrame, feats: List[str]):
    """Z-score with train stats only, clip to [-6, 6], drop zero-variance cols."""
    mu = train[feats].mean()
    sd = train[feats].std().replace(0.0, np.nan)

    keep = sd.notna()
    feats_kept = [f for f in feats if keep.get(f, False)]
    if not feats_kept:
        return train, val, []

    def z(df):
        Z = (df[feats_kept] - mu[feats_kept]) / sd[feats_kept]
        Z = Z.clip(-6, 6).fillna(0.0)
        return Z

    Ztr = z(train)
    Zva = z(val)
    return Ztr, Zva, feats_kept


def _glm_poisson_ridge(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    Xtr: np.ndarray,
    Xva: np.ndarray,
    alpha: float,
):
    """Stable Poisson GLM with L2 penalty (statsmodels.fit_regularized)."""
    ytr = train_df["y_true"].astype("float64").values
    fam = sm.families.Poisson()

    # add intercept
    Xtr_c = sm.add_constant(Xtr, has_constant="add")
    Xva_c = sm.add_constant(Xva, has_constant="add")

    mdl = sm.GLM(ytr, Xtr_c, family=fam)
    # L1_wt=0 → pure ridge; alpha ~ penalty strength
    res = mdl.fit_regularized(alpha=alpha, L1_wt=0.0, maxiter=1000)
    # predict via link inverse to be safe:
    eta_tr = Xtr_c @ res.params
    eta_va = Xva_c @ res.params
    yhat_tr = fam.link.inverse(eta_tr)
    yhat_va = fam.link.inverse(eta_va)
    return np.clip(yhat_tr, 0.0, None), np.clip(yhat_va, 0.0, None)


def _ridge_log1p_fallback(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    Xtr: np.ndarray,
    Xva: np.ndarray,
    lam: float = 1.0,
):
    """Closed-form ridge on log1p(y); very robust fallback."""
    ytr = np.log1p(train_df["y_true"].astype("float64").values)

    # add intercept
    Xtr_c = np.c_[np.ones((Xtr.shape[0], 1)), Xtr]
    Xva_c = np.c_[np.ones((Xva.shape[0], 1)), Xva]

    # (X'X + lam I)^{-1} X'y
    I_mat = np.eye(Xtr_c.shape[1])
    beta = np.linalg.solve(Xtr_c.T @ Xtr_c + lam * I_mat, Xtr_c.T @ ytr)

    mu_tr = np.expm1(Xtr_c @ beta)
    mu_va = np.expm1(Xva_c @ beta)
    return np.clip(mu_tr, 0.0, None), np.clip(mu_va, 0.0, None)


def fit_predict_glm(train_df: pd.DataFrame, val_df: pd.DataFrame, feats: List[str]):
    # train-only standardization & clipping
    Ztr, Zva, kept = _standardize_train_apply(train_df, val_df, feats)
    if not kept:
        # nothing to learn from; predict train mean
        mu = float(np.nanmean(train_df["y_true"]))
        return np.full(len(train_df), mu), np.full(len(val_df), mu)

    Xtr = Ztr.values
    Xva = Zva.values

    # Too few rows or degenerate target → fallback to mean
    if (
        len(train_df) < (Xtr.shape[1] + 5)
        or np.nanstd(train_df["y_true"].values) < 1e-8
    ):
        mu = float(np.nanmean(train_df["y_true"]))
        return np.full(len(train_df), mu), np.full(len(val_df), mu)

    # Cap for safety at evaluation time (keeps metrics sane)
    cap = float(np.nanpercentile(train_df["y_true"], 99.5)) * 3.0 + 10.0

    # Main attempt: Poisson ridge (stable)
    try:
        # penalty strength scaled by features (tune lightly if needed)
        alpha = max(0.5, 0.1 * Xtr.shape[1])
        yhat_tr, yhat_va = _glm_poisson_ridge(train_df, val_df, Xtr, Xva, alpha=alpha)
    except Exception:
        # rock-solid fallback: ridge on log1p
        yhat_tr, yhat_va = _ridge_log1p_fallback(train_df, val_df, Xtr, Xva, lam=1.0)

    # final clips
    yhat_tr = np.nan_to_num(yhat_tr, nan=0.0, posinf=cap, neginf=0.0)
    yhat_va = np.nan_to_num(yhat_va, nan=0.0, posinf=cap, neginf=0.0)
    yhat_tr = np.clip(yhat_tr, 0.0, cap)
    yhat_va = np.clip(yhat_va, 0.0, cap)
    return yhat_tr, yhat_va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True, choices=[f"F{i}" for i in range(1, 6)])
    args = ap.parse_args()

    cfg = load_yaml("configs/data.yaml")
    paths = resolve_paths(cfg)

    design = pd.read_parquet(paths["processed"] / "design_matrix.parquet")
    splits = pd.read_parquet(paths["processed"] / "splits.parquet")
    F = splits[splits["fold_id"] == args.fold].copy()

    XY, feats = build_pairs(design, F)

    out_rows = []
    for h in sorted(XY["horizon"].unique()):
        for geo in sorted(XY["geo_id"].unique()):
            g = XY[(XY["geo_id"] == geo) & (XY["horizon"] == h)].copy()
            tr = g[g["role"] == "train"].copy()
            va = g[g["role"] == "val"].copy()

            if len(tr) < 20 or tr["y_true"].isna().all():
                mu = float(np.nanmean(tr["y_true"])) if len(tr) else 0.0
                yhat_tr = np.full(len(tr), mu)
                yhat_va = np.full(len(va), mu)
            else:
                try:
                    yhat_tr, yhat_va = fit_predict_glm(tr, va, feats)
                except Exception:
                    # final safety net: predict train mean
                    mu = float(np.nanmean(tr["y_true"]))
                    yhat_tr = np.full(len(tr), mu)
                    yhat_va = np.full(len(va), mu)

            tr["y_pred"] = yhat_tr
            va["y_pred"] = yhat_va
            out_rows.append(pd.concat([tr, va], axis=0))

    XYhat = pd.concat(out_rows, axis=0, ignore_index=True)
    XYhat["model"] = "glm_poisson_ridge"

    # forecasts
    fore = XYhat[
        [
            "geo_id",
            "week_start_date",
            "target_week",
            "role",
            "fold_id",
            "horizon",
            "y_true",
            "y_pred",
            "model",
        ]
    ].copy()
    ensure_dir(paths["forecasts"])
    out_fore = paths["forecasts"] / f"glm_{args.fold}.parquet"
    fore.to_parquet(out_fore, index=False)

    # metrics (overall and by horizon)
    def agg(g):
        e = g["y_true"] - g["y_pred"]
        return pd.Series({"MAE": e.abs().mean(), "RMSE": np.sqrt((e**2).mean())})

    m = pd.concat(
        [
            fore.groupby(["role"], dropna=False).apply(agg),
            fore.groupby(["role", "horizon"], dropna=False).apply(agg),
        ]
    ).reset_index()
    m.insert(0, "source", f"metrics_glm_{args.fold}.csv")
    ensure_dir(paths["tables"])
    out_tbl = paths["tables"] / f"metrics_glm_{args.fold}.csv"
    m.to_csv(out_tbl, index=False)

    print(f"[glm] forecasts → {out_fore} ({len(fore):,} rows)")
    print(f"[glm] metrics   → {out_tbl}")


if __name__ == "__main__":
    main()
