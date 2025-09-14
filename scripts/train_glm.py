#!/usr/bin/env python
"""
Negative-Binomial GLM baseline on the design matrix (AR + climate + calendar).

Inputs:
  - data/processed/design_matrix.parquet
  - data/processed/splits.parquet

Outputs:
  - artifacts/forecasts/glm_<FOLD>.parquet
  - artifacts/tables/metrics_glm_<FOLD>.csv

Env toggles (optional):
  - GLM_DIAG=1   -> write diagnostics JSONs to artifacts/debug and print a summary
  - GLM_ALPHA=1  -> ridge strength used in fit_regularized (float)
"""

from __future__ import annotations

import pathlib

# --- repo-root import shim (so `import dengue` works from scripts/) ---
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
# ----------------------------------------------------------------------

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from dengue.utils.io import ensure_dir, load_yaml, resolve_paths  # type: ignore

# --------------------------- helpers: metrics ---------------------------


def build_argparser() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fold",
        required=True,
        choices=["F1", "F2", "F3", "F4", "F5"],
        help="which fold from splits.parquet to train/evaluate",
    )
    return ap.parse_args()


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    drop = {
        "geo_id",
        "geo_name",
        "week_start_date",
        "cases",
        "target_week",
        "role",
        "fold",
        "fold_id",
    }
    feats = [
        c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])
    ]
    return sorted(feats)


def metrics_frame(out: pd.DataFrame, source_name: str) -> pd.DataFrame:
    def _rmse(a):
        return float(np.sqrt(np.mean((a["y_true"] - a["y_pred"]) ** 2)))

    def _mae(a):
        return float(np.mean(np.abs(a["y_true"] - a["y_pred"])))

    rows = []
    # overall by role
    for role, g in out.groupby("role"):
        rows.append(
            dict(
                source=source_name,
                role=role,
                horizon=np.nan,
                MAE=_mae(g),
                RMSE=_rmse(g),
            )
        )
    # by role & horizon
    for (role, h), g in out.groupby(["role", "horizon"]):
        rows.append(
            dict(
                source=source_name,
                role=role,
                horizon=int(h),
                MAE=_mae(g),
                RMSE=_rmse(g),
            )
        )
    return pd.DataFrame(rows)


# --------------------------- helpers: robust GLM ---------------------------


def _write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)


def _estimate_nb_alpha(y: np.ndarray) -> float:
    """Method-of-moments NB2 dispersion estimate; clipped to a small positive."""
    y = np.asarray(y, float)
    m, v = np.mean(y), np.var(y)
    if m <= 0:
        return 1.0
    alpha = (v - m) / (m**2 + 1e-12)
    return float(max(alpha, 1e-6))


def _diagnose_design(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    top_k_corr: int = 12,
) -> dict:
    """
    Basic diagnostics to catch causes of separation / numerical issues.
    - Skips the intercept and zero-variance columns for correlation checks.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, p = X.shape
    if feature_names is None:
        feature_names = [f"x{j}" for j in range(p)]

    # core stats
    diag = {}
    diag["n_rows"] = int(n)
    diag["n_cols"] = int(p)
    diag["y_mean"] = float(np.mean(y))
    diag["y_var"] = float(np.var(y))
    diag["y_min"] = float(np.min(y))
    diag["y_max"] = float(np.max(y))
    diag["prop_y_zero"] = float(np.mean(y == 0))

    # zero-variance detection
    stds = X.std(axis=0)
    zero_var_idx = np.where(stds <= 1e-12)[0]
    diag["constant_cols_idx"] = zero_var_idx.tolist()
    diag["constant_cols"] = [feature_names[j] for j in zero_var_idx]

    # identify intercept if present
    intercept_idx = (
        0
        if feature_names and feature_names[0].lower() in ("const", "intercept")
        else None
    )

    # quasi-perfect predictors (skip intercept and const cols)
    flagged_sep = []
    ypos = y > 0
    for j in range(p):
        if (intercept_idx is not None and j == intercept_idx) or (j in zero_var_idx):
            continue
        xj = X[:, j]
        uniq = np.unique(xj)
        is_binary_like = (uniq.size <= 5) or np.all(np.isin(uniq, [0, 1]))
        if not is_binary_like or uniq.size <= 1:
            continue
        a = np.sum(ypos & (xj == 0))
        b = np.sum(ypos & (xj != 0))
        if (a == 0 and b > 0) or (b == 0 and a > 0):
            flagged_sep.append(feature_names[j])
    diag["quasi_perfect_predictors"] = flagged_sep

    # correlations (drop intercept + zero-variance cols)
    keep_mask = np.ones(p, dtype=bool)
    keep_mask[zero_var_idx] = False
    if intercept_idx is not None:
        keep_mask[intercept_idx] = False
    kept_idx = np.where(keep_mask)[0]
    if kept_idx.size >= 2:
        Xk = X[:, kept_idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            C = np.corrcoef(Xk, rowvar=False)
        # defend against NaNs from any residual near-constant columns
        if np.any(~np.isfinite(C)):
            C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        iu = np.triu_indices_from(C, k=1)
        pairs = list(zip(iu[0], iu[1], C[iu]))
        pairs = sorted(pairs, key=lambda t: -abs(t[2]))
        top = [
            (feature_names[kept_idx[i]], feature_names[kept_idx[j]], float(r))
            for i, j, r in pairs
            if np.isfinite(r) and abs(r) > 0.995
        ][:top_k_corr]
        diag["high_corr_pairs"] = top
        # condition number on correlation of kept features
        try:
            diag["cond_number_corr"] = float(
                np.linalg.cond(C + 1e-10 * np.eye(C.shape[0]))
            )
        except Exception:
            diag["cond_number_corr"] = float("nan")
    else:
        diag["high_corr_pairs"] = []
        diag["cond_number_corr"] = float("nan")

    return diag


def _clean_and_scale(
    X: np.ndarray,
    y: np.ndarray,
    intercept_idx: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler, np.ndarray, np.ndarray, List[int]]:
    """
    - Drop rows with non-finite values
    - Drop near-constant columns (except the intercept)
    - Standardize non-intercept columns
    Returns: Xs, y_clean, scaler, keep_cols_mask, keep_rows_mask, dropped_cols_idx
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)

    row_mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X, y = X[row_mask], y[row_mask]

    col_std = X.std(axis=0)
    keep_cols = col_std > 1e-12
    if intercept_idx is not None:
        keep_cols[intercept_idx] = True  # always keep intercept
    dropped_cols = np.where(~keep_cols)[0].tolist()

    # scale only non-intercept kept columns
    scaler = StandardScaler()
    if intercept_idx is None:
        X_kept = X[:, keep_cols]
        Xs = scaler.fit_transform(X_kept)
    else:
        keep_non_intercept = keep_cols.copy()
        keep_non_intercept[intercept_idx] = False
        Xi = X[:, [intercept_idx]]  # intercept column
        Xn = X[:, keep_non_intercept]
        Xn_s = scaler.fit_transform(Xn) if Xn.shape[1] else Xn
        Xs = np.concatenate([Xi, Xn_s], axis=1)

        # rebuild keep_cols mask order to match Xs columns
        # (still return mask for original X, which the caller expects)
    return Xs, y, scaler, keep_cols, row_mask, dropped_cols


def _ridge_fallback(Xs: np.ndarray, y: np.ndarray, Xs_out: np.ndarray) -> np.ndarray:
    """
    Last-ditch robust predictor:
    fit Ridge on log1p(y) and return expm1 predictions, clipped at 0.
    """
    mdl = Ridge(alpha=1.0)
    mdl.fit(Xs, np.log1p(y))
    pred = np.expm1(mdl.predict(Xs_out))
    return np.clip(pred, 0, None)


def _robust_glm_predict(
    df: pd.DataFrame,
    feats: List[str],
    fam,
    diag_tag: str = "F?",
) -> np.ndarray:
    """
    Robust fit/predict for a *single* (geo, horizon, role subset) DataFrame.
    Follows the user's original behavior: fit on df and predict on df.

    Diagnostics are controlled via GLM_DIAG=1 (env).
    """
    diagnose_mode = os.getenv("GLM_DIAG", "0") == "1"
    alpha_ridge = float(os.getenv("GLM_ALPHA", "0.005"))

    # Build design with explicit intercept as first column
    Xf = df[feats].to_numpy(dtype=float, copy=False)
    intercept = np.ones((len(df), 1), dtype=float)
    X = np.concatenate([intercept, Xf], axis=1)  # [1, feats...]

    y = df["y_true"].to_numpy(dtype=float)

    # Estimate NB alpha if needed
    if isinstance(fam, sm.families.family.NegativeBinomial):
        try:
            if not hasattr(fam, "alpha") or fam.alpha is None:
                fam.alpha = _estimate_nb_alpha(y)
        except Exception:
            fam.alpha = _estimate_nb_alpha(y)

    # names (for diagnostics)
    feat_names = ["const"] + feats

    # pre diagnostics
    pre_diag = (
        _diagnose_design(X, y, feature_names=feat_names) if diagnose_mode else None
    )

    # clean & scale (preserve intercept at idx=0)
    Xs, y_clean, scaler, keep_cols_mask, keep_rows_mask, dropped_cols_idx = (
        _clean_and_scale(X, y, intercept_idx=0)
    )

    if Xs.size == 0 or y_clean.size == 0:
        yhat_full = np.zeros(len(df), dtype=float)
        if diagnose_mode:
            diag = {
                "tag": diag_tag,
                "note": "all features or rows dropped as invalid",
                "pre_diag": pre_diag,
            }
            _write_json(
                diag, Path("artifacts/debug") / f"glm_{diag_tag}_diagnostics.json"
            )
        return yhat_full

    # fit with ridge penalty, try fallbacks
    fit_errs = []
    model_used = None

    try:
        res = sm.GLM(y_clean, Xs, family=fam).fit_regularized(
            alpha=alpha_ridge, L1_wt=0.0, maxiter=200
        )
        model_used = f"GLM({fam.__class__.__name__})-ridge"
        yhat_in = res.predict(Xs)
        yhat_in = np.clip(yhat_in, 0, None)
    except Exception as e1:
        fit_errs.append(f"GLM({fam.__class__.__name__}) ridge failed: {repr(e1)}")
        try:
            fam2 = sm.families.Poisson()
            res = sm.GLM(y_clean, Xs, family=fam2).fit_regularized(
                alpha=alpha_ridge, L1_wt=0.0, maxiter=200
            )
            model_used = "GLM(Poisson)-ridge"
            yhat_in = res.predict(Xs)
            yhat_in = np.clip(yhat_in, 0, None)
        except Exception as e2:
            fit_errs.append(f"Poisson ridge failed: {repr(e2)}")
            try:
                fam3 = sm.families.Tweedie(var_power=1.5, link=sm.families.links.log())
                res = sm.GLM(y_clean, Xs, family=fam3).fit_regularized(
                    alpha=alpha_ridge, L1_wt=0.0, maxiter=200
                )
                model_used = "GLM(Tweedie v=1.5)-ridge"
                yhat_in = res.predict(Xs)
                yhat_in = np.clip(yhat_in, 0, None)
            except Exception as e3:
                fit_errs.append(f"Tweedie ridge failed: {repr(e3)}")
                model_used = "Ridge(log1p)-fallback"
                yhat_in = _ridge_fallback(Xs, y_clean, Xs)

    # map back to original rows (dropped rows -> 0)
    yhat_full = np.zeros(len(df), dtype=float)
    yhat_full[keep_rows_mask] = yhat_in

    if diagnose_mode:
        # names of kept/dropped cols in original order
        kept_names = [n for n, k in zip(feat_names, keep_cols_mask) if k]
        dropped_names = [n for n, k in zip(feat_names, keep_cols_mask) if not k]

        # post diag on kept subset
        X_kept = X[keep_rows_mask][:, keep_cols_mask]
        post_diag = _diagnose_design(
            X_kept, y[keep_rows_mask], feature_names=kept_names
        )

        diag = {
            "tag": diag_tag,
            "model_used": model_used,
            "alpha_ridge": alpha_ridge,
            "nb_alpha_if_used": (
                getattr(fam, "alpha", None)
                if isinstance(fam, sm.families.family.NegativeBinomial)
                else None
            ),
            "fit_errors": fit_errs,
            "pre_diag": pre_diag,
            "dropped_feature_names": dropped_names,
            "kept_feature_names": kept_names,
            "rows_kept": int(np.sum(keep_rows_mask)),
            "rows_total": int(len(df)),
            "post_diag": post_diag,
        }

        out_json = Path("artifacts/debug") / f"glm_{diag_tag}_diagnostics.json"
        _write_json(diag, out_json)

        # console summary
        print("\n=== GLM DIAGNOSTICS ===========================")
        print(f"tag: {diag_tag}")
        print(f"model_used: {model_used}")
        if fit_errs:
            print("fit_errors:")
            for e in fit_errs:
                print(" -", e)
        print(f"#rows kept: {diag['rows_kept']} / {diag['rows_total']}")
        print(f"#features kept: {len(kept_names)} (dropped: {len(dropped_names)})")
        qp = post_diag.get("quasi_perfect_predictors", [])
        if qp:
            print("quasi_perfect_predictors:", qp)
        hp = post_diag.get("high_corr_pairs", [])
        if hp:
            print("top high_corr_pairs (|r|>0.995):")
            for a, b, r in hp:
                print(f"  {a} ~ {b}: r={r:.6f}")
        print("==============================================\n")

    return yhat_full


# --------------------------- main ---------------------------


def main():
    args = build_argparser()

    cfg = load_yaml("configs/data.yaml")
    paths = resolve_paths(cfg)

    # I/O dirs
    ensure_dir(paths["forecasts"])
    ensure_dir(paths["tables"])
    ensure_dir(Path("artifacts/debug"))

    # Data
    design = pd.read_parquet(paths["processed"] / "design_matrix.parquet")
    design["week_start_date"] = pd.to_datetime(design["week_start_date"])

    splits = pd.read_parquet(paths["processed"] / "splits.parquet")
    splits["week_start_date"] = pd.to_datetime(splits["week_start_date"])
    splits["target_week"] = pd.to_datetime(splits["target_week"])

    # Fold column can be "fold" or "fold_id"
    fold_col = "fold" if "fold" in splits.columns else "fold_id"
    S = splits[splits[fold_col] == args.fold].copy()
    if S.empty:
        print(f"[glm] no rows for fold={args.fold}; check splits.")
        return

    # Features (exclude ids/dates/target)
    feats = select_feature_columns(design)

    # Build design X at base week, and labels at target week via merge
    X = design[["geo_id", "week_start_date"] + feats].copy()
    # guard: no duplicate labels on RHS of merges
    if X.columns.duplicated().any():
        X = X.loc[:, ~X.columns.duplicated()]

    y_map = (
        design[["geo_id", "week_start_date", "cases"]]
        .rename(columns={"week_start_date": "target_week", "cases": "y_true"})
        .copy()
    )

    base = S.merge(X, on=["geo_id", "week_start_date"], how="left")
    base = base.merge(y_map, on=["geo_id", "target_week"], how="left")

    # Simple NA handling for features
    for c in feats:
        if c in base.columns:
            base[c] = base[c].fillna(0.0)

    # Per (h, geo) GLM
    out_rows = []
    for h in sorted(base["horizon"].unique()):
        Bh = base[base["horizon"] == h]
        for geo in sorted(Bh["geo_id"].unique()):
            g = Bh[Bh["geo_id"] == geo].copy()
            tr = g[g["role"] == "train"].copy()
            va = g[g["role"] == "val"].copy()

            # if too few points or no labels, fall back to a mean predictor
            if len(tr) < 20 or tr["y_true"].isna().all():
                mean_hat = float(tr["y_true"].mean()) if len(tr) else 0.0
                tr["y_pred"] = mean_hat
                va["y_pred"] = mean_hat
            else:
                # Estimate NB alpha from *train* (more stable), but we keep the
                # original behavior of fitting each subset on itself.
                ytr = tr["y_true"].to_numpy(dtype=float)
                alpha = _estimate_nb_alpha(ytr)

                # Try NB first, then fall back inside helper as needed
                fam_nb = sm.families.NegativeBinomial(alpha=alpha)

                tr_tag = f"{args.fold}_h{h}_geo{geo}_train"
                va_tag = f"{args.fold}_h{h}_geo{geo}_val"

                try:
                    tr["y_pred"] = _robust_glm_predict(
                        tr, feats, fam_nb, diag_tag=tr_tag
                    )
                except Exception:
                    # ultimate fallback path (should be rare)
                    tr["y_pred"] = _robust_glm_predict(
                        tr, feats, sm.families.Poisson(), diag_tag=tr_tag
                    )

                try:
                    va["y_pred"] = _robust_glm_predict(
                        va, feats, fam_nb, diag_tag=va_tag
                    )
                except Exception:
                    va["y_pred"] = _robust_glm_predict(
                        va, feats, sm.families.Poisson(), diag_tag=va_tag
                    )

            out_rows.append(pd.concat([tr, va], axis=0))

    XYhat = pd.concat(out_rows, axis=0, ignore_index=True)
    XYhat["model"] = "glm_nb"

    # Save forecasts
    fore = XYhat[
        [
            "geo_id",
            "week_start_date",
            "target_week",
            "role",
            fold_col,
            "horizon",
            "y_true",
            "y_pred",
            "model",
        ]
    ].copy()
    out_fore = paths["forecasts"] / f"glm_{args.fold}.parquet"
    fore.to_parquet(out_fore, index=False)

    # Save metrics (same schema as GBM)
    m = metrics_frame(fore, f"metrics_glm_{args.fold}.csv")
    out_tbl = paths["tables"] / f"metrics_glm_{args.fold}.csv"
    m.to_csv(out_tbl, index=False)

    print(f"[glm] forecasts → {out_fore} ({len(fore):,} rows)")
    print(f"[glm] metrics   → {out_tbl}")


if __name__ == "__main__":
    main()
