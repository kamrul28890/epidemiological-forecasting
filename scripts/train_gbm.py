#!/usr/bin/env python
"""
Train a GBM per horizon using the design matrix + rolling-origin splits.

Inputs:
  - data/processed/design_matrix.parquet
  - data/processed/splits.parquet

Outputs:
  - artifacts/forecasts/gbm_<FOLD>.parquet
  - artifacts/tables/metrics_gbm_<FOLD>.csv
"""

from __future__ import annotations

import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd

from dengue.utils.io import ensure_dir, load_yaml, resolve_paths


def build_argparser() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fold",
        required=True,
        choices=["F1", "F2", "F3", "F4", "F5"],
        help="which fold from splits.parquet to train/evaluate",
    )
    return ap.parse_args()


def load_data():
    cfg = load_yaml("configs/data.yaml")
    paths = resolve_paths(cfg)

    processed = paths["processed"]
    artifacts = paths["artifacts"]

    # Build paths first; ensure_dir returns None, so don't overwrite the variables
    forecasts_dir = artifacts / "forecasts"
    tables_dir = artifacts / "tables"
    ensure_dir(forecasts_dir)
    ensure_dir(tables_dir)

    dm = pd.read_parquet(processed / "design_matrix.parquet")
    dm["week_start_date"] = pd.to_datetime(dm["week_start_date"])

    splits = pd.read_parquet(processed / "splits.parquet")
    splits["week_start_date"] = pd.to_datetime(splits["week_start_date"])
    splits["target_week"] = pd.to_datetime(splits["target_week"])

    return dm, splits, forecasts_dir, tables_dir


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    # keep engineered features, drop identifiers + raw target
    drop = {"geo_id", "geo_name", "week_start_date", "cases"}
    cols = [c for c in df.columns if c not in drop]
    # Safety: only numeric columns
    num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return sorted(num)


def metrics_frame(out: pd.DataFrame, source_name: str) -> pd.DataFrame:
    def _rmse(a: pd.DataFrame) -> float:
        return float(np.sqrt(np.mean((a["y_true"] - a["y_pred"]) ** 2)))

    def _mae(a: pd.DataFrame) -> float:
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


def main():
    args = build_argparser()
    dm, splits, forecasts_dir, tables_dir = load_data()

    # Which column names the fold?
    fold_col = "fold" if "fold" in splits.columns else "fold_id"

    # Label table: pull the TARGET label at (geo_id, target_week)
    label_map = (
        dm[["geo_id", "week_start_date", "cases"]]
        .rename(columns={"week_start_date": "target_week", "cases": "y_true"})
        .copy()
    )

    feat_cols = select_feature_columns(dm)
    id_cols = ["geo_id", "week_start_date"]

    # rows for the requested fold
    fold_rows = splits[splits[fold_col] == args.fold].copy()
    horizons = sorted(fold_rows["horizon"].unique())

    all_out: list[pd.DataFrame] = []

    # LightGBM: small-sample-friendly params
    params = dict(
        objective="tweedie",  # try "poisson" if strictly integer counts
        tweedie_variance_power=1.3,  # only used for tweedie
        metric=["l2"],
        learning_rate=0.05,
        n_estimators=400,
        num_leaves=15,
        max_depth=3,
        min_data_in_leaf=5,
        min_sum_hessian_in_leaf=0.0,
        min_gain_to_split=0.0,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l1=0.0,
        lambda_l2=0.0,
        force_col_wise=True,
        verbose=-1,
        random_state=42,
    )

    for h in horizons:
        S = fold_rows[fold_rows["horizon"] == h].copy()
        if S.empty:
            continue

        # Join feature rows at BASE week
        base = S.merge(dm[id_cols + feat_cols], on=id_cols, how="left")

        # Join labels at TARGET week
        base = base.merge(label_map, on=["geo_id", "target_week"], how="left")

        # Drop rows without label or with any NA features
        mask_ok = base["y_true"].notna()
        mask_ok &= base[feat_cols].notna().all(axis=1)
        base = base.loc[mask_ok].reset_index(drop=True)
        if base.empty:
            continue

        X = base[feat_cols]
        y = base["y_true"].astype(float)
        role = base["role"]

        X_tr, y_tr = X[role == "train"], y[role == "train"]
        X_va, y_va = X[role == "val"], y[role == "val"]

        model = lgb.LGBMRegressor(**params)

        # Silence eval logging portably (no verbose= kwarg)
        if len(X_va) > 0:
            callbacks = [lgb.early_stopping(50), lgb.log_evaluation(period=0)]
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="l2",
                callbacks=callbacks,
            )
        else:
            callbacks = [lgb.log_evaluation(period=0)]
            model.fit(X_tr, y_tr, callbacks=callbacks)

        # Predictions for both roles
        base["y_pred"] = model.predict(X)
        base["y_pred"] = base["y_pred"].clip(lower=0)  # no negative cases

        all_out.append(
            base[
                [
                    "geo_id",
                    "week_start_date",
                    "target_week",
                    "horizon",
                    "role",
                    "y_true",
                    "y_pred",
                ]
            ]
        )

    if not all_out:
        print(f"[gbm] nothing to save for {args.fold}")
        return

    out = pd.concat(all_out, ignore_index=True)

    # DEFENSIVE: remove accidental duplicates (shouldn’t exist, but safe)
    keys = ["geo_id", "target_week", "horizon", "role"]
    out = out.drop_duplicates(keys, keep="last").reset_index(drop=True)

    # Save forecasts
    fore_path = forecasts_dir / f"gbm_{args.fold}.parquet"
    out.to_parquet(fore_path, index=False)
    print(f"[gbm] forecasts → {fore_path} ({len(out):,} rows)")

    # Save metrics
    m = metrics_frame(out, f"metrics_gbm_{args.fold}.csv")
    met_path = tables_dir / f"metrics_gbm_{args.fold}.csv"
    m.to_csv(met_path, index=False)
    print(f"[gbm] metrics → {met_path}")


if __name__ == "__main__":
    main()
