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
from typing import List

import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm.basic import LightGBMError

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


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    drop = {"geo_id", "geo_name", "week_start_date", "cases"}
    cols = [c for c in df.columns if c not in drop]
    num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return sorted(num)


def metrics_frame(out: pd.DataFrame, source_name: str) -> pd.DataFrame:
    def _rmse(a: pd.DataFrame) -> float:
        return float(np.sqrt(np.mean((a["y_true"] - a["y_pred"]) ** 2)))

    def _mae(a: pd.DataFrame) -> float:
        return float(np.mean(np.abs(a["y_true"] - a["y_pred"])))

    rows = []

    for role, g in out.groupby("role"):
        rows.append(dict(source=source_name, role=role, horizon=np.nan, MAE=_mae(g), RMSE=_rmse(g)))

    for (role, h), g in out.groupby(["role", "horizon"]):
        rows.append(dict(source=source_name, role=role, horizon=int(h), MAE=_mae(g), RMSE=_rmse(g)))

    return pd.DataFrame(rows)


def main():
    args = build_argparser()
    dm, splits, forecasts_dir, tables_dir = load_data()

    fold_col = "fold" if "fold" in splits.columns else "fold_id"

    # labels at target week
    label_map = (
        dm[["geo_id", "week_start_date", "cases"]]
        .rename(columns={"week_start_date": "target_week", "cases": "y_true"})
        .copy()
    )

    feat_cols = select_feature_columns(dm)
    id_cols = ["geo_id", "week_start_date"]

    fold_rows = splits[splits[fold_col] == args.fold].copy()
    horizons = sorted(fold_rows["horizon"].unique())

    all_out: list[pd.DataFrame] = []

    base_params = dict(
        objective="tweedie",
        tweedie_variance_power=1.3,
        metric=["l2"],
        learning_rate=0.05,
        n_estimators=400,
        num_leaves=15,
        max_depth=3,
        min_data_in_leaf=5,
        min_sum_hessian_in_leaf=1e-3,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l1=0.0,
        lambda_l2=0.0,
        force_col_wise=True,
        deterministic=True,
        verbose=-1,
        random_state=42,
    )

    for h in horizons:
        S = fold_rows[fold_rows["horizon"] == h].copy()
        if S.empty:
            continue

        base = S.merge(dm[id_cols + feat_cols], on=id_cols, how="left")
        base = base.merge(label_map, on=["geo_id", "target_week"], how="left")

        # drop rows without label or with any NA features
        ok = base["y_true"].notna()
        ok &= base[feat_cols].notna().all(axis=1)
        base = base.loc[ok].reset_index(drop=True)
        if base.empty:
            continue

        X = base[feat_cols]
        y = base["y_true"].astype(float)
        role = base["role"]

        X_tr, y_tr = X[role == "train"], y[role == "train"]
        X_va, y_va = X[role == "val"], y[role == "val"]

        n_tr = int(len(X_tr))
        n_va = int(len(X_va))

        # If too few training rows, fall back to constant baseline
        if n_tr < 20:
            y_bar = float(y_tr.mean()) if n_tr > 0 else 0.0
            base["y_pred"] = y_bar
            all_out.append(
                base[["geo_id", "week_start_date", "target_week", "horizon", "role", "y_true", "y_pred"]]
            )
            print(f"[gbm] fold={args.fold} h={h}: fallback mean predictor (n_train={n_tr}).")
            continue

        # Safer per-horizon params for tiny samples
        params = dict(base_params)
        params["num_leaves"] = int(max(7, min(params["num_leaves"], n_tr // 3)))
        params["min_data_in_leaf"] = int(max(10, n_tr // 20))
        params["bagging_fraction"] = float(min(1.0, params["bagging_fraction"]))
        params["feature_fraction"] = float(min(1.0, params["feature_fraction"]))

        model = lgb.LGBMRegressor(**params)

        try:
            if n_va > 0:
                callbacks = [lgb.early_stopping(50), lgb.log_evaluation(period=0)]
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="l2", callbacks=callbacks)
            else:
                callbacks = [lgb.log_evaluation(period=0)]
                model.fit(X_tr, y_tr, callbacks=callbacks)

            base["y_pred"] = model.predict(X)
        except LightGBMError as e:
            # Graceful fallback
            y_bar = float(y_tr.mean())
            base["y_pred"] = y_bar
            print(f"[gbm] WARN fold={args.fold} h={h}: LightGBMError -> fallback to mean ({e}).")

        base["y_pred"] = base["y_pred"].clip(lower=0)

        all_out.append(
            base[["geo_id", "week_start_date", "target_week", "horizon", "role", "y_true", "y_pred"]]
        )

    if not all_out:
        print(f"[gbm] nothing to save for {args.fold}")
        return

    out = pd.concat(all_out, ignore_index=True).drop_duplicates(
        ["geo_id", "target_week", "horizon", "role"], keep="last"
    )

    fore_path = forecasts_dir / f"gbm_{args.fold}.parquet"
    out.to_parquet(fore_path, index=False)
    print(f"[gbm] forecasts â†’ {fore_path} ({len(out):,} rows)")

    m = metrics_frame(out, f"metrics_gbm_{args.fold}.csv")
    met_path = tables_dir / f"metrics_gbm_{args.fold}.csv"
    m.to_csv(met_path, index=False)
    print(f"[gbm] metrics â†’ {met_path}")


if __name__ == "__main__":
    main()

