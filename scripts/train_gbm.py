"""
Train LightGBM Tweedie baselines per (geo_id, horizon) using splits.parquet.

Inputs:
  - data/processed/design_matrix.parquet  (features at week t)
  - data/processed/splits.parquet         (maps (geo_id, t, h) -> role + target_week)

Outputs:
  - artifacts/forecasts/gbm_F{fold}.parquet
      [geo_id, week_start_date, target_week, horizon, role, y_true, y_pred]
  - artifacts/tables/metrics_gbm_F{fold}.csv
      [fold_id, geo_id, horizon, n, MAE, RMSE]
"""

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

DESIGN = Path("data/processed/design_matrix.parquet")
SPLITS = Path("data/processed/splits.parquet")
OUT_FORECASTS = Path("artifacts/forecasts")
OUT_TABLES = Path("artifacts/tables")


def build_xy(design: pd.DataFrame, splits: pd.DataFrame, geo: str, h: int):
    # feature rows: (geo, week_start_date=t)
    feats = design.loc[design["geo_id"] == geo].copy()

    # identify numeric feature columns (exclude identifiers)
    drop_cols = {"geo_id", "geo_name", "week_start_date"}
    feat_cols = [
        c
        for c in feats.columns
        if c not in drop_cols and pd.api.types.is_numeric_dtype(feats[c])
    ]

    # split rows for this (geo, horizon)
    pairs = splits[(splits["geo_id"] == geo) & (splits["horizon"] == h)][
        ["geo_id", "week_start_date", "target_week", "role", "fold_id", "horizon"]
    ].copy()

    # join X: (geo, t) → features
    X = pairs.merge(
        feats[["geo_id", "week_start_date"] + feat_cols],
        on=["geo_id", "week_start_date"],
        how="left",
    )

    # y: cases at target_week
    y_lookup = design[["geo_id", "week_start_date", "cases"]].rename(
        columns={"week_start_date": "target_week", "cases": "y_true"}
    )
    XY = X.merge(y_lookup, on=["geo_id", "target_week"], how="left")

    # drop rows without target or with any NA in essential features
    ok = XY.dropna(subset=["y_true"])
    # LightGBM handles NA in features; keep them
    return ok, feat_cols


def train_one(
    geo: str, h: int, df: pd.DataFrame, feat_cols: list[str]
) -> tuple[lgb.LGBMRegressor, dict]:
    # Basic LightGBM Tweedie (mirrors configs/models/gbm.yaml)
    params = dict(
        objective="tweedie",
        tweedie_variance_power=1.3,
        learning_rate=0.05,
        n_estimators=800,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        random_state=123,
    )
    model = lgb.LGBMRegressor(**params)

    tr = df[df["role"] == "train"]
    if tr.empty:
        return None, {"n_train": 0}

    Xtr = tr[feat_cols]
    ytr = tr["y_true"].astype(float)

    # optional early stopping if we have val rows
    va = df[df["role"] == "val"]
    if not va.empty:
        model.set_params(n_estimators=5000)
        model.fit(
            Xtr,
            ytr,
            eval_set=[(va[feat_cols], va["y_true"].astype(float))],
            eval_metric="l2",
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(period=100),
            ],
        )
    else:
        model.fit(Xtr, ytr)

    return model, {"n_train": len(tr), "n_val": len(va)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fold", default="F2", help="Fold ID from splits.parquet (e.g., F1 or F2)"
    )
    args = ap.parse_args()

    OUT_FORECASTS.mkdir(parents=True, exist_ok=True)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)

    design = pd.read_parquet(DESIGN)
    splits = pd.read_parquet(SPLITS)
    splits = splits[splits["fold_id"] == args.fold].copy()

    geos = sorted(design["geo_id"].unique())
    horizons = sorted(splits["horizon"].unique())

    all_preds = []
    metrics_rows = []

    for g in geos:
        for h in horizons:
            XY, feat_cols = build_xy(design, splits, g, h)
            if XY.empty:
                continue

            model, info = train_one(g, h, XY, feat_cols)
            if model is None:
                continue

            # predictions for both train+val rows
            XY = XY.copy()
            XY["y_pred"] = model.predict(XY[feat_cols])

            # collect forecasts
            all_preds.append(
                XY[
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

            # metrics by role (simple MAE/RMSE)
            for role in ["train", "val"]:
                chunk = XY[XY["role"] == role]
                if chunk.empty:
                    continue
                y = chunk["y_true"].astype(float).to_numpy()
                p = np.clip(chunk["y_pred"].astype(float).to_numpy(), 0.0, None)
                mae = np.mean(np.abs(y - p))
                rmse = float(np.sqrt(np.mean((y - p) ** 2)))
                metrics_rows.append(
                    {
                        "fold_id": args.fold,
                        "geo_id": g,
                        "horizon": h,
                        "role": role,
                        "n": len(chunk),
                        "MAE": mae,
                        "RMSE": rmse,
                    }
                )

    if not all_preds:
        raise SystemExit("No predictions produced. Check splits/design inputs.")

    preds = pd.concat(all_preds, ignore_index=True)
    out_fore = OUT_FORECASTS / f"gbm_{args.fold}.parquet"
    preds.to_parquet(out_fore, index=False)
    print(f"[gbm] forecasts → {out_fore} ({len(preds):,} rows)")

    metr = pd.DataFrame(metrics_rows)
    out_metrics = OUT_TABLES / f"metrics_gbm_{args.fold}.csv"
    metr.sort_values(["role", "geo_id", "horizon"]).to_csv(out_metrics, index=False)
    print(f"[gbm] metrics → {out_metrics}")


if __name__ == "__main__":
    main()
