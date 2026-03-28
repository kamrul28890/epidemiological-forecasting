from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, TweedieRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dengue.features.calendar import add_calendar
from dengue.split.rolling_origin import build_split_pairs
from dengue.utils.io import ensure_dir, load_yaml, resolve_paths

WEATHER_COLUMNS = [
    "tmean",
    "humidity",
    "rain",
    "dewpoint",
    "wind10",
    "wind100",
    "winddir10",
    "gust10",
    "soil_temp_0_7cm",
    "soil_moist_0_7cm",
]


@dataclass
class RunArtifacts:
    run_id: str
    run_dir: Path
    data_dir: Path
    forecasts_dir: Path
    metrics_dir: Path
    summaries_dir: Path
    log_path: Path


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        ensure_dir(path.parent)

    def log(self, event: str, **payload: Any) -> None:
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        summary = payload.get("message")
        if summary:
            print(summary)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


def _write_markdown(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def _metric_frame(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n_rows=("y_true", "size"),
            mean_y=("y_true", "mean"),
            mean_pred=("y_pred", "mean"),
            bias=("error", "mean"),
            MAE=("abs_error", "mean"),
            mse=("sq_error", "mean"),
        )
        .reset_index()
    )
    grouped["RMSE"] = np.sqrt(grouped["mse"])
    return grouped.drop(columns=["mse"])


def _macro_average(
    df: pd.DataFrame,
    group_cols: list[str],
    mae_col: str = "MAE",
    rmse_col: str = "RMSE",
    bias_col: str = "bias",
) -> pd.DataFrame:
    out = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n_groups=(mae_col, "size"),
            MAE_macro=(mae_col, "mean"),
            RMSE_macro=(rmse_col, "mean"),
            bias_macro=(bias_col, "mean"),
        )
        .reset_index()
    )
    return out


def _leaderboard_for_group(
    overall_macro: pd.DataFrame, split_group: str
) -> pd.DataFrame:
    return (
        overall_macro.loc[
            (overall_macro["role"] == "val")
            & (overall_macro["split_group"] == split_group)
        ]
        .sort_values(["track", "MAE_macro", "RMSE_macro", "model"])
        .reset_index(drop=True)
    )


def _best_by_horizon_for_group(
    by_fold_horizon_macro: pd.DataFrame, split_group: str
) -> pd.DataFrame:
    frame = by_fold_horizon_macro.loc[
        (by_fold_horizon_macro["role"] == "val")
        & (by_fold_horizon_macro["split_group"] == split_group)
    ].copy()
    if frame.empty:
        return frame
    grouped = (
        frame.groupby(["track", "model", "horizon"], dropna=False)[
            ["MAE_macro", "RMSE_macro"]
        ]
        .mean()
        .reset_index()
    )
    return grouped.sort_values(
        ["track", "horizon", "MAE_macro", "RMSE_macro", "model"]
    ).reset_index(drop=True)


def load_runtime_config(
    config_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    data_cfg = load_yaml("configs/data.yaml")
    split_cfg = load_yaml("configs/split.yaml")
    modeling_cfg = load_yaml(config_path)
    return data_cfg, split_cfg, modeling_cfg


def init_run_artifacts(run_name_prefix: str, artifacts_root: Path) -> RunArtifacts:
    run_id = f"{run_name_prefix}_{_utc_stamp()}"
    run_dir = artifacts_root / "runs" / run_id
    data_dir = run_dir / "data"
    forecasts_dir = run_dir / "forecasts"
    metrics_dir = run_dir / "metrics"
    summaries_dir = run_dir / "summaries"
    for path in (run_dir, data_dir, forecasts_dir, metrics_dir, summaries_dir):
        ensure_dir(path)
    return RunArtifacts(
        run_id=run_id,
        run_dir=run_dir,
        data_dir=data_dir,
        forecasts_dir=forecasts_dir,
        metrics_dir=metrics_dir,
        summaries_dir=summaries_dir,
        log_path=run_dir / "run_log.jsonl",
    )


def load_panel(paths: dict[str, Path]) -> pd.DataFrame:
    panel = pd.read_parquet(paths["interim"] / "panel_raw.parquet")
    panel["week_start_date"] = pd.to_datetime(panel["week_start_date"])
    return panel.sort_values(["geo_id", "week_start_date"]).reset_index(drop=True)


def fix_hierarchy(
    panel: pd.DataFrame,
    hierarchy_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parent_geo = hierarchy_cfg["parent_geo"]
    child_geo = hierarchy_cfg["child_geo"]
    residual_geo_id = hierarchy_cfg["residual_geo_id"]
    residual_geo_name = hierarchy_cfg["residual_geo_name"]

    parent = panel.loc[panel["geo_id"] == parent_geo].copy()
    child = panel.loc[panel["geo_id"] == child_geo, ["week_start_date", "cases"]].copy()

    merged = parent.merge(
        child, on="week_start_date", how="left", suffixes=("_parent", "_child")
    )
    merged["cases_child"] = merged["cases_child"].fillna(0)
    merged["residual_cases_raw"] = merged["cases_parent"] - merged["cases_child"]
    merged["cases"] = merged["residual_cases_raw"].clip(lower=0).round().astype(int)
    merged["geo_id"] = residual_geo_id
    merged["geo_name"] = residual_geo_name
    merged["covariate_proxy_source"] = parent_geo
    merged["covariate_is_proxy"] = 1

    keep_cols = ["geo_id", "geo_name", "week_start_date", "cases"] + [
        c for c in WEATHER_COLUMNS if c in merged.columns
    ]
    residual = merged[
        keep_cols + ["covariate_proxy_source", "covariate_is_proxy"]
    ].copy()

    fixed = panel.loc[panel["geo_id"] != parent_geo].copy()
    fixed["covariate_proxy_source"] = fixed["geo_id"]
    fixed["covariate_is_proxy"] = 0
    fixed = pd.concat([fixed, residual], ignore_index=True)
    fixed = fixed.sort_values(["geo_id", "week_start_date"]).reset_index(drop=True)

    closure = (
        parent[["week_start_date", "cases"]]
        .rename(columns={"cases": "parent_cases"})
        .merge(
            panel.loc[
                panel["geo_id"] == child_geo, ["week_start_date", "cases"]
            ].rename(columns={"cases": "child_cases"}),
            on="week_start_date",
            how="left",
        )
        .merge(
            residual[["week_start_date", "cases"]].rename(
                columns={"cases": "residual_cases"}
            ),
            on="week_start_date",
            how="left",
        )
    )
    closure = closure.fillna(0)
    closure["closure_error"] = closure["parent_cases"] - (
        closure["child_cases"] + closure["residual_cases"]
    )

    summary = pd.DataFrame(
        [
            {
                "parent_geo": parent_geo,
                "child_geo": child_geo,
                "residual_geo_id": residual_geo_id,
                "weeks": int(len(closure)),
                "negative_raw_residual_weeks": int(
                    (merged["residual_cases_raw"] < 0).sum()
                ),
                "max_abs_closure_error": float(closure["closure_error"].abs().max()),
                "parent_total_cases": float(parent["cases"].sum()),
                "child_total_cases": float(
                    panel.loc[panel["geo_id"] == child_geo, "cases"].sum()
                ),
                "residual_total_cases": float(residual["cases"].sum()),
                "fixed_geographies": int(fixed["geo_id"].nunique()),
            }
        ]
    )
    return fixed, summary


def add_case_features(
    panel: pd.DataFrame,
    case_lags: list[int],
    case_roll_mean_windows: list[int],
    case_roll_sum_windows: list[int],
    case_roll_std_windows: list[int],
    growth_lookbacks: list[int],
    outbreak_recent_window: int,
    outbreak_baseline_window: int,
    outbreak_threshold_ratio: float,
) -> pd.DataFrame:
    out = panel.sort_values(["geo_id", "week_start_date"]).copy()
    grouped = out.groupby("geo_id")["cases"]

    for lag in case_lags:
        out[f"cases_lag_{lag}"] = grouped.shift(lag)

    for window in case_roll_mean_windows:
        out[f"cases_rollmean_{window}"] = grouped.transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=w).mean()
        )

    for window in case_roll_sum_windows:
        out[f"cases_recent_sum_{window}"] = grouped.transform(
            lambda s, w=window: s.rolling(w, min_periods=w).sum()
        )

    for window in case_roll_std_windows:
        out[f"cases_recent_std_{window}"] = grouped.transform(
            lambda s, w=window: s.rolling(w, min_periods=w).std()
        )

    out["cases_log1p"] = np.log1p(out["cases"])
    out["cases_nonzero_recent_4"] = grouped.transform(
        lambda s: s.gt(0).astype(float).rolling(4, min_periods=4).sum()
    )
    out["cases_nonzero_recent_8"] = grouped.transform(
        lambda s: s.gt(0).astype(float).rolling(8, min_periods=8).sum()
    )

    for lookback in growth_lookbacks:
        lagged = grouped.shift(lookback)
        out[f"cases_growth_logdiff_{lookback}"] = np.log1p(out["cases"]) - np.log1p(
            lagged
        )

    out["cases_accel_logdiff_1"] = out.get("cases_growth_logdiff_1") - (
        np.log1p(grouped.shift(1)) - np.log1p(grouped.shift(2))
    )

    recent_sum_col = f"cases_recent_sum_{outbreak_recent_window}"
    baseline_mean_col = f"cases_rollmean_{outbreak_baseline_window}"
    if recent_sum_col in out.columns and baseline_mean_col in out.columns:
        baseline_total = out[baseline_mean_col] * float(outbreak_recent_window)
        out["cases_outbreak_index_4_8"] = (out[recent_sum_col] + 1.0) / (
            baseline_total + 1.0
        )
        out["cases_outbreak_flag_4_8"] = (
            out["cases_outbreak_index_4_8"] >= float(outbreak_threshold_ratio)
        ).astype(float)

    def _add_run_features(group: pd.DataFrame) -> pd.DataFrame:
        s = group["cases"].astype(float)
        is_zero = s.eq(0)
        switch_id = is_zero.ne(is_zero.shift(fill_value=False)).cumsum()
        zero_run = is_zero.groupby(switch_id).cumsum().where(is_zero, 0).astype(float)
        last_nonzero_step = np.where(s.gt(0), np.arange(len(s)), np.nan)
        last_nonzero_step = pd.Series(last_nonzero_step, index=group.index).ffill()
        weeks_since_nonzero = np.arange(len(s)) - last_nonzero_step.to_numpy(
            dtype=float
        )
        weeks_since_nonzero = np.where(
            np.isnan(weeks_since_nonzero), np.arange(len(s)) + 1, weeks_since_nonzero
        )
        group["cases_zero_run_length"] = zero_run.to_numpy(dtype=float)
        group["cases_weeks_since_nonzero"] = weeks_since_nonzero.astype(float)
        return group

    out = out.groupby("geo_id", group_keys=False).apply(_add_run_features)
    return out


def add_climate_features(
    panel: pd.DataFrame,
    climate_vars: list[str],
    climate_lags: list[int],
    climate_roll_mean_windows: list[int],
    climate_roll_sum_windows: list[int],
) -> pd.DataFrame:
    out = panel.sort_values(["geo_id", "week_start_date"]).copy()
    for var in climate_vars:
        if var not in out.columns:
            continue
        out[var] = pd.to_numeric(out[var], errors="coerce")
        grouped = out.groupby("geo_id")[var]
        for lag in climate_lags:
            out[f"{var}_lag_{lag}"] = grouped.shift(lag)
        for window in climate_roll_mean_windows:
            out[f"{var}_rollmean_{window}"] = grouped.transform(
                lambda s, w=window: s.shift(1).rolling(w, min_periods=w).mean()
            )
        if var == "rain":
            for window in climate_roll_sum_windows:
                out[f"{var}_rollsum_{window}"] = grouped.transform(
                    lambda s, w=window: s.shift(1).rolling(w, min_periods=w).sum()
                )
    return out


def build_track_feature_panel(
    panel: pd.DataFrame,
    track_name: str,
    track_cfg: dict[str, Any],
    feature_cfg: dict[str, Any],
) -> pd.DataFrame:
    out = add_case_features(
        panel=panel,
        case_lags=list(feature_cfg["case_lags"]),
        case_roll_mean_windows=list(feature_cfg["case_roll_mean_windows"]),
        case_roll_sum_windows=list(feature_cfg["case_roll_sum_windows"]),
        case_roll_std_windows=list(feature_cfg["case_roll_std_windows"]),
        growth_lookbacks=list(feature_cfg["growth_lookbacks"]),
        outbreak_recent_window=int(feature_cfg["outbreak_recent_window"]),
        outbreak_baseline_window=int(feature_cfg["outbreak_baseline_window"]),
        outbreak_threshold_ratio=float(feature_cfg["outbreak_threshold_ratio"]),
    )
    out = add_calendar(out)
    if bool(track_cfg["include_weather"]):
        out = add_climate_features(
            panel=out,
            climate_vars=list(feature_cfg["climate_vars"]),
            climate_lags=list(feature_cfg["climate_lags"]),
            climate_roll_mean_windows=list(feature_cfg["climate_roll_mean_windows"]),
            climate_roll_sum_windows=list(feature_cfg["climate_roll_sum_windows"]),
        )

    required_cols = [f"cases_lag_{lag}" for lag in feature_cfg["case_lags"]]
    if bool(track_cfg["include_weather"]):
        for var in feature_cfg["climate_vars"]:
            required_cols.extend(
                [
                    f"{var}_lag_{lag}"
                    for lag in feature_cfg["climate_lags"]
                    if f"{var}_lag_{lag}" in out.columns
                ]
            )

    required_cols = [c for c in required_cols if c in out.columns]
    feature_panel = out.dropna(subset=required_cols).reset_index(drop=True)
    feature_panel["track"] = track_name
    return feature_panel


def build_track_splits(
    feature_panel: pd.DataFrame, track_cfg: dict[str, Any], split_cfg: dict[str, Any]
) -> pd.DataFrame:
    horizons = list(track_cfg["horizons"])
    fold_frames = []
    for idx, fold_cfg in enumerate(split_cfg["folds"], start=1):
        fold_id = fold_cfg.get("name") or f"F{idx}"
        pairs = build_split_pairs(
            design=feature_panel,
            horizons=horizons,
            train_end=pd.to_datetime(fold_cfg["train_end"]),
            val_end=pd.to_datetime(fold_cfg["val_end"]),
            fold_id=fold_id,
        )
        pairs["split_group"] = str(fold_cfg.get("group", "development"))
        fold_frames.append(pairs)
    return pd.concat(fold_frames, ignore_index=True).sort_values(
        ["split_group", "fold_id", "geo_id", "week_start_date", "horizon"]
    )


def build_supervised_frame(
    feature_panel: pd.DataFrame, splits: pd.DataFrame, panel: pd.DataFrame
) -> pd.DataFrame:
    target_map = (
        panel[["geo_id", "week_start_date", "cases"]]
        .rename(columns={"week_start_date": "target_week", "cases": "y_true"})
        .copy()
    )
    supervised = splits.merge(
        feature_panel,
        on=["geo_id", "week_start_date"],
        how="left",
    )
    supervised = supervised.merge(target_map, on=["geo_id", "target_week"], how="left")
    supervised["target_week"] = pd.to_datetime(supervised["target_week"])
    supervised = supervised.dropna(subset=["y_true"]).reset_index(drop=True)
    supervised["y_true"] = pd.to_numeric(supervised["y_true"], errors="coerce").fillna(
        0.0
    )
    return supervised


def select_feature_columns(
    supervised: pd.DataFrame, include_weather: bool
) -> list[str]:
    protected = {
        "geo_id",
        "geo_name",
        "week_start_date",
        "target_week",
        "role",
        "fold_id",
        "horizon",
        "track",
        "split_group",
        "y_true",
        "covariate_proxy_source",
        "covariate_is_proxy",
    }
    feature_cols = []
    for col in supervised.columns:
        if col in protected:
            continue
        if not pd.api.types.is_numeric_dtype(supervised[col]):
            continue
        if col in WEATHER_COLUMNS:
            continue
        if not include_weather and any(
            col.startswith(f"{w}_") for w in WEATHER_COLUMNS
        ):
            continue
        feature_cols.append(col)
    return sorted(feature_cols)


def _history_lookup(panel: pd.DataFrame) -> pd.Series:
    lookup = panel[["geo_id", "week_start_date", "cases"]].copy()
    lookup["week_start_date"] = pd.to_datetime(lookup["week_start_date"])
    return lookup.set_index(["geo_id", "week_start_date"])["cases"]


def _predict_seasonal_naive(supervised: pd.DataFrame, history: pd.Series) -> np.ndarray:
    preds: list[float] = []
    for row in supervised.itertuples(index=False):
        seasonal_week = pd.Timestamp(row.target_week) - pd.Timedelta(days=364)
        key = (row.geo_id, seasonal_week)
        if key in history.index:
            preds.append(float(history.loc[key]))
        else:
            preds.append(float(row.cases))
    return np.asarray(preds, dtype=float)


def _build_preprocessor(numeric_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "geo",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["geo_id"],
            ),
        ],
        remainder="drop",
    )


def _fit_predict_model(
    model_name: str,
    train_df: pd.DataFrame,
    score_df: pd.DataFrame,
    feature_cols: list[str],
    training_cfg: dict[str, Any],
    train_target: np.ndarray | None = None,
) -> np.ndarray:
    X_train = train_df[["geo_id"] + feature_cols].copy()
    X_score = score_df[["geo_id"] + feature_cols].copy()
    y_train = (
        np.asarray(train_target, dtype=float)
        if train_target is not None
        else train_df["y_true"].astype(float).to_numpy()
    )

    preprocessor = _build_preprocessor(feature_cols)

    if model_name == "ar_tweedie_global":
        estimator = TweedieRegressor(
            power=float(training_cfg["tweedie_power"]),
            alpha=float(training_cfg["tweedie_alpha"]),
            link="log",
            max_iter=1000,
        )
    elif model_name == "residual_elastic_net_global":
        estimator = ElasticNet(
            alpha=float(training_cfg["elastic_net_alpha"]),
            l1_ratio=float(training_cfg["elastic_net_l1_ratio"]),
            max_iter=int(training_cfg["elastic_net_max_iter"]),
            selection="cyclic",
        )
    elif model_name == "panel_hist_gbm":
        estimator = HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=float(training_cfg["hist_gbm_learning_rate"]),
            max_depth=int(training_cfg["hist_gbm_max_depth"]),
            max_iter=int(training_cfg["hist_gbm_max_iter"]),
            min_samples_leaf=int(training_cfg["hist_gbm_min_samples_leaf"]),
            random_state=int(training_cfg["random_state"]),
        )
    elif model_name == "residual_hist_gbm":
        estimator = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=float(training_cfg["residual_hist_gbm_learning_rate"]),
            max_depth=int(training_cfg["residual_hist_gbm_max_depth"]),
            max_iter=int(training_cfg["residual_hist_gbm_max_iter"]),
            min_samples_leaf=int(training_cfg["residual_hist_gbm_min_samples_leaf"]),
            random_state=int(training_cfg["random_state"]),
        )
    elif model_name == "residual_random_forest":
        estimator = RandomForestRegressor(
            n_estimators=int(training_cfg["rf_n_estimators"]),
            max_depth=int(training_cfg["rf_max_depth"]),
            min_samples_leaf=int(training_cfg["rf_min_samples_leaf"]),
            random_state=int(training_cfg["random_state"]),
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported trainable model: {model_name}")

    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_score)
    return np.asarray(y_pred, dtype=float)


def _prediction_instability_reason(
    y_pred: np.ndarray, train_y: np.ndarray
) -> str | None:
    if y_pred.size == 0:
        return None
    if not np.isfinite(y_pred).all():
        return "non_finite_predictions"

    train_max = max(float(np.max(train_y)), 1.0)
    pred_max = float(np.max(y_pred))
    pred_p99 = float(np.quantile(y_pred, 0.99))

    if pred_max > train_max * 25.0 or pred_p99 > train_max * 10.0:
        return (
            "prediction_explosion:"
            f"pred_max={pred_max:.3f},pred_p99={pred_p99:.3f},train_max={train_max:.3f}"
        )
    return None


def _guardrail_model_names(training_cfg: dict[str, Any]) -> set[str]:
    configured = training_cfg.get("low_origin_guardrail_models")
    if configured:
        return {str(name) for name in configured}
    return {
        "ar_tweedie_global",
        "residual_elastic_net_global",
        "panel_hist_gbm",
        "residual_hist_gbm",
        "residual_random_forest",
        "conservative_stack",
    }


def _low_origin_mask(frame: pd.DataFrame, training_cfg: dict[str, Any]) -> pd.Series:
    mask = (
        frame["cases"]
        .astype(float)
        .le(float(training_cfg.get("low_origin_max_cases", 5)))
    )
    if "cases_zero_run_length" in frame.columns:
        mask = mask | frame["cases_zero_run_length"].astype(float).ge(
            float(training_cfg.get("low_origin_zero_run_threshold", 2))
        )
    if "cases_weeks_since_nonzero" in frame.columns:
        mask = mask | frame["cases_weeks_since_nonzero"].astype(float).ge(
            float(training_cfg.get("low_origin_weeks_since_nonzero_threshold", 2))
        )
    if "cases_nonzero_recent_4" in frame.columns:
        mask = mask | frame["cases_nonzero_recent_4"].astype(float).le(
            float(training_cfg.get("low_origin_recent_nonzero_max", 1))
        )
    return mask.fillna(False)


def _apply_low_origin_guardrail(
    model_name: str,
    train_df: pd.DataFrame,
    score_df: pd.DataFrame,
    y_pred: np.ndarray,
    training_cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any] | None]:
    if model_name not in _guardrail_model_names(training_cfg):
        return np.asarray(y_pred, dtype=float), None
    if train_df.empty or score_df.empty:
        return np.asarray(y_pred, dtype=float), None

    low_score_mask = _low_origin_mask(score_df, training_cfg)
    if not bool(low_score_mask.any()):
        return np.asarray(y_pred, dtype=float), None

    low_train = train_df.loc[_low_origin_mask(train_df, training_cfg)].copy()
    if low_train.empty:
        low_train = train_df.copy()
    low_train["future_nonzero"] = low_train["y_true"].astype(float).gt(0).astype(float)
    low_train["positive_delta"] = np.maximum(
        low_train["y_true"].astype(float).to_numpy(dtype=float)
        - low_train["cases"].astype(float).to_numpy(dtype=float),
        0.0,
    )

    cap_quantile = float(training_cfg.get("low_origin_cap_quantile", 0.9))
    min_group_rows = int(training_cfg.get("low_origin_min_group_rows", 12))
    sparse_zero_share_threshold = float(
        training_cfg.get("low_origin_sparse_geo_zero_share_threshold", 0.35)
    )
    sparse_prob_multiplier = float(
        training_cfg.get("low_origin_sparse_geo_prob_multiplier", 0.75)
    )

    global_p_nonzero = float(low_train["future_nonzero"].mean())
    global_y_cap = float(low_train["y_true"].quantile(cap_quantile))
    global_delta_cap = float(
        pd.Series(low_train["positive_delta"]).quantile(cap_quantile)
    )

    zero_share_by_geo = (
        train_df.groupby("geo_id", dropna=False)["cases"]
        .apply(lambda s: float(s.astype(float).le(0).mean()))
        .to_dict()
    )
    geo_stats = (
        low_train.groupby("geo_id", dropna=False)
        .agg(
            n_rows=("y_true", "size"),
            p_nonzero=("future_nonzero", "mean"),
            y_cap=("y_true", lambda s: float(s.quantile(cap_quantile))),
            delta_cap=(
                "positive_delta",
                lambda s: float(pd.Series(s).quantile(cap_quantile)),
            ),
        )
        .reset_index()
    )
    geo_lookup = {
        str(row.geo_id): {
            "n_rows": int(row.n_rows),
            "p_nonzero": float(row.p_nonzero),
            "y_cap": float(row.y_cap),
            "delta_cap": float(row.delta_cap),
        }
        for row in geo_stats.itertuples(index=False)
    }

    adjusted = np.asarray(y_pred, dtype=float).copy()
    original = adjusted.copy()
    applied_rows = 0
    for pos, row in enumerate(score_df.loc[low_score_mask].itertuples(index=False)):
        idx = score_df.loc[low_score_mask].index[pos]
        row_loc = score_df.index.get_loc(idx)
        base_level = float(getattr(row, "cases"))
        pred_level = float(adjusted[row_loc])
        if pred_level <= base_level:
            continue

        geo_id = str(getattr(row, "geo_id"))
        stats = geo_lookup.get(geo_id)
        if stats is not None and int(stats["n_rows"]) >= min_group_rows:
            p_nonzero = float(stats["p_nonzero"])
            y_cap = float(stats["y_cap"])
            delta_cap = float(stats["delta_cap"])
        else:
            p_nonzero = float(global_p_nonzero)
            y_cap = float(global_y_cap)
            delta_cap = float(global_delta_cap)
            if zero_share_by_geo.get(geo_id, 0.0) >= sparse_zero_share_threshold:
                p_nonzero *= sparse_prob_multiplier

        p_nonzero = min(max(p_nonzero, 0.0), 1.0)
        cap_level = max(base_level, min(y_cap, base_level + max(delta_cap, 0.0)))
        shrunk_level = base_level + p_nonzero * (pred_level - base_level)
        adjusted[row_loc] = min(max(shrunk_level, 0.0), cap_level)
        applied_rows += 1

    if applied_rows == 0:
        return adjusted, None

    info = {
        "guardrail_strategy": "low_origin_hurdle_shrink",
        "guardrail_rows": int(applied_rows),
        "guardrail_low_origin_rows": int(low_score_mask.sum()),
        "guardrail_avg_shrink": float((original - adjusted).clip(min=0.0).mean()),
        "guardrail_global_p_nonzero": float(global_p_nonzero),
        "guardrail_global_y_cap": float(global_y_cap),
        "guardrail_global_delta_cap": float(global_delta_cap),
    }
    return adjusted, info


def _stack_identity_columns(predictions: pd.DataFrame) -> list[str]:
    preferred = [
        "track",
        "split_group",
        "geo_id",
        "geo_name",
        "week_start_date",
        "target_week",
        "role",
        "fold_id",
        "horizon",
        "y_true",
        "cases",
        "cases_nonzero_recent_4",
        "cases_zero_run_length",
        "cases_weeks_since_nonzero",
        "cases_outbreak_flag_4_8",
    ]
    return [col for col in preferred if col in predictions.columns]


def _resolve_conservative_stack_settings(
    track_cfg: dict[str, Any],
    available_models: list[str],
) -> dict[str, Any]:
    ensemble_cfg = dict(track_cfg.get("ensemble", {}))
    configured = [
        str(model) for model in ensemble_cfg.get("conservative_stack_base_models", [])
    ]
    default_order = [
        "persistence",
        "residual_elastic_net_global",
        "residual_random_forest",
        "residual_hist_gbm",
        "ar_tweedie_global",
        "panel_hist_gbm",
    ]
    base_models = [model for model in configured if model in available_models]
    if not base_models:
        base_models = [model for model in default_order if model in available_models]
    return {
        "base_models": base_models,
        "persistence_floor": float(ensemble_cfg.get("persistence_floor", 0.0)),
        "inverse_mae_power": float(ensemble_cfg.get("inverse_mae_power", 1.0)),
    }


def _conservative_weight_vector(
    base_models: list[str],
    mae_by_model: dict[str, float],
    persistence_floor: float,
    inverse_mae_power: float,
) -> dict[str, float]:
    epsilon = 1e-6
    raw_scores: dict[str, float] = {}
    for model_name in base_models:
        mae = float(mae_by_model.get(model_name, np.nan))
        if not np.isfinite(mae):
            raw_scores[model_name] = 0.0
            continue
        raw_scores[model_name] = 1.0 / max(mae, epsilon) ** max(
            inverse_mae_power, epsilon
        )

    if sum(raw_scores.values()) <= 0:
        raw_scores = {model_name: 1.0 for model_name in base_models}

    total = float(sum(raw_scores.values()))
    weights = {model_name: score / total for model_name, score in raw_scores.items()}

    if "persistence" in weights and persistence_floor > 0:
        persistence_floor = min(max(persistence_floor, 0.0), 1.0)
        if len(base_models) == 1:
            weights = {"persistence": 1.0}
        elif weights["persistence"] < persistence_floor:
            other_models = [
                model_name for model_name in base_models if model_name != "persistence"
            ]
            other_total = float(sum(weights[model_name] for model_name in other_models))
            target_other_total = max(1.0 - persistence_floor, 0.0)
            if other_total <= 0 or target_other_total <= 0:
                weights = {model_name: 0.0 for model_name in base_models}
                weights["persistence"] = 1.0
            else:
                scale = target_other_total / other_total
                for model_name in other_models:
                    weights[model_name] *= scale
                weights["persistence"] = persistence_floor

    norm = float(sum(weights.values()))
    if norm <= 0:
        return {model_name: 1.0 / len(base_models) for model_name in base_models}
    return {model_name: weight / norm for model_name, weight in weights.items()}


def _build_conservative_stack_predictions(
    track_name: str,
    base_predictions: pd.DataFrame,
    track_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    available_models = sorted(base_predictions["model"].unique().tolist())
    stack_settings = _resolve_conservative_stack_settings(track_cfg, available_models)
    base_models = stack_settings["base_models"]
    if len(base_models) < 2:
        raise ValueError(
            f"conservative_stack requires at least two available base models, got {base_models}"
        )

    working = base_predictions.loc[base_predictions["model"].isin(base_models)].copy()
    id_cols = _stack_identity_columns(working)
    wide = working.pivot_table(
        index=id_cols, columns="model", values="y_pred", aggfunc="first"
    ).reset_index()
    val_rows = working.loc[working["role"] == "val"].copy()
    dev_val_rows = val_rows.loc[val_rows["split_group"] == "development"].copy()

    stack_frames: list[pd.DataFrame] = []
    fit_notes: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []

    for fold_id in sorted(wide["fold_id"].unique().tolist()):
        fold_wide = wide.loc[wide["fold_id"] == fold_id].copy()
        for horizon in sorted(fold_wide["horizon"].unique().tolist()):
            perf_rows = dev_val_rows.loc[
                (dev_val_rows["fold_id"] != fold_id)
                & (dev_val_rows["horizon"] == horizon)
                & (dev_val_rows["model"].isin(base_models))
            ].copy()
            if perf_rows.empty:
                perf_rows = dev_val_rows.loc[
                    (dev_val_rows["horizon"] == horizon)
                    & (dev_val_rows["model"].isin(base_models))
                ].copy()

            if perf_rows.empty:
                mae_lookup = {model_name: np.nan for model_name in base_models}
            else:
                perf_rows["abs_error"] = (
                    perf_rows["y_true"] - perf_rows["y_pred"]
                ).abs()
                mae_lookup = perf_rows.groupby("model")["abs_error"].mean().to_dict()

            weights = _conservative_weight_vector(
                base_models=base_models,
                mae_by_model=mae_lookup,
                persistence_floor=float(stack_settings["persistence_floor"]),
                inverse_mae_power=float(stack_settings["inverse_mae_power"]),
            )

            for model_name in base_models:
                weight_rows.append(
                    {
                        "track": track_name,
                        "fold_id": fold_id,
                        "horizon": int(horizon),
                        "model": model_name,
                        "weight": float(weights.get(model_name, 0.0)),
                        "source_mae": float(mae_lookup.get(model_name, np.nan)),
                        "weight_rule": "leave_one_fold_out_inverse_mae",
                        "persistence_floor": float(stack_settings["persistence_floor"]),
                        "inverse_mae_power": float(stack_settings["inverse_mae_power"]),
                    }
                )

            subset = fold_wide.loc[fold_wide["horizon"] == horizon].copy()
            if subset.empty:
                continue
            fallback_cases = subset["cases"].to_numpy(dtype=float)
            blend = np.zeros(len(subset), dtype=float)
            for model_name in base_models:
                if model_name in subset.columns:
                    model_pred = subset[model_name].to_numpy(dtype=float)
                else:
                    model_pred = fallback_cases
                blend += float(weights[model_name]) * np.nan_to_num(
                    model_pred, nan=fallback_cases
                )

            subset["y_pred"] = np.clip(blend, 0.0, None)
            subset["model"] = "conservative_stack"
            subset["fit_status"] = "stack_inverse_mae"
            train_subset = subset.loc[subset["role"] == "train"].copy()
            adjusted_pred, guardrail_info = _apply_low_origin_guardrail(
                model_name="conservative_stack",
                train_df=train_subset,
                score_df=subset,
                y_pred=subset["y_pred"].to_numpy(dtype=float),
                training_cfg=training_cfg,
            )
            subset["y_pred"] = adjusted_pred
            if guardrail_info is not None:
                subset["fit_status"] = "stack_inverse_mae_low_origin_guardrail"
            stack_frames.append(subset[id_cols + ["y_pred", "model", "fit_status"]])
            note = {
                "track": track_name,
                "model": "conservative_stack",
                "fold_id": fold_id,
                "horizon": int(horizon),
                "fit_strategy": "stack_inverse_mae",
                "base_models": base_models,
                "persistence_floor": float(stack_settings["persistence_floor"]),
                "inverse_mae_power": float(stack_settings["inverse_mae_power"]),
            }
            if guardrail_info is not None:
                note.update(guardrail_info)
            fit_notes.append(note)

    stack_predictions = pd.concat(stack_frames, ignore_index=True)
    return stack_predictions, fit_notes, weight_rows


def run_track_models(
    supervised: pd.DataFrame,
    track_name: str,
    track_cfg: dict[str, Any],
    feature_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    panel: pd.DataFrame,
    logger: JsonlLogger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    history = _history_lookup(panel)
    feature_cols = select_feature_columns(
        supervised, include_weather=bool(track_cfg["include_weather"])
    )
    min_train_rows = int(training_cfg["min_train_rows_per_horizon"])
    requested_models = list(track_cfg["models"])
    base_model_names = [
        model_name
        for model_name in requested_models
        if model_name != "conservative_stack"
    ]
    conservative_stack_requested = "conservative_stack" in requested_models

    predictions: list[pd.DataFrame] = []
    fit_notes: list[dict[str, Any]] = []
    conservative_stack_weights: list[dict[str, Any]] = []

    for model_name in base_model_names:
        logger.log(
            "model_started",
            track=track_name,
            model=model_name,
            message=f"[{track_name}] running {model_name}",
        )

        if model_name == "persistence":
            pred = supervised.copy()
            pred["y_pred"] = pred["cases"].astype(float)
            pred["model"] = model_name
            pred["fit_status"] = "naive_current_cases"
            predictions.append(pred)
            fit_notes.append(
                {
                    "track": track_name,
                    "model": model_name,
                    "fit_strategy": "naive_current_cases",
                }
            )
            continue

        if model_name == "seasonal_naive_52":
            pred = supervised.copy()
            pred["y_pred"] = _predict_seasonal_naive(pred, history)
            pred["model"] = model_name
            pred["fit_status"] = "naive_same_week_last_year"
            predictions.append(pred)
            fit_notes.append(
                {
                    "track": track_name,
                    "model": model_name,
                    "fit_strategy": "naive_same_week_last_year",
                }
            )
            continue

        model_frames: list[pd.DataFrame] = []
        for fold_id in sorted(supervised["fold_id"].unique()):
            fold_df = supervised.loc[supervised["fold_id"] == fold_id].copy()
            for horizon in sorted(fold_df["horizon"].unique()):
                subset = fold_df.loc[fold_df["horizon"] == horizon].copy()
                train_df = subset.loc[subset["role"] == "train"].copy()
                score_df = subset.copy()
                note = {
                    "track": track_name,
                    "model": model_name,
                    "fold_id": fold_id,
                    "horizon": int(horizon),
                    "train_rows": int(len(train_df)),
                    "score_rows": int(len(score_df)),
                    "feature_count": int(len(feature_cols)),
                }
                if len(train_df) < min_train_rows:
                    score_df["y_pred"] = (
                        float(train_df["y_true"].mean()) if len(train_df) else 0.0
                    )
                    score_df["fit_status"] = "fallback_mean"
                    note["fit_strategy"] = "fallback_mean"
                    model_frames.append(score_df)
                    fit_notes.append(note)
                    continue

                try:
                    train_target = None
                    if model_name == "residual_elastic_net_global":
                        train_target = np.log1p(
                            train_df["y_true"].to_numpy(dtype=float)
                        ) - np.log1p(train_df["cases"].to_numpy(dtype=float))
                    elif model_name in {"residual_hist_gbm", "residual_random_forest"}:
                        train_target = train_df["y_true"].to_numpy(
                            dtype=float
                        ) - train_df["cases"].to_numpy(dtype=float)

                    raw_pred = _fit_predict_model(
                        model_name=model_name,
                        train_df=train_df,
                        score_df=score_df,
                        feature_cols=feature_cols,
                        training_cfg=training_cfg,
                        train_target=train_target,
                    )
                    if model_name in {"residual_hist_gbm", "residual_random_forest"}:
                        score_df["y_pred"] = np.clip(
                            score_df["cases"].to_numpy(dtype=float) + raw_pred,
                            0.0,
                            None,
                        )
                    elif model_name == "residual_elastic_net_global":
                        score_df["y_pred"] = np.clip(
                            np.expm1(
                                np.log1p(score_df["cases"].to_numpy(dtype=float))
                                + raw_pred
                            ),
                            0.0,
                            None,
                        )
                    else:
                        score_df["y_pred"] = np.clip(
                            np.asarray(raw_pred, dtype=float), 0.0, None
                        )
                    instability_reason = _prediction_instability_reason(
                        score_df["y_pred"].to_numpy(dtype=float),
                        train_df["y_true"].to_numpy(dtype=float),
                    )
                    if instability_reason is not None:
                        score_df["y_pred"] = score_df["cases"].astype(float)
                        score_df["fit_status"] = (
                            "fallback_persistence_after_instability"
                        )
                        note["fit_strategy"] = "fallback_persistence_after_instability"
                        note["instability_reason"] = instability_reason
                    else:
                        adjusted_pred, guardrail_info = _apply_low_origin_guardrail(
                            model_name=model_name,
                            train_df=train_df,
                            score_df=score_df,
                            y_pred=score_df["y_pred"].to_numpy(dtype=float),
                            training_cfg=training_cfg,
                        )
                        score_df["y_pred"] = adjusted_pred
                        score_df["fit_status"] = (
                            "trained_low_origin_guardrail"
                            if guardrail_info is not None
                            else "trained"
                        )
                        note["fit_strategy"] = str(score_df["fit_status"].iloc[0])
                        if guardrail_info is not None:
                            note.update(guardrail_info)
                except Exception as exc:  # pragma: no cover - defensive fallback
                    score_df["y_pred"] = float(train_df["y_true"].mean())
                    score_df["fit_status"] = "fallback_mean_after_error"
                    note["fit_strategy"] = "fallback_mean_after_error"
                    note["error"] = repr(exc)

                model_frames.append(score_df)
                fit_notes.append(note)

        pred = pd.concat(model_frames, ignore_index=True)
        pred["model"] = model_name
        predictions.append(pred)

    if conservative_stack_requested:
        base_predictions = pd.concat(predictions, ignore_index=True)
        stack_predictions, stack_fit_notes, stack_weights = (
            _build_conservative_stack_predictions(
                track_name=track_name,
                base_predictions=base_predictions,
                track_cfg=track_cfg,
                training_cfg=training_cfg,
            )
        )
        predictions.append(stack_predictions)
        fit_notes.extend(stack_fit_notes)
        conservative_stack_weights.extend(stack_weights)

    prediction_frame = pd.concat(predictions, ignore_index=True)
    prediction_frame["track"] = track_name
    prediction_frame["y_pred"] = (
        prediction_frame["y_pred"].astype(float).clip(lower=0.0)
    )
    metadata = {
        "track": track_name,
        "include_weather": bool(track_cfg["include_weather"]),
        "horizons": list(track_cfg["horizons"]),
        "requested_models": requested_models,
        "feature_columns": feature_cols,
        "fit_notes": fit_notes,
        "case_lags": list(feature_cfg["case_lags"]),
        "conservative_stack_weights": conservative_stack_weights,
    }
    return prediction_frame, metadata


def build_metrics_bundle(predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    scored = predictions.copy()
    scored["error"] = scored["y_true"] - scored["y_pred"]
    scored["abs_error"] = scored["error"].abs()
    scored["sq_error"] = scored["error"] ** 2

    by_fold_geo_horizon = _metric_frame(
        scored,
        ["track", "split_group", "model", "fold_id", "role", "geo_id", "horizon"],
    )
    by_fold_geo = _metric_frame(
        scored,
        ["track", "split_group", "model", "fold_id", "role", "geo_id"],
    )
    by_fold_horizon_pooled = _metric_frame(
        scored,
        ["track", "split_group", "model", "fold_id", "role", "horizon"],
    )
    by_track_model_role_pooled = _metric_frame(
        scored,
        ["track", "split_group", "model", "role"],
    )

    by_fold_horizon_macro = _macro_average(
        by_fold_geo_horizon,
        ["track", "split_group", "model", "fold_id", "role", "horizon"],
    )
    by_fold_macro = _macro_average(
        by_fold_geo_horizon,
        ["track", "split_group", "model", "fold_id", "role"],
    )
    by_geo_horizon_macro = _macro_average(
        by_fold_geo_horizon,
        ["track", "split_group", "model", "role", "geo_id", "horizon"],
    )
    overall_macro = _macro_average(
        by_fold_geo_horizon,
        ["track", "split_group", "model", "role"],
    )

    leaderboard = _leaderboard_for_group(overall_macro, "development")
    leaderboard_selection = _leaderboard_for_group(overall_macro, "selection")
    leaderboard_holdout = _leaderboard_for_group(overall_macro, "holdout")
    leaderboard_diagnostic = _leaderboard_for_group(overall_macro, "diagnostic")
    best_by_horizon = _best_by_horizon_for_group(by_fold_horizon_macro, "development")
    selection_by_horizon = _best_by_horizon_for_group(
        by_fold_horizon_macro, "selection"
    )
    holdout_by_horizon = _best_by_horizon_for_group(by_fold_horizon_macro, "holdout")
    diagnostic_by_horizon = _best_by_horizon_for_group(
        by_fold_horizon_macro, "diagnostic"
    )

    bundle = {
        "predictions_scored": scored,
        "metrics_by_fold_geo_horizon": by_fold_geo_horizon,
        "metrics_by_fold_geo": by_fold_geo,
        "metrics_by_fold_horizon_pooled": by_fold_horizon_pooled,
        "metrics_by_fold_horizon_macro": by_fold_horizon_macro,
        "metrics_by_fold_macro": by_fold_macro,
        "metrics_by_geo_horizon_macro": by_geo_horizon_macro,
        "metrics_overall_pooled": by_track_model_role_pooled,
        "metrics_overall_macro": overall_macro,
        "leaderboard": leaderboard,
        "leaderboard_selection": leaderboard_selection,
        "leaderboard_holdout": leaderboard_holdout,
        "leaderboard_diagnostic": leaderboard_diagnostic,
        "leaderboard_by_horizon": best_by_horizon,
        "leaderboard_selection_by_horizon": selection_by_horizon,
        "leaderboard_holdout_by_horizon": holdout_by_horizon,
        "leaderboard_diagnostic_by_horizon": diagnostic_by_horizon,
    }
    return bundle


def summarize_track(
    feature_panel: pd.DataFrame, splits: pd.DataFrame, metadata: dict[str, Any]
) -> dict[str, Any]:
    split_group_fold_ids = {
        str(group): sorted(frame["fold_id"].dropna().unique().tolist())
        for group, frame in splits.groupby("split_group", dropna=False)
    }
    return {
        "track": metadata["track"],
        "include_weather": metadata["include_weather"],
        "rows": int(len(feature_panel)),
        "geographies": int(feature_panel["geo_id"].nunique()),
        "weeks_start": str(
            pd.to_datetime(feature_panel["week_start_date"]).min().date()
        ),
        "weeks_end": str(pd.to_datetime(feature_panel["week_start_date"]).max().date()),
        "horizons": metadata["horizons"],
        "split_group_fold_ids": split_group_fold_ids,
        "development_fold_ids": split_group_fold_ids.get("development", []),
        "selection_fold_ids": split_group_fold_ids.get("selection", []),
        "holdout_fold_ids": split_group_fold_ids.get("holdout", []),
        "diagnostic_fold_ids": split_group_fold_ids.get("diagnostic", []),
        "feature_count": len(metadata["feature_columns"]),
        "proxy_geographies": sorted(
            feature_panel.loc[feature_panel["covariate_is_proxy"] == 1, "geo_id"]
            .unique()
            .tolist()
        ),
    }


def save_metrics_bundle(bundle: dict[str, pd.DataFrame], metrics_dir: Path) -> None:
    for name, frame in bundle.items():
        if name == "predictions_scored":
            continue
        frame.to_csv(metrics_dir / f"{name}.csv", index=False)


def save_prediction_files(predictions: pd.DataFrame, forecasts_dir: Path) -> None:
    predictions.to_parquet(forecasts_dir / "all_predictions.parquet", index=False)
    for (track, model), frame in predictions.groupby(["track", "model"], dropna=False):
        out_path = forecasts_dir / f"{track}_{model}.parquet"
        frame.to_parquet(out_path, index=False)


def build_run_summary(
    run_id: str,
    hierarchy_summary: pd.DataFrame,
    track_summaries: list[dict[str, Any]],
    leaderboard: pd.DataFrame,
    leaderboard_selection: pd.DataFrame,
    leaderboard_holdout: pd.DataFrame,
    leaderboard_diagnostic: pd.DataFrame,
    leaderboard_by_horizon: pd.DataFrame,
    leaderboard_selection_by_horizon: pd.DataFrame,
    leaderboard_holdout_by_horizon: pd.DataFrame,
    leaderboard_diagnostic_by_horizon: pd.DataFrame,
) -> str:
    lines = [
        "# Modeling Tracks Run Summary",
        "",
        f"- Run ID: `{run_id}`",
        "",
        "## Hierarchy",
        "",
        f"- Residual geography: `{hierarchy_summary.loc[0, 'residual_geo_id']}`",
        f"- Max closure error: `{hierarchy_summary.loc[0, 'max_abs_closure_error']}`",
        f"- Negative raw residual weeks clipped: `{hierarchy_summary.loc[0, 'negative_raw_residual_weeks']}`",
        "",
        "## Track Snapshots",
        "",
    ]

    for summary in track_summaries:
        lines.extend(
            [
                f"### {summary['track'].title()}",
                "",
                f"- Include weather: `{summary['include_weather']}`",
                f"- Horizons: `{summary['horizons']}`",
                f"- Rows: `{summary['rows']}`",
                f"- Geographies: `{summary['geographies']}`",
                f"- Feature count: `{summary['feature_count']}`",
                f"- Development folds: `{summary['development_fold_ids']}`",
                f"- Selection folds: `{summary['selection_fold_ids']}`",
                f"- Holdout folds: `{summary['holdout_fold_ids']}`",
                f"- Diagnostic folds: `{summary['diagnostic_fold_ids']}`",
                f"- Proxy geographies: `{summary['proxy_geographies']}`",
                "",
            ]
        )

    lines.extend(["## Development Leaderboard", ""])
    for track, frame in leaderboard.groupby("track", dropna=False):
        lines.append(f"### {track.title()}")
        lines.append("")
        for row in frame.itertuples(index=False):
            lines.append(
                f"- {row.model}: MAE_macro={row.MAE_macro:.3f}, RMSE_macro={row.RMSE_macro:.3f}"
            )
        lines.append("")

    if not leaderboard_selection.empty:
        lines.extend(["## Selection Leaderboard", ""])
        for track, frame in leaderboard_selection.groupby("track", dropna=False):
            lines.append(f"### {track.title()}")
            lines.append("")
            for row in frame.itertuples(index=False):
                lines.append(
                    f"- {row.model}: MAE_macro={row.MAE_macro:.3f}, RMSE_macro={row.RMSE_macro:.3f}"
                )
            lines.append("")

    if not leaderboard_holdout.empty:
        lines.extend(["## Locked Holdout Leaderboard", ""])
        for track, frame in leaderboard_holdout.groupby("track", dropna=False):
            lines.append(f"### {track.title()}")
            lines.append("")
            for row in frame.itertuples(index=False):
                lines.append(
                    f"- {row.model}: MAE_macro={row.MAE_macro:.3f}, RMSE_macro={row.RMSE_macro:.3f}"
                )
            lines.append("")

    if not leaderboard_diagnostic.empty:
        lines.extend(["## Diagnostic Leaderboard", ""])
        for track, frame in leaderboard_diagnostic.groupby("track", dropna=False):
            lines.append(f"### {track.title()}")
            lines.append("")
            for row in frame.itertuples(index=False):
                lines.append(
                    f"- {row.model}: MAE_macro={row.MAE_macro:.3f}, RMSE_macro={row.RMSE_macro:.3f}"
                )
            lines.append("")

    lines.extend(["## Development Best By Horizon", ""])
    for track, frame in leaderboard_by_horizon.groupby("track", dropna=False):
        lines.append(f"### {track.title()}")
        lines.append("")
        for horizon in sorted(frame["horizon"].unique()):
            best = frame.loc[frame["horizon"] == horizon].iloc[0]
            lines.append(
                f"- Horizon {int(horizon)}: {best['model']} (MAE_macro={best['MAE_macro']:.3f}, RMSE_macro={best['RMSE_macro']:.3f})"
            )
        lines.append("")

    if not leaderboard_selection_by_horizon.empty:
        lines.extend(["## Selection Best By Horizon", ""])
        for track, frame in leaderboard_selection_by_horizon.groupby(
            "track", dropna=False
        ):
            lines.append(f"### {track.title()}")
            lines.append("")
            for horizon in sorted(frame["horizon"].unique()):
                best = frame.loc[frame["horizon"] == horizon].iloc[0]
                lines.append(
                    f"- Horizon {int(horizon)}: {best['model']} (MAE_macro={best['MAE_macro']:.3f}, RMSE_macro={best['RMSE_macro']:.3f})"
                )
            lines.append("")

    if not leaderboard_holdout_by_horizon.empty:
        lines.extend(["## Locked Holdout By Horizon", ""])
        for track, frame in leaderboard_holdout_by_horizon.groupby(
            "track", dropna=False
        ):
            lines.append(f"### {track.title()}")
            lines.append("")
            for horizon in sorted(frame["horizon"].unique()):
                best = frame.loc[frame["horizon"] == horizon].iloc[0]
                lines.append(
                    f"- Horizon {int(horizon)}: {best['model']} (MAE_macro={best['MAE_macro']:.3f}, RMSE_macro={best['RMSE_macro']:.3f})"
                )
            lines.append("")

    if not leaderboard_diagnostic_by_horizon.empty:
        lines.extend(["## Diagnostic Best By Horizon", ""])
        for track, frame in leaderboard_diagnostic_by_horizon.groupby(
            "track", dropna=False
        ):
            lines.append(f"### {track.title()}")
            lines.append("")
            for horizon in sorted(frame["horizon"].unique()):
                best = frame.loc[frame["horizon"] == horizon].iloc[0]
                lines.append(
                    f"- Horizon {int(horizon)}: {best['model']} (MAE_macro={best['MAE_macro']:.3f}, RMSE_macro={best['RMSE_macro']:.3f})"
                )
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def run_modeling_tracks(
    config_path: str | Path = "configs/modeling_tracks.yaml",
) -> dict[str, Any]:
    data_cfg, split_cfg, modeling_cfg = load_runtime_config(config_path)
    paths = resolve_paths(data_cfg)
    processed_modeling_dir = paths["processed"] / "modeling"
    ensure_dir(processed_modeling_dir)

    artifacts = init_run_artifacts(
        run_name_prefix=modeling_cfg["run_name_prefix"],
        artifacts_root=paths["artifacts"],
    )
    logger = JsonlLogger(artifacts.log_path)
    logger.log(
        "run_started",
        run_id=artifacts.run_id,
        config_path=str(config_path),
        message=f"[run] starting {artifacts.run_id}",
    )

    config_snapshot = {
        "data": data_cfg,
        "split": split_cfg,
        "modeling_tracks": modeling_cfg,
    }
    _write_json(artifacts.run_dir / "config_snapshot.json", config_snapshot)

    panel = load_panel(paths)
    fixed_panel, hierarchy_summary = fix_hierarchy(panel, modeling_cfg["hierarchy"])
    fixed_panel.to_parquet(
        processed_modeling_dir / "hierarchy_fixed_panel.parquet", index=False
    )
    fixed_panel.to_parquet(
        artifacts.data_dir / "hierarchy_fixed_panel.parquet", index=False
    )
    hierarchy_summary.to_csv(
        processed_modeling_dir / "hierarchy_summary.csv", index=False
    )
    hierarchy_summary.to_csv(artifacts.data_dir / "hierarchy_summary.csv", index=False)
    logger.log(
        "hierarchy_fixed",
        run_id=artifacts.run_id,
        residual_geo_id=modeling_cfg["hierarchy"]["residual_geo_id"],
        max_abs_closure_error=float(hierarchy_summary.loc[0, "max_abs_closure_error"]),
        message="[run] hierarchy fixed and saved",
    )

    track_summaries: list[dict[str, Any]] = []
    all_predictions: list[pd.DataFrame] = []
    track_metadata_records: dict[str, Any] = {}

    for track_name, track_cfg in modeling_cfg["tracks"].items():
        logger.log(
            "track_started",
            track=track_name,
            message=f"[run] building {track_name} track",
        )

        feature_panel = build_track_feature_panel(
            panel=fixed_panel,
            track_name=track_name,
            track_cfg=track_cfg,
            feature_cfg=modeling_cfg["features"],
        )
        splits = build_track_splits(feature_panel, track_cfg, split_cfg)
        supervised = build_supervised_frame(feature_panel, splits, fixed_panel)

        feature_panel.to_parquet(
            processed_modeling_dir / f"{track_name}_feature_panel.parquet", index=False
        )
        splits.to_parquet(
            processed_modeling_dir / f"{track_name}_splits.parquet", index=False
        )
        supervised.to_parquet(
            processed_modeling_dir / f"{track_name}_supervised.parquet", index=False
        )
        feature_panel.to_parquet(
            artifacts.data_dir / f"{track_name}_feature_panel.parquet", index=False
        )
        splits.to_parquet(
            artifacts.data_dir / f"{track_name}_splits.parquet", index=False
        )
        supervised.to_parquet(
            artifacts.data_dir / f"{track_name}_supervised.parquet", index=False
        )

        predictions, metadata = run_track_models(
            supervised=supervised,
            track_name=track_name,
            track_cfg=track_cfg,
            feature_cfg=modeling_cfg["features"],
            training_cfg=modeling_cfg["training"],
            panel=fixed_panel,
            logger=logger,
        )
        all_predictions.append(predictions)

        track_summary = summarize_track(feature_panel, splits, metadata)
        track_summaries.append(track_summary)
        track_metadata_records[track_name] = {
            "summary": track_summary,
            "metadata": metadata,
        }
        _write_json(
            artifacts.data_dir / f"{track_name}_metadata.json",
            track_metadata_records[track_name],
        )
        stack_weights = metadata.get("conservative_stack_weights") or []
        if stack_weights:
            weights_frame = pd.DataFrame(stack_weights)
            weights_frame.to_csv(
                artifacts.data_dir / f"{track_name}_conservative_stack_weights.csv",
                index=False,
            )
            weights_frame.to_csv(
                processed_modeling_dir / f"{track_name}_conservative_stack_weights.csv",
                index=False,
            )

    predictions = pd.concat(all_predictions, ignore_index=True)
    save_prediction_files(predictions, artifacts.forecasts_dir)

    metrics_bundle = build_metrics_bundle(predictions)
    save_metrics_bundle(metrics_bundle, artifacts.metrics_dir)

    track_summary_path = artifacts.summaries_dir / "run_summary.md"
    summary_text = build_run_summary(
        run_id=artifacts.run_id,
        hierarchy_summary=hierarchy_summary,
        track_summaries=track_summaries,
        leaderboard=metrics_bundle["leaderboard"],
        leaderboard_selection=metrics_bundle["leaderboard_selection"],
        leaderboard_holdout=metrics_bundle["leaderboard_holdout"],
        leaderboard_diagnostic=metrics_bundle["leaderboard_diagnostic"],
        leaderboard_by_horizon=metrics_bundle["leaderboard_by_horizon"],
        leaderboard_selection_by_horizon=metrics_bundle[
            "leaderboard_selection_by_horizon"
        ],
        leaderboard_holdout_by_horizon=metrics_bundle["leaderboard_holdout_by_horizon"],
        leaderboard_diagnostic_by_horizon=metrics_bundle[
            "leaderboard_diagnostic_by_horizon"
        ],
    )
    _write_markdown(track_summary_path, summary_text)

    report_dir = Path("reports/modeling")
    ensure_dir(report_dir)
    _write_markdown(report_dir / f"baseline_suite_{artifacts.run_id}.md", summary_text)

    _write_json(
        artifacts.summaries_dir / "track_metadata.json",
        track_metadata_records,
    )
    logger.log(
        "run_completed",
        run_id=artifacts.run_id,
        summary_path=str(track_summary_path),
        message=f"[run] completed {artifacts.run_id}",
    )

    return {
        "run_id": artifacts.run_id,
        "run_dir": artifacts.run_dir,
        "summary_path": track_summary_path,
        "leaderboard_path": artifacts.metrics_dir / "leaderboard.csv",
    }
