from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dengue.forecasting.hierarchy import bottom_up_sum
from dengue.forecasting.modeling_tracks import (
    JsonlLogger,
    _apply_low_origin_guardrail,
    _conservative_weight_vector,
    _fit_predict_model,
    _history_lookup,
    _predict_seasonal_naive,
    _prediction_instability_reason,
    _resolve_conservative_stack_settings,
    build_track_feature_panel,
    fix_hierarchy,
    load_panel,
    load_runtime_config,
    select_feature_columns,
)
from dengue.utils.io import ensure_dir, resolve_paths


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


def _write_markdown(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def _build_full_training_pairs(
    feature_panel: pd.DataFrame,
    panel: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    target_map = (
        panel[["geo_id", "week_start_date", "cases"]]
        .rename(columns={"week_start_date": "target_week", "cases": "y_true"})
        .copy()
    )
    max_week = pd.to_datetime(panel["week_start_date"]).max()

    frames = []
    for horizon in horizons:
        tmp = feature_panel.copy()
        tmp["horizon"] = horizon
        tmp["target_week"] = pd.to_datetime(tmp["week_start_date"]) + pd.Timedelta(
            days=7 * horizon
        )
        tmp = tmp.loc[tmp["target_week"] <= max_week].copy()
        tmp = tmp.merge(target_map, on=["geo_id", "target_week"], how="inner")
        frames.append(tmp)

    return pd.concat(frames, ignore_index=True)


def _origin_rows(
    feature_panel: pd.DataFrame, origin_week: pd.Timestamp, horizon: int
) -> pd.DataFrame:
    base = feature_panel.loc[feature_panel["week_start_date"] == origin_week].copy()
    base["horizon"] = horizon
    base["target_week"] = pd.to_datetime(origin_week) + pd.Timedelta(days=7 * horizon)
    return base


def _load_operational_rankings(
    backtest_run_dir: Path | None,
) -> tuple[pd.DataFrame | None, str | None]:
    if backtest_run_dir is None:
        return None, None

    for ranking_source, filename in (
        ("holdout", "leaderboard_holdout.csv"),
        ("selection", "leaderboard_selection.csv"),
        ("development", "leaderboard.csv"),
    ):
        leaderboard_path = backtest_run_dir / "metrics" / filename
        if not leaderboard_path.exists():
            continue
        leaderboard = pd.read_csv(leaderboard_path)
        leaderboard = leaderboard.loc[
            (leaderboard["track"] == "operational") & (leaderboard["role"] == "val")
        ].copy()
        if leaderboard.empty:
            continue
        leaderboard = leaderboard.sort_values(
            ["MAE_macro", "RMSE_macro", "model"]
        ).reset_index(drop=True)
        return leaderboard, ranking_source

    return None, None


def _forecast_trainable_model(
    model_name: str,
    train_df: pd.DataFrame,
    score_df: pd.DataFrame,
    feature_cols: list[str],
    training_cfg: dict[str, Any],
) -> tuple[np.ndarray, str]:
    train_target = None
    if model_name == "residual_elastic_net_global":
        train_target = np.log1p(train_df["y_true"].to_numpy(dtype=float)) - np.log1p(
            train_df["cases"].to_numpy(dtype=float)
        )
    elif model_name in {"residual_hist_gbm", "residual_random_forest"}:
        train_target = train_df["y_true"].to_numpy(dtype=float) - train_df[
            "cases"
        ].to_numpy(dtype=float)

    raw_pred = _fit_predict_model(
        model_name=model_name,
        train_df=train_df,
        score_df=score_df,
        feature_cols=feature_cols,
        training_cfg=training_cfg,
        train_target=train_target,
    )

    if model_name in {"residual_hist_gbm", "residual_random_forest"}:
        y_pred = np.clip(score_df["cases"].to_numpy(dtype=float) + raw_pred, 0.0, None)
    elif model_name == "residual_elastic_net_global":
        y_pred = np.clip(
            np.expm1(np.log1p(score_df["cases"].to_numpy(dtype=float)) + raw_pred),
            0.0,
            None,
        )
    else:
        y_pred = np.clip(np.asarray(raw_pred, dtype=float), 0.0, None)

    instability_reason = _prediction_instability_reason(
        y_pred,
        train_df["y_true"].to_numpy(dtype=float),
    )
    if instability_reason is not None:
        return (
            score_df["cases"].to_numpy(dtype=float),
            "fallback_persistence_after_instability",
        )
    adjusted_pred, guardrail_info = _apply_low_origin_guardrail(
        model_name=model_name,
        train_df=train_df,
        score_df=score_df,
        y_pred=y_pred,
        training_cfg=training_cfg,
    )
    fit_status = (
        "trained_full_history_low_origin_guardrail"
        if guardrail_info is not None
        else "trained_full_history"
    )
    return adjusted_pred, fit_status


def _load_backtest_horizon_mae(
    backtest_run_dir: Path | None, track_name: str
) -> dict[tuple[int, str], float]:
    if backtest_run_dir is None:
        return {}
    metrics_path = backtest_run_dir / "metrics" / "metrics_by_fold_horizon_macro.csv"
    if not metrics_path.exists():
        return {}
    metrics = pd.read_csv(metrics_path)
    metrics = metrics.loc[
        (metrics["track"] == track_name)
        & (metrics["role"] == "val")
        & (metrics["split_group"] == "development")
    ].copy()
    if metrics.empty:
        return {}
    averaged = (
        metrics.groupby(["horizon", "model"], dropna=False)["MAE_macro"]
        .mean()
        .reset_index()
    )
    return {
        (int(row.horizon), str(row.model)): float(row.MAE_macro)
        for row in averaged.itertuples(index=False)
    }


def _build_live_conservative_stack(
    base_forecasts: pd.DataFrame,
    training_pairs: pd.DataFrame,
    track_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    backtest_horizon_mae: dict[tuple[int, str], float],
    model_rank: int,
    is_primary_model: bool,
) -> pd.DataFrame:
    available_models = sorted(base_forecasts["model"].unique().tolist())
    stack_settings = _resolve_conservative_stack_settings(track_cfg, available_models)
    base_models = stack_settings["base_models"]
    if len(base_models) < 2:
        raise ValueError(
            f"conservative_stack requires at least two base models for live forecasting, got {base_models}"
        )

    id_cols = [
        "geo_id",
        "geo_name",
        "origin_week",
        "target_week",
        "horizon",
        "cases",
        "cases_nonzero_recent_4",
        "cases_zero_run_length",
        "cases_weeks_since_nonzero",
        "cases_outbreak_flag_4_8",
    ]
    id_cols = [col for col in id_cols if col in base_forecasts.columns]
    wide = (
        base_forecasts.loc[
            base_forecasts["model"].isin(base_models), id_cols + ["model", "y_pred"]
        ]
        .pivot_table(index=id_cols, columns="model", values="y_pred", aggfunc="first")
        .reset_index()
    )

    frames: list[pd.DataFrame] = []
    for horizon in sorted(wide["horizon"].unique().tolist()):
        subset = wide.loc[wide["horizon"] == horizon].copy()
        mae_lookup = {
            model_name: float(
                backtest_horizon_mae.get((int(horizon), model_name), np.nan)
            )
            for model_name in base_models
        }
        weights = _conservative_weight_vector(
            base_models=base_models,
            mae_by_model=mae_lookup,
            persistence_floor=float(stack_settings["persistence_floor"]),
            inverse_mae_power=float(stack_settings["inverse_mae_power"]),
        )
        blend = np.zeros(len(subset), dtype=float)
        for model_name in base_models:
            model_pred = subset[model_name].to_numpy(dtype=float)
            blend += float(weights[model_name]) * np.nan_to_num(model_pred, nan=0.0)

        subset["y_pred"] = np.clip(blend, 0.0, None)
        subset["model"] = "conservative_stack"
        train_df = training_pairs.loc[training_pairs["horizon"] == horizon].copy()
        adjusted_pred, guardrail_info = _apply_low_origin_guardrail(
            model_name="conservative_stack",
            train_df=train_df,
            score_df=subset,
            y_pred=subset["y_pred"].to_numpy(dtype=float),
            training_cfg=training_cfg,
        )
        subset["y_pred"] = adjusted_pred
        subset["fit_status"] = (
            "stack_from_backtest_mae_low_origin_guardrail"
            if guardrail_info is not None
            else "stack_from_backtest_mae"
        )
        subset["model_rank_from_backtest"] = model_rank
        subset["is_primary_model"] = int(is_primary_model)
        frames.append(
            subset[
                id_cols
                + [
                    "y_pred",
                    "model",
                    "fit_status",
                    "model_rank_from_backtest",
                    "is_primary_model",
                ]
            ]
        )

    return pd.concat(frames, ignore_index=True)


def build_operational_forecast_summary(
    forecast_run_id: str,
    origin_week: pd.Timestamp,
    backtest_ranking: pd.DataFrame | None,
    ranking_source: str | None,
    promotion_policy: str,
    forecast_df: pd.DataFrame,
) -> str:
    lines = [
        "# Current Operational Forecast",
        "",
        f"- Forecast run ID: `{forecast_run_id}`",
        f"- Origin week: `{origin_week.date()}`",
        f"- Forecast horizons: `{sorted(forecast_df['horizon'].unique().tolist())}`",
        "",
    ]
    if backtest_ranking is not None and not backtest_ranking.empty:
        ranking_labels = {
            "holdout": "locked holdout",
            "selection": "selection",
            "development": "development",
        }
        ranking_label = ranking_labels.get(ranking_source, str(ranking_source))
        lines.extend(["## Backtest Ranking Used", ""])
        lines.append(f"- Ranking source: `{ranking_label}`")
        lines.append(f"- Promotion policy: `{promotion_policy}`")
        for idx, row in enumerate(backtest_ranking.itertuples(index=False), start=1):
            lines.append(
                f"- {idx}. {row.model}: MAE_macro={row.MAE_macro:.3f}, RMSE_macro={row.RMSE_macro:.3f}"
            )
        lines.append("")

    lines.extend(["## Forecast Bundle", ""])
    for model_name, frame in forecast_df.groupby("model", dropna=False):
        primary_flag = bool(frame["is_primary_model"].iloc[0])
        lines.append(f"### {model_name}")
        lines.append("")
        lines.append(f"- Primary model: `{primary_flag}`")
        lines.append(
            f"- Fit status values: `{sorted(frame['fit_status'].unique().tolist())}`"
        )
        top = frame.loc[frame["geo_id"] == "COUNTRY_TOTAL"].sort_values("horizon")
        if not top.empty:
            country_vals = ", ".join(
                f"h{int(row.horizon)}={row.y_pred:.1f}"
                for row in top.itertuples(index=False)
            )
            lines.append(f"- Country total forecasts: {country_vals}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def run_current_operational_forecast(
    config_path: str | Path = "configs/modeling_tracks.yaml",
    backtest_run_dir: str | Path | None = None,
) -> dict[str, Any]:
    data_cfg, _, modeling_cfg = load_runtime_config(config_path)
    paths = resolve_paths(data_cfg)

    backtest_dir = Path(backtest_run_dir) if backtest_run_dir else None
    backtest_ranking, ranking_source = _load_operational_rankings(backtest_dir)
    backtest_horizon_mae = _load_backtest_horizon_mae(
        backtest_dir, track_name="operational"
    )
    rank_lookup = (
        {
            str(row.model): idx + 1
            for idx, row in enumerate(backtest_ranking.itertuples(index=False))
        }
        if backtest_ranking is not None and not backtest_ranking.empty
        else {}
    )
    promotion_policy = (
        "holdout_winner"
        if ranking_source == "holdout"
        else "cautious_default_persistence_without_true_holdout"
    )
    if (
        ranking_source == "holdout"
        and backtest_ranking is not None
        and not backtest_ranking.empty
    ):
        primary_model = str(backtest_ranking.iloc[0]["model"])
    else:
        primary_model = "persistence"

    forecast_run_id = f"operational_current_{_utc_stamp()}"
    run_dir = paths["artifacts"] / "forecasts" / forecast_run_id
    reports_dir = paths["root"] / "reports" / "forecasts"
    ensure_dir(run_dir)
    ensure_dir(reports_dir)
    logger = JsonlLogger(run_dir / "run_log.jsonl")
    logger.log(
        "forecast_started",
        forecast_run_id=forecast_run_id,
        primary_model=primary_model,
        ranking_source=ranking_source,
        promotion_policy=promotion_policy,
        message=f"[forecast] starting {forecast_run_id}",
    )

    panel = load_panel(paths)
    fixed_panel, hierarchy_summary = fix_hierarchy(panel, modeling_cfg["hierarchy"])
    track_cfg = modeling_cfg["tracks"]["operational"]
    feature_panel = build_track_feature_panel(
        panel=fixed_panel,
        track_name="operational",
        track_cfg=track_cfg,
        feature_cfg=modeling_cfg["features"],
    )
    origin_week = pd.to_datetime(feature_panel["week_start_date"]).max()
    horizons = list(track_cfg["horizons"])
    training_pairs = _build_full_training_pairs(feature_panel, fixed_panel, horizons)
    feature_cols = select_feature_columns(training_pairs, include_weather=False)
    history = _history_lookup(fixed_panel)

    forecast_frames = []
    fit_summary_rows = []

    models_to_run = list(track_cfg["models"])
    base_models_to_run = [
        model_name for model_name in models_to_run if model_name != "conservative_stack"
    ]
    for rank, model_name in enumerate(base_models_to_run, start=1):
        logger.log(
            "forecast_model_started",
            forecast_run_id=forecast_run_id,
            model=model_name,
            message=f"[forecast] running {model_name}",
        )
        for horizon in horizons:
            train_df = training_pairs.loc[training_pairs["horizon"] == horizon].copy()
            score_df = _origin_rows(
                feature_panel, origin_week=origin_week, horizon=horizon
            )
            score_df = score_df.loc[
                :, training_pairs.columns.intersection(score_df.columns).tolist()
            ].copy()
            if score_df.empty:
                continue

            if model_name == "persistence":
                y_pred = score_df["cases"].to_numpy(dtype=float)
                fit_status = "naive_current_cases"
            elif model_name == "seasonal_naive_52":
                y_pred = _predict_seasonal_naive(score_df, history)
                fit_status = "naive_same_week_last_year"
            else:
                y_pred, fit_status = _forecast_trainable_model(
                    model_name=model_name,
                    train_df=train_df,
                    score_df=score_df,
                    feature_cols=feature_cols,
                    training_cfg=modeling_cfg["training"],
                )

            keep_cols = [
                "geo_id",
                "geo_name",
                "week_start_date",
                "target_week",
                "horizon",
                "cases",
                "cases_nonzero_recent_4",
                "cases_zero_run_length",
                "cases_weeks_since_nonzero",
                "cases_outbreak_flag_4_8",
            ]
            keep_cols = [col for col in keep_cols if col in score_df.columns]
            geo_frame = score_df[keep_cols].copy()
            geo_frame = geo_frame.rename(columns={"week_start_date": "origin_week"})
            geo_frame["y_pred"] = np.asarray(y_pred, dtype=float)
            geo_frame["model"] = model_name
            geo_frame["fit_status"] = fit_status
            geo_frame["model_rank_from_backtest"] = rank_lookup.get(model_name, rank)
            geo_frame["is_primary_model"] = int(model_name == primary_model)
            forecast_frames.append(geo_frame)

            fit_summary_rows.append(
                {
                    "model": model_name,
                    "horizon": horizon,
                    "train_rows": int(len(train_df)),
                    "fit_status": fit_status,
                    "origin_week": str(origin_week.date()),
                    "target_week": str(
                        (origin_week + pd.Timedelta(days=7 * horizon)).date()
                    ),
                }
            )

    if "conservative_stack" in models_to_run:
        base_forecast_df = pd.concat(forecast_frames, ignore_index=True)
        stack_frame = _build_live_conservative_stack(
            base_forecasts=base_forecast_df,
            training_pairs=training_pairs,
            track_cfg=track_cfg,
            training_cfg=modeling_cfg["training"],
            backtest_horizon_mae=backtest_horizon_mae,
            model_rank=rank_lookup.get("conservative_stack", len(rank_lookup) + 1),
            is_primary_model=primary_model == "conservative_stack",
        )
        forecast_frames.append(stack_frame)
        for horizon in horizons:
            fit_summary_rows.append(
                {
                    "model": "conservative_stack",
                    "horizon": horizon,
                    "train_rows": int(
                        len(training_pairs.loc[training_pairs["horizon"] == horizon])
                    ),
                    "fit_status": "stack_from_backtest_mae",
                    "origin_week": str(origin_week.date()),
                    "target_week": str(
                        (origin_week + pd.Timedelta(days=7 * horizon)).date()
                    ),
                }
            )

    forecast_df = pd.concat(forecast_frames, ignore_index=True)

    bottom_list = sorted(feature_panel["geo_id"].unique().tolist())
    with_country = []
    for model_name, frame in forecast_df.groupby("model", dropna=False):
        country = bottom_up_sum(
            df_pred=frame,
            geo_col="geo_id",
            target_col="y_pred",
            time_col="target_week",
            country_geo="COUNTRY_TOTAL",
            bottom_list=bottom_list,
        )
        country["model"] = model_name
        country["fit_status"] = country["fit_status"].fillna(
            frame["fit_status"].iloc[0]
            if "fit_status" in frame.columns
            else "aggregated"
        )
        country["model_rank_from_backtest"] = country[
            "model_rank_from_backtest"
        ].fillna(frame["model_rank_from_backtest"].iloc[0])
        country["is_primary_model"] = country["is_primary_model"].fillna(
            frame["is_primary_model"].iloc[0]
        )
        if "origin_week" in country.columns:
            country["origin_week"] = country["origin_week"].fillna(
                frame["origin_week"].iloc[0]
            )
        if "geo_name" in country.columns:
            country.loc[country["geo_id"] == "COUNTRY_TOTAL", "geo_name"] = (
                "Bangladesh Total"
            )
        with_country.append(country)
    forecast_df = pd.concat(with_country, ignore_index=True)

    forecast_df = forecast_df.sort_values(
        ["model_rank_from_backtest", "model", "geo_id", "horizon"]
    ).reset_index(drop=True)
    fit_summary = (
        pd.DataFrame(fit_summary_rows)
        .sort_values(["model", "horizon"])
        .reset_index(drop=True)
    )

    forecast_parquet = run_dir / "forecast_operational_current.parquet"
    forecast_csv = run_dir / "forecast_operational_current.csv"
    forecast_df.to_parquet(forecast_parquet, index=False)
    forecast_df.to_csv(forecast_csv, index=False)
    fit_summary.to_csv(run_dir / "fit_summary.csv", index=False)
    hierarchy_summary.to_csv(run_dir / "hierarchy_summary.csv", index=False)

    report_csv = reports_dir / f"operational_current_{forecast_run_id}.csv"
    report_md = reports_dir / f"operational_current_{forecast_run_id}.md"
    forecast_df.to_csv(report_csv, index=False)
    summary_text = build_operational_forecast_summary(
        forecast_run_id=forecast_run_id,
        origin_week=origin_week,
        backtest_ranking=backtest_ranking,
        ranking_source=ranking_source,
        promotion_policy=promotion_policy,
        forecast_df=forecast_df,
    )
    _write_markdown(report_md, summary_text)

    metadata = {
        "forecast_run_id": forecast_run_id,
        "origin_week": str(origin_week.date()),
        "primary_model": primary_model,
        "ranking_source": ranking_source,
        "promotion_policy": promotion_policy,
        "config_path": str(config_path),
        "backtest_run_dir": str(backtest_dir) if backtest_dir else None,
        "horizons": horizons,
        "models": models_to_run,
        "artifacts": {
            "forecast_csv": str(forecast_csv),
            "forecast_parquet": str(forecast_parquet),
            "fit_summary": str(run_dir / "fit_summary.csv"),
            "report_csv": str(report_csv),
            "report_md": str(report_md),
        },
    }
    _write_json(run_dir / "metadata.json", metadata)

    logger.log(
        "forecast_completed",
        forecast_run_id=forecast_run_id,
        report_csv=str(report_csv),
        message=f"[forecast] completed {forecast_run_id}",
    )
    return metadata
