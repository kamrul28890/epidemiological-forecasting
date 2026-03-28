from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dengue.forecasting.modeling_tracks import _low_origin_mask
from dengue.utils.io import ensure_dir, load_yaml

ORIGIN_BUCKETS = [-1.0, 0.0, 5.0, 20.0, 100.0, np.inf]
ORIGIN_LABELS = ["zero", "1_5", "6_20", "21_100", "100p"]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze operational forecasting failure modes for a tracked modeling run."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to the modeling run directory under artifacts/runs/.",
    )
    parser.add_argument(
        "--config",
        default="configs/modeling_tracks.yaml",
        help="Path to the modeling-track config YAML.",
    )
    return parser


def _score(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["error"] = out["y_true"] - out["y_pred"]
    out["abs_error"] = out["error"].abs()
    out["sq_error"] = out["error"] ** 2
    return out


def _segment_metrics(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = (
        frame.groupby(group_cols, dropna=False)
        .agg(
            n_rows=("y_true", "size"),
            mean_origin=("cases", "mean"),
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


def _macro_geo(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    geo_level = _segment_metrics(frame, group_cols + ["geo_id"])
    return (
        geo_level.groupby(group_cols, dropna=False)
        .agg(
            n_geos=("geo_id", "size"),
            MAE_macro=("MAE", "mean"),
            RMSE_macro=("RMSE", "mean"),
            bias_macro=("bias", "mean"),
        )
        .reset_index()
    )


def build_report(
    run_id: str,
    scored: pd.DataFrame,
    analysis_group: str,
    dev_low_origin: pd.DataFrame,
    holdout_low_origin: pd.DataFrame,
    holdout_by_origin_bucket: pd.DataFrame,
    zero_origin_by_horizon: pd.DataFrame,
    holdout_by_regime: pd.DataFrame,
    holdout_geo_winners: pd.DataFrame,
) -> str:
    analysis_label = analysis_group.replace("_", " ").title()
    lines = [
        f"# Operational Failure-Mode Analysis - {run_id}",
        "",
        "This report breaks the operational validation results into the regimes that mattered most in the active analysis split:",
        "low-origin windows, zero-origin windows, sparse-activity behavior, and outbreak-flagged periods.",
        "",
    ]

    lines.extend(["## Scope", ""])
    lines.append(
        f"- Rows analyzed: `{len(scored)}` operational validation predictions."
    )
    lines.append(
        f"- Split groups present: `{sorted(scored['split_group'].dropna().unique().tolist())}`"
    )
    lines.append(f"- Active analysis split: `{analysis_group}`")
    lines.append(
        f"- Models present: `{sorted(scored['model'].dropna().unique().tolist())}`"
    )
    lines.append("")

    if not dev_low_origin.empty:
        lines.extend(["## Development Low-Origin Leaderboard", ""])
        for row in dev_low_origin.sort_values(
            ["MAE_macro", "RMSE_macro", "model"]
        ).itertuples(index=False):
            lines.append(
                f"- {row.model}: MAE_macro={row.MAE_macro:.3f}, RMSE_macro={row.RMSE_macro:.3f}"
            )
        lines.append("")

    if not holdout_low_origin.empty:
        lines.extend([f"## {analysis_label} Low-Origin Leaderboard", ""])
        for row in holdout_low_origin.sort_values(
            ["MAE_macro", "RMSE_macro", "model"]
        ).itertuples(index=False):
            lines.append(
                f"- {row.model}: MAE_macro={row.MAE_macro:.3f}, RMSE_macro={row.RMSE_macro:.3f}"
            )
        lines.append("")

    if not holdout_by_origin_bucket.empty:
        lines.extend([f"## {analysis_label} By Origin Bucket", ""])
        for bucket in ORIGIN_LABELS:
            bucket_frame = holdout_by_origin_bucket.loc[
                holdout_by_origin_bucket["origin_bucket"] == bucket
            ].sort_values(["MAE", "RMSE", "model"])
            if bucket_frame.empty:
                continue
            best = bucket_frame.iloc[0]
            lines.append(
                f"- `{bucket}` best model: {best['model']} "
                f"(MAE={best['MAE']:.3f}, bias={best['bias']:.3f}, mean_pred={best['mean_pred']:.3f}, mean_y={best['mean_y']:.3f})"
            )
        lines.append("")

    if not zero_origin_by_horizon.empty:
        lines.extend([f"## {analysis_label} Zero-Origin By Horizon", ""])
        for horizon in sorted(zero_origin_by_horizon["horizon"].unique()):
            frame = zero_origin_by_horizon.loc[
                zero_origin_by_horizon["horizon"] == horizon
            ].sort_values(["MAE", "RMSE", "model"])
            if frame.empty:
                continue
            best = frame.iloc[0]
            lines.append(
                f"- Horizon {int(horizon)} best model under zero-origin starts: {best['model']} "
                f"(MAE={best['MAE']:.3f}, mean_pred={best['mean_pred']:.3f}, mean_y={best['mean_y']:.3f})"
            )
        lines.append("")

    if not holdout_by_regime.empty:
        lines.extend([f"## {analysis_label} By Regime Flag", ""])
        for regime_name in [
            "low_origin",
            "not_low_origin",
            "outbreak_flag",
            "not_outbreak_flag",
        ]:
            frame = holdout_by_regime.loc[
                holdout_by_regime["regime"] == regime_name
            ].sort_values(["MAE_macro", "RMSE_macro", "model"])
            if frame.empty:
                continue
            best = frame.iloc[0]
            lines.append(
                f"- `{regime_name}` best model: {best['model']} "
                f"(MAE_macro={best['MAE_macro']:.3f}, RMSE_macro={best['RMSE_macro']:.3f})"
            )
        lines.append("")

    if not holdout_geo_winners.empty:
        lines.extend([f"## {analysis_label} Geography Winners", ""])
        for row in holdout_geo_winners.sort_values("geo_id").itertuples(index=False):
            lines.append(
                f"- {row.geo_id}: {row.model} (MAE={row.MAE:.3f}, RMSE={row.RMSE:.3f}, bias={row.bias:.3f})"
            )
        lines.append("")

    lines.extend(["## Interpretation", ""])
    lines.append(
        "- The key operational question is whether challengers are overpredicting rebound after low or zero observed incidence."
    )
    lines.append(
        "- The low-origin and zero-origin tables make that visible directly instead of hiding it inside overall MAE."
    )
    lines.append(
        "- These segmented results should be read together with the main analysis-group leaderboard before promoting any challenger."
    )
    lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    args = build_argparser().parse_args()
    run_dir = Path(args.run_dir)
    run_id = run_dir.name
    config = load_yaml(args.config)
    training_cfg = dict(config.get("training", {}))

    forecast_path = run_dir / "forecasts" / "all_predictions.parquet"
    if not forecast_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {forecast_path}")

    analysis_dir = run_dir / "analysis"
    ensure_dir(analysis_dir)
    report_dir = Path("reports/modeling")
    ensure_dir(report_dir)

    predictions = pd.read_parquet(forecast_path)
    predictions = predictions.loc[
        (predictions["track"] == "operational") & (predictions["role"] == "val")
    ].copy()
    predictions = _score(predictions)
    predictions["origin_bucket"] = pd.cut(
        predictions["cases"].astype(float),
        bins=ORIGIN_BUCKETS,
        labels=ORIGIN_LABELS,
    )
    predictions["low_origin"] = _low_origin_mask(predictions, training_cfg).astype(int)
    outbreak_source = (
        predictions["cases_outbreak_flag_4_8"]
        if "cases_outbreak_flag_4_8" in predictions.columns
        else pd.Series(0, index=predictions.index, dtype=float)
    )
    predictions["outbreak_flag"] = outbreak_source.fillna(0).astype(int)
    predictions["zero_origin"] = predictions["cases"].astype(float).le(0).astype(int)

    analysis_group = (
        "holdout" if (predictions["split_group"] == "holdout").any() else "diagnostic"
    )
    holdout = predictions.loc[predictions["split_group"] == analysis_group].copy()
    development = predictions.loc[predictions["split_group"] == "development"].copy()

    dev_low_origin = _macro_geo(
        development.loc[development["low_origin"] == 1],
        ["model"],
    ).sort_values(["MAE_macro", "RMSE_macro", "model"])
    holdout_low_origin = _macro_geo(
        holdout.loc[holdout["low_origin"] == 1],
        ["model"],
    ).sort_values(["MAE_macro", "RMSE_macro", "model"])
    holdout_by_origin_bucket = _segment_metrics(
        holdout,
        ["model", "origin_bucket"],
    ).sort_values(["origin_bucket", "MAE", "RMSE", "model"])
    zero_origin_by_horizon = _segment_metrics(
        holdout.loc[holdout["zero_origin"] == 1],
        ["model", "horizon"],
    ).sort_values(["horizon", "MAE", "RMSE", "model"])

    regime_frames: list[pd.DataFrame] = []
    for regime_name, mask in (
        ("low_origin", holdout["low_origin"] == 1),
        ("not_low_origin", holdout["low_origin"] == 0),
        ("outbreak_flag", holdout["outbreak_flag"] == 1),
        ("not_outbreak_flag", holdout["outbreak_flag"] == 0),
    ):
        frame = _macro_geo(holdout.loc[mask], ["model"])
        frame["regime"] = regime_name
        regime_frames.append(frame)
    holdout_by_regime = pd.concat(regime_frames, ignore_index=True)

    holdout_geo_metrics = _segment_metrics(holdout, ["model", "geo_id"])
    holdout_geo_winners = (
        holdout_geo_metrics.sort_values(["geo_id", "MAE", "RMSE", "model"])
        .groupby("geo_id", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    dev_low_origin.to_csv(
        analysis_dir / "development_low_origin_leaderboard.csv", index=False
    )
    holdout_low_origin.to_csv(
        analysis_dir / "holdout_low_origin_leaderboard.csv", index=False
    )
    holdout_by_origin_bucket.to_csv(
        analysis_dir / "holdout_by_origin_bucket.csv", index=False
    )
    zero_origin_by_horizon.to_csv(
        analysis_dir / "holdout_zero_origin_by_horizon.csv", index=False
    )
    holdout_by_regime.to_csv(analysis_dir / "holdout_by_regime.csv", index=False)
    holdout_geo_winners.to_csv(
        analysis_dir / "holdout_best_model_by_geo.csv", index=False
    )

    report_text = build_report(
        run_id=run_id,
        scored=predictions,
        analysis_group=analysis_group,
        dev_low_origin=dev_low_origin,
        holdout_low_origin=holdout_low_origin,
        holdout_by_origin_bucket=holdout_by_origin_bucket,
        zero_origin_by_horizon=zero_origin_by_horizon,
        holdout_by_regime=holdout_by_regime,
        holdout_geo_winners=holdout_geo_winners,
    )
    report_path = report_dir / f"failure_mode_analysis_{run_id}.md"
    report_path.write_text(report_text, encoding="utf-8")

    print(f"[analysis] run_id      -> {run_id}")
    print(f"[analysis] analysis_dir -> {analysis_dir}")
    print(f"[analysis] report       -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
