from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dengue.utils.io import ensure_dir


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _latest_dir(root: Path, prefix: str) -> Path:
    matches = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix))
    if not matches:
        raise FileNotFoundError(f"No directory under {root} starts with {prefix}")
    return matches[-1]


def _heatmap(
    data: pd.DataFrame,
    title: str,
    out_path: Path,
    fmt: str = ".1f",
    cmap: str = "YlOrRd",
    xlabel: str = "",
    ylabel: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(1.4 * max(4, data.shape[1]), 0.6 * max(4, data.shape[0]) + 2))
    im = ax.imshow(data.values, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(list(data.columns), rotation=45, ha="right")
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(list(data.index))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, format(float(data.iloc[i, j]), fmt), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _line_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    title: str,
    out_path: Path,
    xlabel: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, frame in df.groupby(group_col, dropna=False):
        frame = frame.sort_values(x_col)
        ax.plot(frame[x_col], frame[y_col], marker="o", linewidth=2, label=str(key))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _bar_plot(
    df: pd.DataFrame,
    label_col: str,
    value_cols: list[str],
    title: str,
    out_path: Path,
) -> None:
    n = len(value_cols)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    labels = df[label_col].tolist()
    ypos = np.arange(len(labels))
    for ax, value_col in zip(axes, value_cols):
        ax.barh(ypos, df[value_col], color="#2a7f62")
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_title(value_col)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


def _write_markdown(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def _study_design_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"item": "Forecast target", "value": "Weekly dengue case counts"},
            {"item": "Temporal grain", "value": "Weekly"},
            {"item": "Spatial grain", "value": "geo_id"},
            {"item": "Operational horizon", "value": "1 to 4 weeks ahead"},
            {"item": "Experimental horizon", "value": "5 to 12 weeks ahead"},
            {"item": "Primary KPI", "value": "Macro-averaged MAE across geography and horizon"},
            {"item": "Secondary KPI", "value": "Macro-averaged RMSE"},
            {"item": "Forecast strategy", "value": "Direct multi-horizon, not recursive"},
            {"item": "Operational covariates", "value": "Case-history and calendar only"},
            {"item": "Experimental covariates", "value": "Case-history, calendar, and climate"},
            {"item": "Hierarchy repair", "value": "DHA_DIV replaced by DHA_DIV_OUT_METRO"},
        ]
    )


def _dataset_summary_table(eda_dir: Path) -> pd.DataFrame:
    case_summary = pd.read_csv(eda_dir / "tables" / "case_series_summary.csv")
    seasonality = pd.read_csv(eda_dir / "tables" / "seasonality_summary.csv")
    gap = pd.read_csv(eda_dir / "tables" / "climate_gap_summary.csv")
    merged = case_summary.merge(seasonality, on="geo_id", how="left").merge(gap[["geo_id", "gap_weeks"]], on="geo_id", how="left")
    return merged[
        [
            "geo_id",
            "weeks",
            "zero_pct",
            "mean_cases",
            "variance_to_mean",
            "peak_week_of_year",
            "stl_seasonal_strength",
            "acf_lag_1",
            "acf_lag_52",
            "gap_weeks",
        ]
    ].sort_values("mean_cases", ascending=False)


def _fold_winners_table(metrics_by_fold_macro: pd.DataFrame) -> pd.DataFrame:
    val = metrics_by_fold_macro.loc[metrics_by_fold_macro["role"] == "val"].copy()
    winners = (
        val.sort_values(["track", "fold_id", "MAE_macro", "RMSE_macro", "model"])
        .groupby(["track", "fold_id"], as_index=False)
        .first()
    )
    return winners


def _geo_model_summary(metrics_by_geo_horizon_macro: pd.DataFrame, track: str) -> pd.DataFrame:
    frame = metrics_by_geo_horizon_macro.loc[
        (metrics_by_geo_horizon_macro["track"] == track) & (metrics_by_geo_horizon_macro["role"] == "val")
    ].copy()
    out = (
        frame.groupby(["geo_id", "model"], as_index=False)[["MAE_macro", "RMSE_macro", "bias_macro"]]
        .mean()
        .sort_values(["geo_id", "MAE_macro", "RMSE_macro"])
    )
    return out


def _current_forecast_tables(forecast_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    forecast = pd.read_csv(forecast_csv)
    country = forecast.loc[forecast["geo_id"] == "COUNTRY_TOTAL"].copy()
    primary = forecast.loc[forecast["is_primary_model"] == 1].copy()
    return country.sort_values(["model_rank_from_backtest", "horizon"]), primary.sort_values(["geo_id", "horizon"])


def build_results_snapshot(
    backtest_run_id: str,
    forecast_run_id: str,
    operational_leaderboard: pd.DataFrame,
    experimental_leaderboard: pd.DataFrame,
    fold_winners: pd.DataFrame,
    current_country_forecast: pd.DataFrame,
) -> str:
    lines = [
        "# Paper Assets Results Snapshot",
        "",
        f"- Backtest run: `{backtest_run_id}`",
        f"- Forecast run: `{forecast_run_id}`",
        "",
        "## Main Result",
        "",
        f"- Operational winner: `{operational_leaderboard.iloc[0]['model']}` with MAE_macro={operational_leaderboard.iloc[0]['MAE_macro']:.3f} and RMSE_macro={operational_leaderboard.iloc[0]['RMSE_macro']:.3f}.",
        f"- Experimental winner: `{experimental_leaderboard.iloc[0]['model']}` with MAE_macro={experimental_leaderboard.iloc[0]['MAE_macro']:.3f} and RMSE_macro={experimental_leaderboard.iloc[0]['RMSE_macro']:.3f}.",
        "",
        "## Fold Story",
        "",
    ]

    for track, frame in fold_winners.groupby("track", dropna=False):
        lines.append(f"### {track.title()}")
        lines.append("")
        for row in frame.itertuples(index=False):
            lines.append(
                f"- {row.fold_id}: {row.model} (MAE_macro={row.MAE_macro:.3f}, RMSE_macro={row.RMSE_macro:.3f})"
            )
        lines.append("")

    lines.extend(["## Current Country Forecasts", ""])
    for model, frame in current_country_forecast.groupby("model", dropna=False):
        values = ", ".join(f"h{int(r.horizon)}={r.y_pred:.1f}" for r in frame.sort_values("horizon").itertuples(index=False))
        primary = bool(frame["is_primary_model"].iloc[0])
        lines.append(f"- {model} (primary={primary}): {values}")

    lines.extend(
        [
            "",
            "## Recommended Paper Framing",
            "",
            "- The strongest applied result is that persistence remains the most reliable short-horizon operational model under realistic data constraints.",
            "- The best challenger class is persistence-anchored correction models, but they do not yet beat persistence consistently enough for promotion.",
            "- Weather is scientifically relevant but not operationally dependable in the current coverage window.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def build_asset_guide(asset_dir: Path, tables: list[Path], figures: list[Path]) -> str:
    lines = [
        "# Paper Asset Guide",
        "",
        f"- Asset bundle: `{asset_dir.name}`",
        "",
        "## Tables",
        "",
    ]
    for path in tables:
        lines.append(f"- `{path.name}`")
    lines.extend(["", "## Figures", ""])
    for path in figures:
        lines.append(f"- `{path.name}`")
    lines.extend(
        [
            "",
            "## Suggested Use",
            "",
            "- Use the study-design and dataset-summary tables in Methods.",
            "- Use the leaderboard, fold-winner, and geo-summary tables in Results.",
            "- Use the current-forecast tables in the operational forecasting subsection or appendix.",
            "- Use the leaderboard, horizon, fold, and country-forecast figures as the core result figures.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def generate_paper_assets(backtest_run_dir: Path, forecast_run_dir: Path, out_root: Path) -> dict[str, Any]:
    stamp = _utc_stamp()
    asset_dir = out_root / f"paper_assets_{stamp}"
    tables_dir = asset_dir / "tables"
    figures_dir = asset_dir / "figures"
    ensure_dir(tables_dir)
    ensure_dir(figures_dir)

    eda_dir = Path("reports/eda")
    backtest_metrics = backtest_run_dir / "metrics"
    forecast_metadata = json.loads((forecast_run_dir / "metadata.json").read_text(encoding="utf-8"))
    forecast_csv = Path(forecast_metadata["artifacts"]["report_csv"])

    leaderboard = pd.read_csv(backtest_metrics / "leaderboard.csv")
    leaderboard_by_horizon = pd.read_csv(backtest_metrics / "leaderboard_by_horizon.csv")
    metrics_by_fold_macro = pd.read_csv(backtest_metrics / "metrics_by_fold_macro.csv")
    metrics_by_geo_horizon_macro = pd.read_csv(backtest_metrics / "metrics_by_geo_horizon_macro.csv")

    study_design = _study_design_table()
    dataset_summary = _dataset_summary_table(eda_dir)
    operational_leaderboard = leaderboard.loc[(leaderboard["track"] == "operational") & (leaderboard["role"] == "val")].sort_values(["MAE_macro", "RMSE_macro"])
    experimental_leaderboard = leaderboard.loc[(leaderboard["track"] == "experimental") & (leaderboard["role"] == "val")].sort_values(["MAE_macro", "RMSE_macro"])
    fold_winners = _fold_winners_table(metrics_by_fold_macro)
    horizon_winners = (
        leaderboard_by_horizon.sort_values(["track", "horizon", "MAE_macro", "RMSE_macro", "model"])
        .groupby(["track", "horizon"], as_index=False)
        .first()
    )
    operational_geo_model_summary = _geo_model_summary(metrics_by_geo_horizon_macro, track="operational")
    experimental_geo_model_summary = _geo_model_summary(metrics_by_geo_horizon_macro, track="experimental")
    current_country_forecast, current_primary_forecast = _current_forecast_tables(forecast_csv)

    table_map = {
        "table_01_study_design.csv": study_design,
        "table_02_dataset_summary.csv": dataset_summary,
        "table_03_operational_leaderboard.csv": operational_leaderboard,
        "table_04_experimental_leaderboard.csv": experimental_leaderboard,
        "table_05_fold_winners.csv": fold_winners,
        "table_06_horizon_winners.csv": horizon_winners,
        "table_07_operational_geo_model_summary.csv": operational_geo_model_summary,
        "table_08_experimental_geo_model_summary.csv": experimental_geo_model_summary,
        "table_09_current_country_forecast.csv": current_country_forecast,
        "table_10_current_primary_forecast.csv": current_primary_forecast,
    }
    for name, frame in table_map.items():
        frame.to_csv(tables_dir / name, index=False)

    _bar_plot(
        df=operational_leaderboard.reset_index(drop=True),
        label_col="model",
        value_cols=["MAE_macro", "RMSE_macro"],
        title="Operational Validation Leaderboard",
        out_path=figures_dir / "fig_01_operational_leaderboard.png",
    )
    _line_plot(
        df=leaderboard_by_horizon.loc[leaderboard_by_horizon["track"] == "operational"].copy(),
        x_col="horizon",
        y_col="MAE_macro",
        group_col="model",
        title="Operational Validation MAE by Horizon",
        out_path=figures_dir / "fig_02_operational_horizon_mae.png",
        xlabel="Horizon (weeks)",
        ylabel="MAE (macro)",
    )
    fold_heat = (
        metrics_by_fold_macro.loc[(metrics_by_fold_macro["track"] == "operational") & (metrics_by_fold_macro["role"] == "val")]
        .pivot(index="model", columns="fold_id", values="MAE_macro")
        .sort_index()
    )
    _heatmap(
        data=fold_heat,
        title="Operational Validation MAE by Fold",
        out_path=figures_dir / "fig_03_operational_fold_mae_heatmap.png",
        xlabel="Fold",
        ylabel="Model",
    )
    geo_heat = (
        operational_geo_model_summary.pivot(index="geo_id", columns="model", values="MAE_macro").sort_index()
    )
    _heatmap(
        data=geo_heat,
        title="Operational Validation MAE by Geography",
        out_path=figures_dir / "fig_04_operational_geo_mae_heatmap.png",
        xlabel="Model",
        ylabel="Geography",
    )
    _line_plot(
        df=current_country_forecast.copy(),
        x_col="horizon",
        y_col="y_pred",
        group_col="model",
        title="Current Country Forecast by Model",
        out_path=figures_dir / "fig_05_current_country_forecasts.png",
        xlabel="Horizon (weeks)",
        ylabel="Forecasted weekly cases",
    )
    _line_plot(
        df=leaderboard_by_horizon.loc[leaderboard_by_horizon["track"] == "experimental"].copy(),
        x_col="horizon",
        y_col="MAE_macro",
        group_col="model",
        title="Experimental Validation MAE by Horizon",
        out_path=figures_dir / "fig_s1_experimental_horizon_mae.png",
        xlabel="Horizon (weeks)",
        ylabel="MAE (macro)",
    )

    snapshot = build_results_snapshot(
        backtest_run_id=backtest_run_dir.name,
        forecast_run_id=forecast_run_dir.name,
        operational_leaderboard=operational_leaderboard.reset_index(drop=True),
        experimental_leaderboard=experimental_leaderboard.reset_index(drop=True),
        fold_winners=fold_winners,
        current_country_forecast=current_country_forecast,
    )
    snapshot_path = asset_dir / "results_snapshot.md"
    _write_markdown(snapshot_path, snapshot)

    table_paths = [tables_dir / name for name in table_map]
    figure_paths = sorted(figures_dir.glob("*.png"))
    asset_guide = build_asset_guide(asset_dir, table_paths, figure_paths)
    _write_markdown(asset_dir / "asset_guide.md", asset_guide)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "backtest_run_dir": str(backtest_run_dir),
        "forecast_run_dir": str(forecast_run_dir),
        "forecast_report_csv": str(forecast_csv),
        "tables": [str(path) for path in table_paths],
        "figures": [str(path) for path in figure_paths],
        "results_snapshot": str(snapshot_path),
    }
    _write_json(asset_dir / "asset_manifest.json", metadata)
    return {"asset_dir": asset_dir, "metadata": metadata}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate manuscript-facing tables and figures from the tracked runs.")
    parser.add_argument("--backtest-run-dir", default=None, help="Backtest run directory. Defaults to latest modeling_tracks run.")
    parser.add_argument("--forecast-run-dir", default=None, help="Forecast run directory. Defaults to latest operational_current forecast run.")
    parser.add_argument("--outdir", default="reports/paper", help="Root directory for the paper asset bundle.")
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    runs_root = Path("artifacts/runs")
    forecasts_root = Path("artifacts/forecasts")
    backtest_run_dir = Path(args.backtest_run_dir) if args.backtest_run_dir else _latest_dir(runs_root, "modeling_tracks_")
    forecast_run_dir = Path(args.forecast_run_dir) if args.forecast_run_dir else _latest_dir(forecasts_root, "operational_current_")
    out_root = Path(args.outdir)
    ensure_dir(out_root)

    result = generate_paper_assets(backtest_run_dir=backtest_run_dir, forecast_run_dir=forecast_run_dir, out_root=out_root)
    print(f"[paper] asset_dir -> {result['asset_dir']}")
    print(f"[paper] manifest  -> {result['asset_dir'] / 'asset_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
