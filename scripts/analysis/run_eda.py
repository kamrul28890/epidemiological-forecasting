#!/usr/bin/env python
"""
Comprehensive EDA for weekly dengue forecasting.

This script is designed for panel count time series with exogenous weather covariates.
It produces:
  - dataset overview and metadata
  - case-series summaries
  - missingness and covariate coverage diagnostics
  - stationarity tests
  - seasonality and STL diagnostics
  - annual peak and outbreak concentration summaries
  - baseline forecastability diagnostics
  - climate lead-lag correlation tables
  - cross-geo correlation diagnostics
  - a Markdown summary plus figures and CSV outputs

Recommended usage:
  python -m scripts.analysis.run_eda
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import warnings
from pathlib import Path
from typing import Iterable

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

from dengue.ingest.cases_reader import load_cases_wide_to_long
from dengue.ingest.weather_reader import load_weather_folder
from dengue.utils.io import ensure_dir

CLIMATE_COLUMNS = [
    "tmean",
    "humidity",
    "rain",
    "dewpoint",
    "wind10",
    "wind100",
    "winddir10",
    "gust10",
    "soil_moist_0_7cm",
]
CORE_GEO_COLUMNS = [
    "dhaka_metro_EW_new_cases",
    "dhaka_div_out_metro_EW_new_cases",
    "Mymensingh_EW_new_cases",
    "Chattogram_EW_new_cases",
    "Khulna_EW_new_cases",
    "Rajshahi_EW_new_cases",
    "Rangpur_EW_new_cases",
    "Barishal_EW_new_cases",
    "Sylhet_EW_new_cases",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cases-path",
        default="data/raw/1_df_final_weekly_Jan_1_2022_to_August_23_2025.csv",
    )
    ap.add_argument("--weather-dir", default="data/raw")
    ap.add_argument("--output-dir", default="reports/eda")
    ap.add_argument("--season-length", type=int, default=52)
    ap.add_argument("--max-climate-lag", type=int, default=12)
    return ap.parse_args()


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den not in (0, 0.0) else float("nan")


def canonical_week(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, errors="coerce").dt.to_period("W-SUN").dt.start_time


def complete_series(df_geo: pd.DataFrame) -> pd.DataFrame:
    df_geo = df_geo.sort_values("week_start_date").copy()
    idx = pd.date_range(
        df_geo["week_start_date"].min(),
        df_geo["week_start_date"].max(),
        freq="W-MON",
    )
    out = (
        df_geo.set_index("week_start_date")
        .reindex(idx)
        .rename_axis("week_start_date")
        .reset_index()
    )
    for col in ["geo_id", "geo_name"]:
        if col in df_geo.columns:
            out[col] = df_geo[col].dropna().iloc[0]
    return out


def load_inputs(cases_path: str | Path, weather_dir: str | Path) -> tuple[pd.DataFrame, ...]:
    raw_cases = pd.read_csv(cases_path, parse_dates=["date"])
    raw_cases["date"] = canonical_week(raw_cases["date"])

    cases_long = load_cases_wide_to_long(cases_path)
    cases_long["week_start_date"] = canonical_week(cases_long["week_start_date"])

    weather_long = load_weather_folder(weather_dir)
    weather_long["week_start_date"] = canonical_week(weather_long["week_start_date"])

    panel = pd.merge(
        cases_long,
        weather_long,
        on=["geo_id", "geo_name", "week_start_date"],
        how="left",
    )
    panel = panel.sort_values(["geo_id", "week_start_date"]).reset_index(drop=True)
    return raw_cases, cases_long, weather_long, panel


def output_paths(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    figures = root / "figures"
    tables = root / "tables"
    ensure_dir(root)
    ensure_dir(figures)
    ensure_dir(tables)
    return {"root": root, "figures": figures, "tables": tables}


def dataset_overview(
    raw_cases: pd.DataFrame,
    cases_long: pd.DataFrame,
    weather_long: pd.DataFrame,
    panel: pd.DataFrame,
    season_length: int,
    max_climate_lag: int,
) -> dict[str, object]:
    weather_max = pd.to_datetime(weather_long["week_start_date"]).max()
    case_max = pd.to_datetime(cases_long["week_start_date"]).max()
    return {
        "n_case_rows_raw": int(len(raw_cases)),
        "n_case_rows_long": int(len(cases_long)),
        "n_weather_rows_long": int(len(weather_long)),
        "n_panel_rows": int(len(panel)),
        "n_geos": int(cases_long["geo_id"].nunique()),
        "case_date_min": str(pd.to_datetime(cases_long["week_start_date"]).min().date()),
        "case_date_max": str(case_max.date()),
        "weather_date_min": str(pd.to_datetime(weather_long["week_start_date"]).min().date()),
        "weather_date_max": str(weather_max.date()),
        "weather_gap_weeks_to_latest_cases": int((case_max - weather_max).days // 7),
        "season_length": int(season_length),
        "max_climate_lag_tested": int(max_climate_lag),
        "geos": sorted(cases_long["geo_id"].unique().tolist()),
    }


def summarize_case_series(cases_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for geo_id, g in cases_long.groupby("geo_id"):
        s = complete_series(g)
        y = pd.to_numeric(s["cases"], errors="coerce")
        top_k = max(1, int(np.ceil(len(y.dropna()) * 0.1)))
        top_share = safe_div(y.nlargest(top_k).sum(), y.sum())
        rows.append(
            {
                "geo_id": geo_id,
                "start_date": s["week_start_date"].min().date().isoformat(),
                "end_date": s["week_start_date"].max().date().isoformat(),
                "weeks": int(len(s)),
                "missing_weeks": int(y.isna().sum()),
                "zero_pct": float((y.fillna(0) == 0).mean() * 100),
                "mean_cases": float(y.mean()),
                "median_cases": float(y.median()),
                "std_cases": float(y.std()),
                "cv": safe_div(float(y.std()), float(y.mean())),
                "max_cases": float(y.max()),
                "p95_cases": float(y.quantile(0.95)),
                "p99_cases": float(y.quantile(0.99)),
                "sum_cases": float(y.sum()),
                "skewness": float(skew(y.dropna(), bias=False)),
                "variance_to_mean": safe_div(float(y.var()), float(y.mean())),
                "top_10pct_weeks_share": float(top_share),
            }
        )
    return pd.DataFrame(rows).sort_values("sum_cases", ascending=False)


def covariate_coverage(panel: pd.DataFrame, climate_columns: Iterable[str]) -> pd.DataFrame:
    rows = []
    for geo_id, g in panel.groupby("geo_id"):
        for col in climate_columns:
            if col not in g.columns:
                continue
            valid = g.loc[g[col].notna(), "week_start_date"]
            rows.append(
                {
                    "geo_id": geo_id,
                    "covariate": col,
                    "non_null_weeks": int(g[col].notna().sum()),
                    "pct_non_null": float(g[col].notna().mean() * 100),
                    "first_non_null_date": (
                        valid.min().date().isoformat() if not valid.empty else None
                    ),
                    "last_non_null_date": (
                        valid.max().date().isoformat() if not valid.empty else None
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_adf(series: pd.Series) -> dict[str, float]:
    series = series.dropna()
    if len(series) < 20 or series.nunique() < 3:
        return {"adf_stat": np.nan, "adf_pvalue": np.nan}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            stat, pvalue, *_ = adfuller(series, autolag="AIC")
        except Exception:
            return {"adf_stat": np.nan, "adf_pvalue": np.nan}
    return {"adf_stat": float(stat), "adf_pvalue": float(pvalue)}


def run_kpss(series: pd.Series) -> dict[str, float]:
    series = series.dropna()
    if len(series) < 20 or series.nunique() < 3:
        return {"kpss_stat": np.nan, "kpss_pvalue": np.nan}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            stat, pvalue, *_ = kpss(series, regression="c", nlags="auto")
        except Exception:
            return {"kpss_stat": np.nan, "kpss_pvalue": np.nan}
    return {"kpss_stat": float(stat), "kpss_pvalue": float(pvalue)}


def stationarity_tests(cases_long: pd.DataFrame, season_length: int) -> pd.DataFrame:
    rows = []
    transforms = {
        "raw": lambda s: s,
        "log1p": lambda s: np.log1p(s),
        "diff_1": lambda s: s.diff(1),
        f"diff_{season_length}": lambda s: s.diff(season_length),
        "log1p_diff_1": lambda s: np.log1p(s).diff(1),
        f"log1p_diff_{season_length}": lambda s: np.log1p(s).diff(season_length),
    }

    for geo_id, g in cases_long.groupby("geo_id"):
        s = complete_series(g)["cases"].astype(float)
        for transform_name, transform in transforms.items():
            transformed = transform(s)
            adf = run_adf(transformed)
            kpss_res = run_kpss(transformed)
            rows.append(
                {
                    "geo_id": geo_id,
                    "transform": transform_name,
                    "n_obs": int(transformed.dropna().shape[0]),
                    **adf,
                    **kpss_res,
                    "stationary_candidate": bool(
                        pd.notna(adf["adf_pvalue"])
                        and pd.notna(kpss_res["kpss_pvalue"])
                        and adf["adf_pvalue"] < 0.05
                        and kpss_res["kpss_pvalue"] > 0.05
                    ),
                }
            )
    return pd.DataFrame(rows)


def seasonal_strength(stl_result: STL) -> tuple[float, float]:
    resid = pd.Series(stl_result.resid)
    trend = pd.Series(stl_result.trend)
    seasonal = pd.Series(stl_result.seasonal)
    seasonal_strength_value = max(
        0.0,
        1.0 - safe_div(float(resid.var()), float((resid + seasonal).var())),
    )
    trend_strength_value = max(
        0.0,
        1.0 - safe_div(float(resid.var()), float((resid + trend).var())),
    )
    return seasonal_strength_value, trend_strength_value


def seasonality_summary(cases_long: pd.DataFrame, season_length: int) -> pd.DataFrame:
    rows = []
    for geo_id, g in cases_long.groupby("geo_id"):
        s = complete_series(g)
        y = s["cases"].astype(float)
        iso_weeks = s["week_start_date"].dt.isocalendar().week.astype(int)
        weekly_profile = (
            pd.DataFrame({"week": iso_weeks, "cases": y})
            .groupby("week")["cases"]
            .mean()
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stl = STL(y, period=season_length, robust=True).fit()
        season_strength_value, trend_strength_value = seasonal_strength(stl)
        rows.append(
            {
                "geo_id": geo_id,
                "peak_week_of_year": int(weekly_profile.idxmax()),
                "seasonality_index": safe_div(
                    float(weekly_profile.max()), float(weekly_profile.mean())
                ),
                "stl_seasonal_strength": float(season_strength_value),
                "stl_trend_strength": float(trend_strength_value),
                "acf_lag_1": float(y.autocorr(lag=1)),
                "acf_lag_4": float(y.autocorr(lag=4)),
                "acf_lag_13": float(y.autocorr(lag=13)),
                "acf_lag_26": float(y.autocorr(lag=26)),
                "acf_lag_52": float(y.autocorr(lag=52)),
            }
        )
    return pd.DataFrame(rows).sort_values("stl_seasonal_strength", ascending=False)


def annual_peak_summary(cases_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    tmp = cases_long.copy()
    tmp["year"] = pd.to_datetime(tmp["week_start_date"]).dt.year
    for (geo_id, year), g in tmp.groupby(["geo_id", "year"]):
        peak_idx = g["cases"].idxmax()
        peak_row = g.loc[peak_idx]
        rows.append(
            {
                "geo_id": geo_id,
                "year": int(year),
                "annual_total_cases": float(g["cases"].sum()),
                "peak_week_start": pd.to_datetime(peak_row["week_start_date"]).date().isoformat(),
                "peak_cases": float(peak_row["cases"]),
                "peak_share_of_year": safe_div(float(peak_row["cases"]), float(g["cases"].sum())),
            }
        )
    return pd.DataFrame(rows).sort_values(["geo_id", "year"])


def baseline_forecastability(
    cases_long: pd.DataFrame,
    season_length: int,
    horizons: Iterable[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for geo_id, g in cases_long.groupby("geo_id"):
        s = complete_series(g)
        s = s.set_index("week_start_date")["cases"].astype(float)
        for horizon in horizons:
            y_true = s.shift(-horizon)
            persistence = s
            seasonal_naive = y_true.copy()
            for target_week in y_true.index:
                ref_week = target_week - pd.Timedelta(weeks=season_length)
                seasonal_naive.loc[target_week] = s.get(ref_week, np.nan)

            for model_name, y_pred in {
                "persistence": persistence,
                "seasonal_naive": seasonal_naive,
            }.items():
                eval_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
                if eval_df.empty:
                    continue
                err = eval_df["y_true"] - eval_df["y_pred"]
                rows.append(
                    {
                        "geo_id": geo_id,
                        "horizon": int(horizon),
                        "model": model_name,
                        "n_obs": int(len(eval_df)),
                        "MAE": float(err.abs().mean()),
                        "RMSE": float(np.sqrt((err**2).mean())),
                    }
                )
    by_geo = pd.DataFrame(rows).sort_values(["model", "horizon", "geo_id"])
    overall = (
        by_geo.groupby(["model", "horizon"], as_index=False)
        .agg(
            MAE_macro=("MAE", "mean"),
            RMSE_macro=("RMSE", "mean"),
            n_geos=("geo_id", "nunique"),
        )
        .sort_values(["model", "horizon"])
    )
    return by_geo, overall


def climate_lead_lag(
    panel: pd.DataFrame,
    climate_columns: Iterable[str],
    max_lag: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for geo_id, g in panel.groupby("geo_id"):
        g = g.sort_values("week_start_date").copy()
        for covariate in climate_columns:
            if covariate not in g.columns:
                continue
            for lag in range(0, max_lag + 1):
                eval_df = pd.DataFrame(
                    {
                        "cases": g["cases"].astype(float),
                        "covariate": pd.to_numeric(g[covariate], errors="coerce").shift(lag),
                    }
                ).dropna()
                if len(eval_df) < 30 or eval_df["covariate"].nunique() < 3:
                    continue
                rows.append(
                    {
                        "geo_id": geo_id,
                        "covariate": covariate,
                        "lag_weeks": int(lag),
                        "n_obs": int(len(eval_df)),
                        "pearson_corr": float(eval_df["cases"].corr(eval_df["covariate"])),
                        "spearman_corr": float(
                            eval_df["cases"].corr(eval_df["covariate"], method="spearman")
                        ),
                    }
                )
    all_corr = pd.DataFrame(rows).sort_values(["covariate", "geo_id", "lag_weeks"])
    if all_corr.empty:
        return all_corr, all_corr
    best = (
        all_corr.assign(abs_spearman=lambda d: d["spearman_corr"].abs())
        .sort_values(["geo_id", "covariate", "abs_spearman"], ascending=[True, True, False])
        .groupby(["geo_id", "covariate"], as_index=False)
        .first()
        .drop(columns=["abs_spearman"])
    )
    return all_corr, best


def cross_geo_correlations(cases_long: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        cases_long.pivot(index="week_start_date", columns="geo_id", values="cases")
        .sort_index()
    )
    return pivot.corr(method="spearman")


def hierarchy_consistency(raw_cases: pd.DataFrame) -> pd.DataFrame:
    other_sum = raw_cases[CORE_GEO_COLUMNS].sum(axis=1)
    checks = [
        {
            "check": "dhaka_division_equals_metro_plus_out_metro",
            "max_abs_diff": float(
                (
                    raw_cases["dhaka_division_EW_new_cases_computed"]
                    - raw_cases["dhaka_metro_EW_new_cases"]
                    - raw_cases["dhaka_div_out_metro_EW_new_cases"]
                )
                .abs()
                .max()
            ),
        },
        {
            "check": "country_total_equals_bottom_level_sum",
            "max_abs_diff": float(
                (raw_cases["country_wide_total_EW_new_cases"] - other_sum).abs().max()
            ),
        },
    ]
    return pd.DataFrame(checks)


def climate_gap_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    latest_case_week = pd.to_datetime(panel["week_start_date"]).max()
    for geo_id, g in panel.groupby("geo_id"):
        valid = (
            g.loc[g["rain"].notna(), "week_start_date"]
            if "rain" in g.columns
            else pd.Series(dtype="datetime64[ns]")
        )
        last_cov = pd.to_datetime(valid).max() if not valid.empty else pd.NaT
        rows.append(
            {
                "geo_id": geo_id,
                "latest_case_week": latest_case_week.date().isoformat(),
                "last_covariate_week": last_cov.date().isoformat() if pd.notna(last_cov) else None,
                "gap_weeks": int((latest_case_week - last_cov).days // 7) if pd.notna(last_cov) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def plot_case_small_multiples(cases_long: pd.DataFrame, path: Path) -> None:
    geos = sorted(cases_long["geo_id"].unique().tolist())
    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True)
    axes = axes.flatten()
    for ax, geo_id in zip(axes, geos):
        g = cases_long[cases_long["geo_id"] == geo_id].sort_values("week_start_date")
        ax.plot(g["week_start_date"], g["cases"], color="tab:red", linewidth=1.5)
        ax.set_title(geo_id)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Weekly Dengue Cases by Geography", fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_weekly_profiles(cases_long: pd.DataFrame, path: Path) -> None:
    geos = sorted(cases_long["geo_id"].unique().tolist())
    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True, sharey=False)
    axes = axes.flatten()
    for ax, geo_id in zip(axes, geos):
        g = cases_long[cases_long["geo_id"] == geo_id].copy()
        g["iso_week"] = pd.to_datetime(g["week_start_date"]).dt.isocalendar().week.astype(int)
        profile = g.groupby("iso_week")["cases"].mean()
        ax.plot(profile.index, profile.values, color="tab:blue", linewidth=1.5)
        ax.set_title(geo_id)
        ax.set_xlim(1, 53)
    fig.suptitle("Average Cases by ISO Week", fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_bar(summary: pd.DataFrame, value_col: str, path: Path, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    order = summary.sort_values(value_col, ascending=False)
    ax.bar(order["geo_id"], order[value_col], color="tab:green")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_corr_heatmap(corr_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr_df.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(corr_df.index)))
    ax.set_yticklabels(corr_df.index)
    ax.set_title("Spearman Correlation Across Geographies")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_climate_gap(gap_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    order = gap_df.sort_values("gap_weeks", ascending=False)
    ax.bar(order["geo_id"], order["gap_weeks"], color="tab:orange")
    ax.set_title("Weeks Between Latest Cases and Latest Climate Covariates")
    ax.set_ylabel("Gap (weeks)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_baselines(overall_df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    for model, g in overall_df.groupby("model"):
        axes[0].plot(g["horizon"], g["MAE_macro"], marker="o", label=model)
        axes[1].plot(g["horizon"], g["RMSE_macro"], marker="o", label=model)
    axes[0].set_title("Baseline MAE by Horizon")
    axes[0].set_ylabel("MAE (macro)")
    axes[1].set_title("Baseline RMSE by Horizon")
    axes[1].set_ylabel("RMSE (macro)")
    for ax in axes:
        ax.set_xlabel("Horizon (weeks)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_climate_heatmap(best_corr_df: pd.DataFrame, value_col: str, path: Path, title: str) -> None:
    pivot = best_corr_df.pivot(index="geo_id", columns="covariate", values=value_col).sort_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    values = pivot.values.astype(float)
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    if np.allclose(vmin, vmax, equal_nan=True):
        vmin, vmax = 0.0, 1.0
    im = ax.imshow(values, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_stl_and_acf(cases_long: pd.DataFrame, figures_dir: Path, season_length: int) -> None:
    for geo_id, g in cases_long.groupby("geo_id"):
        s = complete_series(g)
        y = s["cases"].astype(float)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stl = STL(y, period=season_length, robust=True).fit()

        fig = stl.plot()
        fig.set_size_inches(12, 8)
        fig.suptitle(f"STL Decomposition - {geo_id}", fontsize=14)
        fig.tight_layout()
        fig.savefig(figures_dir / f"stl_{geo_id}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4))
        plot_acf(y.dropna(), ax=ax, lags=min(60, len(y.dropna()) - 1), zero=False)
        ax.set_title(f"ACF - {geo_id}")
        fig.tight_layout()
        fig.savefig(figures_dir / f"acf_{geo_id}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        rolling_mean = y.rolling(8, min_periods=4).mean()
        rolling_std = y.rolling(8, min_periods=4).std()
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(s["week_start_date"], y, color="tab:red", alpha=0.5, label="cases")
        axes[0].plot(
            s["week_start_date"], rolling_mean, color="black", linewidth=1.5, label="8-week mean"
        )
        axes[0].set_title(f"Rolling Mean - {geo_id}")
        axes[0].legend()
        axes[1].plot(s["week_start_date"], rolling_std, color="tab:purple", linewidth=1.5)
        axes[1].set_title(f"Rolling Std - {geo_id}")
        fig.tight_layout()
        fig.savefig(figures_dir / f"rolling_moments_{geo_id}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def write_summary_markdown(
    out_path: Path,
    overview: dict[str, object],
    case_summary: pd.DataFrame,
    seasonality_df: pd.DataFrame,
    stationarity_df: pd.DataFrame,
    hierarchy_df: pd.DataFrame,
    gap_df: pd.DataFrame,
    baseline_overall: pd.DataFrame,
    best_climate_df: pd.DataFrame,
) -> None:
    top_bursty = case_summary.sort_values("variance_to_mean", ascending=False).head(3)
    top_zero = case_summary.sort_values("zero_pct", ascending=False).head(3)
    most_seasonal = seasonality_df.sort_values("stl_seasonal_strength", ascending=False).head(3)
    stationary_candidates = stationarity_df[stationarity_df["stationary_candidate"]]
    best_baseline = baseline_overall.sort_values(["horizon", "MAE_macro"]).groupby("horizon").first()

    lines = [
        "# EDA Summary",
        "",
        "## Dataset Overview",
        f"- Geographies: {overview['n_geos']}",
        f"- Case range: {overview['case_date_min']} to {overview['case_date_max']}",
        f"- Weather range: {overview['weather_date_min']} to {overview['weather_date_max']}",
        f"- Climate coverage gap to latest cases: {overview['weather_gap_weeks_to_latest_cases']} weeks",
        "",
        "## Key Findings",
        f"- Highest zero-inflation geographies: {', '.join(top_zero['geo_id'].tolist())}",
        f"- Highest overdispersion geographies: {', '.join(top_bursty['geo_id'].tolist())}",
        f"- Strongest STL seasonality: {', '.join(most_seasonal['geo_id'].tolist())}",
        f"- Number of geo/transform pairs flagged as stationary candidates: {len(stationary_candidates)}",
        "",
        "## Hierarchy Checks",
    ]
    for _, row in hierarchy_df.iterrows():
        lines.append(f"- {row['check']}: max_abs_diff = {row['max_abs_diff']:.3f}")

    lines.extend(["", "## Climate Coverage Gaps"])
    for _, row in gap_df.sort_values("gap_weeks", ascending=False).iterrows():
        lines.append(f"- {row['geo_id']}: {int(row['gap_weeks'])} weeks")

    lines.extend(["", "## Best Baseline By Horizon (MAE)"])
    for horizon, row in best_baseline.iterrows():
        lines.append(
            f"- Horizon {int(horizon)}: {row['model']} "
            f"(MAE={row['MAE_macro']:.2f}, RMSE={row['RMSE_macro']:.2f})"
        )

    if not best_climate_df.empty:
        strongest = (
            best_climate_df.assign(abs_spearman=lambda d: d["spearman_corr"].abs())
            .sort_values("abs_spearman", ascending=False)
            .head(5)
        )
        lines.extend(["", "## Strongest Climate Lead-Lag Associations"])
        for _, row in strongest.iterrows():
            lines.append(
                f"- {row['geo_id']} / {row['covariate']}: lag {int(row['lag_weeks'])}, "
                f"spearman={row['spearman_corr']:.3f}"
            )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = output_paths(args.output_dir)
    raw_cases, cases_long, weather_long, panel = load_inputs(args.cases_path, args.weather_dir)

    overview = dataset_overview(
        raw_cases,
        cases_long,
        weather_long,
        panel,
        args.season_length,
        args.max_climate_lag,
    )
    (out["root"] / "metadata.json").write_text(
        json.dumps(overview, indent=2),
        encoding="utf-8",
    )

    case_summary = summarize_case_series(cases_long)
    coverage_df = covariate_coverage(panel, CLIMATE_COLUMNS)
    stationarity_df = stationarity_tests(cases_long, args.season_length)
    seasonality_df = seasonality_summary(cases_long, args.season_length)
    peak_df = annual_peak_summary(cases_long)
    baseline_geo, baseline_overall = baseline_forecastability(
        cases_long,
        args.season_length,
        horizons=range(1, 13),
    )
    climate_corr_all, climate_corr_best = climate_lead_lag(
        panel,
        CLIMATE_COLUMNS,
        args.max_climate_lag,
    )
    cross_geo_corr = cross_geo_correlations(cases_long)
    hierarchy_df = hierarchy_consistency(raw_cases)
    gap_df = climate_gap_summary(panel)

    save_dataframe(case_summary, out["tables"] / "case_series_summary.csv")
    save_dataframe(coverage_df, out["tables"] / "covariate_coverage.csv")
    save_dataframe(stationarity_df, out["tables"] / "stationarity_tests.csv")
    save_dataframe(seasonality_df, out["tables"] / "seasonality_summary.csv")
    save_dataframe(peak_df, out["tables"] / "annual_peak_summary.csv")
    save_dataframe(baseline_geo, out["tables"] / "baseline_forecastability_by_geo_horizon.csv")
    save_dataframe(baseline_overall, out["tables"] / "baseline_forecastability_overall.csv")
    save_dataframe(climate_corr_all, out["tables"] / "climate_lead_lag_correlations.csv")
    save_dataframe(climate_corr_best, out["tables"] / "climate_lead_lag_best.csv")
    save_dataframe(hierarchy_df, out["tables"] / "hierarchy_consistency.csv")
    save_dataframe(gap_df, out["tables"] / "climate_gap_summary.csv")
    cross_geo_corr.to_csv(out["tables"] / "cross_geo_spearman_correlation.csv")

    plot_case_small_multiples(cases_long, out["figures"] / "cases_by_geo_small_multiples.png")
    plot_weekly_profiles(cases_long, out["figures"] / "weekly_seasonal_profiles.png")
    plot_bar(
        case_summary,
        "zero_pct",
        out["figures"] / "zero_share_by_geo.png",
        "Zero Share by Geography",
        "Zero weeks (%)",
    )
    plot_bar(
        case_summary,
        "variance_to_mean",
        out["figures"] / "overdispersion_by_geo.png",
        "Variance-to-Mean Ratio by Geography",
        "Variance / mean",
    )
    plot_corr_heatmap(cross_geo_corr, out["figures"] / "cross_geo_correlation_heatmap.png")
    plot_climate_gap(gap_df, out["figures"] / "climate_gap_weeks.png")
    plot_baselines(baseline_overall, out["figures"] / "baseline_forecastability.png")
    if not climate_corr_best.empty:
        plot_climate_heatmap(
            climate_corr_best,
            "spearman_corr",
            out["figures"] / "climate_best_spearman_heatmap.png",
            "Best Climate Spearman Correlation by Geography and Covariate",
        )
        plot_climate_heatmap(
            climate_corr_best,
            "lag_weeks",
            out["figures"] / "climate_best_lag_heatmap.png",
            "Best Climate Lag (weeks) by Geography and Covariate",
        )
    plot_stl_and_acf(cases_long, out["figures"], args.season_length)

    write_summary_markdown(
        out["root"] / "summary.md",
        overview,
        case_summary,
        seasonality_df,
        stationarity_df,
        hierarchy_df,
        gap_df,
        baseline_overall,
        climate_corr_best,
    )

    print(f"[eda] wrote outputs under {out['root']}")
    print(f"[eda] tables: {out['tables']}")
    print(f"[eda] figures: {out['figures']}")
    print(f"[eda] summary: {out['root'] / 'summary.md'}")


if __name__ == "__main__":
    main()
