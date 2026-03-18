"""
Builds the weekly panel by joining cases and weather on (geo_id, week_start_date).

Outputs:
  - data/interim/panel_raw.parquet
  - data/interim/panel_sample.csv  (first 200 rows)

Notes:
  * We canonicalize all week_start_date values to **Monday** after the join.
  * We sort and drop duplicates on (geo_id, week_start_date).
"""

import sys
from pathlib import Path

import pandas as pd

from dengue.ingest.cases_reader import load_cases_wide_to_long
from dengue.ingest.weather_reader import load_weather_folder
from dengue.utils.io import load_yaml, resolve_paths


def find_cases_csv(raw_dir: Path) -> Path:
    """Auto-detect a likely weekly cases CSV in data/raw."""
    patterns = [
        "*final*weekly*.csv",
        "*cases*weekly*.csv",
        "*EW*new*cases*.csv",
    ]
    for pat in patterns:
        hits = sorted(raw_dir.glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"Could not auto-detect cases file in {raw_dir}. Put it there and rerun."
    )


def main() -> int:
    # --- paths & configs ---
    cfg = load_yaml("configs/data.yaml")
    paths = resolve_paths(cfg)
    raw = paths["raw"]
    interim = paths["interim"]

    # --- cases ---
    cases_csv = find_cases_csv(raw)
    print(f"[build_panel] Cases file: {cases_csv.name}")

    print("[build_panel] Loading cases…")
    df_cases = load_cases_wide_to_long(cases_csv)
    print(
        "[build_panel] Cases: "
        f"{df_cases['geo_id'].nunique()} geos, {len(df_cases):,} records, "
        f"{df_cases['week_start_date'].min():%Y-%m-%d} → {df_cases['week_start_date'].max():%Y-%m-%d}"
    )

    # --- weather ---
    print("[build_panel] Loading weather…")
    df_wx = load_weather_folder(raw)
    print(
        "[build_panel] Weather: "
        f"{df_wx['geo_id'].nunique()} geos, {len(df_wx):,} records, "
        f"{df_wx['week_start_date'].min():%Y-%m-%d} → {df_wx['week_start_date'].max():%Y-%m-%d}"
    )

    # --- join ---
    print("[build_panel] Joining on (geo_id, week_start_date)…")
    panel = pd.merge(
        df_cases, df_wx, on=["geo_id", "geo_name", "week_start_date"], how="left"
    )

    before_n = len(panel)

    # Canonicalize to Monday-anchored week starts everywhere
    panel["week_start_date"] = pd.to_datetime(panel["week_start_date"], errors="coerce")
    panel["week_start_date"] = (
        panel["week_start_date"].dt.to_period("W-SUN").dt.start_time
    )

    # Sort & drop any accidental duplicates
    panel = panel.sort_values(["geo_id", "week_start_date"]).drop_duplicates(
        ["geo_id", "week_start_date"]
    )

    after_n = len(panel)

    # Monotone + Monday check per geo (Series-only to avoid FutureWarning)
    def _geo_ok(s: pd.Series) -> bool:
        return bool(s.is_monotonic_increasing and (s.dt.dayofweek == 0).all())

    chk = (
        panel.groupby("geo_id", group_keys=False)["week_start_date"]
        .apply(_geo_ok)
        .rename("ok")
    )
    if not chk.all():
        bad = chk[~chk].index.tolist()
        print(
            f"[WARN] Non-monotone or non-Monday weeks for geos: {bad}", file=sys.stderr
        )

    # Final date range (after canonicalization)
    global_start = panel["week_start_date"].min()
    global_end = panel["week_start_date"].max()
    print(
        f"[build_panel] Canonicalized to Mondays. Rows: {before_n:,} → {after_n:,}. "
        f"Date range: {global_start:%Y-%m-%d} → {global_end:%Y-%m-%d}"
    )

    # --- save ---
    out_parquet = interim / "panel_raw.parquet"
    out_sample = interim / "panel_sample.csv"
    panel.to_parquet(out_parquet, index=False)
    panel.head(200).to_csv(out_sample, index=False)

    # quick column peek
    col_list = list(panel.columns)
    preview = ", ".join(col_list[: min(20, len(col_list))])
    print(f"[build_panel] Saved: {out_parquet} ({len(panel):,} rows)")
    print(f"[build_panel] Sample: {out_sample}")
    print("[build_panel] Columns:", preview, "…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
