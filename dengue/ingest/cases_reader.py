"""Load & normalize cases (stub)."""
"""
Load weekly dengue cases and convert wide → tidy long:
columns: [geo_id, geo_name, week_start_date, cases]
"""

import re
from pathlib import Path
import pandas as pd

# map wide column names → standardized geo_id + pretty name
GEO_MAP = {
    "dhaka_metro_ew_new_cases": ("DHA_METRO", "Dhaka Metro"),
    "dhaka_division_ew_new_cases_computed": ("DHA_DIV", "Dhaka Division"),
    "mymensingh_ew_new_cases": ("MYM", "Mymensingh"),
    "chattogram_ew_new_cases": ("CHA", "Chattogram"),
    "khulna_ew_new_cases": ("KHU", "Khulna"),
    "rajshahi_ew_new_cases": ("RAJ", "Rajshahi"),
    "rangpur_ew_new_cases": ("RAN", "Rangpur"),
    "barishal_ew_new_cases": ("BAR", "Barishal"),
    "sylhet_ew_new_cases": ("SYL", "Sylhet"),
    # optional aggregates if you want them later:
    "tot_division_except_dhaka_metro_ew_new_cases": ("NON_DHA_METRO", "All Except Dhaka Metro"),
    "country_wide_total_ew_new_cases": ("NATIONAL", "Bangladesh Total"),
}

def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

def load_cases_wide_to_long(csv_path: str | Path, date_col: str = "date") -> pd.DataFrame:
    p = Path(csv_path)
    df = pd.read_csv(p)
    if date_col not in df.columns:
        raise ValueError(f"Expected date column '{date_col}' in {p.name}; got {df.columns.tolist()}")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["week_start_date"] = df[date_col].dt.to_period("W-SUN").dt.start_time

    

    # find case columns that match our GEO_MAP keys
    cols_norm = {c: _normalize(c) for c in df.columns}
    inv = {v: k for k, v in cols_norm.items()}  # normalized -> original

    selected = []
    for norm_key, (geo_id, geo_name) in GEO_MAP.items():
        if norm_key in inv:
            selected.append((inv[norm_key], geo_id, geo_name))

    if not selected:
        raise ValueError("Could not find any known case columns. Check column names or update GEO_MAP.")

    out_frames = []
    for orig_col, geo_id, geo_name in selected:
        tmp = df[[ "week_start_date", orig_col ]].copy()
        tmp.rename(columns={orig_col: "cases"}, inplace=True)
        tmp["geo_id"] = geo_id
        tmp["geo_name"] = geo_name
        out_frames.append(tmp)

    long_df = pd.concat(out_frames, ignore_index=True)
    # basic QC
    long_df = long_df.dropna(subset=["week_start_date"])
    # enforce integers for cases when possible
    long_df["cases"] = pd.to_numeric(long_df["cases"], errors="coerce")
    long_df = long_df.dropna(subset=["cases"])
    long_df["cases"] = long_df["cases"].astype(int).clip(lower=0)
    # keep only the 8 primary geos for now
    keep = {"BAR","CHA","DHA_DIV","DHA_METRO","KHU","MYM","RAJ","RAN","SYL"}
    long_df = long_df[long_df["geo_id"].isin(keep)]
    return long_df[["geo_id","geo_name","week_start_date","cases"]].sort_values(["geo_id","week_start_date"])
