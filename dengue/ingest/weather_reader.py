"""Load weekly weather CSVs (stub)."""

"""
Load weekly weather CSVs into a tidy long table:
columns: [geo_id, geo_name, week_start_date, tmean, humidity, rain, dewpoint, wind10, wind100, soil_temp_0_7cm, soil_moist_0_7cm, ...]
(Only keeps columns that exist in each file.)
"""

import re
from pathlib import Path

import pandas as pd

# filename detection to geo_id/name
FILENAME_TO_GEO = {
    "barisal_division": ("BAR", "Barishal"),
    "chattogram_division": ("CHA", "Chattogram"),
    "dhaka_division": ("DHA_DIV", "Dhaka Division"),
    "dhaka_metro_only": ("DHA_METRO", "Dhaka Metro"),
    "khulna_division": ("KHU", "Khulna"),
    "mymensingh_division": ("MYM", "Mymensingh"),
    "rajshahi_division": ("RAJ", "Rajshahi"),
    "rangpur_division": ("RAN", "Rangpur"),
    "sylhet_division": ("SYL", "Sylhet"),
}

RENAME_MAP = {
    "temperature_2m": "tmean",
    "relative_humidity_2m": "humidity",
    "total_precipitation": "rain",
    "dew_point_2m": "dewpoint",
    "wind_speed_10m": "wind10",
    "wind_speed_100m": "wind100",
    "wind_direction_10m": "winddir10",
    "wind_gusts_10m": "gust10",
    "soil_temperature_0_to_7cm": "soil_temp_0_7cm",
    "soil_moisture_0_to_7cm": "soil_moist_0_7cm",
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _geo_from_filename(path: Path):
    s = _norm(path.stem)
    for key, meta in FILENAME_TO_GEO.items():
        if key in s:
            return meta
    return None


def load_weather_folder(folder: str | Path) -> pd.DataFrame:
    folder = Path(folder)
    files = sorted(folder.glob("df_weather_weekly_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No weather files found in {folder} matching 'df_weather_weekly_*.csv'"
        )

    frames = []
    for f in files:
        meta = _geo_from_filename(f)
        if meta is None:
            continue
        geo_id, geo_name = meta
        dfw = pd.read_csv(f)
        # find date col
        date_col = None
        for c in dfw.columns:
            if "week_start" in c.lower() or c.lower() == "date":
                date_col = c
                break
        if date_col is None:
            raise ValueError(f"Could not find a week/date column in {f.name}")
        dfw[date_col] = pd.to_datetime(dfw[date_col], errors="coerce")
        dfw["week_start_date"] = dfw[date_col].dt.to_period("W-SUN").dt.start_time

        # rename if present
        cols = {c: RENAME_MAP.get(c, c) for c in dfw.columns}
        dfw = dfw.rename(columns=cols)

        keep_cols = [
            c
            for c in [
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
            if c in dfw.columns
        ]

        out = dfw[["week_start_date"] + keep_cols].copy()
        out["geo_id"] = geo_id
        out["geo_name"] = geo_name
        frames.append(out)

    if not frames:
        raise RuntimeError(
            "Weather files were found, but none matched expected naming patterns."
        )
    long_wx = pd.concat(frames, ignore_index=True)
    # drop NA dates
    long_wx = long_wx.dropna(subset=["week_start_date"])
    # sort
    cols_order = ["geo_id", "geo_name", "week_start_date"] + [
        c for c in long_wx.columns if c not in {"geo_id", "geo_name", "week_start_date"}
    ]
    return long_wx[cols_order].sort_values(["geo_id", "week_start_date"])
