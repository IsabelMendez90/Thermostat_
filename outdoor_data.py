from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from app_config import OUTDOOR_LOCATIONS_PATH, OUTDOOR_TEMPS_DIR


def load_outdoor_locations() -> Optional[pd.DataFrame]:
    try:
        if OUTDOOR_LOCATIONS_PATH.exists():
            return pd.read_csv(OUTDOOR_LOCATIONS_PATH)
    except Exception:
        return None
    return None


def find_nearest_outdoor_location(lat: float, lon: float) -> Tuple[Optional[int], float]:
    locations = load_outdoor_locations()
    if locations is None or locations.empty:
        return None, float("inf")

    latitudes = locations["lat"].values
    longitudes = locations["lon"].values

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    latitudes_rad = np.radians(latitudes)
    longitudes_rad = np.radians(longitudes)

    dlat = latitudes_rad - lat_rad
    dlon = longitudes_rad - lon_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_rad) * np.cos(latitudes_rad) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    distance_km = 6371 * c

    min_idx = int(np.argmin(distance_km))
    return int(locations.iloc[min_idx]["location_id"]), float(distance_km[min_idx])


def load_outdoor_temps_for_location(location_id: int, year: int) -> Optional[pd.DataFrame]:
    try:
        path = OUTDOOR_TEMPS_DIR / f"location_id={location_id:04d}" / f"year={year}.parquet"
        if not path.exists():
            path = OUTDOOR_TEMPS_DIR / f"location_id={location_id}" / f"year={year}.parquet"
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
        return df
    except Exception:
        return None


def available_years_for_location(location_id: int) -> List[int]:
    base_dir = OUTDOOR_TEMPS_DIR / f"location_id={location_id:04d}"
    if not base_dir.exists():
        base_dir = OUTDOOR_TEMPS_DIR / f"location_id={location_id}"
    if not base_dir.exists():
        return []
    years: List[int] = []
    for path in base_dir.glob("year=*.parquet"):
        stem = path.stem  # year=YYYY
        try:
            years.append(int(stem.split("=", 1)[1]))
        except (IndexError, ValueError):
            continue
    return sorted(set(years))


def load_full_year_outdoor_temps(lat: float, lon: float, year: int = 2023) -> Tuple[Optional[pd.DataFrame], Optional[int], float]:
    location_id, distance_km = find_nearest_outdoor_location(lat, lon)
    if location_id is None:
        return None, None, float("inf")
    temps = load_outdoor_temps_for_location(location_id, year)
    return temps, location_id, distance_km


def load_day_outdoor_temps_from_nearest_location(
    lat: float,
    lon: float,
    date_str: str,
) -> Tuple[Optional[pd.DataFrame], Optional[int], float]:
    location_id, distance_km = find_nearest_outdoor_location(lat, lon)
    if location_id is None:
        return None, None, float("inf")
    year = int(date_str[:4])
    df = load_outdoor_temps_for_location(location_id, year)
    if df is None or df.empty:
        return None, location_id, distance_km
    day = pd.to_datetime(date_str).date()
    df_day = df[df["time"].dt.date == day]
    return df_day, location_id, distance_km
