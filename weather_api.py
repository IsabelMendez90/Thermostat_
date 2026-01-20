from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd
import requests

from app_config import FORECAST_URL, GEOCODE_URL


def geocode_place(place: str) -> Tuple[List[dict], str]:
    q = (place or "").strip()
    if not q:
        return [], "Enter a location first"
    try:
        r = requests.get(GEOCODE_URL, params={"name": q, "count": 5, "format": "json"}, timeout=12)
        r.raise_for_status()
        results = r.json().get("results", []) or []
        if not results:
            return [], f"No matches for '{q}'"
        return results, f"Found {len(results)} location(s)"
    except Exception as e:
        return [], f"Error: {e}"


def nice_place(r: dict) -> str:
    parts = [r.get("name", ""), r.get("admin1", ""), r.get("country", "")]
    return ", ".join(p for p in parts if p) or "Unknown"


def fetch_current_weather(lat: float, lon: float) -> Tuple[Optional[float], Optional[float], str]:
    try:
        r = requests.get(
            FORECAST_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m",
                "temperature_unit": "fahrenheit",
            },
            timeout=12,
        )
        r.raise_for_status()
        cur = r.json().get("current", {}) or {}
        temp = cur.get("temperature_2m", None)
        rh = cur.get("relative_humidity_2m", None)
        if temp is None:
            return None, None, "No temperature data"
        return float(temp), (float(rh) if rh is not None else None), "OK"
    except Exception as e:
        return None, None, f"Error: {e}"


def fetch_hourly_forecast(lat: float, lon: float, days: int = 7) -> Optional[pd.DataFrame]:
    try:
        r = requests.get(
            FORECAST_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": "temperature_2m,relative_humidity_2m",
                "temperature_unit": "fahrenheit",
                "forecast_days": days,
            },
            timeout=12,
        )
        r.raise_for_status()
        hourly = r.json().get("hourly", {}) or {}
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        hums = hourly.get("relative_humidity_2m", [])
        if not times:
            return None
        df = pd.DataFrame({
            "time": pd.to_datetime(times),
            "outdoor_temp": temps,
            "outdoor_humidity": hums,
        })
        return df
    except Exception:
        return None


def fetch_historical_weather(lat: float, lon: float, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    try:
        r = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": start_date,
                "end_date": end_date,
                "hourly": "temperature_2m,relative_humidity_2m",
                "temperature_unit": "fahrenheit",
            },
            timeout=15,
        )
        r.raise_for_status()
        hourly = r.json().get("hourly", {}) or {}
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        hums = hourly.get("relative_humidity_2m", [])
        if not times:
            return None
        df = pd.DataFrame({
            "time": pd.to_datetime(times),
            "outdoor_temp": temps,
            "outdoor_humidity": hums,
        })
        return df
    except Exception:
        return None


def fetch_single_day_weather(lat: float, lon: float, date_str: str) -> Optional[pd.DataFrame]:
    try:
        r = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": date_str,
                "end_date": date_str,
                "hourly": "temperature_2m,relative_humidity_2m",
                "temperature_unit": "fahrenheit",
            },
            timeout=15,
        )
        r.raise_for_status()
        hourly = r.json().get("hourly", {}) or {}
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        hums = hourly.get("relative_humidity_2m", [])
        if not times:
            return None
        df = pd.DataFrame({
            "time": pd.to_datetime(times),
            "outdoor_temp": temps,
            "outdoor_humidity": hums,
        })
        return df
    except Exception:
        return None
