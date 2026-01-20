from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pandas as pd

from models_runtime import estimate_runtime_simple, estimate_runtime_with_predictor
from models_tin import predict_tin_extended
from outdoor_data import load_full_year_outdoor_temps


def compute_annual_runtime_with_historical_weather(
    schedules: Dict[str, Dict],
    tin_model: Any,
    runtime_predictor: Any,
    lat: float,
    lon: float,
    indoor_seed: float = 72.0,
    building_age: Optional[int] = None,
    building_type: Optional[str] = None,
    floor_area: Optional[float] = None,
    climate_code: str = "",
    hvac_mode: str = "Auto",
    year: int = 2023,
    schedule_priorities: Optional[Dict[str, int]] = None,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Compute runtime predictions for a full year using actual historical weather data.
    """
    weather_df, location_id, distance_km = load_full_year_outdoor_temps(lat, lon, year)

    if weather_df is None or weather_df.empty:
        return None, {"error": f"Could not load weather data for year {year}", "location_id": location_id}

    if tin_model is None:
        return None, {"error": "Indoor temperature model not available"}

    weather_df["month"] = weather_df["time"].dt.month
    weather_df["hour"] = weather_df["time"].dt.hour
    weather_df["day_of_week"] = (weather_df["time"].dt.dayofweek + 1) % 7 + 1
    weather_df["date"] = weather_df["time"].dt.date

    all_predictions = []
    monthly_summary = {
        "location_id": location_id,
        "distance_km": distance_km,
        "year": year,
        "months": {},
    }

    for month in range(1, 13):
        month_weather = weather_df[weather_df["month"] == month].copy()
        if month_weather.empty:
            continue

        month_forecast = predict_tin_extended(
            weather_df=month_weather,
            schedules=schedules,
            indoor_seed=indoor_seed,
            building_age=building_age,
            building_type=building_type,
            floor_area=floor_area,
            climate_code=climate_code,
            tin_model=tin_model,
            schedule_priorities=schedule_priorities,
        )

        if month_forecast is None or month_forecast.empty:
            continue

        if runtime_predictor:
            month_runtime = estimate_runtime_with_predictor(
                month_forecast,
                runtime_predictor,
                hvac_mode=hvac_mode,
                use_ev=False,
            )
        else:
            month_runtime = estimate_runtime_simple(month_forecast)

        if month_runtime is not None and not month_runtime.empty:
            month_runtime["month"] = month
            all_predictions.append(month_runtime)

            month_name = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][month - 1]

            month_stats = {
                "total_runtime_hours": float(month_runtime["runtime_hours"].sum()),
                "avg_outdoor_temp": float(month_weather["outdoor_temp"].mean()),
            }

            for comp in ["heat_sec_hours", "cool_sec_hours", "aux_sec_hours", "fan_sec_hours"]:
                if comp in month_runtime.columns:
                    month_stats[comp] = float(month_runtime[comp].sum())

            monthly_summary["months"][month_name] = month_stats

    if not all_predictions:
        return None, {"error": "No predictions generated"}

    full_year_predictions = pd.concat(all_predictions, ignore_index=True)

    monthly_summary["annual_runtime_hours"] = float(full_year_predictions["runtime_hours"].sum())
    monthly_summary["annual_kwh"] = monthly_summary["annual_runtime_hours"] * 3.5

    return full_year_predictions, monthly_summary
