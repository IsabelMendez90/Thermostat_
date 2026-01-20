from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

from app_config import SUGGESTIONS_PATH, TIMING_HOURLY_PATH
from neighbors import infer_location_id_from_latlon
from outdoor_data import load_full_year_outdoor_temps
from text_utils import fuzzy_match, norm_text


@st.cache_data(show_spinner=False)
def load_schedule_suggestions() -> pd.DataFrame:
    if not SUGGESTIONS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(SUGGESTIONS_PATH)
    if "schedule_name_norm" in df.columns:
        df["schedule_name_norm"] = df["schedule_name_norm"].astype(str).apply(norm_text)
    return df


@st.cache_data(show_spinner=False)
def load_timing_suggestions() -> pd.DataFrame:
    if not TIMING_HOURLY_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(TIMING_HOURLY_PATH)
    if "schedule_name_norm" in df.columns:
        df["schedule_name_norm"] = df["schedule_name_norm"].astype(str).apply(norm_text)

    for c in ["hour", "is_weekend", "location_id"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    if "p_active" in df.columns:
        df["p_active"] = pd.to_numeric(df["p_active"], errors="coerce")
    if "heat_p50_h" in df.columns:
        df["heat_p50_h"] = pd.to_numeric(df["heat_p50_h"], errors="coerce")
    if "cool_p50_h" in df.columns:
        df["cool_p50_h"] = pd.to_numeric(df["cool_p50_h"], errors="coerce")
    return df


def suggest_setpoints(schedule_name: str, suggestions_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if suggestions_df.empty or "schedule_name_norm" not in suggestions_df.columns:
        return None

    norm_name = norm_text(schedule_name)
    exact = suggestions_df[suggestions_df["schedule_name_norm"] == norm_name]
    if not exact.empty:
        row = exact.iloc[0]
        # Safely handle n_homes which might be NaN
        n_homes_val = row.get("n_homes", 0)
        if pd.isna(n_homes_val):
            n_homes_val = 0
        return {
            "heat_sp": float(row["heat_p50"]) if pd.notna(row.get("heat_p50")) else None,
            "cool_sp": float(row["cool_p50"]) if pd.notna(row.get("cool_p50")) else None,
            "n_homes": int(n_homes_val),
            "source": "exact",
            "match_score": 100,
        }

    choices = suggestions_df["schedule_name_norm"].dropna().unique().tolist()
    matches = fuzzy_match(schedule_name, choices, threshold=60)
    if matches:
        best_match, score = matches[0]
        row = suggestions_df[suggestions_df["schedule_name_norm"] == norm_text(best_match)].iloc[0]
        # Safely handle n_homes which might be NaN
        n_homes_val = row.get("n_homes", 0)
        if pd.isna(n_homes_val):
            n_homes_val = 0
        return {
            "heat_sp": float(row["heat_p50"]) if pd.notna(row.get("heat_p50")) else None,
            "cool_sp": float(row["cool_p50"]) if pd.notna(row.get("cool_p50")) else None,
            "n_homes": int(n_homes_val),
            "source": "fuzzy",
            "match_score": int(score),
            "matched_name": best_match,
        }
    return None


def suggest_timing(schedule_name: str, location_id: int, is_weekend: int, timing_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if timing_df.empty:
        return None
    norm_name = norm_text(schedule_name)

    req_cols = {"schedule_name_norm", "location_id", "is_weekend", "hour", "p_active"}
    if not req_cols.issubset(set(timing_df.columns)):
        return None

    subset = timing_df[
        (timing_df["schedule_name_norm"] == norm_name) &
        (timing_df["location_id"] == location_id) &
        (timing_df["is_weekend"] == is_weekend)
    ]
    if subset.empty:
        return None

    p = np.zeros(24, dtype=float)
    for h in range(24):
        vv = subset.loc[subset["hour"] == h, "p_active"]
        p[h] = float(vv.mean()) if not vv.empty else 0.0

    if np.max(p) <= 0:
        return None

    peak_hour = int(np.argmax(p))
    thr = 0.5 * float(np.max(p))
    high = np.where(p >= thr)[0]
    if len(high) == 0:
        return None

    start = int(high[0])
    end = int((high[-1] + 1) % 24)

    return {"start_hour": start, "end_hour": end, "peak_hour": peak_hour, "source": "location_timing"}


def compute_suggestion_energy_impact(
    schedule_name: str,
    current_heat_sp: int,
    current_cool_sp: int,
    suggested_heat_sp: int,
    suggested_cool_sp: int,
    all_schedules: Dict[str, Dict],
    tin_model: Any,
    runtime_predictor: Any,
    lat: float,
    lon: float,
    building_age: Optional[int] = None,
    building_type: Optional[str] = None,
    floor_area: Optional[float] = None,
    climate_code: str = "",
    hvac_mode: str = "Auto",
    schedule_priorities: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Compute the energy impact of changing a schedule's setpoints.
    """
    if tin_model is None or runtime_predictor is None:
        return {"error": "Models not loaded", "recommendation": "unknown"}

    name_lower = schedule_name.lower()
    if any(kw in name_lower for kw in ["sleep", "night", "morning", "wake"]):
        analysis_months = [1, 12]  # Winter months - heating matters most
        season_note = "winter months (heating-dominant)"
    elif any(kw in name_lower for kw in ["away", "work", "office", "school", "peak"]):
        analysis_months = [7, 8]  # Summer months - cooling during away/peak
        season_note = "summer months (cooling-dominant)"
    else:
        analysis_months = [4, 10]  # Shoulder seasons
        season_note = "shoulder seasons (mixed heating/cooling)"

    weather_df, location_id, dist_km = load_full_year_outdoor_temps(lat, lon, year=2023)
    if weather_df is None:
        return {"error": "Could not load weather data", "recommendation": "unknown"}

    weather_df["month"] = weather_df["time"].dt.month
    analysis_weather = weather_df[weather_df["month"].isin(analysis_months)].copy()

    if analysis_weather.empty:
        return {"error": "No weather data for analysis period", "recommendation": "unknown"}

    current_schedules = {k: dict(v) for k, v in all_schedules.items()}
    suggested_schedules = {k: dict(v) for k, v in all_schedules.items()}
    suggested_schedules[schedule_name]["heat_sp"] = suggested_heat_sp
    suggested_schedules[schedule_name]["cool_sp"] = suggested_cool_sp

    def compute_runtime_for_schedules(schedules: Dict, weather: pd.DataFrame) -> float:
        from models_tin import predict_tin_extended
        from models_runtime import estimate_runtime_with_predictor

        forecast = predict_tin_extended(
            weather_df=weather,
            schedules=schedules,
            indoor_seed=72.0,
            building_age=building_age,
            building_type=building_type,
            floor_area=floor_area,
            climate_code=climate_code,
            tin_model=tin_model,
            schedule_priorities=schedule_priorities,
        )
        if forecast is None or forecast.empty:
            return 0.0

        runtime_df = estimate_runtime_with_predictor(
            forecast,
            runtime_predictor,
            hvac_mode=hvac_mode,
            use_ev=False,
        )
        if runtime_df is None or runtime_df.empty:
            return 0.0

        return float(runtime_df["runtime_hours"].sum())

    current_runtime = compute_runtime_for_schedules(current_schedules, analysis_weather)
    suggested_runtime = compute_runtime_for_schedules(suggested_schedules, analysis_weather)

    if analysis_months == [1, 12]:
        seasonal_weight = 6  # ~6 months of heating-like conditions
    elif analysis_months == [7, 8]:
        seasonal_weight = 4  # ~4 months of cooling-like conditions
    else:
        seasonal_weight = 2  # ~2 months of shoulder conditions

    current_kwh = current_runtime * 3.5 * seasonal_weight
    suggested_kwh = suggested_runtime * 3.5 * seasonal_weight

    savings_kwh = current_kwh - suggested_kwh
    savings_cost = savings_kwh * 0.30

    if savings_kwh > 50:
        recommendation = "save"
        rec_text = f"Saves ~{savings_kwh:.0f} kWh/yr (${savings_cost:.0f}/yr)"
    elif savings_kwh < -50:
        recommendation = "cost"
        rec_text = f"Costs ~{abs(savings_kwh):.0f} kWh/yr more (+${abs(savings_cost):.0f}/yr)"
    else:
        recommendation = "neutral"
        rec_text = f"Minimal change: {savings_kwh:+.0f} kWh/yr (${savings_cost:+.0f}/yr)"

    heat_change = suggested_heat_sp - current_heat_sp
    cool_change = suggested_cool_sp - current_cool_sp

    if heat_change < 0 and cool_change > 0:
        strategy = "wider deadband (energy efficient)"
    elif heat_change > 0 and cool_change < 0:
        strategy = "tighter deadband (more comfort, more energy)"
    elif heat_change > 0:
        strategy = "higher heating (warmer, more heating energy)"
    elif heat_change < 0:
        strategy = "lower heating (cooler, less heating energy)"
    elif cool_change > 0:
        strategy = "higher cooling (less cooling energy)"
    elif cool_change < 0:
        strategy = "lower cooling (more cooling energy)"
    else:
        strategy = "no change"

    return {
        "annual_kwh_current": round(current_kwh, 0),
        "annual_kwh_suggested": round(suggested_kwh, 0),
        "annual_savings_kwh": round(savings_kwh, 0),
        "annual_savings_cost": round(savings_cost, 2),
        "recommendation": recommendation,
        "recommendation_text": rec_text,
        "strategy": strategy,
        "analysis_note": f"Based on {season_note} weather patterns",
        "heat_change": heat_change,
        "cool_change": cool_change,
    }


def semantic_schedule_intent(schedule_name: str) -> Dict[str, Any]:
    """
    Detect schedule intent from name and suggest appropriate setpoints and hours.
    """
    name_lower = (schedule_name or "").lower()
    patterns = {
        "peak": {
            "intent": "time-of-use / demand response",
            "typical_hours": (16, 21),
            "setpoints": {"heat_sp": 65, "cool_sp": 80},
            "keywords": ["peak", "tou", "demand", "expensive", "grid", "4-9", "4–9", "caiso", "flex"],
        },
        "work": {
            "intent": "away during work hours",
            "typical_hours": (8, 17),
            "setpoints": {"heat_sp": 62, "cool_sp": 84},
            "keywords": ["work", "office", "job", "away", "out", "gone", "school"],
        },
        "sleep": {
            "intent": "overnight comfort",
            "typical_hours": (22, 7),
            "setpoints": {"heat_sp": 66, "cool_sp": 78},
            "keywords": ["sleep", "night", "bed", "rest", "nap", "siesta"],
        },
        "morning": {
            "intent": "morning routine",
            "typical_hours": (6, 9),
            "setpoints": {"heat_sp": 70, "cool_sp": 75},
            "keywords": ["morning", "wake", "breakfast", "am", "dawn"],
        },
        "evening": {
            "intent": "evening relaxation",
            "typical_hours": (18, 23),
            "setpoints": {"heat_sp": 69, "cool_sp": 76},
            "keywords": ["evening", "dinner", "relax", "pm", "dusk", "home"],
        },
        "weekend": {
            "intent": "home all day",
            "typical_hours": (9, 23),
            "setpoints": {"heat_sp": 69, "cool_sp": 76},
            "keywords": ["weekend", "saturday", "sunday", "day off", "holiday"],
        },
        "pet": {
            "intent": "pet comfort while away",
            "typical_hours": (8, 17),
            "setpoints": {"heat_sp": 65, "cool_sp": 80},
            "keywords": ["pet", "dog", "cat", "puppy", "kitty", "animal"],
        },
        "baby": {
            "intent": "baby/child comfort",
            "typical_hours": (0, 24),
            "setpoints": {"heat_sp": 68, "cool_sp": 76},
            "keywords": ["baby", "infant", "child", "kid", "nursery", "toddler", "daddy", "mommy"],
        },
        "workout": {
            "intent": "exercise time",
            "typical_hours": (6, 8),
            "setpoints": {"heat_sp": 64, "cool_sp": 72},
            "keywords": ["workout", "exercise", "gym", "fitness", "run", "cardio", "training"],
        },
        "yoga": {
            "intent": "yoga/meditation practice",
            "typical_hours": (6, 8),
            "setpoints": {"heat_sp": 68, "cool_sp": 74},
            "keywords": ["yoga", "meditation", "mindfulness", "stretch", "pilates"],
        },
        "studying": {
            "intent": "focused studying or work",
            "typical_hours": (9, 17),
            "setpoints": {"heat_sp": 68, "cool_sp": 74},
            "keywords": ["study", "studying", "homework", "reading", "learning", "focus"],
        },
        "cooking": {
            "intent": "cooking time (extra heat)",
            "typical_hours": (17, 19),
            "setpoints": {"heat_sp": 66, "cool_sp": 72},
            "keywords": ["cook", "cooking", "baking", "meal", "kitchen"],
        },
        "guests": {
            "intent": "guests visiting",
            "typical_hours": (10, 22),
            "setpoints": {"heat_sp": 69, "cool_sp": 75},
            "keywords": ["guest", "visitor", "company", "party", "entertain"],
        },
        "movie": {
            "intent": "movie/entertainment time",
            "typical_hours": (19, 23),
            "setpoints": {"heat_sp": 69, "cool_sp": 75},
            "keywords": ["movie", "tv", "gaming", "entertainment", "streaming", "watch"],
        },
        "remote_work": {
            "intent": "working from home",
            "typical_hours": (8, 17),
            "setpoints": {"heat_sp": 68, "cool_sp": 74},
            "keywords": ["wfh", "remote", "telework", "home office"],
        },
        "elderly": {
            "intent": "elderly comfort (higher warmth)",
            "typical_hours": (0, 24),
            "setpoints": {"heat_sp": 70, "cool_sp": 76},
            "keywords": ["elderly", "senior", "grandparent", "old"],
        },
    }
    for key, info in patterns.items():
        if any(kw in name_lower for kw in info["keywords"]):
            return {
                "category": key,
                "intent": info["intent"],
                "suggested_hours": info["typical_hours"],
                "suggested_setpoints": info["setpoints"],
            }
    return {
        "category": "custom",
        "intent": "custom schedule",
        "suggested_hours": (9, 17),
        "suggested_setpoints": {"heat_sp": 68, "cool_sp": 76},
    }


def location_p50_setpoints_for_block(
    timing_df: pd.DataFrame,
    location_id: int,
    schedule_norm: str,
    is_weekend: int,
    start_hour: int,
    end_hour: int,
) -> Optional[Dict[str, int]]:
    if timing_df.empty:
        return None
    req = {"location_id", "is_weekend", "schedule_name_norm", "hour", "heat_p50_h", "cool_p50_h"}
    if not req.issubset(set(timing_df.columns)):
        return None

    df = timing_df[
        (timing_df["location_id"] == location_id) &
        (timing_df["is_weekend"] == is_weekend) &
        (timing_df["schedule_name_norm"] == norm_text(schedule_norm))
    ]
    if df.empty:
        return None

    if start_hour == end_hour:
        hours = list(range(24))
    elif start_hour < end_hour:
        hours = list(range(start_hour, end_hour))
    else:
        hours = list(range(start_hour, 24)) + list(range(0, end_hour))

    df = df[df["hour"].isin(hours)]
    if df.empty:
        return None

    heat = int(round(pd.to_numeric(df["heat_p50_h"], errors="coerce").median()))
    cool = int(round(pd.to_numeric(df["cool_p50_h"], errors="coerce").median()))
    if not np.isfinite(heat) or not np.isfinite(cool):
        return None
    return {"heat_sp": heat, "cool_sp": cool}


def bootstrap_default_schedules_from_location() -> None:
    ss = st.session_state
    if ss.get("did_bootstrap_location_setpoints", False):
        return

    timing_df = load_timing_suggestions()
    if timing_df.empty:
        ss.did_bootstrap_location_setpoints = True
        return

    loc_id = infer_location_id_from_latlon(ss.user_lat, ss.user_lon)
    if loc_id is None:
        ss.location_id = None
        ss.did_bootstrap_location_setpoints = True
        return

    mapping = {
        "Home":  ("awake", ss.schedules["Home"]["start_hour"],  ss.schedules["Home"]["end_hour"]),
        "Away":  ("away",  ss.schedules["Away"]["start_hour"],  ss.schedules["Away"]["end_hour"]),
        "Sleep": ("sleep", ss.schedules["Sleep"]["start_hour"], ss.schedules["Sleep"]["end_hour"]),
    }
    for sched_name, (norm_name, sh, eh) in mapping.items():
        sp = location_p50_setpoints_for_block(
            timing_df=timing_df,
            location_id=loc_id,
            schedule_norm=norm_name,
            is_weekend=0,
            start_hour=int(sh),
            end_hour=int(eh),
        )
        if sp:
            ss.schedules[sched_name]["heat_sp"] = sp["heat_sp"]
            ss.schedules[sched_name]["cool_sp"] = sp["cool_sp"]

    ss.location_id = loc_id
    ss.did_bootstrap_location_setpoints = True
