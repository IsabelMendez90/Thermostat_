import pandas as pd
import streamlit as st


def init_app_state() -> None:
    ss = st.session_state

    # Navigation
    ss.setdefault("view", "Setup")  # Setup | Home | Schedules | Forecast | Reports
    ss.setdefault("setup_step", 1)  # 1=Location, 2=Building, 3=Schedules, 4=Matching
    ss.setdefault("setup_complete", False)

    # Location
    ss.setdefault("location_name", "")
    ss.setdefault("user_lat", 37.8715)     # Berkeley default
    ss.setdefault("user_lon", -122.2730)   # Berkeley default
    ss.setdefault("geo_results", [])
    ss.setdefault("location_id", None)

    # Building (optional)
    ss.setdefault("building_age_yrs", None)
    ss.setdefault("building_type", None)
    ss.setdefault("floor_area_sqft", None)
    ss.setdefault("climate_code", "")

    # Schedules
    ss.setdefault("schedules", {
        "Home": {"heat_sp": 68, "cool_sp": 76, "start_hour": 17, "end_hour": 23},
        "Away": {"heat_sp": 64, "cool_sp": 82, "start_hour": 8, "end_hour": 17},
        "Sleep": {"heat_sp": 66, "cool_sp": 78, "start_hour": 23, "end_hour": 7},
    })
    ss.setdefault("active_comfort", "Home")
    ss.setdefault("schedule_priorities", {})

    # HVAC
    ss.setdefault("hvac_mode", "Auto")
    ss.setdefault("fan_on", False)

    # Current state
    ss.setdefault("indoor_temp", 72.0)
    ss.setdefault("indoor_humidity", 51)
    ss.setdefault("outdoor_temp", None)
    ss.setdefault("outdoor_humidity", None)
    ss.setdefault("weather_updated", False)

    # Personalization
    ss.setdefault("neighbors_df", pd.DataFrame())
    ss.setdefault("match_score", 0)
    ss.setdefault("match_confidence", "Unknown")
    ss.setdefault("similar_homes_count", 0)
    ss.setdefault("neighbor_priors", {})
    ss.setdefault("neighbors_ready", False)

    # Suggestions
    ss.setdefault("schedule_suggestions", {})
    ss.setdefault("did_bootstrap_location_setpoints", False)

    # Models
    ss.setdefault("model_loaded", False)
    ss.setdefault("tin_model", None)
    ss.setdefault("runtime_predictor", None)
    ss.setdefault("runtime_model_loaded", False)

    # Forecast
    ss.setdefault("forecast_data", None)

    # Scenarios (sandbox)
    ss.setdefault("custom_scenarios", {})
    ss.setdefault("selected_scenario", "Baseline (current schedules)")

    # Assistant
    ss.setdefault("assistant_messages", [])
    ss.setdefault("pending_action", None)
    ss.setdefault("pending_explainer", "")
