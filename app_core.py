#!/usr/bin/env python3
"""
Smart Ecobee Thermostat - Personalized Energy Dashboard (Streamlit)
===================================================================
One-file Streamlit app that mimics an Ecobee-like UI + personalization:
- Setup wizard (location, building, schedules, neighbor matching)
- Home screen (Tin prediction optional)
- Schedule manager with name-based + location-based suggestions
- 7-day forecast with scenario comparison (sandbox scenarios)
- Optional OpenRouter assistant + scenario builder

SECURITY:
- DO NOT hardcode API keys.
- Set OPENROUTER_API_KEY in environment OR Streamlit secrets.

Run:
  pip install -r requirements.txt
  streamlit run app.py

Suggested requirements:
  streamlit pandas numpy requests plotly pyarrow
Optional:
  rapidfuzz xgboost joblib streamlit-js-eval openai
"""

from __future__ import annotations

from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from app_config import MONTH_NAMES, TIN_MODEL_PATH
from assistant_tools import (
    apply_action,
    assistant_bar,
    build_location_context_for_llm,
    call_llm_for_schedule_suggestion,
    call_openrouter_for_scenario,
)
from cluster_utils import find_user_cluster
from energy_analysis import compute_annual_runtime_with_historical_weather
from models_runtime import (
    estimate_energy_from_runtime,
    estimate_runtime_simple,
    estimate_runtime_with_predictor,
    load_runtime_predictor,
)
from models_tin import build_tin_features, get_2r2c_params_for_home, load_tin_model, predict_tin_7days
from neighbors import (
    ensure_neighbors_loaded,
    extract_neighbor_priors,
    find_neighbors,
    infer_location_id_from_latlon,
    load_neighbor_data,
)
from outdoor_data import (
    find_nearest_outdoor_location,
    load_day_outdoor_temps_from_nearest_location,
)
from schedule_suggestions import (
    bootstrap_default_schedules_from_location,
    compute_suggestion_energy_impact,
    load_schedule_suggestions,
    load_timing_suggestions,
    semantic_schedule_intent,
    suggest_setpoints,
    suggest_timing,
)
from schedule_utils import ensure_schedule_priorities, normalize_schedule_priorities
from scenarios import apply_custom_scenario, build_scenario_schedules
from ui_components import bottom_nav, render_mini_metric, render_report_metric, topbar
from text_utils import norm_text
from weather_api import (
    fetch_current_weather,
    fetch_historical_weather,
    fetch_hourly_forecast,
    fetch_single_day_weather,
    geocode_place,
    nice_place,
)

try:
    from streamlit_js_eval import get_geolocation
    GEOLOC_AVAILABLE = True
except Exception:
    GEOLOC_AVAILABLE = False


# =========================================================
# =========================================================
# SETUP FLOW
# =========================================================
def render_setup():
    bootstrap_default_schedules_from_location()
    step = st.session_state.setup_step

    st.markdown('<div class="frame">', unsafe_allow_html=True)

    if step == 1:
        render_setup_location()
    elif step == 2:
        render_setup_building()
    elif step == 3:
        render_setup_schedules()
    # Step 4 (Finalizing Setup) removed - go directly to home from schedules

    st.markdown("</div>", unsafe_allow_html=True)

def render_setup_location():
    topbar("Welcome")
    st.markdown("### Let's personalize your thermostat")
    st.write("First, where is your home located?")

    st.markdown('<div class="card">', unsafe_allow_html=True)

    # Use form to enable Enter key for search
    with st.form(key="location_search_form", clear_on_submit=False):
        location = st.text_input(
            "Location",
            value=st.session_state.location_name,
            placeholder="Type your location (e.g., Berkeley, CA)",
            label_visibility="collapsed",
        )
        search_clicked = st.form_submit_button("🔍 Search", use_container_width=True)

    if search_clicked and location:
        results, msg = geocode_place(location)
        if results:
            st.session_state.geo_results = results
        else:
            st.error(msg)
        st.rerun()

    if st.session_state.geo_results:
        st.write("**Select your location:**")
        for i, r in enumerate(st.session_state.geo_results):
            if st.button(nice_place(r), key=f"geo_{i}", use_container_width=True):
                st.session_state.user_lat = float(r["latitude"])
                st.session_state.user_lon = float(r["longitude"])
                st.session_state.location_name = nice_place(r)
                st.session_state.geo_results = []
                st.session_state.did_bootstrap_location_setpoints = False
                st.session_state.location_id = None
                st.session_state.neighbors_df = pd.DataFrame()
                st.session_state.neighbors_ready = False
                st.session_state.similar_homes_count = 0
                st.session_state.match_score = 0
                st.session_state.outdoor_temp = None
                st.session_state.outdoor_humidity = None
                st.session_state.weather_updated = False
                st.success(f"Location set: {st.session_state.location_name}")
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.location_name:
        st.caption(f"📍 Current: {st.session_state.location_name}")
    else:
        st.caption(f"📍 Current: {st.session_state.user_lat:.4f}, {st.session_state.user_lon:.4f} (default)")

    if st.button("Next →", use_container_width=True, type="primary"):
        if not st.session_state.location_name:
            st.session_state.location_name = f"Location ({st.session_state.user_lat:.2f}, {st.session_state.user_lon:.2f})"
        st.session_state.setup_step = 3  # Skip building info, go directly to schedules
        st.rerun()

def render_setup_building():
    topbar("About Your Home")
    st.markdown("### Help us understand your home better")
    st.write("Optional, but improves suggestions and predictions.")

    st.markdown('<div class="card">', unsafe_allow_html=True)

    age_options = ["I don't know", "0-5 years", "5-10 years", "10-20 years", "20-40 years", "40+ years"]
    age_choice = st.selectbox("Building age", age_options, index=0)
    if age_choice != "I don't know":
        age_map = {"0-5 years": 2.5, "5-10 years": 7.5, "10-20 years": 15, "20-40 years": 30, "40+ years": 50}
        st.session_state.building_age_yrs = age_map[age_choice]
    else:
        st.session_state.building_age_yrs = None

    type_options = ["I don't know", "Detached", "Townhouse", "Apartment", "Condominium", "Other"]
    type_choice = st.selectbox("Building type", type_options, index=0)
    st.session_state.building_type = None if type_choice == "I don't know" else type_choice

    st.markdown("</div>", unsafe_allow_html=True)

    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Back", use_container_width=True):
            st.session_state.setup_step = 1
            st.rerun()
    with col_next:
        if st.button("Next →", use_container_width=True, type="primary"):
            st.session_state.setup_step = 3
            st.rerun()

def render_setup_schedules():
    bootstrap_default_schedules_from_location()
    topbar("Your Schedules")
    st.markdown("### Set up your comfort schedules")
    st.write("Start with these three. You can add more later.")

    for name in ["Home", "Away", "Sleep"]:
        st.markdown('<div class="schedule-item">', unsafe_allow_html=True)
        st.markdown(f"**{name}**")
        sched = st.session_state.schedules[name]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🔥 HEAT")
            sched["heat_sp"] = st.slider(f"Heat (°F)", 55, 75, int(sched["heat_sp"]), key=f"setup_heat_{name}")
        with col2:
            st.markdown("#### ❄️ COOL")
            sched["cool_sp"] = st.slider(f"Cool (°F)", 70, 90, int(sched["cool_sp"]), key=f"setup_cool_{name}")

        col3, col4 = st.columns(2)
        with col3:
            sched["start_hour"] = st.selectbox(
                "Start time", range(24),
                index=int(sched["start_hour"]) % 24,
                format_func=lambda h: f"{h:02d}:00",
                key=f"setup_start_{name}",
            )
        with col4:
            sched["end_hour"] = st.selectbox(
                "End time", range(24),
                index=int(sched["end_hour"]) % 24,
                format_func=lambda h: f"{h:02d}:00",
                key=f"setup_end_{name}",
            )
        st.markdown("</div>", unsafe_allow_html=True)

    col_back, col_done = st.columns(2)
    with col_back:
        if st.button("← Back", use_container_width=True):
            st.session_state.setup_step = 1
            st.rerun()
    with col_done:
        if st.button("Done! → Home", use_container_width=True, type="primary"):
            # Run neighbor matching in background
            meta, idx, _, _ = load_neighbor_data()
            if meta is not None and idx is not None:
                neighbors_df, stats = find_neighbors(st.session_state.user_lat, st.session_state.user_lon)
                st.session_state.neighbors_df = neighbors_df
                st.session_state.match_score = stats.get("match_score", 0)
                st.session_state.match_confidence = stats.get("confidence", "Unknown")
                st.session_state.similar_homes_count = stats.get("total_homes", 0)
                st.session_state.neighbor_priors = extract_neighbor_priors(neighbors_df) if not neighbors_df.empty else {}
                st.session_state.neighbors_ready = True
            else:
                st.session_state.neighbors_df = pd.DataFrame()
                st.session_state.neighbors_ready = False
                st.session_state.similar_homes_count = 0

            # Mark setup as complete and go to home
            st.session_state.setup_complete = True
            st.session_state.view = "Home"
            temp, rh, status = fetch_current_weather(st.session_state.user_lat, st.session_state.user_lon)
            if status == "OK":
                st.session_state.outdoor_temp = temp
                st.session_state.outdoor_humidity = rh
                st.session_state.weather_updated = True
            st.rerun()

def render_setup_matching():
    topbar("Finalizing Setup")
    st.markdown("### Setting up personalization...")

    meta, idx, _, _ = load_neighbor_data()

    # Run matching in background without displaying results
    if meta is not None and idx is not None:
        neighbors_df, stats = find_neighbors(st.session_state.user_lat, st.session_state.user_lon)
        st.session_state.neighbors_df = neighbors_df
        st.session_state.match_score = stats.get("match_score", 0)
        st.session_state.match_confidence = stats.get("confidence", "Unknown")
        st.session_state.similar_homes_count = stats.get("total_homes", 0)
        st.session_state.neighbor_priors = extract_neighbor_priors(neighbors_df) if not neighbors_df.empty else {}
        st.session_state.neighbors_ready = True
    else:
        # No neighbor data available
        st.session_state.neighbors_df = pd.DataFrame()
        st.session_state.neighbors_ready = False
        st.session_state.similar_homes_count = 0

    st.success("✓ Setup complete!")

    col_back, col_done = st.columns(2)
    with col_back:
        if st.button("← Back", use_container_width=True):
            st.session_state.setup_step = 3
            st.rerun()
    with col_done:
        if st.button("Done! → Home", use_container_width=True, type="primary"):
            st.session_state.setup_complete = True
            st.session_state.view = "Home"
            temp, rh, status = fetch_current_weather(st.session_state.user_lat, st.session_state.user_lon)
            if status == "OK":
                st.session_state.outdoor_temp = temp
                st.session_state.outdoor_humidity = rh
                st.session_state.weather_updated = True
            st.rerun()


# =========================================================
# HOME VIEW
# =========================================================
def render_home():
    st.markdown('<div class="frame">', unsafe_allow_html=True)
    topbar("My Thermostat")

    # Weather refresh once
    if not st.session_state.weather_updated:
        temp, rh, _ = fetch_current_weather(st.session_state.user_lat, st.session_state.user_lon)
        if temp is not None:
            st.session_state.outdoor_temp = temp
            st.session_state.outdoor_humidity = rh
            st.session_state.weather_updated = True

    ensure_neighbors_loaded()

    # Load model once
    if not st.session_state.model_loaded:
        st.session_state.tin_model = load_tin_model()
        st.session_state.model_loaded = True

    # Predict Tin for "now" (optional)
    if st.session_state.tin_model and st.session_state.outdoor_temp is not None:
        now = datetime.now()
        active = st.session_state.active_comfort
        sched = st.session_state.schedules[active]
        X = build_tin_features(
            outdoor_temp=float(st.session_state.outdoor_temp),
            outdoor_humidity=float(st.session_state.outdoor_humidity or 50),
            heat_sp=float(sched["heat_sp"]),
            cool_sp=float(sched["cool_sp"]),
            dt=now,
            indoor_temp_lag=float(st.session_state.indoor_temp),
            outdoor_temp_lag=float(st.session_state.outdoor_temp),
            building_age=st.session_state.building_age_yrs,
            building_type=st.session_state.building_type,
            floor_area=st.session_state.floor_area_sqft,
            climate_code=st.session_state.climate_code,
        )
        try:
            pred = st.session_state.tin_model.predict(X)
            st.session_state.indoor_temp = float(pred[0])
        except Exception:
            pass

    outdoor_text = "—" if st.session_state.outdoor_temp is None else f"{st.session_state.outdoor_temp:.0f}°F"
    st.markdown(
        f"""
        <div class="status-row">
          <div class="status-chip">💧 <b>{st.session_state.indoor_humidity}%</b></div>
          <div class="status-chip">📍 <b>{st.session_state.location_name or "—"}</b></div>
          <div class="status-chip">🌡️ <b>Outside: {outdoor_text}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f'<div class="big-temp">{int(round(st.session_state.indoor_temp))}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="comfort-badge">🏠 {st.session_state.active_comfort}</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.5, 1])
    with col1:
        modes = ["Off", "Heat", "Cool", "Auto", "Aux"]
        st.session_state.hvac_mode = st.selectbox("HVAC Mode", modes, index=modes.index(st.session_state.hvac_mode), label_visibility="collapsed")
    with col2:
        st.session_state.fan_on = st.toggle("Fan On", value=st.session_state.fan_on)

    active_sched = st.session_state.schedules[st.session_state.active_comfort]
    st.markdown(
        f"""
        <div class="setpoint-pills">
          <div class="pill heat">
            <div class="label">🔥 HEAT</div>
            <div class="value">{int(active_sched['heat_sp'])}</div>
          </div>
          <div class="pill cool">
            <div class="label">❄️ COOL</div>
            <div class="value">{int(active_sched['cool_sp'])}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("**Switch comfort:**")

    # Define schedule icons
    schedule_icons = {
        "Sleep": "🌙",
        "Away": "🚶",
        "Home": "🏠",
        "Awake": "☀️",
        "Evening": "🌆",
        "Morning": "🌅",
    }

    cols = st.columns(len(st.session_state.schedules))
    for i, name in enumerate(list(st.session_state.schedules.keys())):
        with cols[i]:
            # Get icon for schedule name, default to 🏠 if not found
            icon = schedule_icons.get(name, "🏠")
            button_label = f"{icon} {name}"
            if st.button(button_label, key=f"comfort_{name}", use_container_width=True):
                st.session_state.active_comfort = name
                st.rerun()

    # Removed "Personalized for You" section per user request

    # Show 2R2C thermal parameters if available (building physics transparency)
    if hasattr(st.session_state, "identifier") and st.session_state.identifier:
        thermal_params = get_2r2c_params_for_home(st.session_state.identifier)
        if thermal_params:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🏗️ Building Thermal Characteristics")
            st.write("Physics-based model of your home's thermal behavior:")

            # Display key parameters
            tau_env = thermal_params.get("tau_env")
            tau_mass = thermal_params.get("tau_mass")
            confidence = thermal_params.get("confidence", "unknown")
            rmse = thermal_params.get("rmse_openloop")

            col_a, col_b = st.columns(2)
            with col_a:
                if tau_env and np.isfinite(tau_env):
                    st.metric("Envelope Time Constant", f"{tau_env:.1f} hrs", help="How quickly your home responds to outdoor temperature changes")
            with col_b:
                if tau_mass and np.isfinite(tau_mass):
                    st.metric("Thermal Mass", f"{tau_mass:.1f} hrs", help="How long your home retains heat/coolness")

            # Confidence and quality indicators
            if confidence != "unknown":
                conf_emoji = {"high": "🟢", "medium": "🟡", "low": "🟠", "discard": "🔴"}.get(confidence, "⚪")
                st.caption(f"{conf_emoji} Model confidence: **{confidence}**")

            if rmse and np.isfinite(rmse):
                st.caption(f"Prediction accuracy: ±{rmse:.1f}°F RMSE")

            with st.expander("📐 Advanced Parameters", expanded=False):
                R_ao = thermal_params.get("R_ao")
                R_am = thermal_params.get("R_am")
                C_a = thermal_params.get("C_a")
                C_m = thermal_params.get("C_m")

                st.markdown("**2R2C Model Parameters:**")
                if R_ao and np.isfinite(R_ao):
                    st.caption(f"R_ao (air-outdoor resistance): {R_ao:.3f}")
                if R_am and np.isfinite(R_am):
                    st.caption(f"R_am (air-mass resistance): {R_am:.3f}")
                if C_a and np.isfinite(C_a):
                    st.caption(f"C_a (air capacitance): {C_a:.3f}")
                if C_m and np.isfinite(C_m):
                    st.caption(f"C_m (mass capacitance): {C_m:.3f}")

                st.markdown("These parameters describe your building's thermal dynamics using a physics-based resistance-capacitance model.")

            st.markdown("</div>", unsafe_allow_html=True)

    # Show cluster matching info (personalization based on behavior patterns)
    if hasattr(st.session_state, "location_id") and st.session_state.location_id:
        cluster_info = find_user_cluster(st.session_state.location_id, st.session_state.schedules)
        if cluster_info:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 👥 Your Usage Pattern")
            st.markdown(
                f'<p style="color: white; font-size: 14px; margin-bottom: 10px;">'
                f'Matched to a local usage profile from {cluster_info["n_homes"]:,} nearby homes with similar setpoints.'
                f"</p>",
                unsafe_allow_html=True,
            )

            schedule_names = list(st.session_state.schedules.keys())
            schedule_count = len(schedule_names)
            schedule_list = ", ".join(schedule_names[:4]) + ("..." if schedule_count > 4 else "")
            location_label = st.session_state.location_name or "your selected location"
            st.markdown(
                f'<p style="color: white; font-size: 13px; margin-bottom: 16px;">'
                f'Your setup: {schedule_count} schedule{"s" if schedule_count != 1 else ""} '
                f'({schedule_list}); location: {location_label}'
                f"</p>",
                unsafe_allow_html=True,
            )
            st.caption("Match basis: your setpoints, schedule timing (from names), schedule count/diversity, and local patterns.")

            col_c1, col_c2, col_c3 = st.columns(3)
            with col_c1:
                st.metric("Typical Heating", f"{cluster_info['heat_p50']}°F")
            with col_c2:
                st.metric("Typical Cooling", f"{cluster_info['cool_p50']}°F")
            with col_c3:
                st.metric("Similar Homes", f"{cluster_info['n_homes']:,}")

            st.markdown(
                f'<p style="color: white; font-size: 14px;">'
                f'📅 Common timing in your area: '
                f'Home {cluster_info["awake_start"]}-{cluster_info["awake_end"]}h, '
                f'Away {cluster_info["away_start"]}-{cluster_info["away_end"]}h, '
                f'Sleep {cluster_info["sleep_start"]}-{cluster_info["sleep_end"]}h'
                f"</p>",
                unsafe_allow_html=True,
            )

            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 👥 Your Usage Pattern")
            st.caption("No local usage profile available for this location.")
            st.markdown("</div>", unsafe_allow_html=True)
    elif st.session_state.location_name:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 👥 Your Usage Pattern")
        st.caption("Local usage profiles are only available for the current dataset region.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    bottom_nav()
    assistant_bar()


# =========================================================
# SCHEDULES VIEW
# =========================================================
def render_schedule_card(name: str, suggestions_df: pd.DataFrame, timing_df: pd.DataFrame, location_id: Optional[int]):
    sched = st.session_state.schedules[name]
    is_active = (name == st.session_state.active_comfort)

    # Define colors for each schedule
    schedule_colors = {
        "Home": "rgba(100, 181, 246, 0.2)",      # Light blue
        "Away": "rgba(186, 104, 200, 0.2)",      # Purple
        "Sleep": "rgba(144, 164, 174, 0.2)",     # Gray
        "Morning": "rgba(255, 183, 77, 0.2)",    # Orange
    }
    card_color = schedule_colors.get(name, "rgba(76, 175, 80, 0.15)")  # Default green

    # Create colored header box with collapsed info
    border_style = "2px solid rgba(255, 255, 255, 0.3)" if is_active else "1px solid rgba(255, 255, 255, 0.1)"
    st.markdown(
        f'''<div style="background: {card_color}; padding: 16px; border-radius: 12px; margin-bottom: 16px; border: {border_style};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h3 style="margin: 0; padding: 0; color: white;">{name} {'✨' if is_active else ''}</h3>
                    <p style="margin: 4px 0 0 0; font-size: 14px; color: rgba(255, 255, 255, 0.9);">
                        <strong>Current:</strong> 🔥 Heat {sched['heat_sp']}°F  |  ❄️ Cool {sched['cool_sp']}°F  |  {int(sched['start_hour']):02d}:00-{int(sched['end_hour']):02d}:00
                    </p>
                </div>
            </div>
        </div>''',
        unsafe_allow_html=True
    )

    # Delete button (outside the colored box)
    if len(st.session_state.schedules) > 1:
        if st.button(f"🗑️ Delete {name}", key=f"del_{name}", use_container_width=False):
            del st.session_state.schedules[name]
            if st.session_state.active_comfort == name:
                st.session_state.active_comfort = list(st.session_state.schedules.keys())[0]
            st.session_state.schedule_priorities = normalize_schedule_priorities(
                st.session_state.schedules,
                st.session_state.get("schedule_priorities", {}),
            )
            st.rerun()

    with st.expander("📊 Energy-Validated Suggestions", expanded=False):
        # Ensure models are loaded for energy validation
        if not st.session_state.get("model_loaded"):
            st.session_state.tin_model = load_tin_model()
            st.session_state.model_loaded = True
        if not st.session_state.get("runtime_model_loaded"):
            st.session_state.runtime_predictor = load_runtime_predictor()
            st.session_state.runtime_model_loaded = True

        if not suggestions_df.empty:
            sp = suggest_setpoints(name, suggestions_df)
            if sp and sp.get("heat_sp") and sp.get("cool_sp"):
                suggested_heat = int(sp["heat_sp"])
                suggested_cool = int(sp["cool_sp"])
                current_heat = int(sched["heat_sp"])
                current_cool = int(sched["cool_sp"])

                # Check if suggestion is different from current
                if suggested_heat == current_heat and suggested_cool == current_cool:
                    st.success("✅ Your current setpoints match California patterns!")
                    st.markdown("**Try energy-saving alternatives:**")

                    # Offer alternatives to analyze
                    tin_model = st.session_state.get("tin_model")
                    runtime_predictor = st.session_state.get("runtime_predictor")

                    if tin_model and runtime_predictor:
                        col_alt1, col_alt2 = st.columns(2)
                        with col_alt1:
                            if st.button(f"📊 Analyze -2°F Heat, +2°F Cool", key=f"alt_wider_{name}"):
                                alt_heat = max(55, current_heat - 2)
                                alt_cool = min(90, current_cool + 2)
                                with st.spinner("Computing energy impact..."):
                                    impact = compute_suggestion_energy_impact(
                                        schedule_name=name,
                                        current_heat_sp=current_heat,
                                        current_cool_sp=current_cool,
                                        suggested_heat_sp=alt_heat,
                                        suggested_cool_sp=alt_cool,
                                        all_schedules=st.session_state.schedules,
                                        tin_model=tin_model,
                                        runtime_predictor=runtime_predictor,
                                        lat=st.session_state.user_lat,
                                        lon=st.session_state.user_lon,
                                        building_age=st.session_state.building_age_yrs,
                                        building_type=st.session_state.building_type,
                                        floor_area=st.session_state.floor_area_sqft,
                                        climate_code=st.session_state.climate_code,
                                        hvac_mode=st.session_state.hvac_mode,
                                        schedule_priorities=st.session_state.schedule_priorities,
                                    )
                                    if "error" not in impact:
                                        st.info(f"Wider deadband ({alt_heat}°F / {alt_cool}°F): {impact['recommendation_text']}")
                        with col_alt2:
                            if st.button(f"📊 Analyze +2°F Heat, -2°F Cool", key=f"alt_tighter_{name}"):
                                alt_heat = min(75, current_heat + 2)
                                alt_cool = max(70, current_cool - 2)
                                with st.spinner("Computing energy impact..."):
                                    impact = compute_suggestion_energy_impact(
                                        schedule_name=name,
                                        current_heat_sp=current_heat,
                                        current_cool_sp=current_cool,
                                        suggested_heat_sp=alt_heat,
                                        suggested_cool_sp=alt_cool,
                                        all_schedules=st.session_state.schedules,
                                        tin_model=tin_model,
                                        runtime_predictor=runtime_predictor,
                                        lat=st.session_state.user_lat,
                                        lon=st.session_state.user_lon,
                                        building_age=st.session_state.building_age_yrs,
                                        building_type=st.session_state.building_type,
                                        floor_area=st.session_state.floor_area_sqft,
                                        climate_code=st.session_state.climate_code,
                                        hvac_mode=st.session_state.hvac_mode,
                                        schedule_priorities=st.session_state.schedule_priorities,
                                    )
                                    if "error" not in impact:
                                        st.info(f"Tighter deadband ({alt_heat}°F / {alt_cool}°F): {impact['recommendation_text']}")
                    else:
                        st.caption("⚠️ Energy models not loaded - cannot analyze alternatives")
                else:
                    st.markdown("**🌡️ Suggested Setpoints (based on similar schedule names)**")
                    if sp["source"] == "fuzzy":
                        st.caption(f"Pattern match: _{sp.get('matched_name')}_ ({sp.get('match_score')}% similarity)")

                    # Show current vs suggested
                    col_cur, col_sug = st.columns(2)
                    with col_cur:
                        st.markdown("**Current:**")
                        st.write(f"🔥 Heat: {current_heat}°F | ❄️ Cool: {current_cool}°F")
                    with col_sug:
                        st.markdown("**Suggested:**")
                        st.write(f"🔥 Heat: {suggested_heat}°F | ❄️ Cool: {suggested_cool}°F")

                    # Compute energy impact
                    tin_model = st.session_state.get("tin_model")
                    runtime_predictor = st.session_state.get("runtime_predictor")

                    # Cache key for this suggestion's energy analysis
                    impact_cache_key = f"impact_{name}_{current_heat}_{current_cool}_{suggested_heat}_{suggested_cool}"
                    if not hasattr(st.session_state, "suggestion_impact_cache"):
                        st.session_state.suggestion_impact_cache = {}

                    if impact_cache_key in st.session_state.suggestion_impact_cache:
                        impact = st.session_state.suggestion_impact_cache[impact_cache_key]
                    else:
                        impact = None

                    if impact is None and tin_model and runtime_predictor:
                        if st.button("📊 Analyze Energy Impact", key=f"analyze_impact_{name}"):
                            with st.spinner("Computing energy impact..."):
                                impact = compute_suggestion_energy_impact(
                                    schedule_name=name,
                                    current_heat_sp=current_heat,
                                    current_cool_sp=current_cool,
                                    suggested_heat_sp=suggested_heat,
                                    suggested_cool_sp=suggested_cool,
                                    all_schedules=st.session_state.schedules,
                                    tin_model=tin_model,
                                    runtime_predictor=runtime_predictor,
                                    lat=st.session_state.user_lat,
                                    lon=st.session_state.user_lon,
                                    building_age=st.session_state.building_age_yrs,
                                    building_type=st.session_state.building_type,
                                    floor_area=st.session_state.floor_area_sqft,
                                    climate_code=st.session_state.climate_code,
                                    hvac_mode=st.session_state.hvac_mode,
                                    schedule_priorities=st.session_state.schedule_priorities,
                                )
                                st.session_state.suggestion_impact_cache[impact_cache_key] = impact
                                st.rerun()

                    if impact and "error" not in impact:
                        st.markdown("---")
                        st.markdown("**⚡ Energy Impact Analysis:**")

                        rec = impact.get("recommendation", "unknown")
                        if rec == "save":
                            st.success(f"✅ {impact['recommendation_text']}")
                            st.caption(f"Strategy: {impact['strategy']}")
                            apply_label = "✅ Apply (Saves Energy)"
                            apply_type = "primary"
                        elif rec == "cost":
                            st.warning(f"⚠️ {impact['recommendation_text']}")
                            st.caption(f"Strategy: {impact['strategy']} - This is a comfort-focused suggestion, not energy-saving.")
                            apply_label = "⚠️ Apply Anyway"
                            apply_type = "secondary"
                        else:
                            st.info(f"➡️ {impact['recommendation_text']}")
                            st.caption(f"Strategy: {impact['strategy']}")
                            apply_label = "Apply"
                            apply_type = "secondary"

                        st.caption(f"📊 {impact['analysis_note']}")

                        # Apply button with energy context
                        if st.button(apply_label, key=f"apply_sp_{name}", type=apply_type):
                            sched["heat_sp"] = suggested_heat
                            sched["cool_sp"] = suggested_cool
                            st.success("Applied.")
                            # Clear impact cache for this schedule
                            keys_to_clear = [k for k in st.session_state.suggestion_impact_cache if k.startswith(f"impact_{name}_")]
                            for k in keys_to_clear:
                                del st.session_state.suggestion_impact_cache[k]
                            st.rerun()
                    elif impact and "error" in impact:
                        st.warning(f"Could not analyze energy impact: {impact['error']}")
                        # Still show apply button but warn user
                        if st.button("Apply (unvalidated)", key=f"apply_sp_{name}"):
                            sched["heat_sp"] = suggested_heat
                            sched["cool_sp"] = suggested_cool
                            st.success("Applied.")
                            st.rerun()
                    elif not tin_model or not runtime_predictor:
                        st.caption("⚠️ Energy models not loaded - cannot validate energy impact")
                        if st.button("Apply (unvalidated)", key=f"apply_sp_{name}"):
                            sched["heat_sp"] = suggested_heat
                            sched["cool_sp"] = suggested_cool
                            st.success("Applied.")
                            st.rerun()

                    st.caption(f"Pattern based on {sp.get('n_homes', 0):,} California homes")
            elif not sp:
                st.info("No setpoint pattern matches found for this schedule name.")

        if location_id is not None and not timing_df.empty:
            st.markdown("**⏰ Timing suggestions (nearby homes)**")
            tt = suggest_timing(name, location_id, is_weekend=0, timing_df=timing_df)
            if tt:
                c1, c2, c3 = st.columns([2, 2, 1])
                with c1:
                    st.metric("Typical Start", f"{tt['start_hour']:02d}:00")
                with c2:
                    st.metric("Typical End", f"{tt['end_hour']:02d}:00")
                with c3:
                    if st.button("Apply", key=f"apply_time_{name}"):
                        sched["start_hour"] = int(tt["start_hour"])
                        sched["end_hour"] = int(tt["end_hour"])
                        st.success("Applied.")
                        st.rerun()
                st.caption(f"Peak activity ~ {tt['peak_hour']:02d}:00")
            else:
                st.info("No timing data for this schedule in your area.")

        st.markdown("**🤖 Name intent analysis**")
        intent = semantic_schedule_intent(name)
        st.write(f"Detected: _{intent['intent']}_")

    with st.expander("✏️ Edit Schedule", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🔥 HEAT")
            new_heat = st.slider("Heat setpoint (°F)", 55, 75, int(sched["heat_sp"]), key=f"edit_heat_{name}")
        with c2:
            st.markdown("#### ❄️ COOL")
            new_cool = st.slider("Cool setpoint (°F)", 70, 90, int(sched["cool_sp"]), key=f"edit_cool_{name}")

        c3, c4 = st.columns(2)
        with c3:
            new_start = st.selectbox("Start time", range(24), index=int(sched["start_hour"]) % 24, format_func=lambda h: f"{h:02d}:00", key=f"edit_start_{name}")
        with c4:
            new_end = st.selectbox("End time", range(24), index=int(sched["end_hour"]) % 24, format_func=lambda h: f"{h:02d}:00", key=f"edit_end_{name}")

        if st.button("💾 Save Changes", key=f"save_{name}", use_container_width=True):
            sched["heat_sp"] = int(new_heat)
            sched["cool_sp"] = int(new_cool)
            sched["start_hour"] = int(new_start)
            sched["end_hour"] = int(new_end)
            st.success("Saved.")
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

def render_schedules():
    st.markdown('<div class="frame">', unsafe_allow_html=True)
    ensure_neighbors_loaded()
    topbar("My Schedules")
    st.markdown("### Your Comfort Schedules")
    st.write("Edit your schedules or add new ones. Suggestions come from name patterns and (if available) nearby homes.")

    suggestions_df = load_schedule_suggestions()
    timing_df = load_timing_suggestions()

    # location_id: prefer session inferred; else from neighbors
    location_id = st.session_state.location_id
    if location_id is None and not st.session_state.neighbors_df.empty and "location_id" in st.session_state.neighbors_df.columns:
        try:
            topn = st.session_state.neighbors_df.nsmallest(10, "dist_km")
            if not topn.empty:
                location_id = int(topn["location_id"].mode().iloc[0])
        except Exception:
            location_id = None

    if location_id is not None:
        st.caption(f"Using location cluster #{location_id} for timing priors")

    ensure_schedule_priorities()
    with st.expander("⏭️ Schedule Overrides (Priority)", expanded=False):
        st.write("If schedules overlap, the lower number wins.")
        schedule_names = list(st.session_state.schedules.keys())
        options = list(range(1, len(schedule_names) + 1))
        updated = {}
        cols = st.columns(2)
        for i, name in enumerate(schedule_names):
            current = st.session_state.schedule_priorities.get(name, i + 1)
            current = current if current in options else options[-1]
            with cols[i % 2]:
                updated[name] = st.selectbox(
                    f"{name}",
                    options,
                    index=options.index(current),
                    key=f"priority_{name}",
                )
        st.session_state.schedule_priorities = normalize_schedule_priorities(
            st.session_state.schedules,
            updated,
        )
        st.caption("Tip: Use higher numbers for background schedules like Away.")

    for name in list(st.session_state.schedules.keys()):
        render_schedule_card(name, suggestions_df, timing_df, location_id)

    st.markdown("---")
    st.markdown("### ➕ Add New Schedule")

    with st.expander("Create custom schedule", expanded=False):
        new_name = st.text_input(
            "Schedule name",
            placeholder="e.g., Yoga, Studying, Cooking, Movie Night, Remote Work",
            help="Use descriptive names like 'yoga', 'studying', 'cooking' — the app will suggest appropriate setpoints.",
            key="new_schedule_name",
        )

        suggested_heat, suggested_cool, suggested_start, suggested_end = 68, 78, 9, 17
        source_of_suggestion = "default"
        llm_explanation = ""

        if new_name:
            # Try LLM-powered suggestion first (uses intent + location data)
            location_context = build_location_context_for_llm()
            llm_suggestion = call_llm_for_schedule_suggestion(new_name, location_context)

            suggested_heat = int(llm_suggestion["heat_sp"])
            suggested_cool = int(llm_suggestion["cool_sp"])
            suggested_start = int(llm_suggestion["start_hour"])
            suggested_end = int(llm_suggestion["end_hour"])
            llm_explanation = llm_suggestion.get("explanation", "")
            source_of_suggestion = llm_suggestion.get("source", "llm")

            # Override with exact name match from location database if available
            if not suggestions_df.empty:
                sp = suggest_setpoints(new_name, suggestions_df)
                if sp and sp.get("n_homes", 0) > 0 and sp.get("source") == "exact":
                    if sp.get("heat_sp") is not None:
                        suggested_heat = int(sp["heat_sp"])
                    if sp.get("cool_sp") is not None:
                        suggested_cool = int(sp["cool_sp"])
                    source_of_suggestion = f"exact match ({sp['n_homes']:,} homes in your area)"
                    llm_explanation = f"Found exact schedule name match in your location's database."

            st.markdown('<div class="card" style="background: rgba(34,197,94,0.08);">', unsafe_allow_html=True)
            st.markdown("**🤖 AI-Powered Suggestions**")

            # Get intent for display
            intent = semantic_schedule_intent(new_name)
            st.write(f"Intent detected: _{intent['intent']}_")

            if llm_explanation:
                st.info(f"💡 {llm_explanation}")

            st.caption(f"Suggested hours: {suggested_start:02d}:00–{suggested_end:02d}:00")
            st.caption(f"Suggested setpoints: 🔥 Heat {suggested_heat}°F, ❄️ Cool {suggested_cool}°F")
            st.caption(f"Source: {source_of_suggestion}")

            st.markdown("</div>", unsafe_allow_html=True)

        # Add CSS for colored sliders
        st.markdown("""
        <style>
        /* Heat slider styling (orange) */
        div[data-testid="stSlider"]:has(div[data-baseweb="slider"]) {
            position: relative;
        }
        /* Cool slider styling (blue) - applied via key matching */
        </style>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🔥 HEAT")
            new_heat = st.slider("Heat setpoint (°F)", 55, 75, int(suggested_heat), key="new_heat")
            new_start = st.selectbox("Start time", range(24), index=int(suggested_start) % 24, format_func=lambda h: f"{h:02d}:00", key="new_start")
        with c2:
            st.markdown("### ❄️ COOL")
            new_cool = st.slider("Cool setpoint (°F)", 70, 90, int(suggested_cool), key="new_cool")
            new_end = st.selectbox("End time", range(24), index=int(suggested_end) % 24, format_func=lambda h: f"{h:02d}:00", key="new_end")

        # Show estimated runtime/energy impact if runtime predictor is available
        if new_name and (new_heat != suggested_heat or new_cool != suggested_cool):
            if not st.session_state.runtime_model_loaded:
                st.session_state.runtime_predictor = load_runtime_predictor()
                st.session_state.runtime_model_loaded = True

            runtime_predictor = st.session_state.runtime_predictor
            if runtime_predictor and st.session_state.outdoor_temp is not None:
                # Quick runtime estimate for this schedule
                st.markdown("---")
                st.markdown("**⚡ Estimated Energy Impact**")

                # Simulate 1 hour with these setpoints
                test_data = pd.DataFrame({
                    "time": [datetime.now()],
                    "outdoor_temp": [float(st.session_state.outdoor_temp)],
                    "indoor_temp_pred": [float(st.session_state.indoor_temp)],
                    "heat_sp": [float(new_heat)],
                    "cool_sp": [float(new_cool)],
                    "schedule": [new_name],
                })

                try:
                    result = runtime_predictor.predict(test_data, use_ev=False)
                    runtime_sec = float(result["runtime_sec_pred_sec"].iloc[0])
                    runtime_min = runtime_sec / 60.0

                    # Estimate hourly energy
                    hourly_kwh = estimate_energy_from_runtime(runtime_min / 60.0)
                    daily_kwh = hourly_kwh * (abs(new_end - new_start) if new_end != new_start else 8)

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Est. Runtime/Hour", f"{runtime_min:.1f} min")
                    with col_b:
                        st.metric("Est. Daily Energy", f"{daily_kwh:.1f} kWh")

                    st.caption(f"Based on current conditions: {st.session_state.outdoor_temp:.0f}°F outside, {st.session_state.indoor_temp:.0f}°F inside")
                except Exception:
                    pass  # Silently fail if prediction doesn't work

        if st.button("➕ Add Schedule", use_container_width=True, type="primary"):
            if not new_name.strip():
                st.error("Please enter a schedule name.")
            elif new_name in st.session_state.schedules:
                st.error("Schedule name already exists.")
            else:
                st.session_state.schedules[new_name] = {
                    "heat_sp": int(new_heat),
                    "cool_sp": int(new_cool),
                    "start_hour": int(new_start),
                    "end_hour": int(new_end),
                }
                st.session_state.schedule_priorities = normalize_schedule_priorities(
                    st.session_state.schedules,
                    st.session_state.get("schedule_priorities", {}),
                )
                st.success(f"Added '{new_name}'.")
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    bottom_nav()
    assistant_bar()


# =========================================================
# FORECAST VIEW
# =========================================================
def render_forecast():
    st.markdown('<div class="frame">', unsafe_allow_html=True)
    ensure_neighbors_loaded()
    ensure_schedule_priorities()
    topbar("7-Day Forecast")

    st.markdown("### Energy Impact Prediction")
    st.write("Compare baseline vs. a scenario. Scenarios never change your real schedules (sandbox).")

    if not st.session_state.model_loaded:
        st.session_state.tin_model = load_tin_model()
        st.session_state.model_loaded = True

    if not st.session_state.runtime_model_loaded:
        st.session_state.runtime_predictor = load_runtime_predictor()
        st.session_state.runtime_model_loaded = True

    tin_model = st.session_state.tin_model
    runtime_predictor = st.session_state.runtime_predictor

    if not tin_model:
        st.error("Tin prediction model not available. (Missing file or optional dependencies.)")
        st.caption(f"Expected model at: {TIN_MODEL_PATH}")
        st.markdown("</div>", unsafe_allow_html=True)
        bottom_nav()
        return

    # weather cache
    if st.session_state.forecast_data is None:
        with st.spinner("Fetching weather forecast..."):
            st.session_state.forecast_data = fetch_hourly_forecast(st.session_state.user_lat, st.session_state.user_lon, days=7)

    weather = st.session_state.forecast_data
    if weather is None or weather.empty:
        st.error("Could not fetch weather forecast.")
        if st.button("🔄 Retry", use_container_width=True):
            st.session_state.forecast_data = None
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        bottom_nav()
        return

    # Scenario picker
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Compare Scenarios")

    base_scenarios = [
        "Baseline (current schedules)",
        "Warmer home: +2°F heating/-2°F cooling",
        "Cooler home: -2°F heating/+2°F cooling",
        "Peak Hours Shift (less cooling 4–9pm)",
    ]
    custom = sorted(list(st.session_state.custom_scenarios.keys()))
    scenario_options = base_scenarios + (["— Custom —"] if custom else []) + custom

    # Initialize default scenario if not set
    if "selected_scenario" not in st.session_state or st.session_state.selected_scenario not in scenario_options:
        st.session_state.selected_scenario = "Baseline (current schedules)"

    try:
        default_idx = scenario_options.index(st.session_state.selected_scenario)
    except ValueError:
        default_idx = 0

    # Use selectbox return value directly (updates immediately on user selection)
    scenario = st.selectbox(
        "Test different strategies:",
        scenario_options,
        index=default_idx,
        key="scenario_selector_widget",  # Different key to avoid conflict
    )

    # Update session state for persistence and force rerun if changed
    if scenario != st.session_state.selected_scenario:
        st.session_state.selected_scenario = scenario
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    avg_outdoor = float(weather["outdoor_temp"].mean())
    avg_heat_sp = float(np.mean([s["heat_sp"] for s in st.session_state.schedules.values()]))
    avg_cool_sp = float(np.mean([s["cool_sp"] for s in st.session_state.schedules.values()]))
    if avg_outdoor <= avg_heat_sp + 2:
        peak_mode = "heating"
    elif avg_outdoor >= avg_cool_sp - 2:
        peak_mode = "cooling"
    else:
        peak_mode = "mixed"

    if "Peak Hours" in scenario:
        st.caption(f"Peak Hours Shift adjusts {peak_mode} setpoints based on this week's forecast.")

    # AI Scenario Builder
    st.markdown("### 🧪 Build a custom scenario (sandbox)")
    st.caption("This won't change your real schedules — only the forecast comparison.")
    user_goal = st.text_input(
        "Describe what you want:",
        placeholder="e.g., Reduce runtime 4–9pm, keep Sleep unchanged, relax Home cooling slightly",
        key="scenario_builder_goal",
    )
    if st.button("Build scenario with AI", use_container_width=True):
        if not user_goal.strip():
            st.warning("Type a goal first.")
        else:
            with st.spinner("Building scenario with location-based insights..."):
                location_ctx = build_location_context_for_llm()
                expl, action = call_openrouter_for_scenario(user_goal.strip(), location_context=location_ctx)
            st.write(expl)
            if action and action.get("type") == "create_scenario":
                msg = apply_action(action)
                st.success(msg)
                st.rerun()
            else:
                st.error("No valid create_scenario returned. Try rephrasing.")

    # Baseline - compute only if not cached or if schedules changed
    cache_key = f"baseline_{hash(str(st.session_state.schedules))}_{hash(str(st.session_state.schedule_priorities))}"
    if "baseline_cache_key" not in st.session_state or st.session_state.baseline_cache_key != cache_key:
        with st.spinner("Generating baseline forecast..."):
            baseline_schedules = {k: dict(v) for k, v in st.session_state.schedules.items()}
            baseline_forecast = predict_tin_7days(
                weather_df=weather,
                schedules=baseline_schedules,
                indoor_seed=float(st.session_state.indoor_temp),
                building_age=st.session_state.building_age_yrs,
                building_type=st.session_state.building_type,
                floor_area=st.session_state.floor_area_sqft,
                climate_code=st.session_state.climate_code,
                tin_model=tin_model,
                schedule_priorities=st.session_state.schedule_priorities,
            )
            # Use XGBoost predictor if available, otherwise fall back to simple heuristic
            if runtime_predictor:
                baseline_runtime = estimate_runtime_with_predictor(
                    baseline_forecast,
                    runtime_predictor,
                    hvac_mode=st.session_state.hvac_mode,
                    use_ev=False,
                )
            else:
                baseline_runtime = estimate_runtime_simple(baseline_forecast)

            # Cache the results
            st.session_state.baseline_forecast = baseline_forecast
            st.session_state.baseline_runtime = baseline_runtime
            st.session_state.baseline_cache_key = cache_key
    else:
        # Use cached baseline
        baseline_forecast = st.session_state.baseline_forecast
        baseline_runtime = st.session_state.baseline_runtime

    # Scenario schedules
    scenario_schedules = {k: dict(v) for k, v in st.session_state.schedules.items()}
    is_baseline = (scenario == "Baseline (current schedules)")
    is_custom = (scenario in st.session_state.custom_scenarios)

    scenario_label = "Baseline"
    scenario_color = "#9CA3AF"

    if is_custom:
        rules = st.session_state.custom_scenarios[scenario]
        scenario_schedules = apply_custom_scenario(scenario_schedules, rules)
        scenario_label = scenario
        scenario_color = "#22C55E"
    elif "Warmer home" in scenario:
        for n in scenario_schedules:
            scenario_schedules[n]["cool_sp"] = max(70, int(scenario_schedules[n]["cool_sp"]) - 2)
            scenario_schedules[n]["heat_sp"] = min(75, int(scenario_schedules[n]["heat_sp"]) + 2)
        scenario_label = "Warmer home"
        scenario_color = "#F97316"
    elif "Cooler home" in scenario:
        for n in scenario_schedules:
            scenario_schedules[n]["cool_sp"] = min(90, int(scenario_schedules[n]["cool_sp"]) + 2)
            scenario_schedules[n]["heat_sp"] = max(55, int(scenario_schedules[n]["heat_sp"]) - 2)
        scenario_label = "Cooler home"
        scenario_color = "#22C55E"
    elif "Peak Hours" in scenario:
        target_names = []
        for n in scenario_schedules:
            tokens = set(norm_text(n).split())
            if tokens & {"home", "evening", "awake", "study", "studying", "afternoon", "work", "day"}:
                target_names.append(n)
        if not target_names:
            target_names = list(scenario_schedules.keys())
        for n in target_names:
                if peak_mode == "heating":
                    scenario_schedules[n]["heat_sp"] = max(55, int(scenario_schedules[n]["heat_sp"]) - 2)
                elif peak_mode == "cooling":
                    scenario_schedules[n]["cool_sp"] = min(90, int(scenario_schedules[n]["cool_sp"]) + 3)
                else:
                    scenario_schedules[n]["heat_sp"] = max(55, int(scenario_schedules[n]["heat_sp"]) - 1)
                    scenario_schedules[n]["cool_sp"] = min(90, int(scenario_schedules[n]["cool_sp"]) + 1)
        scenario_label = "Peak Hours Shift"
        scenario_color = "#EF4444"

    if not is_baseline:
        with st.spinner(f"Generating {scenario_label} forecast..."):
            scenario_forecast = predict_tin_7days(
                weather_df=weather,
                schedules=scenario_schedules,
                indoor_seed=float(st.session_state.indoor_temp),
                building_age=st.session_state.building_age_yrs,
                building_type=st.session_state.building_type,
                floor_area=st.session_state.floor_area_sqft,
                climate_code=st.session_state.climate_code,
                tin_model=tin_model,
                schedule_priorities=st.session_state.schedule_priorities,
            )
            # Use XGBoost predictor if available, otherwise fall back to simple heuristic
            if runtime_predictor:
                scenario_runtime = estimate_runtime_with_predictor(
                    scenario_forecast,
                    runtime_predictor,
                    hvac_mode=st.session_state.hvac_mode,
                    use_ev=False,
                )
            else:
                scenario_runtime = estimate_runtime_simple(scenario_forecast)
    else:
        scenario_forecast = baseline_forecast
        scenario_runtime = baseline_runtime

    # Summary
    st.markdown("---")
    st.markdown("### 📊 Impact Summary")

    baseline_total = float(baseline_runtime["runtime_hours"].sum())
    scenario_total = float(scenario_runtime["runtime_hours"].sum())
    delta = scenario_total - baseline_total
    pct_change = (delta / baseline_total * 100) if baseline_total > 0 else 0.0

    if is_baseline:
        # When viewing baseline, show just the baseline metrics
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("7-Day Runtime", f"{baseline_total:.1f} h")
        with c2:
            st.metric("Avg Daily", f"{baseline_total/7:.1f} h/day")
        with c3:
            # Estimate monthly from 7-day
            monthly_estimate = baseline_total / 7 * 30
            st.metric("Est. Monthly", f"{monthly_estimate:.0f} h")
        st.info("ℹ️ Select a scenario above to compare energy impact vs baseline.")
    else:
        # When comparing scenarios, show baseline vs scenario
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Baseline (Current)", f"{baseline_total:.1f} h")
        with c2:
            st.metric(f"{scenario_label}", f"{scenario_total:.1f} h", delta=f"{delta:+.1f} h")
        with c3:
            st.metric("Change", f"{pct_change:+.1f}%")

        if pct_change < -5:
            st.success(f"💰 This strategy could save ~{abs(pct_change):.0f}% runtime.")
        elif pct_change > 5:
            st.warning(f"⚠️ This strategy increases runtime by ~{pct_change:.0f}%.")
        else:
            st.info("Similar runtime to baseline.")

    st.caption(
        "Runtime impact is computed from hourly Tin indoor temperature predictions + "
        "XGBoost runtime estimates, summed over the 7-day forecast and compared to baseline."
    )

    component_cols = [
        c for c in baseline_runtime.columns
        if c.endswith("_hours") and c not in ["runtime_hours"]
    ]
    if not is_baseline and component_cols:
        rows = []
        for col in component_cols:
            if col not in scenario_runtime.columns:
                continue
            label = col.replace("_sec_hours", "").replace("_hours", "")
            if "aux" in label:
                label = "Aux Heat"
            elif "heat" in label:
                label = "Heat"
            elif "cool" in label:
                label = "Cool"
            elif "fan" in label:
                label = "Fan"
            else:
                label = label.replace("_", " ").title()
            b = float(baseline_runtime[col].sum())
            s = float(scenario_runtime[col].sum())
            rows.append({
                "System": label,
                "Baseline (h)": round(b, 1),
                "Scenario (h)": round(s, 1),
                "Δ (h)": round(s - b, 1),
            })
        if rows:
            st.markdown("#### 🔎 Runtime Breakdown (7-Day)")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Temperature plot
    st.markdown("### 🌡️ Indoor Temperature Forecast")
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(
        x=baseline_forecast["time"], y=baseline_forecast["outdoor_temp"],
        name="Outdoor", line=dict(color="#9CA3AF", width=2, dash="dot")
    ))
    fig_temp.add_trace(go.Scatter(
        x=baseline_forecast["time"], y=baseline_forecast["indoor_temp_pred"],
        name="Indoor (Baseline)", line=dict(color="#60A5FA", width=3)
    ))
    if not is_baseline:
        fig_temp.add_trace(go.Scatter(
            x=scenario_forecast["time"], y=scenario_forecast["indoor_temp_pred"],
            name=f"Indoor ({scenario_label})", line=dict(color=scenario_color, width=3)
        ))
    fig_temp.update_layout(template="plotly_dark", height=420, xaxis_title="Date", yaxis_title="Temperature (°F)")
    st.plotly_chart(fig_temp, use_container_width=True)

    # Daily runtime plot
    st.markdown("### 🧊 Estimated Runtime (Daily)")
    fig_rt = go.Figure()
    fig_rt.add_trace(go.Bar(
        x=baseline_runtime["date"], y=baseline_runtime["runtime_hours"], name="Baseline"
    ))
    if not is_baseline:
        fig_rt.add_trace(go.Bar(
            x=scenario_runtime["date"], y=scenario_runtime["runtime_hours"], name=scenario_label
        ))
    fig_rt.update_layout(template="plotly_dark", barmode="group", height=380, xaxis_title="Day", yaxis_title="Runtime (hours/day)")
    st.plotly_chart(fig_rt, use_container_width=True)

    # Downloads - hidden per user request
    # st.markdown("### 📥 Download")
    # colA, colB = st.columns(2)
    # with colA:
    #     st.download_button(
    #         "Download baseline hourly CSV",
    #         data=baseline_forecast.to_csv(index=False).encode("utf-8"),
    #         file_name="baseline_hourly_forecast.csv",
    #         mime="text/csv",
    #         use_container_width=True,
    #     )
    # with colB:
    #     st.download_button(
    #         f"Download {scenario_label} hourly CSV",
    #         data=scenario_forecast.to_csv(index=False).encode("utf-8"),
    #         file_name="scenario_hourly_forecast.csv",
    #         mime="text/csv",
    #         use_container_width=True,
    #     )

    st.markdown("</div>", unsafe_allow_html=True)
    bottom_nav()
    assistant_bar()


# =========================================================
# ENERGY CONSUMPTION HELPERS
# =========================================================
def compute_monthly_energy_breakdown(
    months_data: Dict[str, Dict[str, float]],
    heat_kw: float,
    cool_kw: float,
    aux_kw: float,
    fan_kw: float,
    cost_per_kwh: float,
    gas_btu_per_hr: float = 0.0,
    gas_cost_per_therm: float = 0.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Convert monthly runtime components into energy and cost.
    """
    rows = []
    totals = {
        "heat_kwh": 0.0,
        "cool_kwh": 0.0,
        "aux_kwh": 0.0,
        "fan_kwh": 0.0,
        "electric_kwh": 0.0,
        "gas_therms": 0.0,
        "electric_cost": 0.0,
        "gas_cost": 0.0,
        "total_cost": 0.0,
    }

    for m in MONTH_NAMES:
        month = months_data.get(m, {})
        heat_hours = float(month.get("heat_sec_hours", 0))
        cool_hours = float(month.get("cool_sec_hours", 0))
        aux_hours = float(month.get("aux_sec_hours", 0))
        fan_hours = float(month.get("fan_sec_hours", 0))
        avg_temp = float(month.get("avg_outdoor_temp", 0))

        if gas_btu_per_hr > 0:
            heat_therms = heat_hours * gas_btu_per_hr / 100000.0
            heat_kwh = 0.0
        else:
            heat_therms = 0.0
            heat_kwh = heat_hours * heat_kw

        cool_kwh = cool_hours * cool_kw
        aux_kwh = aux_hours * aux_kw
        fan_kwh = fan_hours * fan_kw
        electric_kwh = heat_kwh + cool_kwh + aux_kwh + fan_kwh
        electric_cost = electric_kwh * cost_per_kwh
        gas_cost = heat_therms * gas_cost_per_therm
        total_cost = electric_cost + gas_cost
        heat_kwh_equiv = heat_therms * 29.3 if heat_therms > 0 else heat_kwh

        rows.append({
            "Month": m,
            "heat_kwh": heat_kwh,
            "heat_therms": heat_therms,
            "heat_kwh_equiv": heat_kwh_equiv,
            "cool_kwh": cool_kwh,
            "aux_kwh": aux_kwh,
            "fan_kwh": fan_kwh,
            "electric_kwh": electric_kwh,
            "electric_cost": electric_cost,
            "gas_cost": gas_cost,
            "total_cost": total_cost,
            "avg_outdoor_temp": avg_temp,
        })

        totals["heat_kwh"] += heat_kwh
        totals["cool_kwh"] += cool_kwh
        totals["aux_kwh"] += aux_kwh
        totals["fan_kwh"] += fan_kwh
        totals["electric_kwh"] += electric_kwh
        totals["gas_therms"] += heat_therms
        totals["electric_cost"] += electric_cost
        totals["gas_cost"] += gas_cost
        totals["total_cost"] += total_cost

    return rows, totals

def estimate_monthly_energy(daily_runtime_df: pd.DataFrame, days_in_month: int = 30) -> Dict[str, float]:
    """
    Project 7-day runtime to monthly energy consumption.

    Returns dict with:
    - total_kwh: Total energy consumption for the month
    - avg_daily_kwh: Average daily energy
    - monthly_cost: Estimated cost (at $0.14/kWh avg US residential rate)
    """
    if daily_runtime_df.empty or "runtime_hours" not in daily_runtime_df.columns:
        return {"total_kwh": 0.0, "avg_daily_kwh": 0.0, "monthly_cost": 0.0}

    # Calculate average daily runtime from the 7-day forecast
    avg_daily_runtime = float(daily_runtime_df["runtime_hours"].mean())

    # Convert to energy
    avg_daily_kwh = estimate_energy_from_runtime(avg_daily_runtime)

    # Project to full month
    total_monthly_kwh = avg_daily_kwh * days_in_month

    # Estimate cost at average US residential rate of $0.14/kWh (EIA 2024)
    monthly_cost = total_monthly_kwh * 0.14

    return {
        "total_kwh": round(total_monthly_kwh, 1),
        "avg_daily_kwh": round(avg_daily_kwh, 2),
        "monthly_cost": round(monthly_cost, 2),
        "avg_daily_runtime": round(avg_daily_runtime, 2),
    }

# =========================================================
# REPORTS VIEW - Primary Energy Analysis using Historical Weather
# =========================================================
def render_reports():
    st.markdown('<div class="frame">', unsafe_allow_html=True)
    ensure_neighbors_loaded()
    ensure_schedule_priorities()
    topbar("Reports")

    # Energy Consumption Analysis - PRIMARY METHOD: Historical Weather
    st.markdown("---")
    st.markdown("### ⚡ Annual Energy Analysis")
    st.markdown('<p style="color: white; font-size: 14px; margin-bottom: 16px;">Accurate energy predictions using actual historical outdoor temperature data (8,760 hours/year)</p>', unsafe_allow_html=True)

    # Load models
    if not st.session_state.get("model_loaded"):
        st.session_state.tin_model = load_tin_model()
        st.session_state.model_loaded = True

    if not st.session_state.get("runtime_model_loaded"):
        st.session_state.runtime_predictor = load_runtime_predictor()
        st.session_state.runtime_model_loaded = True

    tin_model = st.session_state.tin_model
    runtime_predictor = st.session_state.runtime_predictor

    # Year selector and equipment settings
    col_year, col_equip, col_rate = st.columns([1, 1, 1])
    with col_year:
        analysis_year = 2025
        st.markdown(
            '<div class="static-field"><div class="static-field-label">Analysis Year</div>'
            f'<div class="static-field-value">{analysis_year}</div></div>',
            unsafe_allow_html=True,
        )
    with col_equip:
        equipment_presets = {
            "Heat Pump (electric)": {"heat_kw": 3.0, "cool_kw": 3.5, "aux_kw": 7.5, "fan_kw": 0.6, "gas_btu_per_hr": 0},
            "Gas Furnace + Central AC": {"heat_kw": 0.0, "cool_kw": 3.5, "aux_kw": 0.0, "fan_kw": 0.6, "gas_btu_per_hr": 60000},
            "Electric Resistance + AC": {"heat_kw": 7.5, "cool_kw": 3.5, "aux_kw": 7.5, "fan_kw": 0.6, "gas_btu_per_hr": 0},
            "Mini-Split Heat Pump": {"heat_kw": 2.0, "cool_kw": 2.0, "aux_kw": 0.0, "fan_kw": 0.4, "gas_btu_per_hr": 0},
            "Custom": {"heat_kw": 3.5, "cool_kw": 3.5, "aux_kw": 7.5, "fan_kw": 0.6, "gas_btu_per_hr": 0},
        }

        equipment_type = st.selectbox(
            "Equipment Type",
            options=list(equipment_presets.keys()),
            key="equipment_type",
            help="Pick a typical system to seed power assumptions. You can fine-tune below.",
        )

        preset = equipment_presets[equipment_type]
        if st.session_state.get("equipment_preset_last") != equipment_type:
            st.session_state["heat_kw"] = preset["heat_kw"]
            st.session_state["cool_kw"] = preset["cool_kw"]
            st.session_state["aux_kw"] = preset["aux_kw"]
            st.session_state["fan_kw"] = preset["fan_kw"]
            st.session_state["gas_kbtu_per_hr"] = preset["gas_btu_per_hr"] / 1000 if preset["gas_btu_per_hr"] else 0.0
            st.session_state["equipment_preset_last"] = equipment_type

    with col_rate:
        cost_per_kwh = st.number_input(
            "Electricity Rate ($/kWh)",
            value=0.30,  # California average is higher
            min_value=0.05,
            max_value=1.0,
            step=0.01,
            help="California avg: $0.25-0.35/kWh"
        )

    with st.expander("⚙️ Equipment details", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            heat_kw = st.number_input("Heat (kW)", min_value=0.0, max_value=15.0, step=0.5, key="heat_kw")
        with c2:
            cool_kw = st.number_input("Cool (kW)", min_value=0.0, max_value=15.0, step=0.5, key="cool_kw")
        with c3:
            aux_kw = st.number_input("Aux (kW)", min_value=0.0, max_value=20.0, step=0.5, key="aux_kw")
        with c4:
            fan_kw = st.number_input("Fan (kW)", min_value=0.0, max_value=5.0, step=0.1, key="fan_kw")

        gas_btu_per_hr = 0.0
        gas_rate_per_therm = 0.0
        if "Gas Furnace" in equipment_type:
            c5, c6 = st.columns(2)
            with c5:
                gas_kbtu = st.number_input(
                    "Gas heat input (kBtu/h)",
                    min_value=10.0,
                    max_value=200.0,
                    step=5.0,
                    key="gas_kbtu_per_hr",
                )
                gas_btu_per_hr = gas_kbtu * 1000.0
            with c6:
                gas_rate_per_therm = st.number_input(
                    "Gas rate ($/therm)",
                    min_value=0.50,
                    max_value=5.0,
                    step=0.05,
                    value=1.80,
                    key="gas_rate_per_therm",
                )

    # Check cache
    cache_key_annual = f"annual_{analysis_year}_{hash(str(st.session_state.schedules))}_{hash(str(st.session_state.schedule_priorities))}_{st.session_state.user_lat}_{st.session_state.user_lon}"
    if not hasattr(st.session_state, "annual_energy_cache"):
        st.session_state.annual_energy_cache = {}

    # Find nearest weather station
    location_id, distance_km = find_nearest_outdoor_location(st.session_state.user_lat, st.session_state.user_lon)

    if location_id:
        st.info(f"📍 Weather station **#{location_id}** ({distance_km:.1f} km from your location)")

    # Auto-calculate or use cached
    need_calculation = cache_key_annual not in st.session_state.annual_energy_cache

    if need_calculation:
        if tin_model is None or runtime_predictor is None:
            st.error("Models not loaded. Cannot compute energy analysis.")
        else:
            with st.spinner(f"Computing energy using {analysis_year} historical weather (8,760 hours)... This takes ~1-2 minutes."):
                annual_predictions, annual_summary = compute_annual_runtime_with_historical_weather(
                    schedules=st.session_state.schedules,
                    tin_model=tin_model,
                    runtime_predictor=runtime_predictor,
                    lat=st.session_state.user_lat,
                    lon=st.session_state.user_lon,
                    indoor_seed=float(st.session_state.indoor_temp),
                    building_age=st.session_state.building_age_yrs,
                    building_type=st.session_state.building_type,
                    floor_area=st.session_state.floor_area_sqft,
                    climate_code=st.session_state.climate_code,
                    hvac_mode=st.session_state.hvac_mode,
                    year=analysis_year,
                    schedule_priorities=st.session_state.schedule_priorities,
                )

            if annual_predictions is not None:
                st.session_state.annual_energy_cache[cache_key_annual] = {
                    "predictions": annual_predictions,
                    "summary": annual_summary,
                }
            else:
                st.error(f"Could not compute: {annual_summary.get('error', 'Unknown error')}")

    # Display results if cached
    if cache_key_annual in st.session_state.annual_energy_cache:
        cached = st.session_state.annual_energy_cache[cache_key_annual]
        annual_predictions = cached["predictions"]
        annual_summary = cached["summary"]

        # =====================================================
        # PRIMARY METRICS - Annual Summary
        # =====================================================
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"#### Annual Energy Summary ({analysis_year})")

        # Calculate totals with equipment-specific power settings
        months_data = annual_summary.get("months", {})
        baseline_rows, baseline_totals = compute_monthly_energy_breakdown(
            months_data=months_data,
            heat_kw=heat_kw,
            cool_kw=cool_kw,
            aux_kw=aux_kw,
            fan_kw=fan_kw,
            cost_per_kwh=cost_per_kwh,
            gas_btu_per_hr=gas_btu_per_hr,
            gas_cost_per_therm=gas_rate_per_therm,
        )

        annual_kwh = baseline_totals["electric_kwh"]
        annual_gas_therms = baseline_totals["gas_therms"]
        annual_cost = baseline_totals["total_cost"]
        if annual_gas_therms > 0:
            r1 = st.columns(4)
            with r1[0]:
                render_report_metric("Electric Energy", f"{annual_kwh:,.0f} kWh")
            with r1[1]:
                render_report_metric("Gas Energy", f"{annual_gas_therms:,.0f} therms")
            with r1[2]:
                render_report_metric("Annual Cost", f"${annual_cost:,.0f}")
            with r1[3]:
                render_report_metric("Avg Monthly", f"${annual_cost/12:,.0f}")
        else:
            r1 = st.columns(3)
            with r1[0]:
                render_report_metric("Total Energy", f"{annual_kwh:,.0f} kWh")
            with r1[1]:
                render_report_metric("Annual Cost", f"${annual_cost:,.0f}")
            with r1[2]:
                render_report_metric("Avg Monthly", f"${annual_cost/12:,.0f}")

        r2 = st.columns(4)
        if annual_gas_therms > 0:
            heat_label = "Heating (Gas)"
            heat_value = f"{annual_gas_therms:,.0f} therms"
        else:
            heat_label = "Heating"
            heat_value = f"{baseline_totals['heat_kwh']:,.0f} kWh"
        with r2[0]:
            render_report_metric(heat_label, heat_value)
        with r2[1]:
            render_report_metric("Cooling", f"{baseline_totals['cool_kwh']:,.0f} kWh")
        with r2[2]:
            render_report_metric("Aux Heat", f"{baseline_totals['aux_kwh']:,.0f} kWh")
        with r2[3]:
            render_report_metric("Fan", f"{baseline_totals['fan_kwh']:,.0f} kWh")

        st.markdown("</div>", unsafe_allow_html=True)

        # =====================================================
        # Monthly Breakdown Chart
        # =====================================================
        st.markdown("---")
        st.markdown("### 📊 Monthly Energy Breakdown")

        monthly_heat_kwh = [r["heat_kwh_equiv"] for r in baseline_rows]
        monthly_cool_kwh = [r["cool_kwh"] for r in baseline_rows]
        monthly_aux_kwh = [r["aux_kwh"] for r in baseline_rows]
        monthly_fan_kwh = [r["fan_kwh"] for r in baseline_rows]
        monthly_temps = [r["avg_outdoor_temp"] for r in baseline_rows]
        monthly_costs = [r["total_cost"] for r in baseline_rows]
        monthly_gas_therms = [r["heat_therms"] for r in baseline_rows]
        heat_label = "Heating (gas equiv)" if annual_gas_therms > 0 else "Heating"

        # Stacked bar chart
        fig_monthly = go.Figure()

        if any(monthly_heat_kwh):
            heat_hover = "Heating: %{y:.0f} kWh ($%{customdata:.0f})<extra></extra>"
            heat_custom = [h * cost_per_kwh for h in monthly_heat_kwh]
            if annual_gas_therms > 0:
                heat_hover = "Heating (gas equiv): %{y:.0f} kWh-eq<extra></extra>"
                heat_custom = None
            fig_monthly.add_trace(go.Bar(
                name=heat_label, x=MONTH_NAMES, y=monthly_heat_kwh,
                marker_color="#F97316",
                hovertemplate=heat_hover,
                customdata=heat_custom,
            ))

        if any(monthly_cool_kwh):
            fig_monthly.add_trace(go.Bar(
                name="Cooling", x=MONTH_NAMES, y=monthly_cool_kwh,
                marker_color="#60A5FA",
                hovertemplate="Cooling: %{y:.0f} kWh ($%{customdata:.0f})<extra></extra>",
                customdata=[c * cost_per_kwh for c in monthly_cool_kwh],
            ))

        if any(monthly_aux_kwh):
            fig_monthly.add_trace(go.Bar(
                name="Aux Heat", x=MONTH_NAMES, y=monthly_aux_kwh,
                marker_color="#EF4444",
                hovertemplate="Aux Heat: %{y:.0f} kWh ($%{customdata:.0f})<extra></extra>",
                customdata=[a * cost_per_kwh for a in monthly_aux_kwh],
            ))

        if any(monthly_fan_kwh):
            fig_monthly.add_trace(go.Bar(
                name="Fan", x=MONTH_NAMES, y=monthly_fan_kwh,
                marker_color="#22C55E",
                hovertemplate="Fan: %{y:.0f} kWh ($%{customdata:.0f})<extra></extra>",
                customdata=[f * cost_per_kwh for f in monthly_fan_kwh],
            ))

        # Temperature line
        fig_monthly.add_trace(go.Scatter(
            name="Outdoor Temp", x=MONTH_NAMES, y=monthly_temps,
            mode="lines+markers", line=dict(color="#9CA3AF", width=2),
            marker=dict(size=8), yaxis="y2",
            hovertemplate="Avg: %{y:.1f}°F<extra></extra>",
        ))

        fig_monthly.update_layout(
            barmode="stack",
            title=f"Monthly HVAC Energy - {analysis_year}",
            xaxis_title="Month",
            yaxis_title="Energy (kWh)",
            yaxis2=dict(title="Temp (°F)", overlaying="y", side="right", showgrid=False),
            template="plotly_dark",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

        # Monthly cost table
        cost_data = []
        for i, m in enumerate(MONTH_NAMES):
            total_kwh = monthly_heat_kwh[i] + monthly_cool_kwh[i] + monthly_aux_kwh[i] + monthly_fan_kwh[i]
            row = {
                "Month": m,
                "Cooling (kWh)": round(monthly_cool_kwh[i], 0),
                "Aux (kWh)": round(monthly_aux_kwh[i], 0),
                "Fan (kWh)": round(monthly_fan_kwh[i], 0),
                "Electric kWh": round(total_kwh, 0),
                "Cost": f"${monthly_costs[i]:,.0f}",
                "Avg °F": round(monthly_temps[i], 0),
            }
            if annual_gas_therms > 0:
                row["Heat (therms)"] = round(monthly_gas_therms[i], 1)
            else:
                row["Heat (kWh)"] = round(monthly_heat_kwh[i], 0)
            cost_data.append(row)

        st.dataframe(pd.DataFrame(cost_data), use_container_width=True, hide_index=True)

        if annual_gas_therms > 0:
            st.caption("Gas heating is shown as kWh-equivalent in the chart; use therms in the table for billing.")

        # Component summary
        st.markdown("---")
        st.markdown("### 🔍 Energy Component Summary")

        col_comp = st.columns(4)
        total_cost = baseline_totals["total_cost"] if baseline_totals["total_cost"] > 0 else 1.0
        components = []
        if annual_gas_therms > 0:
            components.append(("🔥 Gas Heating", annual_gas_therms, "therms", baseline_totals["gas_cost"]))
        else:
            heat_kwh = baseline_totals["heat_kwh"]
            components.append(("🔥 Heating", heat_kwh, "kWh", heat_kwh * cost_per_kwh))

        cool_kwh = baseline_totals["cool_kwh"]
        aux_kwh = baseline_totals["aux_kwh"]
        fan_kwh = baseline_totals["fan_kwh"]
        components.extend([
            ("❄️ Cooling", cool_kwh, "kWh", cool_kwh * cost_per_kwh),
            ("🔴 Aux Heat", aux_kwh, "kWh", aux_kwh * cost_per_kwh),
            ("💨 Fan", fan_kwh, "kWh", fan_kwh * cost_per_kwh),
        ])

        for i, (label, value, unit, cost) in enumerate(components):
            with col_comp[i]:
                pct = (cost / total_cost * 100) if total_cost > 0 else 0
                st.markdown(f"**{label}**")
                st.markdown(f"{value:,.0f} {unit} ({pct:.0f}%)")
                st.markdown(f"${cost:,.0f}/yr")

        # Recalculate button
        if st.button("🔄 Recalculate", key="recalc_annual"):
            del st.session_state.annual_energy_cache[cache_key_annual]
            st.rerun()

        # =====================================================
        # SCENARIO COMPARISON - What-if Analysis
        # =====================================================
        st.markdown("---")
        st.markdown("### 🔄 What-If Scenario Comparison")
        st.markdown('<p style="color: white; font-size: 14px; margin-bottom: 16px;">See how different strategies affect your annual energy consumption</p>', unsafe_allow_html=True)

        # Define scenarios
        scenario_options = {
            "Cooler home (-2°F heat, +2°F cool)": {"heat_delta": -2, "cool_delta": +2, "desc": "Energy saving: wider deadband"},
            "Warmer home (+2°F heat, -2°F cool)": {"heat_delta": +2, "cool_delta": -2, "desc": "More comfort: tighter deadband"},
            "Night setback (-4°F heat at night)": {"heat_delta": -4, "cool_delta": 0, "schedules": ["Sleep", "Night"], "desc": "Lower heating during sleep"},
            "Away savings (+4°F cool, -4°F heat when Away)": {"heat_delta": -4, "cool_delta": +4, "schedules": ["Away"], "desc": "Let home float when nobody's there"},
        }

        selected_scenarios = st.multiselect(
            "Select scenarios to compare:",
            list(scenario_options.keys()),
            default=["Cooler home (-2°F heat, +2°F cool)"],
            help="Choose one or more scenarios to compare against your current baseline"
        )

        if selected_scenarios and st.button("📊 Compare Scenarios", type="primary", key="compare_scenarios"):
            comparison_results = []
            scenario_energy_results = {}

            # Add baseline
            comparison_results.append({
                "Scenario": "Current (Baseline)",
                "Electric kWh": annual_kwh,
                "Gas therms": annual_gas_therms,
                "Annual Cost": annual_cost,
                "vs Baseline": "$0",
                "Change %": "0%",
            })

            for scenario_name in selected_scenarios:
                scenario_config = scenario_options[scenario_name]

                # Create modified schedules
                modified_schedules = {k: dict(v) for k, v in st.session_state.schedules.items()}

                # Apply modifications
                target_schedules = scenario_config.get("schedules", list(modified_schedules.keys()))
                for sched_name in modified_schedules:
                    if sched_name in target_schedules or not scenario_config.get("schedules"):
                        modified_schedules[sched_name]["heat_sp"] = max(55, min(75, int(modified_schedules[sched_name]["heat_sp"]) + scenario_config["heat_delta"]))
                        modified_schedules[sched_name]["cool_sp"] = max(70, min(90, int(modified_schedules[sched_name]["cool_sp"]) + scenario_config["cool_delta"]))

                # Compute energy for this scenario (simplified - use cached weather, quick calculation)
                with st.spinner(f"Computing {scenario_name}..."):
                    scenario_cache_key = f"scenario_{scenario_name}_{analysis_year}_{hash(str(modified_schedules))}_{hash(str(st.session_state.schedule_priorities))}"

                    if scenario_cache_key not in st.session_state.annual_energy_cache:
                        _, scenario_summary = compute_annual_runtime_with_historical_weather(
                            schedules=modified_schedules,
                            tin_model=tin_model,
                            runtime_predictor=runtime_predictor,
                            lat=st.session_state.user_lat,
                            lon=st.session_state.user_lon,
                            indoor_seed=float(st.session_state.indoor_temp),
                            building_age=st.session_state.building_age_yrs,
                            building_type=st.session_state.building_type,
                            floor_area=st.session_state.floor_area_sqft,
                            climate_code=st.session_state.climate_code,
                            hvac_mode=st.session_state.hvac_mode,
                            year=analysis_year,
                            schedule_priorities=st.session_state.schedule_priorities,
                        )
                        st.session_state.annual_energy_cache[scenario_cache_key] = scenario_summary
                    else:
                        scenario_summary = st.session_state.annual_energy_cache[scenario_cache_key]

                    if "months" in scenario_summary:
                        s_months = scenario_summary.get("months", {})
                        scenario_rows, scenario_totals = compute_monthly_energy_breakdown(
                            months_data=s_months,
                            heat_kw=heat_kw,
                            cool_kw=cool_kw,
                            aux_kw=aux_kw,
                            fan_kw=fan_kw,
                            cost_per_kwh=cost_per_kwh,
                            gas_btu_per_hr=gas_btu_per_hr,
                            gas_cost_per_therm=gas_rate_per_therm,
                        )

                        scenario_energy_results[scenario_name] = {
                            "rows": scenario_rows,
                            "totals": scenario_totals,
                        }

                        s_kwh = scenario_totals["electric_kwh"]
                        s_gas = scenario_totals["gas_therms"]
                        s_cost = scenario_totals["total_cost"]

                        diff_cost = s_cost - annual_cost
                        diff_pct = (diff_cost / annual_cost * 100) if annual_cost > 0 else 0

                        comparison_results.append({
                            "Scenario": scenario_name.split("(")[0].strip(),
                            "Electric kWh": s_kwh,
                            "Gas therms": s_gas,
                            "Annual Cost": s_cost,
                            "vs Baseline": f"${diff_cost:+,.0f}",
                            "Change %": f"{diff_pct:+.1f}%",
                        })

            # Display comparison table
            if len(comparison_results) > 1:
                st.markdown("**Scenario Comparison Results:**")

                # Create a formatted dataframe
                comparison_df = pd.DataFrame(comparison_results)
                comparison_df["Electric kWh"] = comparison_df["Electric kWh"].apply(lambda x: f"{x:,.0f}")
                if annual_gas_therms > 0:
                    comparison_df["Gas therms"] = comparison_df["Gas therms"].apply(lambda x: f"{x:,.0f}")
                else:
                    comparison_df = comparison_df.drop(columns=["Gas therms"])
                comparison_df["Annual Cost"] = comparison_df["Annual Cost"].apply(lambda x: f"${x:,.0f}")

                st.dataframe(comparison_df, use_container_width=True, hide_index=True)

                # Find best scenario
                best_idx = 1  # Start from first non-baseline
                best_savings = 0
                for i, row in enumerate(comparison_results[1:], start=1):
                    savings = annual_cost - row["Annual Cost"] if isinstance(row["Annual Cost"], (int, float)) else 0
                    if savings > best_savings:
                        best_savings = savings
                        best_idx = i

                if best_savings > 0:
                    best_scenario = comparison_results[best_idx]["Scenario"]
                    st.success(f"💰 **Best option: {best_scenario}** - Could save ${best_savings:,.0f}/year!")
                else:
                    st.info("Your current settings are already energy-efficient for these scenarios.")

                if scenario_energy_results:
                    st.markdown("### 📆 Monthly Baseline vs Scenario")
                    scenario_choice = st.selectbox(
                        "Choose a scenario to compare by month:",
                        options=list(scenario_energy_results.keys()),
                        key="monthly_scenario_choice",
                    )
                    scenario_rows = scenario_energy_results[scenario_choice]["rows"]

                    compare_rows = []
                    for idx, m in enumerate(MONTH_NAMES):
                        base_cost = baseline_rows[idx]["total_cost"]
                        scen_cost = scenario_rows[idx]["total_cost"]
                        compare_rows.append({
                            "Month": m,
                            "Baseline Cost": f"${base_cost:,.0f}",
                            "Scenario Cost": f"${scen_cost:,.0f}",
                            "Δ Cost": f"${(scen_cost - base_cost):+,.0f}",
                        })

                    st.dataframe(pd.DataFrame(compare_rows), use_container_width=True, hide_index=True)

                    fig_compare = go.Figure()
                    fig_compare.add_trace(go.Bar(
                        name="Baseline", x=MONTH_NAMES, y=[r["total_cost"] for r in baseline_rows],
                        marker_color="#9CA3AF",
                        hovertemplate="Baseline: $%{y:.0f}<extra></extra>",
                    ))
                    fig_compare.add_trace(go.Bar(
                        name="Scenario", x=MONTH_NAMES, y=[r["total_cost"] for r in scenario_rows],
                        marker_color="#22C55E",
                        hovertemplate="Scenario: $%{y:.0f}<extra></extra>",
                    ))
                    fig_compare.update_layout(
                        barmode="group",
                        title=f"Monthly Cost Comparison - {scenario_choice.split('(')[0].strip()}",
                        xaxis_title="Month",
                        yaxis_title="Cost ($)",
                        template="plotly_dark",
                        height=360,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )
                    st.plotly_chart(fig_compare, use_container_width=True)

    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Annual energy analysis unavailable")
        st.markdown('<p style="color: white;">We could not compute your annual energy analysis yet. Check model loading or weather data availability.</p>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Store baseline_monthly for compatibility with other sections
    # (compute a simple estimate for sections that still need it)
    if not hasattr(st.session_state, "baseline_runtime") or st.session_state.baseline_runtime is None:
        # Quick 7-day forecast for other sections
        weather = fetch_hourly_forecast(st.session_state.user_lat, st.session_state.user_lon, days=7)
        if weather is not None and tin_model:
            baseline_forecast = predict_tin_7days(
                weather_df=weather,
                schedules=st.session_state.schedules,
                indoor_seed=float(st.session_state.indoor_temp),
                building_age=st.session_state.building_age_yrs,
                building_type=st.session_state.building_type,
                floor_area=st.session_state.floor_area_sqft,
                climate_code=st.session_state.climate_code,
                tin_model=tin_model,
                schedule_priorities=st.session_state.schedule_priorities,
            )
            if runtime_predictor:
                baseline_runtime = estimate_runtime_with_predictor(baseline_forecast, runtime_predictor, hvac_mode=st.session_state.hvac_mode, use_ev=False)
            else:
                baseline_runtime = estimate_runtime_simple(baseline_forecast)
            st.session_state.baseline_forecast = baseline_forecast
            st.session_state.baseline_runtime = baseline_runtime

    # Continue with remaining sections only if we have baseline data
    if hasattr(st.session_state, "baseline_runtime") and st.session_state.baseline_runtime is not None:
        baseline_runtime = st.session_state.baseline_runtime
        baseline_monthly = estimate_monthly_energy(baseline_runtime)
        base_scenarios = [
            "Baseline (current schedules)",
            "Warmer home: +2°F heating/-2°F cooling",
            "Cooler home: -2°F heating/+2°F cooling",
            "Peak Hours Shift (less cooling 4–9pm)",
        ]
        custom = sorted(list(st.session_state.custom_scenarios.keys()))
        scenario_options = base_scenarios + custom

        # Weekly snapshot cards
        st.markdown("---")
        st.markdown("### 📆 Weekly Snapshot")

        baseline_forecast = st.session_state.baseline_forecast

        avg_indoor = float(baseline_forecast["indoor_temp_pred"].mean())
        avg_outdoor = float(baseline_forecast["outdoor_temp"].mean())
        indoor_high = float(baseline_forecast["indoor_temp_pred"].max())
        indoor_low = float(baseline_forecast["indoor_temp_pred"].min())

        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(
            x=baseline_forecast["time"],
            y=baseline_forecast["outdoor_temp"],
            name="Outdoor",
            line=dict(color="#9CA3AF", width=2, dash="dot"),
        ))
        fig_temp.add_trace(go.Scatter(
            x=baseline_forecast["time"],
            y=baseline_forecast["indoor_temp_pred"],
            name="Indoor",
            line=dict(color="#0B8375", width=2, dash="dot"),
        ))
        fig_temp.add_trace(go.Scatter(
            x=baseline_forecast["time"],
            y=baseline_forecast["heat_sp"],
            name="Heat SP",
            line=dict(color="#EF4444", width=2, shape="hv"),
        ))
        fig_temp.add_trace(go.Scatter(
            x=baseline_forecast["time"],
            y=baseline_forecast["cool_sp"],
            name="Cool SP",
            line=dict(color="#60A5FA", width=2, shape="hv"),
        ))
        fig_temp.update_layout(
            template="plotly_dark",
            height=260,
            margin=dict(l=20, r=20, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title="",
            yaxis_title="",
        )

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🌡️ Temperature Summary (7-Day)")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Avg Indoor", f"{avg_indoor:.1f}°F")
        with c2:
            st.metric("Avg Outdoor", f"{avg_outdoor:.1f}°F")
        with c3:
            st.metric("Indoor High/Low", f"{indoor_high:.1f}°F / {indoor_low:.1f}°F")
        st.plotly_chart(fig_temp, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        total_runtime = float(baseline_runtime["runtime_hours"].sum()) if "runtime_hours" in baseline_runtime.columns else 0.0

        # Savings card (scenario vs baseline)
        def week_energy_from_runtime(rt: pd.DataFrame) -> Dict[str, float]:
            heat_hours = float(rt["heat_sec_hours"].sum()) if "heat_sec_hours" in rt.columns else 0.0
            cool_hours = float(rt["cool_sec_hours"].sum()) if "cool_sec_hours" in rt.columns else 0.0
            aux_hours = float(rt["aux_sec_hours"].sum()) if "aux_sec_hours" in rt.columns else 0.0
            fan_hours = float(rt["fan_sec_hours"].sum()) if "fan_sec_hours" in rt.columns else 0.0

            if gas_btu_per_hr > 0:
                heat_therms = heat_hours * gas_btu_per_hr / 100000.0
                heat_kwh = 0.0
            else:
                heat_therms = 0.0
                heat_kwh = heat_hours * heat_kw

            cool_kwh = cool_hours * cool_kw
            aux_kwh = aux_hours * aux_kw
            fan_kwh = fan_hours * fan_kw
            electric_kwh = heat_kwh + cool_kwh + aux_kwh + fan_kwh
            electric_cost = electric_kwh * cost_per_kwh
            gas_cost = heat_therms * gas_rate_per_therm
            total_cost = electric_cost + gas_cost

            if electric_kwh == 0 and heat_therms == 0:
                total_cost = float(rt["runtime_hours"].sum()) * max(heat_kw, cool_kw, 1.0) * cost_per_kwh
                electric_kwh = total_cost / max(cost_per_kwh, 1e-9)

            return {
                "electric_kwh": electric_kwh,
                "gas_therms": heat_therms,
                "total_cost": total_cost,
            }

        week_avg_outdoor = float(baseline_forecast["outdoor_temp"].mean())
        avg_heat_sp = float(np.mean([s["heat_sp"] for s in st.session_state.schedules.values()]))
        avg_cool_sp = float(np.mean([s["cool_sp"] for s in st.session_state.schedules.values()]))
        if week_avg_outdoor <= avg_heat_sp + 2:
            week_mode = "heating"
        elif week_avg_outdoor >= avg_cool_sp - 2:
            week_mode = "cooling"
        else:
            week_mode = "mixed"

        scenario_choice_week = st.selectbox(
            "Scenario for weekly savings:",
            scenario_options,
            index=2 if len(scenario_options) > 2 else 0,
            key="weekly_savings_scenario",
        )

        scenario_schedules_week = build_scenario_schedules(
            st.session_state.schedules,
            scenario_choice_week,
            week_mode,
            st.session_state.custom_scenarios,
        )

        scenario_forecast_week = predict_tin_7days(
            weather_df=baseline_forecast[["time", "outdoor_temp", "outdoor_humidity"]],
            schedules=scenario_schedules_week,
            indoor_seed=float(st.session_state.indoor_temp),
            building_age=st.session_state.building_age_yrs,
            building_type=st.session_state.building_type,
            floor_area=st.session_state.floor_area_sqft,
            climate_code=st.session_state.climate_code,
            tin_model=tin_model,
            schedule_priorities=st.session_state.schedule_priorities,
        )
        if runtime_predictor:
            scenario_runtime_week = estimate_runtime_with_predictor(
                scenario_forecast_week,
                runtime_predictor,
                hvac_mode=st.session_state.hvac_mode,
                use_ev=False,
            )
        else:
            scenario_runtime_week = estimate_runtime_simple(scenario_forecast_week)

        base_energy = week_energy_from_runtime(baseline_runtime)
        scenario_energy = week_energy_from_runtime(scenario_runtime_week)
        base_cost = base_energy["total_cost"]
        scenario_cost = scenario_energy["total_cost"]
        cost_delta = scenario_cost - base_cost
        runtime_delta = float(scenario_runtime_week["runtime_hours"].sum()) - total_runtime

        scenario_total_runtime = float(scenario_runtime_week["runtime_hours"].sum()) if "runtime_hours" in scenario_runtime_week.columns else 0.0
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 💰 Savings vs Scenario (7-Day)")
        st.markdown(f'<p style="color: white; font-size: 13px; margin-bottom: 8px;">Scenario: {scenario_choice_week}</p>', unsafe_allow_html=True)
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            render_mini_metric("Baseline Cost", f"${base_cost:.2f}")
        with s2:
            render_mini_metric("Scenario Cost", f"${scenario_cost:.2f}")
        with s3:
            render_mini_metric("Δ Cost", f"${cost_delta:+.2f}")
        with s4:
            render_mini_metric("Δ Runtime", f"{runtime_delta:+.1f} h")
        st.caption(f"Baseline runtime: {total_runtime:.1f} h | Scenario runtime: {scenario_total_runtime:.1f} h")
        st.markdown("</div>", unsafe_allow_html=True)

        # Extreme Day Analysis (Coldest and Hottest)
        st.markdown("---")
        st.markdown("### 🌡️ Extreme Day Analysis (2025)")
        st.markdown(
            '<p style="color: white; font-size: 14px; margin-bottom: 16px;">'
            "Compare your current schedules to another scenario on the coldest and hottest days of 2025."
            "</p>",
            unsafe_allow_html=True,
        )

        scenario_choice = st.selectbox(
            "Scenario to compare:",
            scenario_options,
            index=1 if len(scenario_options) > 1 else 0,
            key="extreme_day_scenario_choice",
        )
        st.caption(f"Scenario: {scenario_choice}")

        start_date = "2025-01-01"
        end_date = "2025-12-31"

        if not hasattr(st.session_state, "historical_weather_2025") or st.session_state.historical_weather_2025 is None:
            with st.spinner("Loading 2025 historical weather data..."):
                st.session_state.historical_weather_2025 = fetch_historical_weather(
                    st.session_state.user_lat,
                    st.session_state.user_lon,
                    start_date,
                    end_date
                )

        historical_weather = st.session_state.historical_weather_2025

        if historical_weather is not None and not historical_weather.empty:
            daily_temps = historical_weather.groupby(historical_weather["time"].dt.date)["outdoor_temp"].agg(["min", "max", "mean"])
            coldest_day = daily_temps["mean"].idxmin()
            hottest_day = daily_temps["mean"].idxmax()

            def load_day_weather(date_obj: date) -> Tuple[Optional[pd.DataFrame], str]:
                date_str = date_obj.strftime("%Y-%m-%d")
                day_df = load_day_outdoor_temps_from_nearest_location(
                    st.session_state.user_lat,
                    st.session_state.user_lon,
                    date_str,
                )
                if day_df is not None and not day_df.empty:
                    return day_df, "Nearest station"

                day_df = fetch_single_day_weather(
                    st.session_state.user_lat,
                    st.session_state.user_lon,
                    date_str,
                )
                if day_df is not None and not day_df.empty:
                    return day_df, "Open-Meteo (archive)"

                return None, "Unavailable"

            def compute_day_metrics(day_weather: pd.DataFrame, schedules: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
                if day_weather is None or day_weather.empty:
                    return None

                forecast = predict_tin_7days(
                    weather_df=day_weather,
                    schedules=schedules,
                    indoor_seed=float(st.session_state.indoor_temp),
                    building_age=st.session_state.building_age_yrs,
                    building_type=st.session_state.building_type,
                    floor_area=st.session_state.floor_area_sqft,
                    climate_code=st.session_state.climate_code,
                    tin_model=tin_model,
                    schedule_priorities=st.session_state.schedule_priorities,
                )
                avg_indoor = float(forecast["indoor_temp_pred"].mean())
                avg_heat_sp = float(forecast["heat_sp"].mean())
                avg_cool_sp = float(forecast["cool_sp"].mean())

                if runtime_predictor:
                    runtime_df = estimate_runtime_with_predictor(
                        forecast,
                        runtime_predictor,
                        hvac_mode=st.session_state.hvac_mode,
                        use_ev=False,
                    )
                    if runtime_df is None or runtime_df.empty:
                        return None
                    row = runtime_df.iloc[0]
                    heat_hours = float(row.get("heat_sec_hours", 0))
                    cool_hours = float(row.get("cool_sec_hours", 0))
                    aux_hours = float(row.get("aux_sec_hours", 0))
                    fan_hours = float(row.get("fan_sec_hours", 0))
                    fallback_kwh = 0.0
                else:
                    runtime_df = estimate_runtime_simple(forecast)
                    if runtime_df is None or runtime_df.empty:
                        return None
                    row = runtime_df.iloc[0]
                    heat_hours = 0.0
                    cool_hours = 0.0
                    aux_hours = 0.0
                    fan_hours = 0.0
                    runtime_hours = float(row.get("runtime_hours", 0))
                    fallback_kwh = runtime_hours * max(heat_kw, cool_kw, 1.0)

                if gas_btu_per_hr > 0:
                    heat_therms = heat_hours * gas_btu_per_hr / 100000.0
                    heat_kwh = 0.0
                else:
                    heat_therms = 0.0
                    heat_kwh = heat_hours * heat_kw

                cool_kwh = cool_hours * cool_kw
                aux_kwh = aux_hours * aux_kw
                fan_kwh = fan_hours * fan_kw
                electric_kwh = heat_kwh + cool_kwh + aux_kwh + fan_kwh + fallback_kwh
                electric_cost = electric_kwh * cost_per_kwh
                gas_cost = heat_therms * gas_rate_per_therm
                total_cost = electric_cost + gas_cost

                return {
                    "avg_temp": float(day_weather["outdoor_temp"].mean()),
                    "avg_indoor": avg_indoor,
                    "avg_heat_sp": avg_heat_sp,
                    "avg_cool_sp": avg_cool_sp,
                    "electric_kwh": electric_kwh,
                    "gas_therms": heat_therms,
                    "total_cost": total_cost,
                }

            coldest_weather, coldest_source = load_day_weather(coldest_day)
            hottest_weather, hottest_source = load_day_weather(hottest_day)

            cold_mode = "heating"
            hot_mode = "cooling"
            cold_schedules = build_scenario_schedules(
                st.session_state.schedules,
                scenario_choice,
                cold_mode,
                st.session_state.custom_scenarios,
            )
            hot_schedules = build_scenario_schedules(
                st.session_state.schedules,
                scenario_choice,
                hot_mode,
                st.session_state.custom_scenarios,
            )

            cold_base = compute_day_metrics(coldest_weather, st.session_state.schedules)
            cold_scenario = compute_day_metrics(coldest_weather, cold_schedules)
            hot_base = compute_day_metrics(hottest_weather, st.session_state.schedules)
            hot_scenario = compute_day_metrics(hottest_weather, hot_schedules)

            col_cold, col_hot = st.columns(2)
            with col_cold:
                st.markdown('<div class="card card-cold">', unsafe_allow_html=True)
                st.markdown("#### ❄️ Coldest Day")
                st.markdown(f'<p style="color: white; font-size: 14px; margin-bottom: 12px;">{coldest_day.strftime("%A, %B %d")}</p>', unsafe_allow_html=True)
                if cold_base:
                    st.metric("Avg Outdoor Temp", f"{cold_base['avg_temp']:.1f}°F")
                    st.metric("Baseline Indoor Temp", f"{cold_base['avg_indoor']:.1f}°F")
                    if cold_scenario:
                        st.metric("Scenario Indoor Temp", f"{cold_scenario['avg_indoor']:.1f}°F")
                    st.metric("Baseline Heat Setpoint", f"{int(round(cold_base['avg_heat_sp']))}°F")
                    if cold_scenario:
                        st.metric("Scenario Heat Setpoint", f"{int(round(cold_scenario['avg_heat_sp']))}°F")
                    st.metric("Baseline Cost", f"${cold_base['total_cost']:.2f}")
                    st.metric("Scenario Cost", f"${cold_scenario['total_cost']:.2f}" if cold_scenario else "—")
                    if cold_scenario:
                        delta = cold_scenario["total_cost"] - cold_base["total_cost"]
                        st.metric("Δ Cost", f"${delta:+.2f}")
                st.caption(f"Hourly temps source: {coldest_source}")
                st.markdown("</div>", unsafe_allow_html=True)

            with col_hot:
                st.markdown('<div class="card card-hot">', unsafe_allow_html=True)
                st.markdown("#### 🔥 Hottest Day")
                st.markdown(f'<p style="color: white; font-size: 14px; margin-bottom: 12px;">{hottest_day.strftime("%A, %B %d")}</p>', unsafe_allow_html=True)
                if hot_base:
                    st.metric("Avg Outdoor Temp", f"{hot_base['avg_temp']:.1f}°F")
                    st.metric("Baseline Indoor Temp", f"{hot_base['avg_indoor']:.1f}°F")
                    if hot_scenario:
                        st.metric("Scenario Indoor Temp", f"{hot_scenario['avg_indoor']:.1f}°F")
                    st.metric("Baseline Cool Setpoint", f"{int(round(hot_base['avg_cool_sp']))}°F")
                    if hot_scenario:
                        st.metric("Scenario Cool Setpoint", f"{int(round(hot_scenario['avg_cool_sp']))}°F")
                    st.metric("Baseline Cost", f"${hot_base['total_cost']:.2f}")
                    st.metric("Scenario Cost", f"${hot_scenario['total_cost']:.2f}" if hot_scenario else "—")
                    if hot_scenario:
                        delta = hot_scenario["total_cost"] - hot_base["total_cost"]
                        st.metric("Δ Cost", f"${delta:+.2f}")
                st.caption(f"Hourly temps source: {hottest_source}")
                st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.info("💡 Unable to load historical weather data. Extreme day analysis unavailable.")

    else:
        st.info("💡 Visit the **Forecast** tab to generate energy projections based on your schedules and local weather.")

    st.markdown("</div>", unsafe_allow_html=True)
    bottom_nav()
    assistant_bar()


# =========================================================
# ACKNOWLEDGEMENT
# =========================================================
def show_acknowledgement():
    """Display Ecobee acknowledgement on all pages."""
    st.markdown(
        '<div style="text-align: center; padding: 10px; margin-top: 20px; font-size: 0.85em; color: rgba(255, 255, 255, 0.6);">'
        'We gratefully acknowledge the support and contribution of ecobee and ecobee customers to this research.'
        '</div>',
        unsafe_allow_html=True
    )

# =========================================================
# MAIN ROUTER
# =========================================================
def main():
    if not st.session_state.setup_complete or st.session_state.view == "Setup":
        st.session_state.view = "Setup"
        render_setup()
        show_acknowledgement()
        return

    if st.session_state.view == "Home":
        render_home()
    elif st.session_state.view == "Schedules":
        render_schedules()
    elif st.session_state.view == "Forecast":
        render_forecast()
    elif st.session_state.view == "Reports":
        render_reports()
    else:
        st.session_state.view = "Home"
        st.rerun()

    show_acknowledgement()
