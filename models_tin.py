from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from app_config import R2C2_PROFILES_PATH, TIN_FEATURES, TIN_MODEL_PATH
from schedule_utils import _active_schedule_for_hour

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    from joblib import load as joblib_load
    JOBLIB_AVAILABLE = True
except Exception:
    JOBLIB_AVAILABLE = False


class TinModelWrapper:
    def __init__(self, model: Any, kind: str):
        self.model = model
        self.kind = kind

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.kind == "sklearn":
            return np.asarray(self.model.predict(X), dtype=float)
        if self.kind == "xgb_booster":
            dmat = xgb.DMatrix(X.values, feature_names=list(X.columns))
            return np.asarray(self.model.predict(dmat), dtype=float)
        raise ValueError(f"Unknown model kind: {self.kind}")


@st.cache_resource(show_spinner=False)
def load_tin_model() -> Optional[TinModelWrapper]:
    if not TIN_MODEL_PATH.exists():
        return None

    # joblib/pickle
    if JOBLIB_AVAILABLE and TIN_MODEL_PATH.suffix.lower() in [".pkl", ".joblib"]:
        try:
            m = joblib_load(TIN_MODEL_PATH)
            if hasattr(m, "predict"):
                return TinModelWrapper(m, "sklearn")
        except Exception:
            pass

    # xgb booster
    if XGB_AVAILABLE and TIN_MODEL_PATH.suffix.lower() in [".json", ".ubj", ".bin"]:
        try:
            booster = xgb.Booster()
            booster.load_model(str(TIN_MODEL_PATH))
            return TinModelWrapper(booster, "xgb_booster")
        except Exception:
            pass

    return None


@st.cache_data(show_spinner=False)
def load_2r2c_profiles() -> Optional[pd.DataFrame]:
    """
    Load 2R2C thermal profiles if available.
    """
    if not R2C2_PROFILES_PATH.exists():
        return None
    try:
        if R2C2_PROFILES_PATH.suffix == ".parquet":
            df = pd.read_parquet(R2C2_PROFILES_PATH)
        else:
            df = pd.read_csv(R2C2_PROFILES_PATH)
        return df
    except Exception:
        return None


def get_2r2c_params_for_home(identifier: str) -> Optional[Dict[str, Any]]:
    """
    Look up 2R2C thermal parameters for a specific home identifier.
    """
    profiles = load_2r2c_profiles()
    if profiles is None or profiles.empty:
        return None

    if "identifier" not in profiles.columns:
        return None

    match = profiles[profiles["identifier"] == identifier]
    if match.empty:
        return None

    row = match.iloc[0]
    result = {
        "R_ao": row.get("R_ao"),
        "R_am": row.get("R_am"),
        "C_a": row.get("C_a"),
        "C_m": row.get("C_m"),
        "tau_env": row.get("tau_env"),
        "tau_mass": row.get("tau_mass"),
        "confidence": row.get("confidence", "unknown"),
        "rmse_openloop": row.get("rmse_openloop"),
        "use_profile_bool": row.get("use_profile_bool", False),
    }

    if result.get("use_profile_bool") is False:
        return None

    return result


def build_tin_features(
    outdoor_temp: float,
    outdoor_humidity: float,
    heat_sp: float,
    cool_sp: float,
    dt: datetime,
    indoor_temp_lag: float,
    outdoor_temp_lag: float,
    building_age: Optional[float] = None,
    building_type: Optional[str] = None,
    floor_area: Optional[float] = None,
    climate_code: str = "",
) -> pd.DataFrame:
    deadband = cool_sp - heat_sp
    temp_diff = outdoor_temp - indoor_temp_lag
    temp_diff_lag = outdoor_temp_lag - indoor_temp_lag

    hour_sin, hour_cos = np.sin(2 * np.pi * dt.hour / 24), np.cos(2 * np.pi * dt.hour / 24)
    month_sin, month_cos = np.sin(2 * np.pi * dt.month / 12), np.cos(2 * np.pi * dt.month / 12)
    is_weekend = 1 if dt.weekday() >= 5 else 0

    hvac_heat, hvac_cool, hvac_auto, hvac_off = 0, 0, 1, 0

    heat_eff_off = 1 if heat_sp <= 50 else 0
    cool_eff_off = 1 if cool_sp >= 90 else 0
    hvac_idle = 1 if (heat_eff_off and cool_eff_off) else 0

    age = float(building_age) if building_age is not None and np.isfinite(building_age) else 0.0
    area = float(floor_area) if floor_area is not None and np.isfinite(floor_area) else 0.0

    climate_keys = ["3B", "4B", "3C", "2B", "5B", "4C", "6B"]
    climate = {f"climate_{k}": (1 if (climate_code or "").upper() == k else 0) for k in climate_keys}

    bldg_keys = ["Townhouse", "Apartment", "Detached", "Other", "Row House", "Semi-Detached",
                 "Condominium", "Multi-plex", "Loft"]
    bldg = {f"bldg_{k}": (1 if building_type == k else 0) for k in bldg_keys}

    feat = {
        "outdoor_temp": outdoor_temp,
        "outdoor_temp_lag1": outdoor_temp_lag,
        "outdoor_humidity": outdoor_humidity,
        "heat_sp": heat_sp,
        "cool_sp": cool_sp,
        "deadband": deadband,
        "temp_diff": temp_diff,
        "temp_diff_lag1": temp_diff_lag,
        "indoor_temp_lag1": indoor_temp_lag,
        "hvac_heat": hvac_heat,
        "hvac_cool": hvac_cool,
        "hvac_auto": hvac_auto,
        "hvac_off": hvac_off,
        "heat_effectively_off": heat_eff_off,
        "cool_effectively_off": cool_eff_off,
        "hvac_effectively_idle": hvac_idle,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "is_weekend": is_weekend,
        "building_age_yrs": age,
        "floor_area_sqft": area,
        **climate,
        **bldg,
    }
    return pd.DataFrame([{k: feat.get(k, 0.0) for k in TIN_FEATURES}])


def predict_tin_7days(
    weather_df: pd.DataFrame,
    schedules: Dict[str, Dict[str, Any]],
    indoor_seed: float,
    building_age: Optional[float],
    building_type: Optional[str],
    floor_area: Optional[float],
    climate_code: str,
    tin_model: TinModelWrapper,
    schedule_priorities: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    results = []
    tin_lag = float(indoor_seed)
    tout_lag = float(weather_df.iloc[0]["outdoor_temp"])

    for _, row in weather_df.iterrows():
        dt = pd.to_datetime(row["time"])
        tout = float(row["outdoor_temp"])
        rh = float(row["outdoor_humidity"]) if pd.notna(row["outdoor_humidity"]) else 50.0

        active = _active_schedule_for_hour(dt, schedules, schedule_priorities)
        heat_sp = float(schedules[active]["heat_sp"])
        cool_sp = float(schedules[active]["cool_sp"])

        X = build_tin_features(
            outdoor_temp=tout,
            outdoor_humidity=rh,
            heat_sp=heat_sp,
            cool_sp=cool_sp,
            dt=dt.to_pydatetime(),
            indoor_temp_lag=tin_lag,
            outdoor_temp_lag=tout_lag,
            building_age=building_age,
            building_type=building_type,
            floor_area=floor_area,
            climate_code=climate_code,
        )

        tin_pred = float(tin_model.predict(X)[0])
        results.append({
            "time": dt,
            "outdoor_temp": tout,
            "outdoor_humidity": rh,
            "indoor_temp_pred": tin_pred,
            "heat_sp": heat_sp,
            "cool_sp": cool_sp,
            "schedule": active,
        })
        tin_lag = tin_pred
        tout_lag = tout

    return pd.DataFrame(results)


def predict_tin_extended(
    weather_df: pd.DataFrame,
    schedules: Dict[str, Dict],
    indoor_seed: float,
    building_age: Optional[int],
    building_type: Optional[str],
    floor_area: Optional[float],
    climate_code: str,
    tin_model: Any,
    schedule_priorities: Optional[Dict[str, int]] = None,
) -> Optional[pd.DataFrame]:
    """
    Extended version of predict_tin_7days that handles arbitrary length weather data.
    """
    if weather_df is None or weather_df.empty or tin_model is None:
        return None

    df = weather_df.copy()

    if "time" not in df.columns:
        return None

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    df["hour"] = df["time"].dt.hour
    df["month"] = df["time"].dt.month
    df["is_weekend"] = df["time"].dt.dayofweek.isin([5, 6]).astype(int)
    df["date"] = df["time"].dt.date

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    def get_schedule_for_time(ts: pd.Timestamp) -> Tuple[str, int, int]:
        name = _active_schedule_for_hour(ts, schedules, schedule_priorities)
        sched = schedules[name]
        return name, int(sched["heat_sp"]), int(sched["cool_sp"])

    schedule_info = df["time"].apply(get_schedule_for_time)
    df["schedule"] = [s[0] for s in schedule_info]
    df["heat_sp"] = [s[1] for s in schedule_info]
    df["cool_sp"] = [s[2] for s in schedule_info]
    df["deadband"] = df["cool_sp"] - df["heat_sp"]

    if "outdoor_humidity" not in df.columns or df["outdoor_humidity"].isna().all():
        df["outdoor_humidity"] = 50.0
    df["outdoor_humidity"] = df["outdoor_humidity"].fillna(50.0)

    df["building_age_yrs"] = building_age if building_age else 25
    df["floor_area_sqft"] = floor_area if floor_area else 1800

    climate_cols = ["climate_3B", "climate_4B", "climate_3C", "climate_2B",
                    "climate_5B", "climate_4C", "climate_6B"]
    for c in climate_cols:
        df[c] = 1 if climate_code and c.split("_")[1] == climate_code else 0

    bldg_cols = ["bldg_Townhouse", "bldg_Apartment", "bldg_Detached", "bldg_Other",
                 "bldg_Row House", "bldg_Semi-Detached", "bldg_Condominium",
                 "bldg_Multi-plex", "bldg_Loft"]
    for c in bldg_cols:
        df[c] = 1 if building_type and c.split("_", 1)[1] == building_type else 0

    df["hvac_heat"] = 0
    df["hvac_cool"] = 0
    df["hvac_auto"] = 1
    df["hvac_off"] = 0

    df["heat_effectively_off"] = (df["outdoor_temp"] > df["heat_sp"] + 5).astype(int)
    df["cool_effectively_off"] = (df["outdoor_temp"] < df["cool_sp"] - 5).astype(int)
    df["hvac_effectively_idle"] = (df["heat_effectively_off"] & df["cool_effectively_off"]).astype(int)

    df["indoor_temp_pred"] = indoor_seed

    df["outdoor_temp_lag1"] = df["outdoor_temp"].shift(1).fillna(df["outdoor_temp"].iloc[0])
    df["indoor_temp_lag1"] = indoor_seed
    df["temp_diff"] = df["outdoor_temp"] - indoor_seed
    df["temp_diff_lag1"] = df["temp_diff"].shift(1).fillna(0)

    predictions = []
    current_indoor = indoor_seed

    for idx in range(len(df)):
        row = df.iloc[idx:idx + 1].copy()
        row["indoor_temp_lag1"] = current_indoor
        row["temp_diff"] = row["outdoor_temp"].values[0] - current_indoor
        row["temp_diff_lag1"] = df.iloc[max(0, idx - 1)]["temp_diff"] if idx > 0 else 0

        features = row[TIN_FEATURES].values

        try:
            if hasattr(tin_model, "predict"):
                pred = float(tin_model.predict(features)[0])
            else:
                pred = current_indoor
        except Exception:
            pred = current_indoor

        pred = max(55, min(95, pred))
        predictions.append(pred)
        current_indoor = pred

    df["indoor_temp_pred"] = predictions

    return df
