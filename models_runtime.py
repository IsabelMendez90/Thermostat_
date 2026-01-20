from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from app_config import RUNTIME_MODEL_DIR

try:
    from runtime_predictor import EndUseRuntimePredictor
    RUNTIME_PREDICTOR_AVAILABLE = True
except Exception:
    RUNTIME_PREDICTOR_AVAILABLE = False


@st.cache_resource(show_spinner=False)
def load_runtime_predictor() -> Optional[EndUseRuntimePredictor]:
    """Load the XGBoost runtime predictor model."""
    if not RUNTIME_PREDICTOR_AVAILABLE:
        return None
    if not RUNTIME_MODEL_DIR.exists():
        return None
    try:
        return EndUseRuntimePredictor(str(RUNTIME_MODEL_DIR))
    except Exception:
        return None


def estimate_runtime_simple(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fallback simple heuristic when XGBoost predictor is not available.
    """
    df = forecast_df.copy()
    df["heat_load"] = np.maximum(0, df["heat_sp"] - df["indoor_temp_pred"])
    df["cool_load"] = np.maximum(0, df["indoor_temp_pred"] - df["cool_sp"])
    df["runtime_hours"] = (df["heat_load"] + df["cool_load"]) / 10.0
    df["runtime_hours"] = np.clip(df["runtime_hours"], 0, 1)
    df["date"] = df["time"].dt.date
    daily = df.groupby("date", as_index=False).agg(
        runtime_hours=("runtime_hours", "sum"),
        heat_load=("heat_load", "mean"),
        cool_load=("cool_load", "mean"),
    )
    return daily


def estimate_runtime_with_predictor(
    forecast_df: pd.DataFrame,
    runtime_predictor: EndUseRuntimePredictor,
    hvac_mode: str = "auto",
    use_ev: bool = False,
) -> pd.DataFrame:
    """
    Use trained XGBoost runtime predictor to estimate runtime from forecast.
    """
    df = forecast_df.copy()

    df["indoor_temp"] = df["indoor_temp_pred"]
    df["month"] = pd.to_datetime(df["time"]).dt.month
    df["hour"] = pd.to_datetime(df["time"]).dt.hour
    df["day_of_week"] = (pd.to_datetime(df["time"]).dt.dayofweek + 1) % 7 + 1  # Sun=1, Sat=7
    df["hvac_mode"] = hvac_mode
    df["schedule_name"] = df["schedule"]

    predictions = runtime_predictor.predict(df, use_ev=use_ev)
    predictions["date"] = pd.to_datetime(predictions["time"]).dt.date

    agg_dict = {}

    if "runtime_sec_pred_sec" in predictions.columns:
        agg_dict["runtime_hours"] = ("runtime_sec_pred_sec", lambda x: x.sum() / 3600.0)

    for target in runtime_predictor.targets:
        col = f"{target}_pred_sec"
        if col in predictions.columns:
            agg_dict[f"{target}_hours"] = (col, lambda x: x.sum() / 3600.0)

    if "heat_sp" in predictions.columns and "indoor_temp" in predictions.columns:
        predictions["heat_load"] = np.maximum(0, predictions["heat_sp"] - predictions["indoor_temp"])
        predictions["cool_load"] = np.maximum(0, predictions["indoor_temp"] - predictions["cool_sp"])
        agg_dict["heat_load"] = ("heat_load", "mean")
        agg_dict["cool_load"] = ("cool_load", "mean")

    daily = predictions.groupby("date", as_index=False).agg(**agg_dict)
    return daily


def estimate_energy_from_runtime(runtime_hours: float, system_type: str = "central_ac") -> float:
    """
    Convert runtime hours to energy consumption (kWh).
    """
    if system_type == "heat_pump":
        power_kw = 3.0
    elif system_type == "gas_furnace":
        power_kw = 0.6
    else:
        power_kw = 3.5
    return runtime_hours * power_kw
