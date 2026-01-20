from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from app_config import (
    FP_HOME_ALL_PATH,
    ID_TO_LOC_PATH,
    IDX_ALL_PATH,
    LOCATIONS_PATH,
    META_PATH,
    PEAKS_ALL_PATH,
)


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    R = 6371.0
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


@st.cache_data(show_spinner=False)
def load_neighbor_data() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    # core checks
    if not (LOCATIONS_PATH.exists() and ID_TO_LOC_PATH.exists() and IDX_ALL_PATH.exists()):
        return None, None, None, None

    # 1A locations
    locations = pd.read_csv(LOCATIONS_PATH)
    loc_cols = locations.columns.tolist()

    loc_id_col = next((c for c in loc_cols if c.lower() in ["location_id", "loc_id", "id"]), None)
    lat_col = next((c for c in loc_cols if c.lower() in ["lat", "latitude"]), None)
    lon_col = next((c for c in loc_cols if c.lower() in ["lon", "longitude", "long"]), None)
    if not (loc_id_col and lat_col and lon_col):
        return None, None, None, None

    locations = locations.rename(columns={loc_id_col: "location_id", lat_col: "lat", lon_col: "lon"})
    locations["location_id"] = pd.to_numeric(locations["location_id"], errors="coerce").astype("Int64")
    locations["lat"] = pd.to_numeric(locations["lat"], errors="coerce")
    locations["lon"] = pd.to_numeric(locations["lon"], errors="coerce")
    locations = locations.dropna(subset=["location_id", "lat", "lon"])

    # 1B id->loc
    id_to_loc = pd.read_csv(ID_TO_LOC_PATH)
    id_to_loc["identifier"] = id_to_loc["identifier"].astype(str)
    map_cols = id_to_loc.columns.tolist()
    map_loc_id_col = next((c for c in map_cols if c.lower() in ["location_id", "loc_id", "id"]), None)
    if not map_loc_id_col:
        return None, None, None, None
    id_to_loc = id_to_loc.rename(columns={map_loc_id_col: "location_id"})
    id_to_loc["location_id"] = pd.to_numeric(id_to_loc["location_id"], errors="coerce").astype("Int64")
    id_to_loc = id_to_loc.dropna(subset=["location_id"])

    # merge
    meta = id_to_loc.merge(locations, on="location_id", how="inner")
    if meta.empty:
        return None, None, None, None
    meta = meta[["identifier", "location_id", "lat", "lon"]].drop_duplicates("identifier")

    # indices
    idx = pd.read_parquet(IDX_ALL_PATH)
    idx["identifier"] = idx["identifier"].astype(str)

    # optional building meta
    if META_PATH.exists():
        try:
            extra = pd.read_csv(META_PATH)
            extra["identifier"] = extra["identifier"].astype(str)
            building_cols = ["identifier"]
            for col in ["building_type_pre", "floor_area_sqft", "building_age_yrs", "ASHRAE"]:
                if col in extra.columns:
                    building_cols.append(col)
            if len(building_cols) > 1:
                meta = meta.merge(extra[building_cols], on="identifier", how="left")
        except Exception:
            pass

    peaks = None
    if PEAKS_ALL_PATH.exists():
        try:
            peaks = pd.read_parquet(PEAKS_ALL_PATH)
            peaks["identifier"] = peaks["identifier"].astype(str)
        except Exception:
            peaks = None

    fp = None
    if FP_HOME_ALL_PATH.exists():
        try:
            fp = pd.read_parquet(FP_HOME_ALL_PATH)
            fp["identifier"] = fp["identifier"].astype(str)
        except Exception:
            fp = None

    return meta, idx, peaks, fp


def find_neighbors(lat: float, lon: float, top_n: int = 250, min_homes: int = 120) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        return pd.DataFrame(), {"match_score": 0, "confidence": "Low", "total_homes": 0}

    meta, idx, peaks, fp = load_neighbor_data()
    if meta is None or idx is None:
        return pd.DataFrame(), {"match_score": 0, "confidence": "Low", "total_homes": 0}

    meta = meta.copy()
    meta["dist_km"] = haversine_km(lat, lon, meta["lat"].values, meta["lon"].values)
    if float(meta["dist_km"].min()) > 300.0:
        return pd.DataFrame(), {"match_score": 0, "confidence": "Low", "total_homes": 0}

    # expanding radius
    candidates = meta
    for radius in range(25, 301, 25):
        c = meta[meta["dist_km"] <= radius]
        if len(c) >= min_homes:
            candidates = c
            break

    df = candidates.merge(idx, on="identifier", how="inner")
    if df.empty:
        return df, {"match_score": 0, "confidence": "Low", "total_homes": 0}

    if peaks is not None:
        df = df.merge(peaks, on="identifier", how="left")
    if fp is not None:
        df = df.merge(fp, on="identifier", how="left")

    # similarity score: geo + indices
    geo_score = 1.0 - np.clip(df["dist_km"] / 200.0, 0, 1)

    if "location_id" in df.columns:
        closest = df.nsmallest(1, "dist_km").iloc[0]
        user_location_id = int(closest["location_id"])
        same_loc = (df["location_id"] == user_location_id).values
        geo_score = np.where(same_loc, 1.0, geo_score)

    idx_cols = ["TSI_med", "II_med", "SDI_med", "SE_top3_med", "schedule_transition_rate_med", "n_schedules_med"]
    available = [c for c in idx_cols if c in df.columns]

    if available:
        X = df[available].apply(pd.to_numeric, errors="coerce")
        z = (X - X.mean()) / X.std().replace(0, np.nan)
        idx_score = 1.0 - np.clip(np.sqrt((z ** 2).mean(axis=1)) / 3.0, 0, 1)
        idx_score = np.where(np.isnan(idx_score), 0.35, idx_score)
        df["sim_score"] = 0.35 * geo_score + 0.65 * idx_score
    else:
        df["sim_score"] = geo_score

    df = df.sort_values("sim_score", ascending=False).head(top_n).reset_index(drop=True)

    weights = df["sim_score"].to_numpy()
    weights = weights / max(weights.sum(), 1e-9)
    n_eff = float((weights.sum() ** 2) / np.sum(weights ** 2))
    avg_dist = float(np.sum(df["dist_km"] * weights))
    confidence_score = 0.5 * np.clip(n_eff / 150, 0, 1) + 0.5 * (1 - np.clip(avg_dist / 200, 0, 1))

    if confidence_score >= 0.70:
        confidence = "High"
    elif confidence_score >= 0.45:
        confidence = "Medium"
    else:
        confidence = "Low"

    stats = {
        "n_eff": n_eff,
        "avg_dist_km": avg_dist,
        "confidence": confidence,
        "confidence_score": confidence_score,
        "total_homes": int(len(df)),
        "match_score": int(confidence_score * 100),
    }
    return df, stats


def extract_neighbor_priors(neighbors_df: pd.DataFrame) -> Dict[str, float]:
    if neighbors_df.empty or "sim_score" not in neighbors_df.columns:
        return {}
    w = neighbors_df["sim_score"].to_numpy()
    w = w / max(w.sum(), 1e-9)

    def weighted_median(col: str) -> float:
        if col not in neighbors_df.columns:
            return np.nan
        x = pd.to_numeric(neighbors_df[col], errors="coerce").to_numpy()
        mask = np.isfinite(x) & np.isfinite(w)
        if not mask.any():
            return np.nan
        xx, ww = x[mask], w[mask]
        idx = np.argsort(xx)
        cdf = np.cumsum(ww[idx]) / max(ww.sum(), 1e-9)
        return float(np.interp(0.5, cdf, xx[idx]))

    return {
        "TSI_med": weighted_median("TSI_med"),
        "SDI_med": weighted_median("SDI_med"),
        "II_med": weighted_median("II_med"),
        "n_schedules_med": weighted_median("n_schedules_med"),
        "schedule_transition_rate_med": weighted_median("schedule_transition_rate_med"),
    }


@st.cache_data(show_spinner=False)
def infer_location_id_from_latlon(lat: float, lon: float) -> Optional[int]:
    if not LOCATIONS_PATH.exists():
        return None
    loc = pd.read_csv(LOCATIONS_PATH)
    loc = loc.rename(columns={c: c.lower() for c in loc.columns})
    if not {"location_id", "lat", "lon"}.issubset(set(loc.columns)):
        # try flexible naming
        cols = list(loc.columns)
        idcol = next((c for c in cols if c.lower() in ["location_id", "loc_id", "id"]), None)
        latcol = next((c for c in cols if c.lower() in ["lat", "latitude"]), None)
        loncol = next((c for c in cols if c.lower() in ["lon", "longitude", "long"]), None)
        if not (idcol and latcol and loncol):
            return None
        loc = loc.rename(columns={idcol: "location_id", latcol: "lat", loncol: "lon"})
    loc = loc.dropna(subset=["location_id", "lat", "lon"]).copy()
    loc["location_id"] = pd.to_numeric(loc["location_id"], errors="coerce")
    loc["lat"] = pd.to_numeric(loc["lat"], errors="coerce")
    loc["lon"] = pd.to_numeric(loc["lon"], errors="coerce")
    loc = loc.dropna(subset=["location_id", "lat", "lon"])
    if loc.empty:
        return None
    d = haversine_km(lat, lon, loc["lat"].to_numpy(), loc["lon"].to_numpy())
    i = int(np.argmin(d))
    if float(d[i]) > 300.0:
        return None
    return int(loc.iloc[i]["location_id"])


def ensure_neighbors_loaded() -> None:
    if st.session_state.neighbors_ready:
        return

    meta, idx, _, _ = load_neighbor_data()
    if meta is None or idx is None:
        st.session_state.neighbors_ready = True
        st.session_state.similar_homes_count = 0
        st.session_state.match_score = 0
        st.session_state.neighbor_priors = {}
        return

    with st.spinner("Personalizing with nearby patterns..."):
        neighbors_df, stats = find_neighbors(st.session_state.user_lat, st.session_state.user_lon)
        st.session_state.neighbors_df = neighbors_df
        st.session_state.match_score = stats.get("match_score", 0)
        st.session_state.match_confidence = stats.get("confidence", "Unknown")
        st.session_state.similar_homes_count = stats.get("total_homes", 0)
        if not neighbors_df.empty:
            st.session_state.neighbor_priors = extract_neighbor_priors(neighbors_df)

    st.session_state.neighbors_ready = True
