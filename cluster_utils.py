from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from app_config import CLUSTER_PROFILES_PATH
from schedule_utils import _active_schedule_for_hour
from text_utils import norm_text


def load_cluster_profiles() -> Optional[pd.DataFrame]:
    """Load cluster profiles from clustering analysis."""
    if not CLUSTER_PROFILES_PATH.exists():
        return None
    try:
        return pd.read_parquet(CLUSTER_PROFILES_PATH)
    except Exception:
        return None


def _schedule_category(name: str) -> Optional[str]:
    """
    Map a user-facing schedule name to a coarse category for clustering alignment.
    """
    tokens = set(norm_text(name).split())
    if not tokens:
        return None

    if {"sleep", "bed", "night"}.intersection(tokens):
        return "sleep"
    if {"away", "vacation", "travel", "work", "out", "outside"}.intersection(tokens):
        return "away"
    if {"home", "awake", "morning", "evening", "day", "daytime"}.intersection(tokens):
        return "awake"
    return None


def _compute_schedule_diversity(schedules: Dict[str, Any]) -> float:
    """
    Approximate schedule diversity index (SDI) using 24-hour schedule coverage.
    """
    if not schedules:
        return 0.0

    hour_counts: Dict[str, int] = {}
    for hour in range(24):
        dt = pd.Timestamp(year=2025, month=1, day=1, hour=hour)
        active = _active_schedule_for_hour(dt, schedules)
        hour_counts[active] = hour_counts.get(active, 0) + 1

    n_sched = len(hour_counts)
    if n_sched <= 1:
        return 0.0

    total = sum(hour_counts.values())
    probs = [c / total for c in hour_counts.values() if c > 0]
    sdi = -sum(p * math.log(p + 1e-9) for p in probs)
    return float(sdi / math.log(n_sched + 1))


def _compute_user_timing(schedules: Dict[str, Any]) -> Dict[str, Optional[int]]:
    """
    Derive awake/away/sleep timing from schedule names and start/end hours.
    """
    buckets = {"awake": {"start": [], "end": []}, "away": {"start": [], "end": []}, "sleep": {"start": [], "end": []}}
    for name, sched in schedules.items():
        category = _schedule_category(name)
        if not category:
            continue
        start = int(sched["start_hour"]) % 24
        end = int(sched["end_hour"]) % 24
        buckets[category]["start"].append(start)
        buckets[category]["end"].append(end)

    timing: Dict[str, Optional[int]] = {}
    for cat in ["awake", "away", "sleep"]:
        starts = buckets[cat]["start"]
        ends = buckets[cat]["end"]
        timing[f"{cat}_start"] = int(np.median(starts)) if starts else None
        timing[f"{cat}_end"] = int(np.median(ends)) if ends else None
    return timing


def find_user_cluster(location_id: int, schedules: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Match user's location and schedules to a cluster profile.
    Returns cluster info with typical setpoints and timing.
    """
    profiles = load_cluster_profiles()
    if profiles is None or profiles.empty:
        return None

    # Use most recent year
    latest_year = profiles["year"].max()
    profiles = profiles[profiles["year"] == latest_year].copy()

    # Match by location
    location_profiles = profiles[profiles["location_id_mode"] == location_id]
    if location_profiles.empty:
        # Fallback: use any cluster (maybe average)
        location_profiles = profiles

    # Compute user's schedule diversity, count, timing, and setpoints
    n_schedules = len(schedules)
    user_sdi = _compute_schedule_diversity(schedules)
    user_timing = _compute_user_timing(schedules)
    heat_vals = [s["heat_sp"] for s in schedules.values()]
    cool_vals = [s["cool_sp"] for s in schedules.values()]
    user_heat_median = int(np.median(heat_vals))
    user_cool_median = int(np.median(cool_vals))

    # Match by weighted distance across setpoints, diversity, schedule count, and timing
    best_cluster = None
    best_distance = float("inf")

    for _, row in location_profiles.iterrows():
        cluster_heat = row.get("heat_p50_p50", user_heat_median)
        cluster_cool = row.get("cool_p50_p50", user_cool_median)
        cluster_sdi = row.get("SDI_p50", user_sdi)
        cluster_n = row.get("n_schedules_p50", n_schedules)

        distance = 0.0
        distance += abs(cluster_heat - user_heat_median) * 1.0
        distance += abs(cluster_cool - user_cool_median) * 1.0
        distance += abs(cluster_sdi - user_sdi) * 8.0
        distance += abs(cluster_n - n_schedules) * 2.0

        for key, weight in [
            ("awake_start", 0.5),
            ("awake_end", 0.5),
            ("away_start", 0.5),
            ("away_end", 0.5),
            ("sleep_start", 0.5),
            ("sleep_end", 0.5),
        ]:
            user_val = user_timing.get(key)
            if user_val is None:
                continue
            cluster_val = row.get(f"{key}_p50", user_val)
            distance += abs(cluster_val - user_val) * weight

        if distance < best_distance:
            best_distance = distance
            best_cluster = row

    if best_cluster is None:
        return None

    # Extract cluster info
    return {
        "cluster_id": int(best_cluster["cluster_id"]),
        "n_homes": int(best_cluster["n_homes"]),
        "match_distance": float(best_distance),
        "user_sdi": float(user_sdi),
        "user_n_schedules": int(n_schedules),
        "heat_p50": int(best_cluster.get("heat_p50_p50", user_heat_median)),
        "cool_p50": int(best_cluster.get("cool_p50_p50", user_cool_median)),
        "awake_start": int(best_cluster.get("awake_start_p50", 6)),
        "awake_end": int(best_cluster.get("awake_end_p50", 8)),
        "away_start": int(best_cluster.get("away_start_p50", 8)),
        "away_end": int(best_cluster.get("away_end_p50", 18)),
        "sleep_start": int(best_cluster.get("sleep_start_p50", 23)),
        "sleep_end": int(best_cluster.get("sleep_end_p50", 7)),
    }
