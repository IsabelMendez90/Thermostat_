from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st

from text_utils import norm_text


def _default_schedule_priority(name: str, index: int) -> int:
    """
    Lower value means higher priority when schedules overlap.
    """
    n = norm_text(name)
    if any(k in n for k in ["sleep", "bed", "night"]):
        base = 10
    elif any(k in n for k in ["study", "studying", "work", "office", "focus"]):
        base = 20
    elif any(k in n for k in ["home", "awake", "morning", "evening", "day", "daytime"]):
        base = 30
    elif any(k in n for k in ["away", "vacation", "travel", "out"]):
        base = 40
    else:
        base = 50
    return base * 100 + index


def normalize_schedule_priorities(
    schedules: Dict[str, Any],
    priorities: Optional[Dict[str, Any]] = None,
) -> Dict[str, int]:
    """
    Normalize priority mapping to ranks 1..N (1 = highest priority).
    """
    names = list(schedules.keys())
    raw = priorities if isinstance(priorities, dict) else {}

    scored = []
    for idx, name in enumerate(names):
        if name in raw and isinstance(raw[name], (int, float)):
            score = int(raw[name])
        else:
            score = _default_schedule_priority(name, idx)
        scored.append((name, score, idx))

    ordered = sorted(scored, key=lambda x: (x[1], x[2]))
    return {name: rank + 1 for rank, (name, _, _) in enumerate(ordered)}


def ensure_schedule_priorities() -> Dict[str, int]:
    """
    Ensure session_state.schedule_priorities exists and is normalized.
    """
    try:
        ss = st.session_state
        ss.schedule_priorities = normalize_schedule_priorities(ss.schedules, ss.get("schedule_priorities", {}))
        return ss.schedule_priorities
    except Exception:
        return {}


def _active_schedule_for_hour(
    dt,
    schedules: Dict[str, Dict[str, Any]],
    schedule_priorities: Optional[Dict[str, int]] = None,
) -> Optional[str]:
    if not schedules:
        return None

    if schedule_priorities is None:
        schedule_priorities = ensure_schedule_priorities()
    else:
        schedule_priorities = normalize_schedule_priorities(schedules, schedule_priorities)

    hour = dt.hour
    active = []
    order = {n: i for i, n in enumerate(schedules.keys())}
    for name, sched in schedules.items():
        start = int(sched["start_hour"])
        end = int(sched["end_hour"])
        if start <= end:
            if start <= hour < end:
                active.append(name)
        else:
            if hour >= start or hour < end:
                active.append(name)

    if not active:
        return list(schedules.keys())[0]
    active.sort(key=lambda n: (schedule_priorities.get(n, 9999), order.get(n, 0)))
    return active[0]

