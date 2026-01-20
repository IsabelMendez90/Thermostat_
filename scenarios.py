from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from text_utils import norm_text


def apply_custom_scenario(schedules: Dict[str, Dict[str, Any]], rules: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out = {k: dict(v) for k, v in schedules.items()}
    max_d = int(np.clip(rules.get("max_delta_f", 3), 0, 6))
    min_db = int(np.clip(rules.get("min_deadband_f", 3), 2, 6))
    exclude = set(rules.get("exclude_schedules", ["Sleep"]) or ["Sleep"])
    per = rules.get("per_schedule", {}) or {}
    peak = rules.get("peak_window", None)

    def clamp_deadband(h: int, c: int) -> Tuple[int, int]:
        if (c - h) < min_db:
            c = min(90, h + min_db)
        return h, c

    for name, s in out.items():
        if name in exclude:
            continue

        dd = per.get(name, {}) or {}
        heat_delta = int(np.clip(dd.get("heat_delta", 0), -max_d, max_d))
        cool_delta = int(np.clip(dd.get("cool_delta", 0), -max_d, max_d))

        h = int(np.clip(int(s["heat_sp"]) + heat_delta, 55, 75))
        c = int(np.clip(int(s["cool_sp"]) + cool_delta, 70, 90))
        h, c = clamp_deadband(h, c)
        s["heat_sp"], s["cool_sp"] = h, c

        if peak and isinstance(peak, dict):
            if name.lower() in ["home", "evening"]:
                peak_cool_delta = int(np.clip(peak.get("cool_delta", 0), 0, 6))
                s["cool_sp"] = int(np.clip(int(s["cool_sp"]) + peak_cool_delta, 70, 90))
                s["heat_sp"], s["cool_sp"] = clamp_deadband(int(s["heat_sp"]), int(s["cool_sp"]))

    return out


def build_scenario_schedules(
    base: Dict[str, Dict[str, Any]],
    scenario_name: str,
    mode: str,
    custom_scenarios: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    Build a schedule set for a named scenario.
    """
    schedules = {k: dict(v) for k, v in base.items()}
    if scenario_name == "Baseline (current schedules)":
        return schedules

    if scenario_name in custom_scenarios:
        rules = custom_scenarios[scenario_name]
        return apply_custom_scenario(schedules, rules)

    if "Warmer home" in scenario_name:
        for n in schedules:
            schedules[n]["cool_sp"] = max(70, int(schedules[n]["cool_sp"]) - 2)
            schedules[n]["heat_sp"] = min(75, int(schedules[n]["heat_sp"]) + 2)
        return schedules

    if "Cooler home" in scenario_name:
        for n in schedules:
            schedules[n]["cool_sp"] = min(90, int(schedules[n]["cool_sp"]) + 2)
            schedules[n]["heat_sp"] = max(55, int(schedules[n]["heat_sp"]) - 2)
        return schedules

    if "Peak Hours" in scenario_name:
        target_names = []
        for n in schedules:
            tokens = set(norm_text(n).split())
            if tokens & {"home", "evening", "awake", "study", "studying", "afternoon", "work", "day"}:
                target_names.append(n)
        if not target_names:
            target_names = list(schedules.keys())
        for n in target_names:
            if mode == "heating":
                schedules[n]["heat_sp"] = max(55, int(schedules[n]["heat_sp"]) - 2)
            elif mode == "cooling":
                schedules[n]["cool_sp"] = min(90, int(schedules[n]["cool_sp"]) + 3)
            else:
                schedules[n]["heat_sp"] = max(55, int(schedules[n]["heat_sp"]) - 1)
                schedules[n]["cool_sp"] = min(90, int(schedules[n]["cool_sp"]) + 1)
        return schedules

    return schedules
