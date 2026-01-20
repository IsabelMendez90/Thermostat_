from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from schedule_utils import ensure_schedule_priorities
from schedule_suggestions import semantic_schedule_intent
from text_utils import fuzzy_match

try:
    from openai import OpenAI
    OPENROUTER_AVAILABLE = True
except Exception:
    OPENROUTER_AVAILABLE = False


def get_openrouter_api_key() -> Optional[str]:
    # Streamlit secrets first, then env
    key = None
    try:
        key = st.secrets.get("OPENROUTER_API_KEY")
    except Exception:
        key = None
    return key or os.getenv("OPENROUTER_API_KEY")


def get_openrouter_client():
    if not OPENROUTER_AVAILABLE:
        return None
    key = get_openrouter_api_key()
    if not key:
        return None
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)


def thermostat_state_summary() -> Dict[str, Any]:
    schedules_summary = {}
    for name, sched in st.session_state.schedules.items():
        schedules_summary[name] = {
            "heat": sched["heat_sp"],
            "cool": sched["cool_sp"],
            "hours": f"{int(sched['start_hour']):02d}:00-{int(sched['end_hour']):02d}:00",
        }

    reports_summary = None
    if st.session_state.view == "Reports" and hasattr(st.session_state, "baseline_runtime"):
        baseline_rt = st.session_state.baseline_runtime
        if baseline_rt is not None and len(baseline_rt) > 0:
            reports_summary = {
                "avg_daily_runtime_hours": float(baseline_rt["runtime_hours"].mean()) if "runtime_hours" in baseline_rt.columns else None,
                "avg_indoor_temp": float(baseline_rt["indoor_temp"].mean()) if "indoor_temp" in baseline_rt.columns else None,
            }

            if "heat_sec_hours" in baseline_rt.columns:
                reports_summary["avg_heating_hours_per_day"] = float(baseline_rt["heat_sec_hours"].mean())
            if "cool_sec_hours" in baseline_rt.columns:
                reports_summary["avg_cooling_hours_per_day"] = float(baseline_rt["cool_sec_hours"].mean())
            if "aux_sec_hours" in baseline_rt.columns:
                reports_summary["avg_aux_heat_hours_per_day"] = float(baseline_rt["aux_sec_hours"].mean())
            if "fan_sec_hours" in baseline_rt.columns:
                reports_summary["avg_fan_hours_per_day"] = float(baseline_rt["fan_sec_hours"].mean())

    return {
        "view": st.session_state.view,
        "hvac_mode": st.session_state.hvac_mode,
        "fan_on": st.session_state.fan_on,
        "active_comfort": st.session_state.active_comfort,
        "indoor_temp": st.session_state.indoor_temp,
        "outdoor_temp": st.session_state.outdoor_temp,
        "outdoor_humidity": st.session_state.outdoor_humidity,
        "schedules": schedules_summary,
        "location": st.session_state.location_name,
        "similar_homes_count": st.session_state.similar_homes_count,
        "reports_data": reports_summary,
    }


def parse_action_from_text(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    m = re.search(r"<ACTION>(.*?)</ACTION>", text, re.DOTALL)
    if m:
        try:
            action_json = json.loads(m.group(1))
            cleaned = re.sub(r"<ACTION>.*?</ACTION>", "", text, flags=re.DOTALL).strip()
            return action_json, cleaned
        except json.JSONDecodeError:
            pass

    json_pattern = r'\{[^{}]*"type"\s*:\s*"create_scenario"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.finditer(json_pattern, text, re.DOTALL)

    for match in matches:
        try:
            json_str = match.group(0)
            brace_count = 1
            start_pos = match.start() + 1
            end_pos = start_pos
            for i in range(start_pos, len(text)):
                if text[i] == "{":
                    brace_count += 1
                elif text[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i
                        break
            json_str = text[match.start(): end_pos + 1]
            action_json = json.loads(json_str)
            cleaned = text[:match.start()] + text[end_pos + 1:]
            cleaned = cleaned.strip()
            return action_json, cleaned
        except (json.JSONDecodeError, ValueError):
            continue

    return None, text


def resolve_schedule_key(name: str) -> Optional[str]:
    if not name:
        return None
    if name in st.session_state.schedules:
        return name
    lower_map = {k.lower(): k for k in st.session_state.schedules.keys()}
    if name.lower() in lower_map:
        return lower_map[name.lower()]
    choices = list(st.session_state.schedules.keys())
    hits = fuzzy_match(name, choices, threshold=75)
    if hits:
        return hits[0][0]
    return None


def allowed_actions_for_view(view: str) -> set[str]:
    if view == "Forecast":
        return {"create_scenario"}  # sandbox only
    if view == "Setup":
        return set()
    return {
        "set_hvac_mode",
        "set_fan",
        "set_comfort",
        "set_setpoint",
        "create_schedule",
        "update_schedule",
        "bulk_update_schedules",
        "create_scenario",
    }


def rules_schema_hint() -> str:
    return (
        "Scenario rules schema:\n"
        "- max_delta_f: int (0..6)\n"
        "- min_deadband_f: int (2..6)\n"
        "- exclude_schedules: list[str] (e.g., ['Sleep'])\n"
        "- per_schedule: dict like {\n"
        "    'Home': {'heat_delta': +1, 'cool_delta': +2},\n"
        "    'Away': {'heat_delta': -1, 'cool_delta': +3}\n"
        "  }\n"
        "- peak_window: optional dict like {'start':16,'end':21,'cool_delta': +2}\n"
    )


def apply_action(action: Dict[str, Any]) -> str:
    t = action.get("type", "")
    allowed = allowed_actions_for_view(st.session_state.view)
    if t not in allowed:
        return f"❌ Not allowed in {st.session_state.view}. Forecast is sandbox-only."

    try:
        if t == "set_hvac_mode":
            mode = action.get("mode", "Auto")
            if mode in ["Off", "Heat", "Cool", "Auto", "Aux"]:
                st.session_state.hvac_mode = mode
                return f"✓ HVAC mode set to {mode}"
            return "❌ Invalid HVAC mode"

        if t == "set_fan":
            fan = action.get("fan", "Auto")
            st.session_state.fan_on = (fan == "On")
            return f"✓ Fan set to {fan}"

        if t == "set_comfort":
            comfort = action.get("comfort", "Home")
            key = resolve_schedule_key(comfort)
            if not key:
                return f"❌ Schedule '{comfort}' not found"
            st.session_state.active_comfort = key
            return f"✓ Switched to {key}"

        if t == "set_setpoint":
            target = action.get("target", "heat")
            value = int(action.get("value", 70))
            comfort = action.get("comfort", st.session_state.active_comfort)
            key = resolve_schedule_key(comfort) or comfort
            if key not in st.session_state.schedules:
                return f"❌ Schedule '{comfort}' not found"
            if target == "heat":
                st.session_state.schedules[key]["heat_sp"] = int(np.clip(value, 55, 75))
                return f"✓ Set {key} heat to {value}°F"
            if target == "cool":
                st.session_state.schedules[key]["cool_sp"] = int(np.clip(value, 70, 90))
                return f"✓ Set {key} cool to {value}°F"
            return "❌ Invalid target"

        if t == "create_schedule":
            name = (action.get("name", "Custom") or "Custom").strip()
            if not name:
                return "❌ Missing name"
            if name in st.session_state.schedules:
                return f"❌ Schedule '{name}' already exists"
            heat = int(action.get("heat_sp", 68))
            cool = int(action.get("cool_sp", 76))
            start = int(action.get("start_hour", 9))
            end = int(action.get("end_hour", 17))
            st.session_state.schedules[name] = {"heat_sp": heat, "cool_sp": cool, "start_hour": start, "end_hour": end}
            return f"✓ Created '{name}' schedule"

        if t == "update_schedule":
            raw = action.get("name", "")
            key = resolve_schedule_key(raw)
            if not key:
                return f"❌ Schedule '{raw}' not found"
            s = st.session_state.schedules[key]
            heat = int(action.get("heat_sp", s["heat_sp"]))
            cool = int(action.get("cool_sp", s["cool_sp"]))
            start = int(action.get("start_hour", s["start_hour"]))
            end = int(action.get("end_hour", s["end_hour"]))
            s.update({"heat_sp": heat, "cool_sp": cool, "start_hour": start, "end_hour": end})
            return f"✓ Updated '{key}' schedule"

        if t == "bulk_update_schedules":
            updates = action.get("updates", [])
            creates = action.get("creates", [])
            if not isinstance(updates, list):
                updates = []
            if not isinstance(creates, list):
                creates = []

            updated = []
            created = []
            errors = []

            for item in updates:
                if not isinstance(item, dict):
                    continue
                raw = item.get("name", "")
                key = resolve_schedule_key(raw)
                if not key:
                    errors.append(f"'{raw}' not found")
                    continue
                s = st.session_state.schedules[key]
                if "heat_sp" in item:
                    s["heat_sp"] = int(np.clip(item["heat_sp"], 55, 75))
                if "cool_sp" in item:
                    s["cool_sp"] = int(np.clip(item["cool_sp"], 70, 90))
                if "start_hour" in item:
                    s["start_hour"] = int(np.clip(item["start_hour"], 0, 23))
                if "end_hour" in item:
                    s["end_hour"] = int(np.clip(item["end_hour"], 0, 23))
                updated.append(key)

            for item in creates:
                if not isinstance(item, dict):
                    continue
                name = (item.get("name", "") or "").strip()
                if not name:
                    errors.append("missing schedule name")
                    continue
                if name in st.session_state.schedules:
                    errors.append(f"'{name}' already exists")
                    continue
                heat = int(np.clip(item.get("heat_sp", 68), 55, 75))
                cool = int(np.clip(item.get("cool_sp", 76), 70, 90))
                start = int(np.clip(item.get("start_hour", 9), 0, 23))
                end = int(np.clip(item.get("end_hour", 17), 0, 23))
                st.session_state.schedules[name] = {
                    "heat_sp": heat,
                    "cool_sp": cool,
                    "start_hour": start,
                    "end_hour": end,
                }
                created.append(name)

            if created:
                ensure_schedule_priorities()

            parts = []
            if updated:
                parts.append(f"Updated: {', '.join(sorted(set(updated)))}")
            if created:
                parts.append(f"Created: {', '.join(created)}")
            if errors:
                parts.append(f"Issues: {', '.join(errors)}")
            return "✓ " + (" | ".join(parts) if parts else "No changes applied")

        if t == "create_scenario":
            name = (action.get("name") or "Custom Scenario").strip()
            rules = action.get("rules") or {}
            max_d = int(np.clip(rules.get("max_delta_f", 3), 0, 6))
            min_db = int(np.clip(rules.get("min_deadband_f", 3), 2, 6))
            exclude = rules.get("exclude_schedules", ["Sleep"])
            if not isinstance(exclude, list):
                exclude = ["Sleep"]
            per = rules.get("per_schedule", {})
            if not isinstance(per, dict):
                per = {}
            peak = rules.get("peak_window", None)
            if peak is not None and not isinstance(peak, dict):
                peak = None

            rules_clean = {
                "max_delta_f": max_d,
                "min_deadband_f": min_db,
                "exclude_schedules": exclude,
                "per_schedule": per,
            }
            if peak:
                rules_clean["peak_window"] = {
                    "start": int(np.clip(peak.get("start", 16), 0, 23)),
                    "end": int(np.clip(peak.get("end", 21), 0, 23)),
                    "cool_delta": int(np.clip(peak.get("cool_delta", 0), 0, 6)),
                }

            st.session_state.custom_scenarios[name] = rules_clean
            st.session_state.selected_scenario = name
            return f"✓ Added scenario '{name}' (sandbox)"

        return "✓ Action completed"
    except Exception as e:
        return f"❌ Error: {e}"


def call_openrouter(user_text: str, model: str = "mistralai/devstral-2512:free") -> Tuple[str, Optional[Dict[str, Any]]]:
    client = get_openrouter_client()
    if not client:
        return "Assistant not available (missing OpenRouter key or openai package).", None

    state = thermostat_state_summary()
    view = st.session_state.view
    allowed = allowed_actions_for_view(view)

    allowed_lines = []
    if "set_hvac_mode" in allowed:
        allowed_lines.append('- set_hvac_mode: {"type":"set_hvac_mode","mode":"Off|Heat|Cool|Auto|Aux"}')
    if "set_fan" in allowed:
        allowed_lines.append('- set_fan: {"type":"set_fan","fan":"Auto|On"}')
    if "set_comfort" in allowed:
        allowed_lines.append('- set_comfort: {"type":"set_comfort","comfort":"ScheduleName"}')
    if "set_setpoint" in allowed:
        allowed_lines.append('- set_setpoint: {"type":"set_setpoint","target":"heat|cool","value":INT,"comfort":"optional"}')
    if "create_schedule" in allowed:
        allowed_lines.append('- create_schedule: {"type":"create_schedule","name":"text","heat_sp":INT,"cool_sp":INT,"start_hour":INT,"end_hour":INT}')
    if "update_schedule" in allowed:
        allowed_lines.append('- update_schedule: {"type":"update_schedule","name":"text","heat_sp":INT,"cool_sp":INT,"start_hour":INT,"end_hour":INT}')
    if "bulk_update_schedules" in allowed:
        allowed_lines.append('- bulk_update_schedules: {"type":"bulk_update_schedules","updates":[{"name":"text","heat_sp":INT,"cool_sp":INT,"start_hour":INT,"end_hour":INT}], "creates":[{"name":"text","heat_sp":INT,"cool_sp":INT,"start_hour":INT,"end_hour":INT}]}')
    if "create_scenario" in allowed:
        allowed_lines.append('- create_scenario: {"type":"create_scenario","name":"text","rules":{...}}')

    allowed_block = "\n".join(allowed_lines) if allowed_lines else "(No actions allowed in this view.)"

    system = (
        f"You are an ecobee-style thermostat assistant inside a Streamlit UI.\n"
        f"Current view: {view}\n\n"
        "**Core Purpose:**\n"
        "You help users understand and control their smart thermostat, optimize energy usage, "
        "interpret usage data, and learn about HVAC systems and Ecobee features.\n\n"
        "**Important Scope Guidelines:**\n"
        "- If the user asks questions UNRELATED to thermostats, HVAC, energy, homes, or Ecobee features "
        "(e.g., 'How far is the Earth from the sun?'), politely redirect them:\n"
        '  Example: "That\'s an interesting question! While I focus on helping you with your thermostat '
        'and home comfort, I\'d be happy to help you optimize your heating schedule or answer questions '
        'about your energy usage instead. How can I assist with your thermostat today?"\n\n'
        "**Data Access:**\n"
        "- You have access to the current thermostat state (JSON below).\n"
        "- In Reports view, you can see energy usage data including runtime hours and component breakdowns.\n"
        "- Use this data to answer questions like 'interpret my energy breakdown', 'how much am I using for heating', etc.\n\n"
        "Protocol:\n"
        "1) Give a short helpful answer related to thermostats/energy/home comfort.\n"
        "2) If (and only if) the user requests a change that exists in UI, propose ONE action.\n"
        "3) Put it in exactly one JSON block inside <ACTION>...</ACTION>.\n\n"
        f"Allowed action types:\n{allowed_block}\n\n"
        + rules_schema_hint() +
        "\nRules:\n"
        "- Never claim the change already happened.\n"
        "- In Forecast view, DO NOT propose schedule changes; only propose create_scenario.\n"
        "- If schedule exists, use update_schedule; otherwise create_schedule.\n"
        "- If the user wants multiple schedule edits at once, use bulk_update_schedules.\n"
        "- Be concise and thermostat-focused.\n"
        "- When interpreting Reports data, explain it in simple terms (e.g., 'Your HVAC runs about X hours/day').\n"
    )

    history = st.session_state.assistant_messages[-12:] if st.session_state.assistant_messages else []
    messages = [
        {"role": "system", "content": system},
        {"role": "system", "content": f"Current thermostat state (JSON): {json.dumps(state)}"},
        *history,
        {"role": "user", "content": user_text},
    ]

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            extra_headers={"HTTP-Referer": "https://example.com", "X-Title": "Smart Thermostat"},
        )
        raw = completion.choices[0].message.content.strip()
        action, cleaned = parse_action_from_text(raw)
        return cleaned, action
    except Exception as e:
        return f"Assistant error: {e}", None


def build_location_context_for_llm() -> Optional[Dict[str, Any]]:
    """Build location-based context for LLM scenario builder."""
    location_id = st.session_state.get("location_id")
    if not location_id:
        return None

    context = {
        "location_id": location_id,
        "n_similar_homes": st.session_state.get("similar_homes_count", 0),
    }

    priors = st.session_state.get("neighbor_priors", {})
    if priors:
        context["typical_setpoints"] = {
            "heat_p25": priors.get("heat_sp_p25"),
            "heat_p50": priors.get("heat_sp_p50"),
            "heat_p75": priors.get("heat_sp_p75"),
            "cool_p25": priors.get("cool_sp_p25"),
            "cool_p50": priors.get("cool_sp_p50"),
            "cool_p75": priors.get("cool_sp_p75"),
        }

    schedules = st.session_state.get("schedules", {})
    for name in ["Home", "Evening", "Awake"]:
        if name in schedules:
            context["peak_hours_usage"] = {
                "cool_p50": schedules[name].get("cool_sp", 76),
                "cool_p25": schedules[name].get("cool_sp", 76) - 2,
                "heat_p50": schedules[name].get("heat_sp", 68),
            }
            break

    return context if context.get("typical_setpoints") or context.get("peak_hours_usage") else None


def call_openrouter_for_scenario(
    user_goal: str,
    location_context: Optional[Dict[str, Any]] = None,
    model: str = "mistralai/devstral-2512:free",
) -> Tuple[str, Optional[Dict[str, Any]]]:
    client = get_openrouter_client()
    if not client:
        return "Scenario builder not available (missing OpenRouter key or openai package).", None

    state = thermostat_state_summary()

    location_info = ""
    if location_context and location_context.get("location_id"):
        location_info = f"\n\n**Location-Based Insights (Location #{location_context['location_id']}):**\n"
        location_info += f"Based on {location_context.get('n_similar_homes', 0):,} similar homes in your area:\n"

        if location_context.get("typical_setpoints"):
            typical = location_context["typical_setpoints"]
            location_info += f"- Typical heating setpoints: {typical.get('heat_p25', 'N/A')}°F to {typical.get('heat_p75', 'N/A')}°F (median: {typical.get('heat_p50', 'N/A')}°F)\n"
            location_info += f"- Typical cooling setpoints: {typical.get('cool_p25', 'N/A')}°F to {typical.get('cool_p75', 'N/A')}°F (median: {typical.get('cool_p50', 'N/A')}°F)\n"

        if location_context.get("peak_hours_usage"):
            peak = location_context["peak_hours_usage"]
            location_info += f"- During peak hours (4-9pm), typical users set cooling to {peak.get('cool_p50', 'N/A')}°F\n"
            location_info += f"- For more comfort during peak hours, consider cooling setpoints around {peak.get('cool_p25', 'N/A')}°F\n"

    system = (
        "You are a smart thermostat scenario builder with access to real user behavior data.\n"
        "Return ONE action ONLY: create_scenario.\n"
        "Do NOT propose real schedule changes; scenarios are sandbox.\n\n"
        "When the user requests comfort changes or peak hour strategies:\n"
        "1) Analyze the location-based insights provided below to suggest realistic setpoints\n"
        "2) Explain WHY you chose specific setpoint values based on local user behavior\n"
        "3) Consider comfort vs efficiency tradeoffs\n\n"
        "Output format:\n"
        "1) Natural explanation mentioning:\n"
        "   - What you're doing and why\n"
        "   - Specific setpoint recommendations with reasoning\n"
        "   - Expected impact on comfort and energy\n"
        "2) Exactly one <ACTION>...</ACTION> with JSON:\n"
        '<ACTION>{"type":"create_scenario","name":"...","rules":{...}}</ACTION>\n\n'
        f"{location_info}\n"
        "Rules schema:\n"
        "- max_delta_f: int 0..6 (max temp change allowed)\n"
        "- min_deadband_f: int 2..6 (min gap between heat/cool)\n"
        "- exclude_schedules: list[str] (schedules to skip, e.g. ['Sleep'])\n"
        "- per_schedule: {ScheduleName: {heat_delta:int, cool_delta:int}}\n"
        "  (positive heat_delta = warmer heating, positive cool_delta = warmer cooling/less AC)\n"
        "- peak_window (optional): {start:int, end:int, cool_delta:int}\n\n"
        "Use existing schedule names from state.schedules.\n"
        "Be specific about setpoint values in your explanation, not just deltas.\n"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "system", "content": f"Current thermostat state: {json.dumps(state)}"},
        {"role": "user", "content": user_goal},
    ]
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            extra_headers={"HTTP-Referer": "https://example.com", "X-Title": "Scenario Builder"},
        )
        raw = completion.choices[0].message.content.strip()
        action, cleaned = parse_action_from_text(raw)
        return cleaned, action
    except Exception as e:
        return f"Scenario builder error: {e}", None


def call_llm_for_schedule_suggestion(
    schedule_name: str,
    location_context: Optional[Dict[str, Any]] = None,
    model: str = "mistralai/devstral-2512:free"
) -> Dict[str, Any]:
    """
    Use LLM to suggest setpoints and timing for a new schedule based on intent + location.
    """
    client = get_openrouter_client()
    if not client:
        intent = semantic_schedule_intent(schedule_name)
        return {
            "heat_sp": intent["suggested_setpoints"]["heat_sp"],
            "cool_sp": intent["suggested_setpoints"]["cool_sp"],
            "start_hour": intent["suggested_hours"][0],
            "end_hour": intent["suggested_hours"][1],
            "explanation": f"Based on intent detection: {intent['intent']}",
            "source": "intent_only"
        }

    intent = semantic_schedule_intent(schedule_name)

    location_info = ""
    if location_context and location_context.get("location_id"):
        location_info = f"\n**Location Context (Location #{location_context['location_id']}):**\n"
        location_info += f"- Similar homes in area: {location_context.get('n_similar_homes', 0):,}\n"

        if location_context.get("typical_setpoints"):
            typical = location_context["typical_setpoints"]
            location_info += f"- Typical heating: {typical.get('heat_p25', 'N/A')}-{typical.get('heat_p75', 'N/A')}°F (median {typical.get('heat_p50', 'N/A')}°F)\n"
            location_info += f"- Typical cooling: {typical.get('cool_p25', 'N/A')}-{typical.get('cool_p75', 'N/A')}°F (median {typical.get('cool_p50', 'N/A')}°F)\n"

    system = (
        "You are an HVAC schedule consultant with access to real user behavior data from California homes.\n"
        "Your task: Suggest SPECIFIC setpoint values (heat_sp, cool_sp) and timing (start_hour, end_hour) "
        "for a new thermostat schedule.\n\n"
        "Consider:\n"
        "1. Schedule intent (what activity/time of day)\n"
        "2. Local user patterns (if available)\n"
        "3. Comfort vs. energy efficiency tradeoffs\n"
        "4. Typical HVAC setpoints in California:\n"
        "   - Heating: 62-70°F (away: 62°F, home: 68-70°F)\n"
        "   - Cooling: 72-84°F (away: 80-84°F, home: 72-76°F)\n\n"
        "Output ONLY valid JSON (no markdown, no extra text):\n"
        '{"heat_sp": <int>, "cool_sp": <int>, "start_hour": <int 0-23>, "end_hour": <int 0-23>, '
        '"explanation": "<brief reasoning about why you chose these values>"}\n'
    )

    user_message = (
        f"Schedule name: \"{schedule_name}\"\n"
        f"Detected intent: {intent['intent']}\n"
        f"Intent-based suggestion: Heat {intent['suggested_setpoints']['heat_sp']}°F, "
        f"Cool {intent['suggested_setpoints']['cool_sp']}°F, "
        f"Hours {intent['suggested_hours'][0]:02d}:00-{intent['suggested_hours'][1]:02d}:00\n"
        f"{location_info}\n"
        "Suggest appropriate setpoints and timing."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message},
    ]

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            extra_headers={"HTTP-Referer": "https://example.com", "X-Title": "Schedule Suggester"},
            temperature=0.3,
        )
        raw = completion.choices[0].message.content.strip()

        if raw.startswith("```"):
            raw = re.sub(r"^```[a-zA-Z]*", "", raw).strip().strip("```").strip()

        result = json.loads(raw)
        result["source"] = "llm_enhanced"
        return result
    except Exception as e:
        return {
            "heat_sp": intent["suggested_setpoints"]["heat_sp"],
            "cool_sp": intent["suggested_setpoints"]["cool_sp"],
            "start_hour": intent["suggested_hours"][0],
            "end_hour": intent["suggested_hours"][1],
            "explanation": f"Based on intent detection: {intent['intent']} (LLM error: {str(e)[:50]})",
            "source": "intent_fallback"
        }


def assistant_bar():
    st.markdown("---")
    st.markdown("### 🤖 Assistant")

    key = get_openrouter_api_key()
    if not key:
        st.caption("Tip: set OPENROUTER_API_KEY to enable assistant + AI scenarios.")

    if "assistant_input_value" not in st.session_state:
        st.session_state.assistant_input_value = ""

    with st.form(key="assistant_form", clear_on_submit=True):
        prompt = st.text_input(
            "Ask something…",
            value="",
            placeholder="e.g., set my heat setpoint to 70, make it cooler",
            key="assistant_input_form"
        )

        colA, colB = st.columns([1, 1])
        with colA:
            ask = st.form_submit_button("Send", use_container_width=True)
        with colB:
            pass

    clear = st.button("Clear chat", use_container_width=False)

    if clear:
        st.session_state.assistant_messages = []
        st.session_state.pending_action = None
        st.session_state.pending_explainer = ""
        st.session_state.assistant_input_value = ""
        st.rerun()

    if ask and prompt.strip():
        st.session_state.assistant_messages.append({"role": "user", "content": prompt.strip()})

        with st.spinner("Thinking..."):
            reply, action = call_openrouter(prompt.strip())

        st.session_state.assistant_messages.append({"role": "assistant", "content": reply})
        st.session_state.pending_action = action
        st.session_state.pending_explainer = reply
        st.rerun()

    history = st.session_state.assistant_messages[-12:]
    for m in history:
        if m["role"] == "user":
            st.markdown(f'<div style="background-color: rgba(100, 149, 237, 0.1); padding: 10px; border-radius: 5px; margin: 5px 0;"><b>You:</b> {m["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="background-color: rgba(50, 205, 50, 0.1); padding: 10px; border-radius: 5px; margin: 5px 0;"><b>Assistant:</b> {m["content"]}</div>', unsafe_allow_html=True)

    action = st.session_state.pending_action
    if action:
        action_type = action.get("type", "")

        if action_type == "set_setpoint":
            target = action.get("target", "")
            value = action.get("value", "")
            desc = f"Change {target} setpoint to {value}°F"
        elif action_type == "create_schedule":
            name = action.get("name", "")
            desc = f'Create schedule "{name}"'
        elif action_type == "update_schedule":
            name = action.get("name", "")
            desc = f'Update schedule "{name}"'
        elif action_type == "bulk_update_schedules":
            updates = action.get("updates", []) or []
            creates = action.get("creates", []) or []
            desc = f"Update {len(updates)} schedule(s)"
            if creates:
                desc += f" + create {len(creates)}"
        elif action_type == "set_hvac_mode":
            mode = action.get("mode", "")
            desc = f"Set HVAC mode to {mode}"
        elif action_type == "set_comfort":
            comfort = action.get("comfort", "")
            desc = f'Switch to "{comfort}" schedule'
        elif action_type == "create_scenario":
            name = action.get("name", "")
            desc = f'Create scenario "{name}"'
        else:
            desc = "Apply the suggested change"

        st.markdown('<div style="background-color: rgba(255, 215, 0, 0.15); padding: 15px; border-radius: 10px; border: 2px solid rgba(255, 215, 0, 0.5); margin: 10px 0;">', unsafe_allow_html=True)
        st.markdown(f"**💡 Ready to apply:** {desc}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Apply Change", use_container_width=True, type="primary"):
                msg = apply_action(action)
                st.success(msg)
                st.session_state.pending_action = None
                st.session_state.pending_explainer = ""
                st.rerun()
        with col2:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.pending_action = None
                st.info("Change cancelled.")
                st.session_state.pending_explainer = ""
                st.rerun()

        with st.expander("🔍 Technical details"):
            st.code(json.dumps(action, indent=2), language="json")

        st.markdown('</div>', unsafe_allow_html=True)
