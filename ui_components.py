from __future__ import annotations

from typing import Optional

import streamlit as st


def topbar(title: str):
    st.markdown(
        f"""
        <div class="topbar">
          <div></div>
          <div class="title">{title}</div>
          <div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def bottom_nav():
    view = st.session_state.view

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button(
            "🏠\n\nHome",
            key="nav_home",
            use_container_width=True,
            type="primary" if view == "Home" else "secondary"
        ):
            st.session_state.view = "Home"
            st.rerun()

    with col2:
        if st.button(
            "📅\n\nSchedules",
            key="nav_sched",
            use_container_width=True,
            type="primary" if view == "Schedules" else "secondary"
        ):
            st.session_state.view = "Schedules"
            st.rerun()

    with col3:
        if st.button(
            "📊\n\nForecast",
            key="nav_fore",
            use_container_width=True,
            type="primary" if view == "Forecast" else "secondary"
        ):
            st.session_state.view = "Forecast"
            st.rerun()

    with col4:
        if st.button(
            "📈\n\nReports",
            key="nav_rep",
            use_container_width=True,
            type="primary" if view == "Reports" else "secondary"
        ):
            st.session_state.view = "Reports"
            st.rerun()


def render_report_metric(label: str, value: str, sub_label: Optional[str] = None) -> None:
    st.markdown('<div class="report-metric">', unsafe_allow_html=True)
    st.markdown(f'<div class="report-metric-label">{label}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="report-metric-value">{value}</div>', unsafe_allow_html=True)
    if sub_label:
        st.markdown(f'<div class="mini-metric-label">{sub_label}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_mini_metric(label: str, value: str) -> None:
    st.markdown('<div class="mini-metric">', unsafe_allow_html=True)
    st.markdown(f'<div class="mini-metric-label">{label}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="mini-metric-value">{value}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
