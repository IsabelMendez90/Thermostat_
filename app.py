import streamlit as st

from app_styles import apply_styles
from app_state import init_app_state
from app_core import main
from data_assets import ensure_outdoor_temps_dataset


st.set_page_config(
    page_title="My Smart Thermostat",
    layout="centered",
    initial_sidebar_state="collapsed",
)
apply_styles()
init_app_state()
if not st.session_state.get("outdoor_temps_ready"):
    with st.spinner("Preparing outdoor temperature data (first run may take a while)..."):
        ok, msg = ensure_outdoor_temps_dataset()
    st.session_state.outdoor_temps_ready = ok
    if not ok:
        st.warning(f"Outdoor temperature data unavailable: {msg}")
main()
