import streamlit as st

from app_styles import apply_styles
from app_state import init_app_state
from app_core import main


st.set_page_config(
    page_title="My Smart Thermostat",
    layout="centered",
    initial_sidebar_state="collapsed",
)
apply_styles()
init_app_state()
main()
