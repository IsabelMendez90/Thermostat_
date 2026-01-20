import streamlit as st

from app_config import ACCENT, BASE_BG, COOL_BLUE, MUTED, TEAL, WHITE


def apply_styles() -> None:
    st.markdown(
        f"""
        <style>
          .stApp {{
            background: radial-gradient(1200px 800px at 50% 30%, #0B1220 0%, {BASE_BG} 55%, #0A1020 100%);
            color: {WHITE};
          }}
          #MainMenu {{visibility: hidden;}}
          footer {{visibility: hidden;}}
          header {{visibility: hidden;}}

          .frame {{
            max-width: 450px;
            margin: 0 auto;
            padding: 20px 16px 220px 16px;
          }}

          .topbar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 4px 18px 4px;
          }}
          .topbar .title {{
            font-size: 26px;
            font-weight: 600;
            letter-spacing: 0.3px;
          }}

          .big-temp {{
            text-align: center;
            font-size: 140px;
            line-height: 1.0;
            font-weight: 300;
            margin: 20px 0 12px 0;
            color: {WHITE};
          }}
          .subtle {{
            text-align: center;
            color: {MUTED};
            font-size: 14px;
          }}

          .pill-row {{
            display: flex;
            gap: 14px;
            justify-content: center;
            margin: 24px 0;
          }}
          .pill {{
            flex: 1;
            max-width: 160px;
            padding: 14px 0;
            border-radius: 999px;
            text-align: center;
            border: 2px solid rgba(255,255,255,0.15);
            background: rgba(255,255,255,0.04);
          }}
          .pill.heat {{
            border-color: rgba(249,115,22,0.7);
            background: rgba(249,115,22,0.08);
            color: {ACCENT};
          }}
          .pill.cool {{
            border-color: rgba(96,165,250,0.7);
            background: rgba(96,165,250,0.08);
            color: {COOL_BLUE};
          }}
          .pill .label {{
            font-size: 11px;
            font-weight: 600;
            opacity: 0.75;
            margin-bottom: 4px;
          }}
          .pill .value {{
            font-size: 32px;
            font-weight: 700;
          }}

          .card {{
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 20px;
            padding: 24px;
            margin: 20px 0;
            background: rgba(255,255,255,0.03);
            color: white;
          }}
          .card p, .card span, .card div {{
            color: rgba(255,255,255,0.95) !important;
          }}
          .card-cold {{
            border-color: rgba(96,165,250,0.6);
            background: rgba(96,165,250,0.08);
          }}
          .card-hot {{
            border-color: rgba(249,115,22,0.6);
            background: rgba(249,115,22,0.08);
          }}

          /* Fix Streamlit caption and text visibility */
          .stCaption, [data-testid="stCaptionContainer"] {{
            color: rgba(255,255,255,0.85) !important;
          }}
          .stMarkdown p, .stText {{
            color: rgba(255,255,255,0.9) !important;
          }}
          label[data-testid="stWidgetLabel"],
          label[data-testid="stWidgetLabel"] * {{
            color: {WHITE} !important;
            opacity: 1 !important;
          }}
          span[data-testid="stWidgetHelpText"] {{
            color: rgba(255,255,255,0.9) !important;
          }}
          div[data-baseweb="select"] * {{
            color: {WHITE} !important;
          }}
          div[data-baseweb="select"] > div {{
            background-color: rgba(255,255,255,0.06) !important;
            border-color: rgba(255,255,255,0.12) !important;
          }}
          div[data-baseweb="select"] svg {{
            fill: rgba(255,255,255,0.9) !important;
          }}
          input, textarea {{
            color: {WHITE} !important;
            background-color: rgba(255,255,255,0.06) !important;
          }}
          div[data-testid="stNumberInput"] input {{
            color: {WHITE} !important;
            background-color: rgba(255,255,255,0.06) !important;
            border-color: rgba(255,255,255,0.12) !important;
          }}
          div[data-testid="stNumberInput"] button {{
            color: {WHITE} !important;
            background-color: rgba(255,255,255,0.06) !important;
            border-color: rgba(255,255,255,0.12) !important;
          }}
          div[data-testid="stNumberInput"] button svg {{
            fill: {WHITE} !important;
          }}
          div[data-testid="stNumberInput"] > div {{
            background-color: rgba(255,255,255,0.06) !important;
            border-color: rgba(255,255,255,0.12) !important;
          }}
          div[data-testid="stNumberInput"] div[data-baseweb="input"] > div,
          div[data-testid="stNumberInput"] div[data-baseweb="input"] > div > div {{
            background-color: rgba(255,255,255,0.06) !important;
            border-color: rgba(255,255,255,0.12) !important;
          }}
          div[data-testid="stNumberInput"] div[data-baseweb="input"] input,
          div[data-testid="stNumberInput"] input[type="number"] {{
            color: {WHITE} !important;
            background-color: transparent !important;
          }}
          div[data-testid="stNumberInput"] div[role="group"] {{
            background-color: rgba(255,255,255,0.06) !important;
            border-color: rgba(255,255,255,0.12) !important;
          }}
          div[data-testid="stNumberInput"] div[role="group"] button {{
            background-color: rgba(255,255,255,0.06) !important;
            color: {WHITE} !important;
          }}
          div[data-testid="stTextInput"] div[data-baseweb="input"] > div,
          div[data-testid="stTextInput"] div[data-baseweb="input"] > div > div {{
            background-color: rgba(255,255,255,0.06) !important;
            border-color: rgba(255,255,255,0.12) !important;
          }}
          div[data-testid="stTextInput"] div[data-baseweb="input"] input,
          div[data-testid="stTextInput"] input[type="text"] {{
            color: {WHITE} !important;
            background-color: transparent !important;
          }}
          div[data-baseweb="input"] {{
            background-color: rgba(255,255,255,0.06) !important;
            border-color: rgba(255,255,255,0.12) !important;
          }}
          div[data-baseweb="input"] input {{
            color: {WHITE} !important;
          }}
          input::placeholder, textarea::placeholder {{
            color: rgba(156,163,175,0.85) !important;
            opacity: 1 !important;
          }}
          div[data-testid="stMetricValue"] {{
            color: {WHITE} !important;
          }}
          div[data-testid="stMetricLabel"] {{
            color: {WHITE} !important;
            opacity: 1 !important;
          }}
          div[data-testid="stMetricDelta"] {{
            color: rgba(255,255,255,0.85) !important;
          }}
          div[data-testid="stMetric"] * {{
            color: {WHITE} !important;
          }}
          .report-metric {{
            padding: 6px 0;
          }}
          .report-metric-label {{
            font-size: 12px;
            color: rgba(255,255,255,0.98);
            letter-spacing: 0.2px;
          }}
          .report-metric-value {{
            font-size: 28px;
            font-weight: 700;
            color: {WHITE};
          }}
          .mini-metric {{
            padding: 4px 0;
          }}
          .mini-metric-label {{
            font-size: 11px;
            color: rgba(255,255,255,0.85);
          }}
          .mini-metric-value {{
            font-size: 18px;
            font-weight: 700;
            color: {WHITE};
          }}
          .static-field {{
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 12px;
            padding: 10px 12px;
            margin-top: 4px;
          }}
          .static-field-label {{
            font-size: 12px;
            color: rgba(255,255,255,0.85);
            margin-bottom: 4px;
          }}
          .static-field-value {{
            font-size: 16px;
            font-weight: 600;
            color: {WHITE};
          }}

          .forecast-legend {{
            display: flex;
            gap: 16px;
            align-items: center;
            font-size: 12px;
          }}
          .legend-dot {{
            width: 10px;
            height: 10px;
            border-radius: 999px;
            display: inline-block;
            margin-right: 6px;
          }}

          .nav-pill {{
            padding: 10px 14px;
            border-radius: 999px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.12);
            color: {WHITE};
            font-size: 12px;
            text-align: center;
            min-width: 90px;
          }}
          .nav-pill.active {{
            border-color: {TEAL};
            background: rgba(34,197,94,0.06);
          }}

          .bottom-nav {{
            position: fixed;
            left: 0;
            right: 0;
            bottom: 0;
            padding: 14px 0 18px 0;
            background: rgba(17,24,39,0.9);
            backdrop-filter: blur(12px);
            border-top: 1px solid rgba(255,255,255,0.08);
            z-index: 1000;
          }}
          .nav-inner {{
            max-width: 450px;
            margin: 0 auto;
            display: flex;
            justify-content: space-around;
            padding: 0 24px;
          }}
          .nav-item {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 6px;
            color: {MUTED};
            font-size: 12px;
          }}
          .nav-item.active {{
            color: {TEAL};
          }}
          .nav-dot {{
            width: 8px;
            height: 8px;
            border-radius: 999px;
            background: rgba(255,255,255,0.15);
          }}
          .nav-item.active .nav-dot {{
            background: {TEAL};
          }}

          div.stButton > button,
          div[data-testid="stFormSubmitButton"] > button {{
            border-radius: 999px !important;
            border: 1px solid rgba(255,255,255,0.12) !important;
            background: rgba(255,255,255,0.08) !important;
            color: {WHITE} !important;
            padding: 12px 16px !important;
            font-weight: 500 !important;
          }}
          div.stButton > button:hover,
          div[data-testid="stFormSubmitButton"] > button:hover {{
            background: rgba(255,255,255,0.08) !important;
          }}
          div[data-testid="stFormSubmitButton"] > button:focus {{
            outline: 2px solid rgba(255,255,255,0.25) !important;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )
