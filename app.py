# app.py
# Entry point for Public Sector Data Strategy Explorer (modular version)

import glob
import hashlib
import io
import os
import time

import pandas as pd
import plotly.io as pio
import streamlit as st

import home_tab
import explore_tab
import lenses_tab
import journey_tab
import actions_tab
import resources_tab
import about_tab

APP_VERSION = "v3.3 – 2025-11-13"

# ---------- Page config ----------
st.set_page_config(
    page_title="Public Sector Data Strategy Explorer",
    layout="wide",
)

# ---------- Theme colours ----------
PRIMARY = "#1d70b8"  # GOV-style blue
DARK = "#0b0c0c"
LIGHT = "#f3f2f1"
ACCENT = "#28a197"
RED = "#d4351c"

# ---------- CSS header ----------
st.markdown(
    f"""
<style>
.header-bar {{
  background:{DARK};
  border-bottom:8px solid {PRIMARY};
  padding:0.75rem 1rem;
  margin:-3rem -3rem 1rem -3rem;
}}
.header-bar h1 {{
  color:white; margin:0; font-size:1.6rem; font-weight:700;
  font-family:"Noto Sans","Helvetica Neue",Helvetica,Arial,sans-serif;
}}
.header-bar .sub {{
  color:#dcdcdc; font-size:0.95rem; margin-top:0.2rem;
}}
body, .block-container {{
  color:{DARK};
  font-family:"Noto Sans","Helvetica Neue",Helvetica,Arial,sans-serif;
}}
a, a:visited {{ color:{PRIMARY}; }}
a:hover {{ color:#003078; }}
.card {{
  background:white; border:1px solid #e5e5e5; border-radius:8px;
  padding:16px; box-shadow:0 1px 2px rgba(0,0,0,0.03); height:100%;
}}
.card h3 {{ margin-top:0; }}
.card .desc {{ color:#505a5f; font-size:0.95rem; }}
.info-panel {{
  background:{LIGHT}; border-left:5px solid {PRIMARY};
  padding:1rem; margin:0.5rem 0 1rem 0;
}}
.warn {{
  background:#fef7f7; border-left:5px solid {RED};
  padding:0.6rem 0.8rem; margin:0.3rem 0; color:#6b0f0f;
}}
.badge {{
  display:inline-block; padding:2px 8px; border-radius:999px;
  background:{PRIMARY}15; color:{PRIMARY}; font-size:0.8rem; margin-right:6px;
}}
</style>
<div class="header-bar">
  <h1>Public Sector Data Strategy Explorer</h1>
  <div class="sub">Design better data strategies, faster — balance tensions, align leadership, and plan change.</div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------- Plotly theme ----------
pio.templates["govlook"] = pio.templates["simple_white"]
pio.templates["govlook"].layout.colorway = [
    PRIMARY,
    ACCENT,
    "#d4351c",
    "#f47738",
    "#00703c",
    "#4c2c92",
]
pio.templates["govlook"].layout.font.family = "Noto Sans"
pio.templates["govlook"].layout.font.color = DARK
pio.templates["govlook"].layout.title.font.size = 18
pio.templates.default = "govlook"

st.caption(f"Build: {APP_VERSION}")

# ---------- Data loading ----------
REQUIRED_COLS = [
    "id",
    "title",
    "organisation",
    "org_type",
    "country",
    "year",
    "scope",
    "link",
    "summary",
    "source",
    "date_added",
]


def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


@st.cache_data(show_spinner=False)
def load_data(path: str, file_hash: str, app_version: str) -> pd.DataFrame:
    df = pd.read_csv(path).fillna("")
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    return df


# Find a CSV (prefer strategies.csv)
csv_files = sorted([f for f in glob.glob("*.csv") if os.path.isfile(f)])
if "strategies.csv" in csv_files:
    default_csv = "strategies.csv"
elif csv_files:
    default_csv = csv_files[0]
else:
    default_csv = None

if default_csv:
    try:
        df = load_data(default_csv, file_md5(default_csv), APP_VERSION)
    except Exception as e:
        st.error(f"Error loading CSV '{default_csv}': {e}")
        df = pd.DataFrame(columns=REQUIRED_COLS)
else:
    st.warning("No CSV file found. Place 'strategies.csv' in this folder.")
    df = pd.DataFrame(columns=REQUIRED_COLS)

# ---------- Tabs ----------
tab_home, tab_explore, tab_lenses, tab_journey, tab_actions, tab_resources, tab_about = st.tabs(
    ["Home", "Explore", "Lenses", "Journey", "Actions & Export", "Resources", "About"]
)

with tab_home:
    home_tab.render_home(df, APP_VERSION)

with tab_explore:
    explore_tab.render_explore(df)

with tab_lenses:
    lenses_tab.render_lenses()

with tab_journey:
    journey_tab.render_journey()

with tab_actions:
    actions_tab.render_actions()

with tab_resources:
    resources_tab.render_resources()

with tab_about:
    about_tab.render_about()
