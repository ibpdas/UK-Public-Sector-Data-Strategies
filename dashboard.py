import os
from datetime import date
import pandas as pd
import plotly.express as px
import streamlit as st

# Try to use rapidfuzz if present; fall back to simple contains
try:
    from rapidfuzz import fuzz, process
    HAS_RAPIDFUZZ = True
except Exception:
    HAS_RAPIDFUZZ = False

CSV_PATH = os.path.join("strategies.csv")

@st.cache_data(show_spinner=False)
def load_data(path=CSV_PATH):
    df = pd.read_csv(path).fillna("")
    required = ["id","title","organisation","org_type","country","year","scope","link","themes","pillars","summary","source","date_added"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.warning(f"Missing expected columns: {missing}")
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    return df

def tokenize_semicol(col):
    if not col: return []
    return [t.strip() for t in str(col).split(";") if t.strip()]

def fuzzy_filter(df, query, limit=100):
    if not query:
        return df
    q = query.strip()
    haystack = (df["title"] + " " + df["organisation"] + " " + df["summary"]).fillna("")
    if HAS_RAPIDFUZZ:
        matches = process.extract(q, haystack.tolist(), scorer=fuzz.WRatio, limit=len(haystack))
        keep_idx = [i for _, score, i in matches if score >= 60]
        return df.iloc[keep_idx].head(limit)
    else:
        mask = haystack.str.contains(q, case=False, na=False)
        return df[mask].head(limit)

