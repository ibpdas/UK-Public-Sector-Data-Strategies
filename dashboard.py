# ---------------------------
# Public Sector Data Strategy Explorer (Hardened)
# ---------------------------
import os
from datetime import date
import pandas as pd
import plotly.express as px
import streamlit as st

# Optional fuzzy search: degrade gracefully if not installed
try:
    from rapidfuzz import fuzz, process
    HAS_RAPIDFUZZ = True
except Exception:
    HAS_RAPIDFUZZ = False

CSV_PATH = "strategies.csv"
REQUIRED = [
    "id","title","organisation","org_type","country","year","scope",
    "link","themes","pillars","summary","source","date_added"
]

st.set_page_config(page_title="Public Sector Data Strategy Explorer", layout="wide")
st.title("Public Sector Data Strategy Explorer")
st.caption("Compare UK public sector data strategies, spot patterns, and reuse what works.")

# ------------ Utilities
@st.cache_data(show_spinner=False)
def load_data(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at '{path}'. Add strategies.csv to the repo root.")
    try:
        df = pd.read_csv(path).fillna("")
    except Exception as e:
        raise RuntimeError(f"Could not read CSV: {e}")

    # Column check
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}. Expected: {REQUIRED}")

    # Coerce year to numeric (keep NaN)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    return df

def tokenize_semicol(col):
    if not col: return []
    return [t.strip() for t in str(col).split(";") if t.strip()]

def fuzzy_filter(df, query, limit=200):
    if not query:
        return df
    q = query.strip()
    haystack = (df["title"] + " " + df["organisation"] + " " + df["summary"]).fillna("")
    if HAS_RAPIDFUZZ:
        matches = process.extract(q, haystack.tolist(), scorer=fuzz.WRatio, limit=len(haystack))
        keep_idx = [i for _, score, i in matches if score >= 60]
        if not keep_idx:
            return df.iloc[0:0]
        return df.iloc[keep_idx].head(limit)
    else:
        mask = haystack.str.contains(q, case=False, na=False)
        return df[mask].head(limit)

def explode_semicol(df, col):
    rows = []
    for _, r in df.iterrows():
        items = tokenize_semicol(r.get(col, ""))
        if not items:
            rows.append({**r.to_dict(), col: "(none)"})
        else:
            for it in items:
                rr = r.to_dict(); rr[col] = it
                rows.append(rr)
    return pd.DataFrame(rows)

# ------------ Load data with robust error feedback
try:
    df = load_data(CSV_PATH)
except Exception as e:
    st.error(f"⚠️ {e}")
    st.stop()

# ------------ Sidebar filters
with st.sidebar:
    st.subheader("Filters")

    # Year slider only if we have ANY valid years
    valid_years = sorted(y for y in df["year"].dropna().unique())
    use_year_filter = len(valid_years) > 0
    if use_year_filter:
        min_y, max_y = int(min(valid_years)), int(max(valid_years))
        year_range = st.slider("Year range", min_value=min_y, max_value=max_y, value=(min_y, max_y), step=1)
    else:
        st.info("No valid 'year' values in CSV — skipping year filter.")
        year_range = None

    org_types = sorted([v for v in df["org_type"].unique() if v != ""])
    org_type_sel = st.multiselect("Organisation type", org_types, default=org_types)

    countries = sorted([v for v in df["country"].unique() if v != ""])
    country_sel = st.multiselect("Country", countries, default=countries)

    scopes = sorted([v for v in df["scope"].unique() if v != ""])
    scope_sel = st.multiselect("Scope", scopes, default=scopes)

    q = st.text_input("Search title, organisation, summary", "")

    st.markdown("---")
    debug = st.checkbox("Show debug info")

# ------------ Apply filters (never drop NaN years)
fdf = df.copy()
if year_range and len(fdf):
    yr_mask = (fdf["year"].between(year_range[0], year_range[1], inclusive="both")) | (fdf["year"].isna())
    fdf = fdf[yr_mask]

if org_type_sel:
    fdf = fdf[fdf["org_type"].isin(org_type_sel)]
if country_sel:
    fdf = fdf[fdf["country"].isin(country_sel)]
if scope_sel:
    fdf = fdf[fdf["scope"].isin(scope_sel)]

fdf = fuzzy_filter(fdf, q)

# ------------ Debug panel
if debug:
    with st.expander("🔎 Debug"):
        st.write("Working directory:", os.getcwd())
        st.write("Files:", os.listdir("."))
        st.write("Rows loaded:", len(df))
        st.dataframe(df.head(), use_container_width=True)

# ------------ Empty state
if fdf.empty:
    st.warning("No results match your filters/search. Try clearing the search box or adjusting filters.")
    st.stop()

# ------------ KPIs
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Strategies", len(fdf))
col_b.metric("Org types", fdf["org_type"].nunique())
col_c.metric("Countries", fdf["country"].nunique())
yr_min = int(fdf["year"].min()) if pd.notna(fdf["year"].min()) else "—"
yr_max = int(fdf["year"].max()) if pd.notna(fdf["year"].max()) else "—"
col_d.metric("Year span", f"{yr_min}–{yr_max}")

# ------------ Charts
st.subheader("Patterns")
left, right = st.columns([2,2])

with left:
    if "year" in fdf.columns and fdf["year"].notna().any():
        by_year = fdf[fdf["year"].notna()].groupby("year").size().reset_index(name="count")
        fig1 = px.bar(by_year, x="year", y="count", title="Strategies by year")
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("No numeric 'year' values to chart.")

with right:
    if "themes" in fdf.columns:
        tlong = explode_semicol(fdf, "themes")
        by_theme = tlong.groupby("themes").size().reset_index(name="count").sort_values("count", ascending=False)
        fig2 = px.treemap(by_theme, path=["themes"], values="count", title="Top themes")
        st.plotly_chart(fig2, use_container_width=True)

left2, right2 = st.columns([2,2])
with left2:
    if "org_type" in fdf.columns:
        by_org = fdf.groupby("org_type").size().reset_index(name="count").sort_values("count", ascending=False)
        fig3 = px.bar(by_org, x="org_type", y="count", title="By organisation type")
        st.plotly_chart(fig3, use_container_width=True)

with right2:
    if "pillars" in fdf.columns:
        plong = explode_semicol(fdf, "pillars")
        by_pillar = plong.groupby("pillars").size().reset_index(name="count").sort_values("count", ascending=False)
        fig4 = px.treemap(by_pillar, path=["pillars"], values="count", title="Pillars mentioned")
        st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# ------------ Explorer
st.subheader("Explorer")
st.caption("Click a title to open the official document in a new tab.")
show_cols = ["title","organisation","org_type","country","year","scope","themes","pillars","source"]
st.dataframe(
    fdf[show_cols].sort_values(["year","organisation"], ascending=[False, True]),
    use_container_width=True,
    hide_index=True,
)

st.markdown("### Details")
for _, r in fdf.sort_values("year", ascending=False).iterrows():
    year_txt = int(r["year"]) if pd.notna(r["year"]) else "—"
    with st.expander(f"📄 {r['title']} — {r['organisation']} ({year_txt})"):
        st.write(r["summary"] or "_No summary yet._")
        meta = st.columns(4)
        meta[0].write(f"**Org type:** {r['org_type']}")
        meta[1].write(f"**Country:** {r['country']}")
        meta[2].write(f"**Scope:** {r['scope']}")
        meta[3].write(f"**Source:** {r['source']}")
        st.write(f"**Themes:** {', '.join(tokenize_semicol(r['themes'])) or '—'}")
        st.write(f"**Pillars:** {', '.join(tokenize_semicol(r['pillars'])) or '—'}")
        if r["link"]:
            st.link_button("Open document", r["link"], use_container_width=False)
