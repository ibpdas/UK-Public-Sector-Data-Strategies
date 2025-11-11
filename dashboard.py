# Streamlit dashboard for: UK Public-Sector Data Strategies
# Reads data/strategies.csv and provides filters, charts, and a download.
# Run: streamlit run dashboard.py

import pandas as pd
import streamlit as st
import plotly.express as px
from io import StringIO

st.set_page_config(page_title="UK Public-Sector Data Strategies", layout="wide")

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Basic sanity / normalization
    for col in ["title", "organisation", "year", "scope", "link", "summary"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["scope"] = df["scope"].str.strip().str.lower()
    return df.dropna(subset=["title", "organisation", "year", "scope"])

df = load_data("data/strategies.csv")

# ---- Sidebar filters
st.sidebar.header("Filters")
year_min, year_max = int(df["year"].min()), int(df["year"].max())
year_range = st.sidebar.slider("Year", min_value=year_min, max_value=year_max,
                               value=(year_min, year_max), step=1)
scopes = sorted(df["scope"].dropna().unique())
scope_sel = st.sidebar.multiselect("Scope", scopes, default=scopes)

org_search = st.sidebar.text_input("Search title/organisation/summary", value="").strip().lower()

def apply_filters(d: pd.DataFrame) -> pd.DataFrame:
    m = (d["year"].between(year_range[0], year_range[1])) & (d["scope"].isin(scope_sel))
    if org_search:
        txt_cols = d[["title", "organisation", "summary"]].fillna("").apply(
            lambda s: s.str.lower().str.contains(org_search)
        )
        m &= txt_cols.any(axis=1)
    return d[m].copy()

fdf = apply_filters(df)

# ---- Header
st.title("UK Public-Sector Data Strategies")
st.write("A lightweight meta-view for public-sector data leaders to compare strategies across years, scopes, and organisations.")

# ---- KPIs
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Total strategies (filtered)", f"{len(fdf):,}")
kpi2.metric("Year span (filtered)", f"{int(fdf['year'].min())}–{int(fdf['year'].max())}" if len(fdf) else "—")
kpi3.metric("Scopes represented", f"{len(fdf['scope'].unique())}" if len(fdf) else "—")

# ---- Charts
c1, c2 = st.columns(2)

with c1:
    if len(fdf):
        year_counts = fdf.groupby("year").size().reset_index(name="count")
        fig_year = px.bar(year_counts, x="year", y="count", title="Strategies by Year",
                          labels={"year": "Year", "count": "Count"})
        fig_year.update_layout(margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig_year, use_container_width=True)
    else:
        st.info("No data for selected filters.")

with c2:
    if len(fdf):
        scope_counts = fdf.groupby("scope").size().reset_index(name="count").sort_values("count", ascending=False)
        fig_scope = px.pie(scope_counts, names="scope", values="count", title="Distribution by Scope")
        fig_scope.update_layout(margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig_scope, use_container_width=True)
    else:
        st.info("No data for selected filters.")

# Top organisations (optional)
if len(fdf):
    st.subheader("Top organisations (by number of strategies)")
    top_org = (fdf.groupby("organisation").size()
               .reset_index(name="count")
               .sort_values("count", ascending=False).head(10))
    fig_org = px.bar(top_org, x="organisation", y="count", title="Top organisations",
                     labels={"organisation": "Organisation", "count": "Count"})
    fig_org.update_layout(xaxis_tickangle=-35, margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig_org, use_container_width=True)

# ---- Table (filterable)
st.subheader("Explore strategies")
st.dataframe(
    fdf[["title", "organisation", "year", "scope", "link", "summary"]]
    .sort_values(["year", "organisation", "title"]),
    use_container_width=True,
    height=420
)

# ---- Download filtered CSV
csv_buf = StringIO()
fdf.to_csv(csv_buf, index=False)
st.download_button(
    label="Download filtered CSV",
    data=csv_buf.getvalue(),
    file_name="strategies_filtered.csv",
    mime="text/csv"
)

st.caption("Tip: contribute updates via Issues → “💡 Strategy Submission”.")
