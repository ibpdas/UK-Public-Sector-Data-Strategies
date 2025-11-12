# ---------------------------
# Public Sector Data Strategy Explorer — Option B (Enhanced)
# ---------------------------
import os
from datetime import date
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    "link","themes","archetypes","summary","source","date_added"
]

# Archetype reference text (tooltips + docs)
ARCH_REF = {
    "conceptual": "High-level direction and principles rather than implementation detail.",
    "logical-physical": "Defines logical relationships, models, and concrete implementation patterns.",
    "living": "Evolves with technology, policy, and organisational changes; versioned and iterative.",
    "fixed": "Static for long periods; emphasises stability and formal governance.",
    "essential": "Focus on foundational data management (quality, governance, security, compliance).",
    "transformational": "Focus on high-impact change using AI, advanced analytics, automation.",
    "horizontal": "Builds capability consistently across the enterprise.",
    "breadth": "Targets specific use-cases/quick wins across different domains.",
    "ecosystem-federated": "Encourages collaboration, interoperability, and distributed ownership.",
    "centralised": "Places authority and ownership under a single central function.",
    "value-oriented": "Outcomes and public value first; tech is a means to an end.",
    "technology-oriented": "Infrastructure, platforms, and tools are the central focus.",
    "innovation-driven": "Pushes new methods (AI/ML), experimentation, and flexibility.",
    "compliance-driven": "Prioritises regulation, ethics, risk, and security obligations.",
    "data-democratization": "Wider access to data and tools across the organisation.",
    "controlled-access": "Strict permissions, least-privilege, and need-to-know access.",
    "incremental": "Phased delivery, smaller iterations, learn-and-adapt cycles.",
    "big-bang": "Large, one-off change programmes and rollouts.",
    "data-informed": "Balances data with expert judgement and context.",
    "data-driven": "Heavily privileging analytical outputs for decisions."
}

st.set_page_config(page_title="Public Sector Data Strategy Explorer", layout="wide")
st.title("Public Sector Data Strategy Explorer")
st.caption("Explore public-sector data strategies by themes and strategic archetypes.")

@st.cache_data(show_spinner=False)
def load_data(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at '{path}'. Add strategies.csv to the repo root.")
    try:
        df = pd.read_csv(path).fillna("")
    except Exception as e:
        raise RuntimeError(f"Could not read CSV: {e}")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}. Expected: {REQUIRED}")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    return df

def tokenize_semicol(col):
    if not col: return []
    return [t.strip() for t in str(col).split(";") if t.strip()]

def fuzzy_filter(df, query, limit=500):
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

# Load data
try:
    df = load_data(CSV_PATH)
except Exception as e:
    st.error(f"⚠️ {e}")
    st.stop()

# Tabs for Explore / Compare / About
tab_explore, tab_compare, tab_about = st.tabs(["🔎 Explore", "🆚 Compare", "ℹ️ About"])

with tab_explore:
    # Sidebar controls
    with st.sidebar:
        st.subheader("Filters")

        # Year slider if numeric years exist
        valid_years = sorted(y for y in df["year"].dropna().unique())
        if valid_years:
            min_y, max_y = int(min(valid_years)), int(max(valid_years))
            year_range = st.slider("Year range", min_value=min_y, max_value=max_y, value=(min_y, max_y), step=1, help="Filter strategies by publication year.")
        else:
            year_range = None
            st.info("No valid 'year' values in CSV — skipping year filter.")

        org_types = sorted([v for v in df["org_type"].unique() if v != ""])
        org_type_sel = st.multiselect("Organisation type", org_types, default=org_types)

        countries = sorted([v for v in df["country"].unique() if v != ""])
        country_sel = st.multiselect("Country", countries, default=countries)

        scopes = sorted([v for v in df["scope"].unique() if v != ""])
        scope_sel = st.multiselect("Scope", scopes, default=scopes)

        # Archetype filter (any match) with inline “what is this?”
        arch_all = sorted({a for cell in df["archetypes"] for a in tokenize_semicol(cell)})
        arch_help = "Filter by any matching archetype. See the Framework reference below for definitions."
        arch_sel = st.multiselect("Archetypes (any)", arch_all, default=[], help=arch_help)

        q = st.text_input("Search title, organisation, summary", "", help="Fuzzy search on title, organisation, and summary.")

        st.markdown("---")
        debug = st.checkbox("Show debug info")

    # Apply filters
    fdf = df.copy()
    if year_range:
        yr_mask = (fdf["year"].between(year_range[0], year_range[1], inclusive="both")) | (fdf["year"].isna())
        fdf = fdf[yr_mask]

    if org_type_sel:
        fdf = fdf[fdf["org_type"].isin(org_type_sel)]
    if country_sel:
        fdf = fdf[fdf["country"].isin(country_sel)]
    if scope_sel:
        fdf = fdf[fdf["scope"].isin(scope_sel)]

    # Archetype matching: keep rows that contain ANY of the selected archetypes
    if arch_sel:
        def has_any_arch(s):
            tags = set(tokenize_semicol(s))
            return any(a in tags for a in arch_sel)
        fdf = fdf[fdf["archetypes"].apply(has_any_arch)]

    # Fuzzy search / contains
    fdf = fuzzy_filter(fdf, q)

    # Debug
    if debug:
        with st.expander("🔎 Debug"):
            st.write("Rows loaded:", len(df))
            st.write("Rows after filters:", len(fdf))
            st.dataframe(df.head(), use_container_width=True)

    # Empty state
    if fdf.empty:
        st.warning("No results match your filters/search. Try clearing the search box or adjusting filters.")
        st.stop()

    # KPIs
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Strategies", len(fdf))
    col_b.metric("Org types", fdf["org_type"].nunique())
    col_c.metric("Countries", fdf["country"].nunique())
    yr_min = int(fdf["year"].min()) if pd.notna(fdf["year"].min()) else "—"
    yr_max = int(fdf["year"].max()) if pd.notna(fdf["year"].max()) else "—"
    col_d.metric("Year span", f"{yr_min}–{yr_max}")

    # Charts
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
        if "archetypes" in fdf.columns:
            along = explode_semicol(fdf, "archetypes")
            by_arch = along.groupby("archetypes").size().reset_index(name="count").sort_values("count", ascending=False)
            # Attach short tooltips via legend by renaming with hint
            def with_hint(a):
                return f"{a} — {ARCH_REF.get(a, '')}"
            by_arch["arch_label"] = by_arch["archetypes"].apply(with_hint)
            fig3 = px.bar(by_arch, x="arch_label", y="count", title="Archetype distribution")
            fig3.update_xaxes(tickangle=35)
            st.plotly_chart(fig3, use_container_width=True)

    with right2:
        if "org_type" in fdf.columns:
            by_org = fdf.groupby("org_type").size().reset_index(name="count").sort_values("count", ascending=False)
            fig4 = px.bar(by_org, x="org_type", y="count", title="By organisation type")
            st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")

    # Explorer + Download
    st.subheader("Explorer")
    st.caption("Click a title to open the official document in a new tab.")
    show_cols = ["title","organisation","org_type","country","year","scope","themes","archetypes","source"]
    st.dataframe(
        fdf[show_cols].sort_values(["year","organisation"], ascending=[False, True]),
        use_container_width=True,
        hide_index=True,
    )

    # Download filtered CSV
    st.download_button(
        label="Download filtered CSV",
        data=fdf.to_csv(index=False).encode("utf-8"),
        file_name="strategies_filtered.csv",
        mime="text/csv",
        help="Exports the current filtered dataset."
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
            archs = tokenize_semicol(r["archetypes"])
            if archs:
                lines = [f"- **{a}** — {ARCH_REF.get(a, 'No description')}" for a in archs]
                st.markdown("**Archetypes:**\n" + "\n".join(lines))
            else:
                st.write("**Archetypes:** —")
            if r["link"]:
                st.link_button("Open document", r["link"], use_container_width=False)

with tab_compare:
    st.subheader("Compare strategies")
    st.caption("Select up to 5 strategies to compare their archetype profile (radar).")
    # Select box by title
    titles = df["title"].tolist()
    pick = st.multiselect("Select strategies", titles, default=titles[:2], max_selections=5)

    if not pick:
        st.info("Choose at least one strategy to compare.")
        st.stop()

    comp = df[df["title"].isin(pick)].copy()

    # Build binary matrix of archetypes for radar
    # Define the 20 canonical labels (stable order)
    dims = [
        "conceptual","logical-physical",
        "living","fixed",
        "essential","transformational",
        "horizontal","breadth",
        "ecosystem-federated","centralised",
        "value-oriented","technology-oriented",
        "innovation-driven","compliance-driven",
        "data-democratization","controlled-access",
        "incremental","big-bang",
        "data-informed","data-driven"
    ]

    def row_vector(archetype_cell):
        tags = set(tokenize_semicol(archetype_cell))
        return [1 if d in tags else 0 for d in dims]

    vectors = comp["archetypes"].apply(row_vector).tolist()

    # Radar expects equal-length categories; we'll show as polar with 20 axes
    categories = dims
    fig = go.Figure()
    for i, (title, vec) in enumerate(zip(comp["title"], vectors)):
        fig.add_trace(go.Scatterpolar(
            r=vec + [vec[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name=title
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        showlegend=True,
        title="Archetype radar (1=present, 0=absent)"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Also show a heatmap view for readability
    import numpy as np
    mat = pd.DataFrame([row_vector(a) for a in comp["archetypes"]], columns=dims, index=comp["title"])
    fig_hm = px.imshow(mat, aspect="auto", title="Archetype heatmap (1=present, 0=absent)")
    st.plotly_chart(fig_hm, use_container_width=True)

with tab_about:
    st.subheader("About this Explorer")
    st.markdown("""
This tool classifies public-sector data strategies using a ten‑dimension **archetype** framework:
- **Abstraction Level:** conceptual / logical-physical  
- **Adaptability:** living / fixed  
- **Transformational Depth:** essential / transformational  
- **Breadth of Application:** horizontal / breadth  
- **Governance Structure:** ecosystem-federated / centralised  
- **Strategic Orientation:** value-oriented / technology-oriented  
- **Motivation:** innovation-driven / compliance-driven  
- **Access Philosophy:** data-democratization / controlled-access  
- **Implementation Mode:** incremental / big-bang  
- **Decision-Making Model:** data-informed / data-driven  

**Methodology:** strategies are tagged using human judgement (with optional automation), stored as semicolon-separated labels in the CSV. Visualisations are generated dynamically in Streamlit.
""")

    st.markdown("---")
    st.subheader("Framework reference")
    for key in [
        "conceptual","logical-physical","living","fixed","essential","transformational",
        "horizontal","breadth","ecosystem-federated","centralised","value-oriented","technology-oriented",
        "innovation-driven","compliance-driven","data-democratization","controlled-access","incremental","big-bang",
        "data-informed","data-driven"
    ]:
        st.markdown(f"- **{key}** — {ARCH_REF.get(key, 'No description')}")

    st.caption("© Contributors. Open for reuse under the Open Government Licence (where applicable).")
        st.write(f"**Pillars:** {', '.join(tokenize_semicol(r['pillars'])) or '—'}")
        if r["link"]:
            st.link_button("Open document", r["link"], use_container_width=False)
