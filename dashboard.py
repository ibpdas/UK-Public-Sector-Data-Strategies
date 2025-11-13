# ---------------------------------------------------
# Public Sector Data Strategy Explorer
# ---------------------------------------------------
import os
import glob
import io
import time
import hashlib
import base64

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# --- Optional: semantic embeddings (AI search) ---
try:
    from sentence_transformers import SentenceTransformer

    HAS_EMBED = True
except Exception:
    HAS_EMBED = False
    SentenceTransformer = None

APP_VERSION = "v3.1 – 2025-11-13"

# ---------------- PAGE CONFIG & THEME ----------------
st.set_page_config(
    page_title="Public Sector Data Strategy Explorer",
    layout="wide",
)

PRIMARY = "#1d70b8"  # GOV-style blue
DARK = "#0b0c0c"     # near-black
LIGHT = "#f3f2f1"    # light grey
ACCENT = "#28a197"   # teal
RED = "#d4351c"

st.markdown(
    f"""
<style>
/* Header bar */
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

/* Body */
body, .block-container {{
  color:{DARK};
  font-family:"Noto Sans","Helvetica Neue",Helvetica,Arial,sans-serif;
}}
a, a:visited {{ color:{PRIMARY}; }}
a:hover {{ color:#003078; }}

/* Cards */
.card {{
  background:white; border:1px solid #e5e5e5; border-radius:8px;
  padding:16px; box-shadow:0 1px 2px rgba(0,0,0,0.03); height:100%;
}}
.card h3 {{ margin-top:0; }}
.card .desc {{ color:#505a5f; font-size:0.95rem; }}

/* Info / warning panels */
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
.kv {{
  display:inline-block; padding:2px 6px; border-radius:4px;
  background:{LIGHT}; border:1px solid #e5e5e5; margin-right:6px;
}}

/* Buttons */
.stButton>button {{
  background:{PRIMARY}; color:white; border-radius:0; border:none; font-weight:600;
}}
.stButton>button:hover {{ background:#003078; }}

/* Footer */
.footer {{
  color:#505a5f; font-size:0.85rem; text-align:center; margin-top:1.2rem;
}}
</style>
<div class="header-bar">
  <h1>Public Sector Data Strategy Explorer</h1>
  <div class="sub">Design better data strategies, faster — balance tensions, align leadership, and plan change.</div>
</div>
""",
    unsafe_allow_html=True,
)

# Plotly theme
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

# ---------------- DATA LOADING ----------------
REQUIRED = [
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


def bytes_md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


@st.cache_data(show_spinner=False)
def load_data_from_path(path: str, file_hash: str, app_version: str):
    df = pd.read_csv(path).fillna("")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    return df


# --- Load initial CSV (default or uploaded) ---
csv_files = sorted([f for f in glob.glob("*.csv") if os.path.isfile(f)])
default_csv = (
    "strategies.csv"
    if "strategies.csv" in csv_files
    else (csv_files[0] if csv_files else None)
)

if "uploaded_bytes" in st.session_state:
    content = st.session_state["uploaded_bytes"]
    df = load_data_from_bytes(content, bytes_md5(content), APP_VERSION)
elif default_csv:
    df = load_data_from_path(default_csv, file_md5(default_csv), APP_VERSION)
else:
    df = pd.DataFrame(columns=REQUIRED)

# ---------------- LENSES & MATURITY ----------------

# Government data maturity themes (CDDO)
MATURITY_THEMES = [
    (
        "Uses",
        "How you get value out of data. Making decisions, evidencing impact, improving services.",
    ),
    (
        "Data",
        "Technical aspects of managing data as an asset: collection, quality, cataloguing, interoperability.",
    ),
    (
        "Leadership",
        "How senior and business leaders engage with data: strategy, responsibility, oversight, investment.",
    ),
    (
        "Culture",
        "Attitudes to data across the organisation: awareness, openness, security, responsibility.",
    ),
    (
        "Tools",
        "The systems and tools you use to store, share and work with data.",
    ),
    (
        "Skills",
        "Data and analytical literacy across the organisation, including how people build and maintain those skills.",
    ),
]

# Official government levels 1–5
MATURITY_SCALE = {
    1: "Beginning",
    2: "Emerging",
    3: "Learning",
    4: "Developing",
    5: "Mastering",
}


def maturity_label(avg: float) -> str:
    """
    Map the average (1–5) to the nearest official maturity level.
    """
    idx = int(round(avg))
    idx = max(1, min(5, idx))
    return MATURITY_SCALE[idx]


# Ten Lenses
AXES = [
    ("Abstraction Level", "Conceptual", "Logical / Physical"),
    ("Adaptability", "Living", "Fixed"),
    ("Ambition", "Essential", "Transformational"),
    ("Coverage", "Horizontal", "Use-case-based"),
    ("Governance Structure", "Ecosystem / Federated", "Centralised"),
    ("Orientation", "Technology-focused", "Value-focused"),
    ("Motivation", "Compliance-driven", "Innovation-driven"),
    ("Access Philosophy", "Data-democratised", "Controlled access"),
    ("Delivery Mode", "Incremental", "Big Bang"),
    ("Decision Model", "Data-informed", "Data-driven"),
]
DIMENSIONS = [a[0] for a in AXES]


def radar_trace(values01, dims, name, opacity=0.6, fill=True):
    r = list(values01) + [values01[0]]
    t = list(dims) + [dims[0]]
    return go.Scatterpolar(
        r=r, theta=t, name=name, fill="toself" if fill else None, opacity=opacity
    )


def ensure_sessions():
    if "_maturity_scores" not in st.session_state:
        st.session_state["_maturity_scores"] = {k: 3 for k, _ in MATURITY_THEMES}
    if "_current_scores" not in st.session_state:
        st.session_state["_current_scores"] = {d: 50 for d in DIMENSIONS}
    if "_target_scores" not in st.session_state:
        st.session_state["_target_scores"] = {d: 50 for d in DIMENSIONS}
    if "_actions_df" not in st.session_state:
        st.session_state["_actions_df"] = pd.DataFrame(
            columns=["Priority", "Lens", "Direction", "Owner", "Timeline", "Metric", "Status"]
        )


# ---------------- SEARCH HELPERS ----------------
def simple_search(df_in: pd.DataFrame, query: str) -> pd.DataFrame:
    """
    Simple case-insensitive search over key text columns.
    """
    if not query:
        return df_in

    text_cols = [c for c in ["title", "organisation", "summary", "scope"] if c in df_in.columns]
    if not text_cols:
        return df_in

    text = df_in[text_cols[0]].astype(str)
    for col in text_cols[1:]:
        text = text + " " + df_in[col].astype(str)

    mask = text.str.contains(query, case=False, na=False)
    return df_in[mask]


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    if not HAS_EMBED:
        return None
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data(show_spinner=False)
def compute_strategy_embeddings(df_in: pd.DataFrame, app_version: str):
    if not HAS_EMBED:
        return None
    model = get_embedding_model()
    if model is None:
        return None

    text_cols = [c for c in ["title", "organisation", "summary", "scope", "country"] if c in df_in.columns]
    if not text_cols:
        return None

    texts = df_in[text_cols[0]].astype(str)
    for col in text_cols[1:]:
        texts = texts + " " + df_in[col].astype(str)

    embeddings = model.encode(
        texts.tolist(),
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    emb_df = pd.DataFrame(embeddings, index=df_in.index)
    return emb_df


def semantic_search(fdf: pd.DataFrame, emb_df: pd.DataFrame, query: str, top_k: int = 100) -> pd.DataFrame:
    """
    Semantic search using pre-computed embeddings.
    Respects current filtered subset fdf by aligning on index.
    """
    if not query or emb_df is None or fdf.empty:
        return fdf

    model = get_embedding_model()
    if model is None:
        return fdf

    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    sub_emb = emb_df.loc[fdf.index].values
    sims = sub_emb @ q_emb

    order = np.argsort(-sims)
    order = order[: min(top_k, len(order))]
    result = fdf.iloc[order].copy()
    result["similarity"] = sims[order]
    return result


emb_df = compute_strategy_embeddings(df, APP_VERSION)

# ---------------- HINTS & CONFLICTS ----------------
def hint_for_lens(lens_name, maturity_avg, maturity_level_name=None):
    """
    Give contextual hints based on the organisation's overall maturity level.
    Uses government levels: Beginning, Emerging, Learning, Developing, Mastering.
    """
    level = maturity_level_name or maturity_label(maturity_avg)
    low = level in ("Beginning", "Emerging")
    mid = level in ("Learning", "Developing")
    high = level == "Mastering"

    if lens_name == "Governance Structure":
        if low:
            return "At Beginning/Emerging, stronger central coordination usually works best before moving to federated models."
        if mid:
            return "At Learning/Developing, you can gradually federate – keep common standards and shared services."
        if high:
            return "At Mastering, federation can unlock autonomy – but guard against fragmentation with shared guardrails."
    if lens_name == "Delivery Mode":
        if low:
            return "Favour incremental delivery to build confidence and reduce risk – avoid a single big-bang change."
        if mid:
            return "Blend incremental delivery with a few larger change packages where foundations are solid."
        if high:
            return "At Mastering, big-bang change is possible – but only with strong programme discipline and clear benefits."
    if lens_name == "Access Philosophy":
        if low:
            return "Start with role-based access to a small number of trusted datasets before opening up more widely."
        if mid:
            return "Broaden access with good catalogue/search – keep tight controls around sensitive domains."
        if high:
            return "Push democratisation further – but make sure data protection and audit trails stay robust."
    if lens_name == "Decision Model":
        if low:
            return "Data-informed decisions with clear human oversight are safest while skills and quality are still building."
        if mid:
            return "Increase automation in low-risk areas – keep humans in the loop for high-impact decisions."
        if high:
            return "Mastering orgs can rely more on data-driven decisions – but need strong monitoring and fallback plans."
    if lens_name == "Motivation":
        if low:
            return "Keep compliance at the core while you pilot innovation in tightly scoped sandboxes."
        if mid:
            return "Balance compliance and innovation – use proof-of-concepts to justify broader change."
        if high:
            return "At Mastering, innovation and compliance can reinforce each other via strong governance by design."
    if lens_name == "Ambition":
        if low:
            return "Focus on essentials – data quality, governance, core platforms – before promising transformational change."
        if mid:
            return "You can mix foundational work with some transformational strands where benefits are clear."
        if high:
            return "Aim for transformational impact – but keep benefits and operating model changes clearly articulated."
    if lens_name == "Coverage":
        if low:
            return "Use a few high-impact use-cases to prove value while you build broader capabilities."
        if mid:
            return "Begin to spread capabilities horizontally to avoid islands of excellence."
        if high:
            return "Horizontal coverage makes sense – but choose a few flagship use-cases to anchor the narrative."
    if lens_name == "Orientation":
        if low:
            return "Platform and tooling investments will dominate early – link them clearly to outcomes."
        if mid:
            return "Balance platform work with visible value – avoid tech for tech’s sake."
        if high:
            return "Keep value firmly in the lead, with platforms treated as enablers rather than ends."
    if lens_name == "Adaptability":
        if low:
            return "Keep a stable core with a small living layer – too much churn can confuse people."
        if mid:
            return "Treat the strategy as living – schedule periodic reviews and small course corrections."
        if high:
            return "Mastering orgs can iterate often – just make sure changes are well-governed and communicated."
    if lens_name == "Abstraction Level":
        if low:
            return "Keep the strategy concise and vision-led, but quickly translate into practical roadmaps and controls."
        if mid:
            return "Balance vision with enough logical detail to guide delivery teams."
        if high:
            return "You can afford a more detailed logical/physical description – but avoid over-specifying too early."

    return ""


def conflict_for_target(lens_name, target_score, maturity_avg):
    """
    Flag misalignments between maturity and ambitious targets.
    target_score is 0–100 toward right label.
    """
    level = maturity_label(maturity_avg)
    low = level in ("Beginning", "Emerging")
    highish = level in ("Developing", "Mastering")  # treat Learning as middle

    # Low maturity: warn if target is very ambitious/risky
    if low:
        if lens_name == "Delivery Mode" and target_score >= 70:
            return "Big-bang at Beginning/Emerging maturity is high risk — consider phased delivery."
        if lens_name == "Governance Structure" and target_score <= 30:
            return "Federated at low maturity can fragment standards — strengthen central controls first."
        if lens_name == "Access Philosophy" and target_score <= 30:
            return "Wide democratisation needs strong basics — start with controlled, role-based access."
        if lens_name == "Decision Model" and target_score >= 70:
            return "Highly data-driven decisions need robust data quality, monitoring and skills."
        if lens_name == "Motivation" and target_score >= 70:
            return "Innovation-first without guardrails can raise risk — keep compliance in the loop."

    # High-ish maturity: warn if overly conservative
    if highish:
        if lens_name == "Delivery Mode" and target_score <= 30:
            return "At Developing/Mastering, being too incremental may under-deliver benefits."
        if lens_name == "Governance Structure" and target_score >= 80:
            return "Highly centralised models may slow teams at higher maturity — consider selective federation."
        if lens_name == "Access Philosophy" and target_score >= 80:
            return "Excessive control may limit value realisation — revisit openness where safe."

    return None


# ---------------- EXPLORE CHARTS ----------------
def render_explore_charts(fdf: pd.DataFrame):
    st.markdown("## Explore — landscape & patterns")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Strategies", len(fdf))
    k2.metric("Countries", fdf["country"].nunique() if "country" in fdf.columns else 0)
    k3.metric("Org types", fdf["org_type"].nunique() if "org_type" in fdf.columns else 0)
    if "year" in fdf.columns and fdf["year"].notna().any():
        k4.metric("Year span", f"{int(fdf['year'].min())}–{int(fdf['year'].max())}")
    else:
        k4.metric("Year span", "—")

    st.markdown("---")
    c1, c2 = st.columns(2)

    if "year" in fdf.columns and fdf["year"].notna().any():
        fig_hist = px.histogram(
            fdf[fdf["year"].notna()],
            x="year",
            color="scope" if "scope" in fdf.columns else None,
            nbins=max(10, min(40, fdf["year"].nunique())),
            title="Strategies by year",
        )
        fig_hist.update_layout(bargap=0.05)
        c1.plotly_chart(fig_hist, use_container_width=True)
    else:
        c1.info("No numeric 'year' values to chart. Check your CSV or filters.")

    if "org_type" in fdf.columns and fdf["org_type"].notna().any():
        top_org = (
            fdf.groupby("org_type")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        fig_org = px.bar(
            top_org,
            x="org_type",
            y="count",
            title="Composition by organisation type",
        )
        fig_org.update_xaxes(title=None, tickangle=20)
        c2.plotly_chart(fig_org, use_container_width=True)
    else:
        c2.info("No 'org_type' values to chart.")

    st.markdown("---")
    c3, c4 = st.columns(2)

    if all(col in fdf.columns for col in ["country", "org_type"]):
        if not fdf.empty:
            fig_tree = px.treemap(
                fdf.assign(_value=1),
                path=["country", "org_type", "organisation"],
                values="_value",
                title="Landscape by country → org type → organisation",
            )
            c3.plotly_chart(fig_tree, use_container_width=True)
        else:
            c3.info("No data for treemap.")
    else:
        c3.info("Need 'country' and 'org_type' columns for treemap.")

    if "country" in fdf.columns and fdf["country"].notna().any():
        by_ctry = fdf.groupby("country").size().reset_index(name="count")
        if not by_ctry.empty:
            fig_map = px.choropleth(
                by_ctry,
                locations="country",
                locationmode="country names",
                color="count",
                title="Global distribution of strategies (by country)",
                color_continuous_scale="Blues",
            )
            c4.plotly_chart(fig_map, use_container_width=True)
        else:
            c4.info("No country counts to map.")
    else:
        c4.info("No 'country' values to map.")

    st.markdown("---")
    c5, c6 = st.columns(2)
    if all(col in fdf.columns for col in ["country", "org_type"]):
        top_ctrys = (
            fdf.groupby("country").size().sort_values(ascending=False).head(12).index.tolist()
        )
        sub = fdf[fdf["country"].isin(top_ctrys)]
        if not sub.empty:
            fig_stack = px.bar(
                sub,
                x="country",
                color="org_type",
                title="Top countries by strategies (stacked by org type)",
            )
            fig_stack.update_xaxes(title=None)
            c5.plotly_chart(fig_stack, use_container_width=True)
        else:
            c5.info("No data for stacked bar.")
    else:
        c5.info("Need 'country' and 'org_type' for stacked bar.")

    needed = ["year", "organisation", "title"]
    if all(col in fdf.columns for col in needed) and fdf["year"].notna().any():
        sub = fdf[fdf["year"].notna()].copy()
        fig_scatter = px.scatter(
            sub,
            x="year",
            y="organisation",
            color="country" if "country" in sub.columns else None,
            hover_data=["title", "country", "scope"]
            if "scope" in sub.columns
            else ["title"],
            title="Timeline of strategies by organisation",
        )
        c6.plotly_chart(fig_scatter, use_container_width=True)
    else:
        c6.info("Need 'year', 'organisation', and 'title' columns for timeline.")

    st.markdown("---")
    if "scope" in fdf.columns and fdf["scope"].notna().any():
        by_scope = fdf["scope"].value_counts().reset_index()
        by_scope.columns = ["scope", "count"]
        fig_scope = px.pie(
            by_scope, names="scope", values="count", title="Strategy scope breakdown"
        )
        st.plotly_chart(fig_scope, use_container_width=True)


# ---------------- TABS SETUP ----------------
ensure_sessions()
tab_home, tab_explore, tab_lenses, tab_journey, tab_actions, tab_resources, tab_about = st.tabs(
    ["Home", "Explore", "Lenses", "Journey", "Actions & Export", "Resources", "About"]
)

# ...other tab content...


# ====================================================
# 🏠 HOME
# ====================================================
with tab_home:
    st.markdown(
        """
<div class="info-panel">
<strong>Quick start:</strong> Begin with <strong>Lenses → Maturity</strong> to understand your current readiness,
then set your strategic tensions and review the Journey.
</div>
""",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
<div class="card">
<h3>Explore</h3>
<p class="desc">
See patterns in real strategies — by year, country, organisation type and scope.
Maps, timelines and composition views give fast context.
</p>
</div>
""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
<div class="card">
<h3>Lenses</h3>
<p class="desc">
<strong>Step 1:</strong> Self-diagnose maturity across six government themes.<br>
<strong>Step 2:</strong> Set <em>Current vs Target</em> positions across Ten Lenses,
with hints and conflict flags.
</p>
</div>
""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
<div class="card">
<h3>Journey & Actions</h3>
<p class="desc">
Gap analysis, conflicts and priorities — then turn shifts into an
action log with owners, timelines and metrics.
</p>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rows loaded", len(df))
    k2.metric("Countries", df["country"].nunique() if "country" in df.columns else 0)
    k3.metric("Org types", df["org_type"].nunique() if "org_type" in df.columns else 0)
    k4.metric("Last updated", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    st.markdown("---")
    st.markdown("### How to use this explorer")
    st.markdown(
        """
1. **Explore** — get a feel for what other organisations are doing (by year, country, org type, scope).  
2. **Assess maturity** — use the six government themes (Uses, Data, Leadership, Culture, Tools, Skills) to agree where you are today.  
3. **Set tensions** — define Current vs Target on the Ten Lenses and see where the biggest shifts are.  
4. **Review the journey** — focus on the 3–5 most important shifts and check for conflicts with your maturity.  
5. **Capture actions** — export a small action log and plug it into your programme plan or OKRs.
"""
    )

# ====================================================
# 🔎 EXPLORE
# ====================================================
with tab_explore:
    with st.expander("Manage data (upload / reload)", expanded=False):
        uploaded = st.file_uploader(
            "Upload a strategies CSV", type=["csv"], key="uploader_main"
        )
        st.caption("CSV must include required columns (id, title, organisation, etc.).")
        st.markdown("---")

        csv_files_local = sorted(
            [f for f in glob.glob("*.csv") if os.path.isfile(f)]
        )
        if csv_files_local:
            default_csv_local = (
                "strategies.csv"
                if "strategies.csv" in csv_files_local
                else csv_files_local[0]
            )
            sel = st.selectbox(
                "Or select a CSV from directory",
                options=csv_files_local,
                index=csv_files_local.index(default_csv_local),
            )
            if st.button("Load selected file"):
                st.session_state.pop("uploaded_bytes", None)
                st.cache_data.clear()
                try:
                    df_new = load_data_from_path(
                        sel, file_md5(sel), APP_VERSION
                    )
                    df = df_new
                    st.success(
                        f"Loaded {sel} — {len(df)} rows (MD5 {file_md5(sel)[:12]}…)"
                    )
                except Exception as e:
                    st.error(f"⚠️ {e}")
        else:
            st.info("No CSV files found in directory. Upload one above.")

        cols_reload = st.columns(2)
        if cols_reload[0].button("Reload (clear cache)"):
            st.cache_data.clear()
            st.rerun()
        if cols_reload[1].button("Hard refresh (cache + state)"):
            st.cache_data.clear()
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

        if uploaded is not None:
            content = uploaded.read()
            try:
                df_new = load_data_from_bytes(content, bytes_md5(content), APP_VERSION)
                st.session_state["uploaded_bytes"] = content
                st.cache_data.clear()
                st.success(f"Loaded uploaded CSV — {len(df_new)} rows")
                st.rerun()
            except Exception as e:
                st.error(f"Upload error: {e}")

    with st.sidebar:
        st.subheader("Filters")
        years = sorted(y for y in df["year"].dropna().unique())
        if years:
            yr = st.slider(
                "Year range",
                int(min(years)),
                int(max(years)),
                (int(min(years)), int(max(years))),
            )
        else:
            yr = None

        org_types = sorted([v for v in df["org_type"].unique() if v != ""])
        org_type_sel = st.multiselect("Org type", org_types, default=org_types)

        countries = sorted([v for v in df["country"].unique() if v != ""])
        country_sel = st.multiselect("Country", countries, default=countries)

        scopes = sorted([v for v in df["scope"].unique() if v != ""])
        scope_sel = st.multiselect("Scope", scopes, default=scopes)

        q = st.text_input(
            "Search or describe strategies",
            placeholder="e.g. 'federated data strategy for small countries' or 'AI ethics framework'",
        )

        search_mode = st.radio(
            "Search mode",
            options=["Keyword", "AI semantic"],
            index=1 if emb_df is not None else 0,
            help="Keyword search looks for exact text matches. AI semantic search finds similar strategies by meaning.",
        )
        if emb_df is None and search_mode == "AI semantic":
            st.caption("Install 'sentence-transformers' to enable AI semantic search.")

    fdf = df.copy()
    if yr:
        fdf = fdf[fdf["year"].between(yr[0], yr[1])]
    if org_type_sel:
        fdf = fdf[fdf["org_type"].isin(org_type_sel)]
    if country_sel:
        fdf = fdf[fdf["country"].isin(country_sel)]
    if scope_sel:
        fdf = fdf[fdf["scope"].isin(scope_sel)]

    if q:
        if search_mode == "AI semantic" and emb_df is not None:
            st.caption("Semantic search active (AI-based similarity).")
            fdf = semantic_search(fdf, emb_df, q, top_k=100)
        else:
            fdf = simple_search(fdf, q)
        st.caption(f"{len(fdf)} strategies match your query.")

    if fdf.empty:
        st.warning(
            f"No strategies match the current filters and search term: **{q or '—'}**. "
            "Try broadening filters or removing the search text."
        )
    else:
        render_explore_charts(fdf)
        st.markdown("### Strategy details")
        for _, r in fdf.iterrows():
            year_str = int(r["year"]) if pd.notna(r["year"]) else "—"
            label = f"{r['title']} — {r['organisation']} ({year_str})"
            if "similarity" in fdf.columns:
                label += f"  [similarity {r.get('similarity', 0):.2f}]"
            with st.expander(label):
                st.write(r["summary"] or "_No summary provided._")
                meta = st.columns(4)
                meta[0].write(f"**Org type:** {r['org_type']}")
                meta[1].write(f"**Country:** {r['country']}")
                meta[2].write(f"**Scope:** {r['scope']}")
                meta[3].write(f"**Source:** {r['source']}")
                if r["link"]:
                    st.link_button("Open document", r["link"])

# ====================================================
# 👁️ LENSES (Maturity → Tensions)
# ====================================================
with tab_lenses:
    ensure_sessions()
    st.subheader("Lenses")

    st.caption(
        "First self-diagnose your organisation’s data maturity using the six themes from the "
        "Data Maturity Assessment for Government, then define where your strategy should sit "
        "on key tensions."
    )

    # ------- Section 1: Maturity -------
    st.markdown("### 1) Understand maturity (self-diagnose)")

    st.caption(
        "Based on the six themes in the Data Maturity Assessment for Government framework "
        "(Central Digital and Data Office)."
    )
    st.markdown(
        "[Open the framework in a new tab]"
        "(https://www.gov.uk/government/publications/data-maturity-assessment-for-government-framework/"
        "data-maturity-assessment-for-government-framework-html)"
    )

    cols_theme = st.columns(3)
    for i, (name, desc) in enumerate(MATURITY_THEMES):
        with cols_theme[i % 3]:
            current_val = st.session_state["_maturity_scores"].get(name, 3)
            st.session_state["_maturity_scores"][name] = st.slider(
                name,
                min_value=1,
                max_value=5,
                value=current_val,
                help=desc,
                format="%d",
                key=f"mat_{name}",
            )
            level_name = MATURITY_SCALE[st.session_state["_maturity_scores"][name]]
            st.caption(f"Level: {level_name}")

    # Overall maturity summary + gauge bar + radar
    m_scores = st.session_state["_maturity_scores"]
    m_avg = sum(m_scores.values()) / len(m_scores) if m_scores else 0
    current_level_name = maturity_label(m_avg)

    colA, colB = st.columns([1, 1])

    # LEFT: Gauge-style bar (0–5)
    with colA:
        st.metric("Overall maturity (average)", f"{m_avg:.1f} / 5")
        st.markdown(
            f"<span class='badge'>Level: {current_level_name}</span>",
            unsafe_allow_html=True,
        )

        gauge_df = pd.DataFrame({"Metric": ["Maturity"], "Score": [m_avg]})

        fig_bar = px.bar(
            gauge_df,
            x="Metric",
            y="Score",
            title="Overall maturity (1–5)",
            range_y=[0, 5],
        )
        fig_bar.update_traces(marker_color=PRIMARY)
        fig_bar.update_yaxes(
            tickvals=[1, 2, 3, 4, 5],
            ticktext=["Beginning", "Emerging", "Learning", "Developing", "Mastering"],
            title=None,
        )
        fig_bar.update_xaxes(title=None, showticklabels=False)
        fig_bar.update_layout(margin=dict(l=80, r=10, t=40, b=20))

        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown(
            "_Bar shows your current average position on the government maturity framework._"
        )

    # RIGHT: Radar (themes profile, 1–5 scale)
    with colB:
        dims_m = list(m_scores.keys())
        vals01 = [m_scores[d] / 5 for d in dims_m]
        figm = go.Figure()
        figm.add_trace(radar_trace(vals01, dims_m, "Maturity", opacity=0.6))
        figm.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickvals=[x / 5 for x in [1, 2, 3, 4, 5]],
                    ticktext=["1", "2", "3", "4", "5"],
                )
            ),
            title="Maturity profile across six themes (1–5 scale)",
        )
        st.plotly_chart(figm, use_container_width=True)

        st.markdown(
            "_Bar shows your overall level. Radar shows how that level is distributed across the six themes "
            "(Uses, Data, Leadership, Culture, Tools, Skills)._"
        )

    # Mini-report export for maturity
    maturity_rows = []
    for name, _ in MATURITY_THEMES:
        score = st.session_state["_maturity_scores"][name]
        maturity_rows.append(
            {
                "Theme": name,
                "Score (1–5)": score,
                "Level": MATURITY_SCALE[score],
            }
        )
    maturity_rows.append(
        {
            "Theme": "Overall (average)",
            "Score (1–5)": round(m_avg, 2),
            "Level": current_level_name,
        }
    )
    maturity_df = pd.DataFrame(maturity_rows)
    maturity_csv = maturity_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download maturity snapshot (CSV)",
        data=maturity_csv,
        file_name="maturity_snapshot.csv",
        mime="text/csv",
        help="Download your current maturity assessment for use in slide decks or programme plans.",
    )

    st.markdown("---")

    # ------- Section 2: Tensions -------
    st.markdown("### 2) Determine strategic tensions (current vs target)")
    st.caption(
        "For each lens, 0 = left label and 100 = right label. "
        "Hints and warnings adapt to your maturity profile."
    )

    colL, colR = st.columns(2)

    # Current profile
    with colL:
        st.markdown("#### Current")
        cols = st.columns(2)
        for i, (dim, left_lbl, right_lbl) in enumerate(AXES):
            with cols[i % 2]:
                current_val = st.session_state["_current_scores"].get(dim, 50)
                st.session_state["_current_scores"][dim] = st.slider(
                    f"{dim} (current)",
                    min_value=0,
                    max_value=100,
                    value=current_val,
                    format="%d%%",
                    help=f"{left_lbl} ←→ {right_lbl}",
                    key=f"cur_{dim}",
                )
                st.caption(
                    f"{left_lbl} ←── {st.session_state['_current_scores'][dim]}% → {right_lbl}"
                )

    # Target profile + hints/conflicts
    with colR:
        st.markdown("#### Target")
        cols = st.columns(2)
        for i, (dim, left_lbl, right_lbl) in enumerate(AXES):
            with cols[i % 2]:
                target_val = st.session_state["_target_scores"].get(dim, 50)
                st.session_state["_target_scores"][dim] = st.slider(
                    f"{dim} (target)",
                    min_value=0,
                    max_value=100,
                    value=target_val,
                    format="%d%%",
                    help=f"{left_lbl} ←→ {right_lbl}",
                    key=f"tgt_{dim}",
                )
                st.caption(
                    f"{left_lbl} ←── {st.session_state['_target_scores'][dim]}% → {right_lbl}"
                )

                hint = hint_for_lens(dim, m_avg, current_level_name)
                if hint:
                    st.markdown(
                        f"<div class='info-panel'><strong>Hint:</strong> {hint}</div>",
                        unsafe_allow_html=True,
                    )

                warn = conflict_for_target(
                    dim, st.session_state["_target_scores"][dim], m_avg
                )
                if warn:
                    st.markdown(
                        f"<div class='warn'>⚠️ {warn}</div>",
                        unsafe_allow_html=True,
                    )

    # Twin radar: current vs target
    dims = [a[0] for a in AXES]
    cur01 = [st.session_state["_current_scores"][d] / 100 for d in dims]
    tgt01 = [st.session_state["_target_scores"][d] / 100 for d in dims]
    fig = go.Figure()
    fig.add_trace(radar_trace(cur01, dims, "Current", opacity=0.6))
    fig.add_trace(radar_trace(tgt01, dims, "Target", opacity=0.5))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Current vs Target — strategic fingerprints",
    )
    st.plotly_chart(fig, use_container_width=True)

# ====================================================
# 🧭 JOURNEY
# ====================================================
with tab_journey:
    st.subheader("Journey — compare and prioritise")
    st.caption(
        "Signed change: negative = move toward LEFT label; positive = move toward RIGHT label. "
        "Conflicts highlight ambition that may exceed readiness."
    )

    ensure_sessions()
    dims = [a[0] for a in AXES]
    current = st.session_state.get("_current_scores", {d: 50 for d in dims})
    target = st.session_state.get("_target_scores", {d: 50 for d in dims})
    m_scores = st.session_state.get("_maturity_scores", {k: 3 for k, _ in MATURITY_THEMES})
    m_avg = sum(m_scores.values()) / len(m_scores) if m_scores else 0
    level_name = maturity_label(m_avg)

    rows = []
    for d, left_lbl, right_lbl in AXES:
        diff = target[d] - current[d]
        mag = abs(diff)
        direction = (
            f"→ **{right_lbl}**"
            if diff > 0
            else (f"→ **{left_lbl}**" if diff < 0 else "—")
        )
        conflict = conflict_for_target(d, target[d], m_avg)
        rows.append(
            {
                "Lens": d,
                "Current": current[d],
                "Target": target[d],
                "Change needed": diff,
                "Magnitude": mag,
                "Direction": direction,
                "Conflict": bool(conflict),
                "Conflict note": conflict or "",
            }
        )
    gap_df = pd.DataFrame(rows).sort_values(
        ["Conflict", "Magnitude"], ascending=[False, False]
    )

    # Narrative summary
    moves_left = sum(1 for v in gap_df["Change needed"] if v < 0)
    moves_right = sum(1 for v in gap_df["Change needed"] if v > 0)
    zero_moves = sum(1 for v in gap_df["Change needed"] if v == 0)

    st.markdown(
        f"**Summary:** At overall maturity level **{level_name}** (avg {m_avg:.1f}/5), "
        f"you are planning to move **{moves_left} lens(es) toward the left**, "
        f"**{moves_right} toward the right**, and leaving **{zero_moves} unchanged.**"
    )

    st.markdown("#### Gap by lens (conflicts first)")
    st.dataframe(
        gap_df[["Lens", "Current", "Target", "Change needed", "Direction", "Conflict"]],
        use_container_width=True,
    )

    # bar chart with colour by conflict
    color_series = gap_df["Conflict"].map({True: RED, False: PRIMARY})
    bar = px.bar(
        gap_df.sort_values("Change needed"),
        x="Change needed",
        y="Lens",
        orientation="h",
        title="Signed change needed (− move left • + move right)",
    )
    bar.data[0].marker.color = color_series
    st.plotly_chart(bar, use_container_width=True)

    # Priority list
    TOP_N = 3
    top = gap_df.head(TOP_N)
    if len(top):
        st.markdown(f"#### Priority shifts (top {TOP_N})")
        bullets = []
        for _, row in top.iterrows():
            d = row["Lens"]
            diff = row["Change needed"]
            note = row["Conflict note"]
            left_lbl = [a[1] for a in AXES if a[0] == d][0]
            right_lbl = [a[2] for a in AXES if a[0] == d][0]
            if diff > 0:
                line = f"- **{d}**: shift toward **{right_lbl}** (+{int(diff)} pts)"
            elif diff < 0:
                line = f"- **{d}**: shift toward **{left_lbl}** ({int(diff)} pts)"
            else:
                line = f"- **{d}**: no change"
            if note:
                line += f"  \n  <span class='warn'>⚠️ {note}</span>"
            bullets.append(line)
        st.markdown("\n".join(bullets), unsafe_allow_html=True)

        # Seed actions table for Actions tab
        actions_rows = []
        for i, (_, row) in enumerate(top.iterrows(), start=1):
            d = row["Lens"]
            diff = row["Change needed"]
            left_lbl = [a[1] for a in AXES if a[0] == d][0]
            right_lbl = [a[2] for a in AXES if a[0] == d][0]
            if diff > 0:
                direction = f"toward {right_lbl}"
            elif diff < 0:
                direction = f"toward {left_lbl}"
            else:
                direction = "no change"
            actions_rows.append(
                {
                    "Priority": i,
                    "Lens": d,
                    "Direction": direction,
                    "Owner": "",
                    "Timeline": "",
                    "Metric": "",
                    "Status": "",
                }
            )
        st.session_state["_actions_df"] = pd.DataFrame(actions_rows)
    else:
        st.info(
            "Current and target are identical — no change required. "
            "Adjust the sliders in the Lenses tab to see gaps."
        )

    st.markdown(
        "_Want to go deeper on coherence or pacing? See the **Strategy Kernel** and **Three Horizons** in the Resources tab._"
    )

# ====================================================
# ✅ ACTIONS & EXPORT
# ====================================================
with tab_actions:
    st.subheader("Actions & Export")
    st.caption(
        "Turn your top priority shifts into an action log. "
        "Assign owners, timelines and metrics, then export to CSV."
    )

    ensure_sessions()
    actions_df = st.session_state.get("_actions_df", pd.DataFrame())

    if actions_df.empty:
        st.info(
            "No priority shifts have been generated yet. "
            "Go to the Journey tab to calculate gaps and priorities."
        )
    else:
        st.markdown("### Action log (editable)")
        edited = st.data_editor(
            actions_df,
            num_rows="dynamic",
            use_container_width=True,
            key="actions_editor",
        )
        st.session_state["_actions_df"] = edited

        csv_bytes = edited.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download actions as CSV",
            data=csv_bytes,
            file_name="data_strategy_actions.csv",
            mime="text/csv",
        )

        st.markdown(
            "> Tip: paste this table into your programme plan or OKRs to track progress."
        )
# ---------------------------------------------------
# Public Sector Data Strategy Explorer
# ---------------------------------------------------
import os
import glob
import io
import time
import hashlib
import base64

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# --- Optional: semantic embeddings (AI search) ---
try:
    from sentence_transformers import SentenceTransformer

    HAS_EMBED = True
except Exception:
    HAS_EMBED = False
    SentenceTransformer = None

APP_VERSION = "v3.1 – 2025-11-13"

# ---------------- PAGE CONFIG & THEME ----------------
st.set_page_config(
    page_title="Public Sector Data Strategy Explorer",
    layout="wide",
)

PRIMARY = "#1d70b8"  # GOV-style blue
DARK = "#0b0c0c"     # near-black
LIGHT = "#f3f2f1"    # light grey
ACCENT = "#28a197"   # teal
RED = "#d4351c"

st.markdown(
    f"""
<style>
/* Header bar */
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

/* Body */
body, .block-container {{
  color:{DARK};
  font-family:"Noto Sans","Helvetica Neue",Helvetica,Arial,sans-serif;
}}
a, a:visited {{ color:{PRIMARY}; }}
a:hover {{ color:#003078; }}

/* Cards */
.card {{
  background:white; border:1px solid #e5e5e5; border-radius:8px;
  padding:16px; box-shadow:0 1px 2px rgba(0,0,0,0.03); height:100%;
}}
.card h3 {{ margin-top:0; }}
.card .desc {{ color:#505a5f; font-size:0.95rem; }}

/* Info / warning panels */
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
.kv {{
  display:inline-block; padding:2px 6px; border-radius:4px;
  background:{LIGHT}; border:1px solid #e5e5e5; margin-right:6px;
}}

/* Buttons */
.stButton>button {{
  background:{PRIMARY}; color:white; border-radius:0; border:none; font-weight:600;
}}
.stButton>button:hover {{ background:#003078; }}

/* Footer */
.footer {{
  color:#505a5f; font-size:0.85rem; text-align:center; margin-top:1.2rem;
}}
</style>
<div class="header-bar">
  <h1>Public Sector Data Strategy Explorer</h1>
  <div class="sub">Design better data strategies, faster — balance tensions, align leadership, and plan change.</div>
</div>
""",
    unsafe_allow_html=True,
)

# Plotly theme
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

# ---------------- DATA LOADING ----------------
REQUIRED = [
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


def bytes_md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


@st.cache_data(show_spinner=False)
def load_data_from_path(path: str, file_hash: str, app_version: str):
    df = pd.read_csv(path).fillna("")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_data_from_bytes(content: bytes, file_hash: str, app_version: str):
    df = pd.read_csv(io.BytesIO(content)).fillna("")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    return df


# --- Load initial CSV (default or uploaded) ---
csv_files = sorted([f for f in glob.glob("*.csv") if os.path.isfile(f)])
default_csv = (
    "strategies.csv"
    if "strategies.csv" in csv_files
    else (csv_files[0] if csv_files else None)
)

if "uploaded_bytes" in st.session_state:
    content = st.session_state["uploaded_bytes"]
    df = load_data_from_bytes(content, bytes_md5(content), APP_VERSION)
elif default_csv:
    df = load_data_from_path(default_csv, file_md5(default_csv), APP_VERSION)
else:
    df = pd.DataFrame(columns=REQUIRED)

# ---------------- LENSES & MATURITY ----------------

# Government data maturity themes (CDDO)
MATURITY_THEMES = [
    (
        "Uses",
        "How you get value out of data. Making decisions, evidencing impact, improving services.",
    ),
    (
        "Data",
        "Technical aspects of managing data as an asset: collection, quality, cataloguing, interoperability.",
    ),
    (
        "Leadership",
        "How senior and business leaders engage with data: strategy, responsibility, oversight, investment.",
    ),
    (
        "Culture",
        "Attitudes to data across the organisation: awareness, openness, security, responsibility.",
    ),
    (
        "Tools",
        "The systems and tools you use to store, share and work with data.",
    ),
    (
        "Skills",
        "Data and analytical literacy across the organisation, including how people build and maintain those skills.",
    ),
]

# Official government levels 1–5
MATURITY_SCALE = {
    1: "Beginning",
    2: "Emerging",
    3: "Learning",
    4: "Developing",
    5: "Mastering",
}


def maturity_label(avg: float) -> str:
    """
    Map the average (1–5) to the nearest official maturity level.
    """
    idx = int(round(avg))
    idx = max(1, min(5, idx))
    return MATURITY_SCALE[idx]


# Ten Lenses
AXES = [
    ("Abstraction Level", "Conceptual", "Logical / Physical"),
    ("Adaptability", "Living", "Fixed"),
    ("Ambition", "Essential", "Transformational"),
    ("Coverage", "Horizontal", "Use-case-based"),
    ("Governance Structure", "Ecosystem / Federated", "Centralised"),
    ("Orientation", "Technology-focused", "Value-focused"),
    ("Motivation", "Compliance-driven", "Innovation-driven"),
    ("Access Philosophy", "Data-democratised", "Controlled access"),
    ("Delivery Mode", "Incremental", "Big Bang"),
    ("Decision Model", "Data-informed", "Data-driven"),
]
DIMENSIONS = [a[0] for a in AXES]


def radar_trace(values01, dims, name, opacity=0.6, fill=True):
    r = list(values01) + [values01[0]]
    t = list(dims) + [dims[0]]
    return go.Scatterpolar(
        r=r, theta=t, name=name, fill="toself" if fill else None, opacity=opacity
    )


def ensure_sessions():
    if "_maturity_scores" not in st.session_state:
        st.session_state["_maturity_scores"] = {k: 3 for k, _ in MATURITY_THEMES}
    if "_current_scores" not in st.session_state:
        st.session_state["_current_scores"] = {d: 50 for d in DIMENSIONS}
    if "_target_scores" not in st.session_state:
        st.session_state["_target_scores"] = {d: 50 for d in DIMENSIONS}
    if "_actions_df" not in st.session_state:
        st.session_state["_actions_df"] = pd.DataFrame(
            columns=["Priority", "Lens", "Direction", "Owner", "Timeline", "Metric", "Status"]
        )


# ---------------- SEARCH HELPERS ----------------
def simple_search(df_in: pd.DataFrame, query: str) -> pd.DataFrame:
    """
    Simple case-insensitive search over key text columns.
    """
    if not query:
        return df_in

    text_cols = [c for c in ["title", "organisation", "summary", "scope"] if c in df_in.columns]
    if not text_cols:
        return df_in

    text = df_in[text_cols[0]].astype(str)
    for col in text_cols[1:]:
        text = text + " " + df_in[col].astype(str)

    mask = text.str.contains(query, case=False, na=False)
    return df_in[mask]


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    if not HAS_EMBED:
        return None
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data(show_spinner=False)
def compute_strategy_embeddings(df_in: pd.DataFrame, app_version: str):
    if not HAS_EMBED:
        return None
    model = get_embedding_model()
    if model is None:
        return None

    text_cols = [c for c in ["title", "organisation", "summary", "scope", "country"] if c in df_in.columns]
    if not text_cols:
        return None

    texts = df_in[text_cols[0]].astype(str)
    for col in text_cols[1:]:
        texts = texts + " " + df_in[col].astype(str)

    embeddings = model.encode(
        texts.tolist(),
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    emb_df = pd.DataFrame(embeddings, index=df_in.index)
    return emb_df


def semantic_search(fdf: pd.DataFrame, emb_df: pd.DataFrame, query: str, top_k: int = 100) -> pd.DataFrame:
    """
    Semantic search using pre-computed embeddings.
    Respects current filtered subset fdf by aligning on index.
    """
    if not query or emb_df is None or fdf.empty:
        return fdf

    model = get_embedding_model()
    if model is None:
        return fdf

    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    sub_emb = emb_df.loc[fdf.index].values
    sims = sub_emb @ q_emb

    order = np.argsort(-sims)
    order = order[: min(top_k, len(order))]
    result = fdf.iloc[order].copy()
    result["similarity"] = sims[order]
    return result


emb_df = compute_strategy_embeddings(df, APP_VERSION)

# ---------------- HINTS & CONFLICTS ----------------
def hint_for_lens(lens_name, maturity_avg, maturity_level_name=None):
    """
    Give contextual hints based on the organisation's overall maturity level.
    Uses government levels: Beginning, Emerging, Learning, Developing, Mastering.
    """
    level = maturity_level_name or maturity_label(maturity_avg)
    low = level in ("Beginning", "Emerging")
    mid = level in ("Learning", "Developing")
    high = level == "Mastering"

    if lens_name == "Governance Structure":
        if low:
            return "At Beginning/Emerging, stronger central coordination usually works best before moving to federated models."
        if mid:
            return "At Learning/Developing, you can gradually federate – keep common standards and shared services."
        if high:
            return "At Mastering, federation can unlock autonomy – but guard against fragmentation with shared guardrails."
    if lens_name == "Delivery Mode":
        if low:
            return "Favour incremental delivery to build confidence and reduce risk – avoid a single big-bang change."
        if mid:
            return "Blend incremental delivery with a few larger change packages where foundations are solid."
        if high:
            return "At Mastering, big-bang change is possible – but only with strong programme discipline and clear benefits."
    if lens_name == "Access Philosophy":
        if low:
            return "Start with role-based access to a small number of trusted datasets before opening up more widely."
        if mid:
            return "Broaden access with good catalogue/search – keep tight controls around sensitive domains."
        if high:
            return "Push democratisation further – but make sure data protection and audit trails stay robust."
    if lens_name == "Decision Model":
        if low:
            return "Data-informed decisions with clear human oversight are safest while skills and quality are still building."
        if mid:
            return "Increase automation in low-risk areas – keep humans in the loop for high-impact decisions."
        if high:
            return "Mastering orgs can rely more on data-driven decisions – but need strong monitoring and fallback plans."
    if lens_name == "Motivation":
        if low:
            return "Keep compliance at the core while you pilot innovation in tightly scoped sandboxes."
        if mid:
            return "Balance compliance and innovation – use proof-of-concepts to justify broader change."
        if high:
            return "At Mastering, innovation and compliance can reinforce each other via strong governance by design."
    if lens_name == "Ambition":
        if low:
            return "Focus on essentials – data quality, governance, core platforms – before promising transformational change."
        if mid:
            return "You can mix foundational work with some transformational strands where benefits are clear."
        if high:
            return "Aim for transformational impact – but keep benefits and operating model changes clearly articulated."
    if lens_name == "Coverage":
        if low:
            return "Use a few high-impact use-cases to prove value while you build broader capabilities."
        if mid:
            return "Begin to spread capabilities horizontally to avoid islands of excellence."
        if high:
            return "Horizontal coverage makes sense – but choose a few flagship use-cases to anchor the narrative."
    if lens_name == "Orientation":
        if low:
            return "Platform and tooling investments will dominate early – link them clearly to outcomes."
        if mid:
            return "Balance platform work with visible value – avoid tech for tech’s sake."
        if high:
            return "Keep value firmly in the lead, with platforms treated as enablers rather than ends."
    if lens_name == "Adaptability":
        if low:
            return "Keep a stable core with a small living layer – too much churn can confuse people."
        if mid:
            return "Treat the strategy as living – schedule periodic reviews and small course corrections."
        if high:
            return "Mastering orgs can iterate often – just make sure changes are well-governed and communicated."
    if lens_name == "Abstraction Level":
        if low:
            return "Keep the strategy concise and vision-led, but quickly translate into practical roadmaps and controls."
        if mid:
            return "Balance vision with enough logical detail to guide delivery teams."
        if high:
            return "You can afford a more detailed logical/physical description – but avoid over-specifying too early."

    return ""


def conflict_for_target(lens_name, target_score, maturity_avg):
    """
    Flag misalignments between maturity and ambitious targets.
    target_score is 0–100 toward right label.
    """
    level = maturity_label(maturity_avg)
    low = level in ("Beginning", "Emerging")
    highish = level in ("Developing", "Mastering")  # treat Learning as middle

    # Low maturity: warn if target is very ambitious/risky
    if low:
        if lens_name == "Delivery Mode" and target_score >= 70:
            return "Big-bang at Beginning/Emerging maturity is high risk — consider phased delivery."
        if lens_name == "Governance Structure" and target_score <= 30:
            return "Federated at low maturity can fragment standards — strengthen central controls first."
        if lens_name == "Access Philosophy" and target_score <= 30:
            return "Wide democratisation needs strong basics — start with controlled, role-based access."
        if lens_name == "Decision Model" and target_score >= 70:
            return "Highly data-driven decisions need robust data quality, monitoring and skills."
        if lens_name == "Motivation" and target_score >= 70:
            return "Innovation-first without guardrails can raise risk — keep compliance in the loop."

    # High-ish maturity: warn if overly conservative
    if highish:
        if lens_name == "Delivery Mode" and target_score <= 30:
            return "At Developing/Mastering, being too incremental may under-deliver benefits."
        if lens_name == "Governance Structure" and target_score >= 80:
            return "Highly centralised models may slow teams at higher maturity — consider selective federation."
        if lens_name == "Access Philosophy" and target_score >= 80:
            return "Excessive control may limit value realisation — revisit openness where safe."

    return None


# ---------------- EXPLORE CHARTS ----------------
def render_explore_charts(fdf: pd.DataFrame):
    st.markdown("## Explore — landscape & patterns")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Strategies", len(fdf))
    k2.metric("Countries", fdf["country"].nunique() if "country" in fdf.columns else 0)
    k3.metric("Org types", fdf["org_type"].nunique() if "org_type" in fdf.columns else 0)
    if "year" in fdf.columns and fdf["year"].notna().any():
        k4.metric("Year span", f"{int(fdf['year'].min())}–{int(fdf['year'].max())}")
    else:
        k4.metric("Year span", "—")

    st.markdown("---")
    c1, c2 = st.columns(2)

    if "year" in fdf.columns and fdf["year"].notna().any():
        fig_hist = px.histogram(
            fdf[fdf["year"].notna()],
            x="year",
            color="scope" if "scope" in fdf.columns else None,
            nbins=max(10, min(40, fdf["year"].nunique())),
            title="Strategies by year",
        )
        fig_hist.update_layout(bargap=0.05)
        c1.plotly_chart(fig_hist, use_container_width=True)
    else:
        c1.info("No numeric 'year' values to chart. Check your CSV or filters.")

    if "org_type" in fdf.columns and fdf["org_type"].notna().any():
        top_org = (
            fdf.groupby("org_type")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        fig_org = px.bar(
            top_org,
            x="org_type",
            y="count",
            title="Composition by organisation type",
        )
        fig_org.update_xaxes(title=None, tickangle=20)
        c2.plotly_chart(fig_org, use_container_width=True)
    else:
        c2.info("No 'org_type' values to chart.")

    st.markdown("---")
    c3, c4 = st.columns(2)

    if all(col in fdf.columns for col in ["country", "org_type"]):
        if not fdf.empty:
            fig_tree = px.treemap(
                fdf.assign(_value=1),
                path=["country", "org_type", "organisation"],
                values="_value",
                title="Landscape by country → org type → organisation",
            )
            c3.plotly_chart(fig_tree, use_container_width=True)
        else:
            c3.info("No data for treemap.")
    else:
        c3.info("Need 'country' and 'org_type' columns for treemap.")

    if "country" in fdf.columns and fdf["country"].notna().any():
        by_ctry = fdf.groupby("country").size().reset_index(name="count")
        if not by_ctry.empty:
            fig_map = px.choropleth(
                by_ctry,
                locations="country",
                locationmode="country names",
                color="count",
                title="Global distribution of strategies (by country)",
                color_continuous_scale="Blues",
            )
            c4.plotly_chart(fig_map, use_container_width=True)
        else:
            c4.info("No country counts to map.")
    else:
        c4.info("No 'country' values to map.")

    st.markdown("---")
    c5, c6 = st.columns(2)
    if all(col in fdf.columns for col in ["country", "org_type"]):
        top_ctrys = (
            fdf.groupby("country").size().sort_values(ascending=False).head(12).index.tolist()
        )
        sub = fdf[fdf["country"].isin(top_ctrys)]
        if not sub.empty:
            fig_stack = px.bar(
                sub,
                x="country",
                color="org_type",
                title="Top countries by strategies (stacked by org type)",
            )
            fig_stack.update_xaxes(title=None)
            c5.plotly_chart(fig_stack, use_container_width=True)
        else:
            c5.info("No data for stacked bar.")
    else:
        c5.info("Need 'country' and 'org_type' for stacked bar.")

    needed = ["year", "organisation", "title"]
    if all(col in fdf.columns for col in needed) and fdf["year"].notna().any():
        sub = fdf[fdf["year"].notna()].copy()
        fig_scatter = px.scatter(
            sub,
            x="year",
            y="organisation",
            color="country" if "country" in sub.columns else None,
            hover_data=["title", "country", "scope"]
            if "scope" in sub.columns
            else ["title"],
            title="Timeline of strategies by organisation",
        )
        c6.plotly_chart(fig_scatter, use_container_width=True)
    else:
        c6.info("Need 'year', 'organisation', and 'title' columns for timeline.")

    st.markdown("---")
    if "scope" in fdf.columns and fdf["scope"].notna().any():
        by_scope = fdf["scope"].value_counts().reset_index()
        by_scope.columns = ["scope", "count"]
        fig_scope = px.pie(
            by_scope, names="scope", values="count", title="Strategy scope breakdown"
        )
        st.plotly_chart(fig_scope, use_container_width=True)


# ---------------- TABS SETUP ----------------
ensure_sessions()
tab_home, tab_explore, tab_lenses, tab_journey, tab_actions, tab_resources, tab_about = st.tabs(
    ["Home", "Explore", "Lenses", "Journey", "Actions & Export", "Resources", "About"]
)

# ====================================================
# 🏠 HOME
# ====================================================
with tab_home:
    st.markdown(
        """
<div class="info-panel">
<strong>Quick start:</strong> Begin with <strong>Lenses → Maturity</strong> to understand your current readiness,
then set your strategic tensions and review the Journey.
</div>
""",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
<div class="card">
<h3>Explore</h3>
<p class="desc">
See patterns in real strategies — by year, country, organisation type and scope.
Maps, timelines and composition views give fast context.
</p>
</div>
""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
<div class="card">
<h3>Lenses</h3>
<p class="desc">
<strong>Step 1:</strong> Self-diagnose maturity across six government themes.<br>
<strong>Step 2:</strong> Set <em>Current vs Target</em> positions across Ten Lenses,
with hints and conflict flags.
</p>
</div>
""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
<div class="card">
<h3>Journey & Actions</h3>
<p class="desc">
Gap analysis, conflicts and priorities — then turn shifts into an
action log with owners, timelines and metrics.
</p>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rows loaded", len(df))
    k2.metric("Countries", df["country"].nunique() if "country" in df.columns else 0)
    k3.metric("Org types", df["org_type"].nunique() if "org_type" in df.columns else 0)
    k4.metric("Last updated", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    st.markdown("---")
    st.markdown("### How to use this explorer")
    st.markdown(
        """
1. **Explore** — get a feel for what other organisations are doing (by year, country, org type, scope).  
2. **Assess maturity** — use the six government themes (Uses, Data, Leadership, Culture, Tools, Skills) to agree where you are today.  
3. **Set tensions** — define Current vs Target on the Ten Lenses and see where the biggest shifts are.  
4. **Review the journey** — focus on the 3–5 most important shifts and check for conflicts with your maturity.  
5. **Capture actions** — export a small action log and plug it into your programme plan or OKRs.
"""
    )

# ====================================================
# 🔎 EXPLORE
# ====================================================
with tab_explore:
    with st.expander("Manage data (upload / reload)", expanded=False):
        uploaded = st.file_uploader(
            "Upload a strategies CSV", type=["csv"], key="uploader_main"
        )
        st.caption("CSV must include required columns (id, title, organisation, etc.).")
        st.markdown("---")

        csv_files_local = sorted(
            [f for f in glob.glob("*.csv") if os.path.isfile(f)]
        )
        if csv_files_local:
            default_csv_local = (
                "strategies.csv"
                if "strategies.csv" in csv_files_local
                else csv_files_local[0]
            )
            sel = st.selectbox(
                "Or select a CSV from directory",
                options=csv_files_local,
                index=csv_files_local.index(default_csv_local),
            )
            if st.button("Load selected file"):
                st.session_state.pop("uploaded_bytes", None)
                st.cache_data.clear()
                try:
                    df_new = load_data_from_path(
                        sel, file_md5(sel), APP_VERSION
                    )
                    df = df_new
                    st.success(
                        f"Loaded {sel} — {len(df)} rows (MD5 {file_md5(sel)[:12]}…)"
                    )
                except Exception as e:
                    st.error(f"⚠️ {e}")
        else:
            st.info("No CSV files found in directory. Upload one above.")

        cols_reload = st.columns(2)
        if cols_reload[0].button("Reload (clear cache)"):
            st.cache_data.clear()
            st.rerun()
        if cols_reload[1].button("Hard refresh (cache + state)"):
            st.cache_data.clear()
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

        if uploaded is not None:
            content = uploaded.read()
            try:
                df_new = load_data_from_bytes(content, bytes_md5(content), APP_VERSION)
                st.session_state["uploaded_bytes"] = content
                st.cache_data.clear()
                st.success(f"Loaded uploaded CSV — {len(df_new)} rows")
                st.rerun()
            except Exception as e:
                st.error(f"Upload error: {e}")

    with st.sidebar:
        st.subheader("Filters")
        years = sorted(y for y in df["year"].dropna().unique())
        if years:
            yr = st.slider(
                "Year range",
                int(min(years)),
                int(max(years)),
                (int(min(years)), int(max(years))),
            )
        else:
            yr = None

        org_types = sorted([v for v in df["org_type"].unique() if v != ""])
        org_type_sel = st.multiselect("Org type", org_types, default=org_types)

        countries = sorted([v for v in df["country"].unique() if v != ""])
        country_sel = st.multiselect("Country", countries, default=countries)

        scopes = sorted([v for v in df["scope"].unique() if v != ""])
        scope_sel = st.multiselect("Scope", scopes, default=scopes)

        q = st.text_input(
            "Search or describe strategies",
            placeholder="e.g. 'federated data strategy for small countries' or 'AI ethics framework'",
        )

        search_mode = st.radio(
            "Search mode",
            options=["Keyword", "AI semantic"],
            index=1 if emb_df is not None else 0,
            help="Keyword search looks for exact text matches. AI semantic search finds similar strategies by meaning.",
        )
        if emb_df is None and search_mode == "AI semantic":
            st.caption("Install 'sentence-transformers' to enable AI semantic search.")

    fdf = df.copy()
    if yr:
        fdf = fdf[fdf["year"].between(yr[0], yr[1])]
    if org_type_sel:
        fdf = fdf[fdf["org_type"].isin(org_type_sel)]
    if country_sel:
        fdf = fdf[fdf["country"].isin(country_sel)]
    if scope_sel:
        fdf = fdf[fdf["scope"].isin(scope_sel)]

    if q:
        if search_mode == "AI semantic" and emb_df is not None:
            st.caption("Semantic search active (AI-based similarity).")
            fdf = semantic_search(fdf, emb_df, q, top_k=100)
        else:
            fdf = simple_search(fdf, q)
        st.caption(f"{len(fdf)} strategies match your query.")

    if fdf.empty:
        st.warning(
            f"No strategies match the current filters and search term: **{q or '—'}**. "
            "Try broadening filters or removing the search text."
        )
    else:
        render_explore_charts(fdf)
        st.markdown("### Strategy details")
        for _, r in fdf.iterrows():
            year_str = int(r["year"]) if pd.notna(r["year"]) else "—"
            label = f"{r['title']} — {r['organisation']} ({year_str})"
            if "similarity" in fdf.columns:
                label += f"  [similarity {r.get('similarity', 0):.2f}]"
            with st.expander(label):
                st.write(r["summary"] or "_No summary provided._")
                meta = st.columns(4)
                meta[0].write(f"**Org type:** {r['org_type']}")
                meta[1].write(f"**Country:** {r['country']}")
                meta[2].write(f"**Scope:** {r['scope']}")
                meta[3].write(f"**Source:** {r['source']}")
                if r["link"]:
                    st.link_button("Open document", r["link"])

# ====================================================
# 👁️ LENSES (Maturity → Tensions)
# ====================================================
with tab_lenses:
    ensure_sessions()
    st.subheader("Lenses")

    st.caption(
        "First self-diagnose your organisation’s data maturity using the six themes from the "
        "Data Maturity Assessment for Government, then define where your strategy should sit "
        "on key tensions."
    )

    # ------- Section 1: Maturity -------
    st.markdown("### 1) Understand maturity (self-diagnose)")

    st.caption(
        "Based on the six themes in the Data Maturity Assessment for Government framework "
        "(Central Digital and Data Office)."
    )
    st.markdown(
        "[Open the framework in a new tab]"
        "(https://www.gov.uk/government/publications/data-maturity-assessment-for-government-framework/"
        "data-maturity-assessment-for-government-framework-html)"
    )

    cols_theme = st.columns(3)
    for i, (name, desc) in enumerate(MATURITY_THEMES):
        with cols_theme[i % 3]:
            current_val = st.session_state["_maturity_scores"].get(name, 3)
            st.session_state["_maturity_scores"][name] = st.slider(
                name,
                min_value=1,
                max_value=5,
                value=current_val,
                help=desc,
                format="%d",
                key=f"mat_{name}",
            )
            level_name = MATURITY_SCALE[st.session_state["_maturity_scores"][name]]
            st.caption(f"Level: {level_name}")

    # Overall maturity summary + gauge bar + radar
    m_scores = st.session_state["_maturity_scores"]
    m_avg = sum(m_scores.values()) / len(m_scores) if m_scores else 0
    current_level_name = maturity_label(m_avg)

    colA, colB = st.columns([1, 1])

    # LEFT: Gauge-style bar (0–5)
    with colA:
        st.metric("Overall maturity (average)", f"{m_avg:.1f} / 5")
        st.markdown(
            f"<span class='badge'>Level: {current_level_name}</span>",
            unsafe_allow_html=True,
        )

        gauge_df = pd.DataFrame({"Metric": ["Maturity"], "Score": [m_avg]})

        fig_bar = px.bar(
            gauge_df,
            x="Metric",
            y="Score",
            title="Overall maturity (1–5)",
            range_y=[0, 5],
        )
        fig_bar.update_traces(marker_color=PRIMARY)
        fig_bar.update_yaxes(
            tickvals=[1, 2, 3, 4, 5],
            ticktext=["Beginning", "Emerging", "Learning", "Developing", "Mastering"],
            title=None,
        )
        fig_bar.update_xaxes(title=None, showticklabels=False)
        fig_bar.update_layout(margin=dict(l=80, r=10, t=40, b=20))

        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown(
            "_Bar shows your current average position on the government maturity framework._"
        )

    # RIGHT: Radar (themes profile, 1–5 scale)
    with colB:
        dims_m = list(m_scores.keys())
        vals01 = [m_scores[d] / 5 for d in dims_m]
        figm = go.Figure()
        figm.add_trace(radar_trace(vals01, dims_m, "Maturity", opacity=0.6))
        figm.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickvals=[x / 5 for x in [1, 2, 3, 4, 5]],
                    ticktext=["1", "2", "3", "4", "5"],
                )
            ),
            title="Maturity profile across six themes (1–5 scale)",
        )
        st.plotly_chart(figm, use_container_width=True)

        st.markdown(
            "_Bar shows your overall level. Radar shows how that level is distributed across the six themes "
            "(Uses, Data, Leadership, Culture, Tools, Skills)._"
        )

    # Mini-report export for maturity
    maturity_rows = []
    for name, _ in MATURITY_THEMES:
        score = st.session_state["_maturity_scores"][name]
        maturity_rows.append(
            {
                "Theme": name,
                "Score (1–5)": score,
                "Level": MATURITY_SCALE[score],
            }
        )
    maturity_rows.append(
        {
            "Theme": "Overall (average)",
            "Score (1–5)": round(m_avg, 2),
            "Level": current_level_name,
        }
    )
    maturity_df = pd.DataFrame(maturity_rows)
    maturity_csv = maturity_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download maturity snapshot (CSV)",
        data=maturity_csv,
        file_name="maturity_snapshot.csv",
        mime="text/csv",
        help="Download your current maturity assessment for use in slide decks or programme plans.",
    )

    st.markdown("---")

    # ------- Section 2: Tensions -------
    st.markdown("### 2) Determine strategic tensions (current vs target)")
    st.caption(
        "For each lens, 0 = left label and 100 = right label. "
        "Hints and warnings adapt to your maturity profile."
    )

    colL, colR = st.columns(2)

    # Current profile
    with colL:
        st.markdown("#### Current")
        cols = st.columns(2)
        for i, (dim, left_lbl, right_lbl) in enumerate(AXES):
            with cols[i % 2]:
                current_val = st.session_state["_current_scores"].get(dim, 50)
                st.session_state["_current_scores"][dim] = st.slider(
                    f"{dim} (current)",
                    min_value=0,
                    max_value=100,
                    value=current_val,
                    format="%d%%",
                    help=f"{left_lbl} ←→ {right_lbl}",
                    key=f"cur_{dim}",
                )
                st.caption(
                    f"{left_lbl} ←── {st.session_state['_current_scores'][dim]}% → {right_lbl}"
                )

    # Target profile + hints/conflicts
    with colR:
        st.markdown("#### Target")
        cols = st.columns(2)
        for i, (dim, left_lbl, right_lbl) in enumerate(AXES):
            with cols[i % 2]:
                target_val = st.session_state["_target_scores"].get(dim, 50)
                st.session_state["_target_scores"][dim] = st.slider(
                    f"{dim} (target)",
                    min_value=0,
                    max_value=100,
                    value=target_val,
                    format="%d%%",
                    help=f"{left_lbl} ←→ {right_lbl}",
                    key=f"tgt_{dim}",
                )
                st.caption(
                    f"{left_lbl} ←── {st.session_state['_target_scores'][dim]}% → {right_lbl}"
                )

                hint = hint_for_lens(dim, m_avg, current_level_name)
                if hint:
                    st.markdown(
                        f"<div class='info-panel'><strong>Hint:</strong> {hint}</div>",
                        unsafe_allow_html=True,
                    )

                warn = conflict_for_target(
                    dim, st.session_state["_target_scores"][dim], m_avg
                )
                if warn:
                    st.markdown(
                        f"<div class='warn'>⚠️ {warn}</div>",
                        unsafe_allow_html=True,
                    )

    # Twin radar: current vs target
    dims = [a[0] for a in AXES]
    cur01 = [st.session_state["_current_scores"][d] / 100 for d in dims]
    tgt01 = [st.session_state["_target_scores"][d] / 100 for d in dims]
    fig = go.Figure()
    fig.add_trace(radar_trace(cur01, dims, "Current", opacity=0.6))
    fig.add_trace(radar_trace(tgt01, dims, "Target", opacity=0.5))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Current vs Target — strategic fingerprints",
    )
    st.plotly_chart(fig, use_container_width=True)

# ====================================================
# 🧭 JOURNEY
# ====================================================
with tab_journey:
    st.subheader("Journey — compare and prioritise")
    st.caption(
        "Signed change: negative = move toward LEFT label; positive = move toward RIGHT label. "
        "Conflicts highlight ambition that may exceed readiness."
    )

    ensure_sessions()
    dims = [a[0] for a in AXES]
    current = st.session_state.get("_current_scores", {d: 50 for d in dims})
    target = st.session_state.get("_target_scores", {d: 50 for d in dims})
    m_scores = st.session_state.get("_maturity_scores", {k: 3 for k, _ in MATURITY_THEMES})
    m_avg = sum(m_scores.values()) / len(m_scores) if m_scores else 0
    level_name = maturity_label(m_avg)

    rows = []
    for d, left_lbl, right_lbl in AXES:
        diff = target[d] - current[d]
        mag = abs(diff)
        direction = (
            f"→ **{right_lbl}**"
            if diff > 0
            else (f"→ **{left_lbl}**" if diff < 0 else "—")
        )
        conflict = conflict_for_target(d, target[d], m_avg)
        rows.append(
            {
                "Lens": d,
                "Current": current[d],
                "Target": target[d],
                "Change needed": diff,
                "Magnitude": mag,
                "Direction": direction,
                "Conflict": bool(conflict),
                "Conflict note": conflict or "",
            }
        )
    gap_df = pd.DataFrame(rows).sort_values(
        ["Conflict", "Magnitude"], ascending=[False, False]
    )

    # Narrative summary
    moves_left = sum(1 for v in gap_df["Change needed"] if v < 0)
    moves_right = sum(1 for v in gap_df["Change needed"] if v > 0)
    zero_moves = sum(1 for v in gap_df["Change needed"] if v == 0)

    st.markdown(
        f"**Summary:** At overall maturity level **{level_name}** (avg {m_avg:.1f}/5), "
        f"you are planning to move **{moves_left} lens(es) toward the left**, "
        f"**{moves_right} toward the right**, and leaving **{zero_moves} unchanged.**"
    )

    st.markdown("#### Gap by lens (conflicts first)")
    st.dataframe(
        gap_df[["Lens", "Current", "Target", "Change needed", "Direction", "Conflict"]],
        use_container_width=True,
    )

    # bar chart with colour by conflict
    color_series = gap_df["Conflict"].map({True: RED, False: PRIMARY})
    bar = px.bar(
        gap_df.sort_values("Change needed"),
        x="Change needed",
        y="Lens",
        orientation="h",
        title="Signed change needed (− move left • + move right)",
    )
    bar.data[0].marker.color = color_series
    st.plotly_chart(bar, use_container_width=True)

    # Priority list
    TOP_N = 3
    top = gap_df.head(TOP_N)
    if len(top):
        st.markdown(f"#### Priority shifts (top {TOP_N})")
        bullets = []
        for _, row in top.iterrows():
            d = row["Lens"]
            diff = row["Change needed"]
            note = row["Conflict note"]
            left_lbl = [a[1] for a in AXES if a[0] == d][0]
            right_lbl = [a[2] for a in AXES if a[0] == d][0]
            if diff > 0:
                line = f"- **{d}**: shift toward **{right_lbl}** (+{int(diff)} pts)"
            elif diff < 0:
                line = f"- **{d}**: shift toward **{left_lbl}** ({int(diff)} pts)"
            else:
                line = f"- **{d}**: no change"
            if note:
                line += f"  \n  <span class='warn'>⚠️ {note}</span>"
            bullets.append(line)
        st.markdown("\n".join(bullets), unsafe_allow_html=True)

        # Seed actions table for Actions tab
        actions_rows = []
        for i, (_, row) in enumerate(top.iterrows(), start=1):
            d = row["Lens"]
            diff = row["Change needed"]
            left_lbl = [a[1] for a in AXES if a[0] == d][0]
            right_lbl = [a[2] for a in AXES if a[0] == d][0]
            if diff > 0:
                direction = f"toward {right_lbl}"
            elif diff < 0:
                direction = f"toward {left_lbl}"
            else:
                direction = "no change"
            actions_rows.append(
                {
                    "Priority": i,
                    "Lens": d,
                    "Direction": direction,
                    "Owner": "",
                    "Timeline": "",
                    "Metric": "",
                    "Status": "",
                }
            )
        st.session_state["_actions_df"] = pd.DataFrame(actions_rows)
    else:
        st.info(
            "Current and target are identical — no change required. "
            "Adjust the sliders in the Lenses tab to see gaps."
        )

    st.markdown(
        "_Want to go deeper on coherence or pacing? See the **Strategy Kernel** and **Three Horizons** in the Resources tab._"
    )

# ====================================================
# ✅ ACTIONS & EXPORT
# ====================================================
with tab_actions:
    st.subheader("Actions & Export")
    st.caption(
        "Turn your top priority shifts into an action log. "
        "Assign owners, timelines and metrics, then export to CSV."
    )

    ensure_sessions()
    actions_df = st.session_state.get("_actions_df", pd.DataFrame())

    if actions_df.empty:
        st.info(
            "No priority shifts have been generated yet. "
            "Go to the Journey tab to calculate gaps and priorities."
        )
    else:
        st.markdown("### Action log (editable)")
        edited = st.data_editor(
            actions_df,
            num_rows="dynamic",
            use_container_width=True,
            key="actions_editor",
        )
        st.session_state["_actions_df"] = edited

        csv_bytes = edited.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download actions as CSV",
            data=csv_bytes,
            file_name="data_strategy_actions.csv",
            mime="text/csv",
        )

        st.markdown(
            "> Tip: paste this table into your programme plan or OKRs to track progress."
        )
# ====================================================
# ℹ️ ABOUT — your provided content
# ====================================================

with tab_about:

    import streamlit as st
    import plotly.graph_objects as go
    import pandas as pd

    def render_about_tab_full(container, AXES):
        with container:
            st.subheader("About this Explorer")

    st.markdown("""
### 🎯 Purpose
Help public bodies **design, communicate, and iterate** their data strategy by making
the **key tensions** explicit, comparing **current vs target**, and turning gaps into **prioritised actions**.

The **Public Sector Data Strategy Explorer** helps you understand **how data strategies differ** — in scope, ambition, and governance.  
It combines a searchable dataset of real strategies with a conceptual framework called **The Ten Lenses of Data Strategy**.
""")

    st.markdown("""
### 👥 Who it's for
- **CDOs / Heads of Data** — set direction and align leadership  
- **Policy & Operations leaders** — frame trade-offs and agree priorities  
- **Analysts & Data teams** — translate strategy into delivery  
- **PMOs / Transformation** — track progress and course-correct
""")

    # --- How to use
    st.markdown("""
### 🛠️ How to use this tool
1) **Explore** the landscape of strategies (by year, country, org type) for context.  
2) **Set profiles** using the **Ten Lenses** sliders to define **Current** and **Target** positions.  
3) **Compare** in the **Journey** tab to see directional gaps (left/right) and magnitudes.  
4) **Prioritise** the top shifts and convert them into actions (owners, timelines, measures).  
5) **Re-assess regularly** — treat your strategy as a **living** thing.
""")

    # --- Explanation & Public-Sector Examples (Ten Lenses)
    st.markdown("### 🔍 Explanation and Public-Sector Examples")
    st.markdown("""
| # | Lens | Description | Example |
|---|------|-------------|---------|
| **1** | **Abstraction Level** | **Conceptual** strategies define vision and principles; **Logical / Physical** specify architecture and governance. | A national “Data Vision 2030” is conceptual; a departmental “Data Architecture Strategy” is logical/physical. |
| **2** | **Adaptability** | **Living** evolves with new tech and policy; **Fixed** provides a stable framework. | The UK's AI white paper is living; GDPR is fixed. |
| **3** | **Ambition** | **Essential** ensures foundations; **Transformational** drives innovation and automation. | Strengthening data quality and governance is essential; Estonia’s X-Road is transformational. |
| **4** | **Coverage** | **Horizontal** builds maturity across all functions; **Use-case-based** targets exemplar projects. | Cross-government maturity improvements vs a single use-case pilot. |
| **5** | **Governance Structure** | **Ecosystem / Federated** encourages collaboration; **Centralised** ensures uniform control. | Federated, domain-based ownership vs a fully centralised decision-making model. |
| **6** | **Orientation** | **Technology-focused** emphasises platforms; **Value-focused** prioritises outcomes and citizens. | A cloud migration roadmap vs a policy-impact dashboard. |
| **7** | **Motivation** | **Compliance-driven** manages risk; **Innovation-driven** creates opportunity. | Privacy-by-design vs data-sharing sandboxes and trusts. |
| **8** | **Access Philosophy** | **Democratised** broadens data access; **Controlled** enforces permissions. | Open environmental data portals vs restricted health datasets. |
| **9** | **Delivery Mode** | **Incremental** iterates and tests; **Big Bang** transforms at once. | Local pilots vs national-scale reform. |
| **10** | **Decision Model** | **Data-informed** blends human judgment; **Data-driven** relies on analytics/automation. | Evidence-based policymaking vs automated fraud detection. |
""")

    st.markdown("---")

    # --- FAQs
    st.markdown("""
### ❓ FAQs
**Is one side of a lens better?**  
No — positions reflect context and risk appetite. The goal is **conscious balance**.

**What if Current and Target are far apart?**  
That’s good information: pick **three shifts** to start; avoid Big-Bang unless mandated.

**How do we decide left vs right?**  
Use the **Lenses** tab — each lens includes when to lean left/right and a concrete example.
""")

    # --- Closing tip
    st.markdown(
        "> **“Every data strategy is a balancing act — between governance and growth, "
        "structure and experimentation, control and creativity.”**"
    )

# ---------------- Footer ----------------
st.markdown("""
---
<div class="footer">
This prototype is created for learning and exploration. It is not an official service.
</div>
""", unsafe_allow_html=True)

