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

APP_VERSION = "ALPHA v3.1 – 2025-11-14"

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
  <div class="sub"> Confident Collaboration + Actionable Strategy + Faster Impact </div>
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
    # Hero / intro
    st.markdown(
        """
<div class="info-panel">
  <strong>What this is:</strong> A thinking and workshop tool for public sector data leaders
  to make <strong>maturity</strong>, <strong>strategic tensions</strong>, and <strong>priority shifts</strong> explicit.
  It will not write your strategy for you, but it will help you have a better conversation about it.
</div>
""",
        unsafe_allow_html=True,
    )

    # Three core building blocks
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
<div class="card">
  <h3>Explore</h3>
  <p class="desc">
    Browse real public sector data strategies by <strong>year</strong>, <strong>country</strong>,
    <strong>organisation type</strong> and <strong>scope</strong>. Use this for context and inspiration,
    not as a complete global catalogue.
  </p>
</div>
""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
<div class="card">
  <h3>Lenses &amp; Journey</h3>
  <p class="desc">
    <strong>Step 1:</strong> Self-diagnose maturity using six government data themes.<br>
    <strong>Step 2:</strong> Set <em>Current vs Target</em> positions across Ten Lenses.<br>
    <strong>Step 3:</strong> Use the Journey tab to see gaps, tensions and potential conflicts.
  </p>
</div>
""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
<div class="card">
  <h3>Actions &amp; Resources</h3>
  <p class="desc">
    Turn your top shifts into a simple <strong>action log</strong>, and use the
    <strong>Resources</strong> tab to connect your insights to wider strategy and
    skills frameworks (government and international).
  </p>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Quick dataset snapshot
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Strategies loaded", len(df))
    k2.metric("Countries", df["country"].nunique() if "country" in df.columns else 0)
    k3.metric("Org types", df["org_type"].nunique() if "org_type" in df.columns else 0)
    k4.metric("Last updated", time.strftime("%Y-%m-%d", time.localtime()))

    st.markdown("---")

    # When it is / isn’t useful
    st.markdown("### When this explorer is useful")
    st.markdown(
        """
Use this tool when you want to:

- **Prepare or refine a data strategy** — sense check whether your ambitions match your current maturity.
- **Run a workshop** with leaders or delivery teams (e.g. 60–90 minutes) to surface assumptions and disagreements.
- **Turn vague direction into clearer shifts** — identify 3–5 practical changes in governance, delivery or access.
- **Support learning and development** — use the Lenses, maturity themes and Resources tab as prompts for discussion.
"""
    )

    st.markdown("### When this tool is not designed to be used as")
    st.markdown(
        """
This is not intended to be:

- A formal or official assessment of organisational maturity.
- A complete, up to date catalogue of all public sector data strategies.
- An automatic strategy generator or replacement for professional judgement.
- A benchmarking tool that compares your scores against other named organisations.

Treat the outputs as structured prompts for conversation and planning, not as a single source of truth.
"""
    )

    st.markdown("---")

    # Suggested journey
    st.markdown("### Suggested journey")
    st.markdown(
        """
1. **Explore** — scan strategies by year, country, org type and scope to build a sense of the landscape.  
2. **Assess maturity** — agree where you sit today across the six government data maturity themes.  
3. **Set tensions** — use the ten Lenses to define your Current vs Target positions, with hints tailored to maturity.  
4. **Review the journey** — focus on the biggest and riskiest shifts; sense-check for over- or under-reach.  
5. **Capture actions & learn** — use the Actions tab to create an action log, and the Resources tab to deepen your thinking.
"""
    )

    st.markdown("---")

    # Community, openness, data / licensing
    st.markdown("### Community, openness and data use")
    st.markdown(
        """
- This is a **community project**, created and maintained as a learning and facilitation tool for data strategists.  
- It does **not collect personal data** about users beyond what your hosting platform may collect by default.  
- All calculations are **visible and transparent** — no hidden scoring models or black-box rankings.  
- Strategies are **curated from official, publicly available sources** (for example, government publications),
  typically under the **Open Government Licence (OGL)** or equivalent open licences.  
- The underlying code is **fully open source**, so anyone can inspect, reuse or adapt it for their own context.
"""
    )

    # How people can contribute
    st.markdown("### How you can contribute")
    st.markdown(
        """
If you find this useful, you can help improve it by:

- Sharing links to **new or missing public data strategies**.
- Flagging **errors in the metadata** (country, year, organisation type, etc.).
- Suggesting **better examples** for the Ten Lenses or maturity themes.
- Sharing how you’ve used the tool in **workshops, training or strategy work**.

For now, the easiest way to contribute is via GitHub [![Contribute a Strategy](https://img.shields.io/badge/Contribute-Submit%20New%20Strategy-blue)](https://github.com/ibpdas/Public-Sector-Data-Strategies/issues/new?assignees=&labels=enhancement%2Cresource&template=resource_submission.md&title=%F0%9F%92%A1+Strategy+Submission).
"""
    )

    # Personal note / provenance
    st.markdown(
        """
<small>
This prototype was created by <strong>Bandhu Das</strong>, a public sector data strategist,
as a side project for learning, facilitation and skills development.  
Connect on LinkedIn: <a href="https://www.linkedin.com/in/bandhu-das" target="_blank">linkedin.com/in/bandhu-das</a>.  
</small>
""",
        unsafe_allow_html=True,
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
        st.subheader("Filters for Explore tab")
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
            "**⚠️ Experimental Feature** - Search strategies, see results in Explore tab",
            placeholder="e.g. 'DEFRA' or 'Data ethics'",
        )

        search_mode = st.radio(
            "Search mode",
            options=["Keyword", "AI semantic"],
            index=1 if emb_df is not None else 0,
            help="Keyword search looks for exact text matches. AI semantic search finds similar strategies by meaning. May produce inaccurate results",
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
        "_You can paste maturity snapshots, lens profiles and action logs from this explorer into your slide decks or business cases._"
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
# 📚 RESOURCES
# ====================================================
# ====================================================
# 🧩 SKILLS (Data strategist self-assessment)
# ====================================================
with tab_resources:
    st.subheader("Skills – data strategist self-assessment")

    st.markdown(
        """
Use this page to **reflect on your own skills as a data strategist**, alongside the
strategy and maturity work you do for your organisation.

It is inspired by the idea of a **skills maturity matrix** (for example, the
*Undercurrent Skills Maturity Matrix*), but simplified and adapted for
public sector data leaders.
"""
    )

    st.markdown("### 1) Behaviours and skills")

    st.caption(
        "For each behaviour, pick the statement that feels **most like you today**. "
        "This is for reflection only – nothing is stored or shared."
    )

    # Simple 4-level scale used for all skills
    LEVEL_LABELS = [
        "1 – Early awareness",
        "2 – Practising",
        "3 – Confident and consistent",
        "4 – Leading and coaching others",
    ]

    # Map to UK Civil Service behaviours (rough, indicative)
    BEHAVIOUR_SKILLS = {
        "Seeing the Big Picture": [
            "Connect data work to policy outcomes and citizen impact",
            "Balance short-term delivery with long-term strategic positioning",
        ],
        "Leadership & Communicating": [
            "Tell compelling data stories for senior, non-technical audiences",
            "Frame trade-offs and tensions in clear, human language",
        ],
        "Delivering at Pace": [
            "Shape lean, test-and-learn delivery plans for data initiatives",
            "Balance experimentation with delivery discipline and governance",
        ],
        "Changing & Improving": [
            "Prototype new uses of AI and data safely and responsibly",
            "Spot opportunities to simplify, standardise and reuse",
        ],
        "Collaborating & Partnering": [
            "Broker alignment across digital, data, policy, and operations teams",
            "Work with external partners (academia, vendors, other departments) effectively",
        ],
        "Developing Self & Others": [
            "Build data literacy and confidence in others",
            "Coach teams to think in terms of value, not just tools",
        ],
        "Data Skills (technical and analytical)": [
            "Work with analytical teams on methods, limitations and assumptions",
            "Understand enough of data architecture / engineering to ask the right questions",
        ],
    }

    # Store selections in session
    if "_skills_matrix" not in st.session_state:
        st.session_state["_skills_matrix"] = {}

    skills_data = []

    for behaviour, skills in BEHAVIOUR_SKILLS.items():
        st.markdown(f"#### {behaviour}")
        cols = st.columns(2)
        for i, skill in enumerate(skills):
            with cols[i % 2]:
                key = f"skill_{behaviour}_{i}"
                current_val = st.session_state["_skills_matrix"].get(key, LEVEL_LABELS[1])
                selected = st.selectbox(
                    skill,
                    LEVEL_LABELS,
                    index=LEVEL_LABELS.index(current_val) if current_val in LEVEL_LABELS else 1,
                    key=key,
                )
                st.session_state["_skills_matrix"][key] = selected
                skills_data.append(
                    {
                        "Behaviour": behaviour,
                        "Skill": skill,
                        "Level": selected,
                        "Level_num": int(selected.split("–")[0].strip()),
                    }
                )

    st.markdown("---")

    # Turn into DataFrame for summary and heatmap
    if skills_data:
        skills_df = pd.DataFrame(skills_data)

        # Summary by behaviour
        summary = (
            skills_df.groupby("Behaviour")["Level_num"]
            .mean()
            .reset_index()
            .rename(columns={"Level_num": "Average level"})
        )

        c1, c2 = st.columns([1, 1])

        with c1:
            st.markdown("### 2) Summary by behaviour")
            st.dataframe(summary, use_container_width=True)

        with c2:
            st.markdown("### 3) Heatmap view")
            # Pivot to Behaviour x Skill with numeric levels
            heat = skills_df.pivot_table(
                index="Behaviour",
                columns="Skill",
                values="Level_num",
                aggfunc="mean",
            )

            fig_heat = px.imshow(
                heat,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Blues",
                origin="upper",
                labels=dict(color="Level"),
                title="Skills heatmap (1 = early awareness • 4 = leading / coaching)",
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        # Download option
        csv_skills = skills_df.drop(columns=["Level_num"]).to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download skills self-assessment (CSV)",
            data=csv_skills,
            file_name="data_strategist_skills_self_assessment.csv",
            mime="text/csv",
        )

    st.markdown("---")

       # ====================================================
    # 📚 Strategy & data frameworks 
    # ====================================================
    st.subheader("📚 Frameworks & Case Studies")
    st.markdown("Selected readings that inform strategic thinking and skills development.")

    resources = [
        ("OECD – Data Governance (Policy Sub-Issue)",
         "Policy and governance principles for managing data across its lifecycle.",
         "https://www.oecd.org/en/topics/sub-issues/data-governance.html"),
        ("UK Government – Data Quality Framework (case studies)",
         "Government approach to improving reliability and usability of data.",
         "https://www.gov.uk/government/publications/the-government-data-quality-framework/the-government-data-quality-framework-case-studies"),
        ("NAO – Improving Government Data: A Guide for Senior Leaders",
            "Practical guidance on leadership, culture and maturity.",
            "https://www.nao.org.uk/wp-content/uploads/2022/07/Improving-government-data-a-guide-for-senior-leaders.pdf"),
        ("OECD – A Data-Driven Public Sector (2019)",
         "International maturity model for strategic data use in government.",
         "https://www.oecd.org/content/dam/oecd/en/publications/reports/2019/05/a-data-driven-public-sector_1c183670/09ab162c-en.pdf"),
        ("IMF – Overarching Strategy on Data & Statistics (2018)",
         "Global strategy for standards, access and capacity building.",
         "https://www.imf.org/-/media/Files/Publications/PP/2018/pp020918-overarching-strategy-on-data-and-statistics-at-the-fund-in-the-digital-age.ashx"),
        ("UK – National Data Strategy M&E Framework",
         "Indicator suite to monitor progress and maturity across pillars.",
         "https://www.gov.uk/government/publications/national-data-strategy-monitoring-and-evaluation-update/national-data-strategy-monitoring-and-evaluation-framework"),
        ("OECD – Measuring the Value of Data and Data Flows (2022)",
         "How data creates economic and social value; approaches to valuation.",
         "https://www.oecd.org/content/dam/oecd/en/publications/reports/2022/12/measuring-the-value-of-data-and-data-flows_2561fe7e/923230a6-en.pdf"),
        ("HM Treasury – Public Value Framework (2019)",
         "Assessing how public spending generates measurable value.",
         "https://assets.publishing.service.gov.uk/media/5c883c32ed915d50b3195be3/public_value_framework_and_supplementary_guidance_web.pdf"),
        ("Frontier Economics – The Value of Data Assets (2021)",
         "Estimating the economic value of data assets and use in the UK.",
         "https://assets.publishing.service.gov.uk/media/6399f93d8fa8f50de138f220/Frontier_Economics_-_value_of_data_assets_-_Dec_2021.pdf"),
        ("OECD – Measuring Data as an Asset (2021)",
         "Methods linking data maturity to national accounts and productivity.",
         "https://www.oecd-ilibrary.org/economics/measuring-data-as-an-asset_b840fb01-en"),
    ]

    # Helper to render resources as cards
    def render_resource_cards(resources, cols_per_row=3):
        for i in range(0, len(resources), cols_per_row):
            row_items = resources[i : i + cols_per_row]
            cols = st.columns(len(row_items))
            for col, (title, desc, url) in zip(cols, row_items):
                with col:
                    st.markdown(
                        f"""
                        <div style="
                            background-color: #0f172a;
                            border-radius: 12px;
                            padding: 16px 16px 14px 16px;
                            margin-bottom: 12px;
                            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.35);
                            border: 1px solid rgba(148, 163, 184, 0.4);
                        ">
                            <div style="font-weight: 600; font-size: 0.95rem; color: #e5e7eb; margin-bottom: 6px;">
                                {title}
                            </div>
                            <div style="font-size: 0.85rem; color: #cbd5f5; margin-bottom: 10px;">
                                {desc}
                            </div>
                            <a href="{url}" target="_blank" style="
                                display: inline-flex;
                                align-items: center;
                                gap: 6px;
                                font-size: 0.85rem;
                                font-weight: 500;
                                color: #38bdf8;
                                text-decoration: none;
                            ">
                                <span>Open resource</span>
                                <span style="font-size: 0.9rem;">↗</span>
                            </a>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    render_resource_cards(resources)
# ====================================================
# ℹ️ ABOUT
# ====================================================
with tab_about:
    st.subheader("About this explorer")

    # Why it exists
    st.markdown(
        """
<div class="info-panel">
  <strong>Why this exists:</strong> this explorer started as a side project by a public-sector
  data strategist to make conversations about <em>data maturity</em> and <em>strategic tensions</em>
  more concrete. It is a learning and facilitation tool, not a benchmarking product or official view.
</div>
""",
        unsafe_allow_html=True,
    )

    # Conceptual foundations
    st.markdown("### Conceptual foundations")

    st.markdown(
        """
This prototype combines three main ingredients:

1. **Government data maturity themes**  
   Based on the six themes in the Data Maturity Assessment for Government framework:  
   **Uses, Data, Leadership, Culture, Tools, Skills** — each rated from **1 (Beginning)** to **5 (Mastering)**.  
   The average score gives a simple sense of overall readiness.

2. **Ten lenses of tensions**  
   A set of paired tensions that describe different ways a data strategy can be configured.  
   The goal is not to pick a “correct” side, but to make trade-offs explicit, for example:
"""
    )

    st.markdown(
        """
| # | Lens | Left | Right | What it’s probing |
|---|------|------|-------|-------------------|
| 1 | **Abstraction level** | Conceptual | Logical / physical | Vision vs. detailed architecture & governance |
| 2 | **Adaptability** | Living | Fixed | How often you change the strategy |
| 3 | **Ambition** | Essential | Transformational | Foundations vs. innovation and automation |
| 4 | **Coverage** | Horizontal | Use-case-based | Whole-org maturity vs. flagship projects |
| 5 | **Governance structure** | Ecosystem / federated | Centralised | Distributed ownership vs. central control |
| 6 | **Orientation** | Technology-focused | Value-focused | Platforms and tools vs. policy / citizen outcomes |
| 7 | **Motivation** | Compliance-driven | Innovation-driven | Risk management vs. opportunity creation |
| 8 | **Access philosophy** | Data-democratised | Controlled access | Broad access vs. tight permissions |
| 9 | **Delivery mode** | Incremental | Big bang | Iterative change vs. large programmes |
| 10 | **Decision model** | Data-informed | Data-driven | Human-in-the-loop vs. automation |
"""
    )

    st.markdown(
        """
3. **Resources and skills**  
   The **Resources** tab links these ideas to wider strategy and skills material
   (for example, strategy “kernels”, horizons, DAMA, Theory of Change, Logic Model),
   so that insights from this tool can feed into **personal development** as well as **business level planning and influencing**.
"""
    )

    st.markdown("---")

    # How the tabs fit together (under the hood)
    st.markdown("### How the pieces fit together")

    st.markdown(
        """
- **Explore** uses a curated dataset of public data strategies (country, org type, scope, year)
  so you can see patterns and examples.  
- **Lenses → Maturity** captures a quick, self-reported view of where your organisation sits today
  across the six government data themes.  
- **Lenses → Tensions** then lets you set **Current vs Target** positions on the ten lenses
  (0 = left label, 100 = right label).  
- **Journey** compares these positions to show gaps and direction of travel, and flags
  when targets may be misaligned with your current maturity (for example, “big-bang delivery”
  at low readiness).  
- **Actions & Export** lets you turn the top shifts into a small, editable action log.  
- **Resources** connects this view to broader strategy and skills frameworks for further reading and self-development.
"""
    )

    st.markdown("---")

    # Methods / calculations
    st.markdown("### How the calculations work (in plain English)")

    st.markdown(
        """
- **Maturity scores** are simple averages of your 1–5 ratings; the level
  (Beginning / Emerging / Learning / Developing / Mastering) is just the rounded average.  
- **Lens sliders** run from 0–100, with 0 = left label and 100 = right label.  
- **Change needed** is `target − current`; negative values mean “move left”, positive values mean “move right”.  
- **Conflicts** are highlighted when there is a big ambition on a lens that usually requires
  higher maturity (for example, very democratised access or big-bang delivery at low overall readiness).  
- **Search** defaults to keyword matching; if you enable semantic search locally (with the optional
  `sentence-transformers` library installed) it will also offer a meaning based search option.
"""
    )

    st.markdown(
        """
There are no hidden scores or external benchmarks: everything you see is derived directly
from the inputs you provide in the app.
"""
    )

    st.markdown("---")

    # Responsible use / limits
    st.markdown("### Limits and responsible use")

    st.markdown(
        """
- All inputs are **self reported** and should be treated as prompts, not formal audit evidence.  
- The tool is **not designed for ranking or comparing** named organisations.  
- It does not express any **official government position** and should not be quoted as such.  
- Use the outputs to structure **conversations, workshops and action planning**, alongside
  other evidence, stakeholder views and professional judgement.
"""
    )

    st.markdown("---")

    # Reuse and adaptation
    st.markdown("### Reuse, forking and adaptation")

    st.markdown(
        """
- The code is intended to be **forked and adapted** via GitHub repository for different sectors, countries or frameworks.  
- If you adapt it, please:  
  - Be clear about your own **data sources** and **licensing** (for example, OGL or other open licences).  
  - Keep the **logic transparent**, so users can see how scores and visuals are produced.  
  - Avoid turning subjective, self-reported scores into hard performance rankings.
"""
    )

# ---------------- Footer ----------------

st.markdown("""
---
<div class="footer">
<p>This is a community learning project. It collects no personal data. 
All strategy documents are drawn from publicly available sources under the Open Government Licence.</p>

<p>
<img src="https://img.shields.io/badge/Open%20Source-Yes-1d70b8" height="22">
<img src="https://img.shields.io/badge/Content-OGL%20v3.0-0b0c0c" height="22">
<img src="https://img.shields.io/badge/No%20Personal%20Data%20Collected-✓-28a197" height="22">
<img src="https://img.shields.io/badge/Community%20Project-Open%20Collaboration-f47738" height="22">
<img src="https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B" height="22">
</p>
</div>
""", unsafe_allow_html=True)
