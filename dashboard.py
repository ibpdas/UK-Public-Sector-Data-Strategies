
# ---------------------------------------------------
# Public Sector Data Strategy Explorer — No Presets + Lenses Explainer
# ---------------------------------------------------
import os, glob, time, json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from about_full_snippet import render_about_tab_full

# Optional fuzzy search
try:
    from rapidfuzz import fuzz, process
    HAS_RAPIDFUZZ = True
except Exception:
    HAS_RAPIDFUZZ = False

REQUIRED = [
    "id","title","organisation","org_type","country","year","scope",
    "link","summary","source","date_added"
]

st.set_page_config(page_title="Public Sector Data Strategy Explorer", layout="wide")
st.title("Public Sector Data Strategy Explorer")
st.caption("Lenses = tensions to manage • Profile = your chosen balance • Journey = current → target. (No presets for maximum clarity)")


# ---------------- DATA LOAD ----------------
csv_files = sorted([f for f in glob.glob("*.csv") if os.path.isfile(f)])
default_csv = "strategies.csv" if "strategies.csv" in csv_files else (csv_files[0] if csv_files else None)
if not csv_files:
    st.error("No CSV files found in this folder. Please add a CSV (e.g., strategies.csv).")
    st.stop()

with st.sidebar:
    st.subheader("Data source")
    csv_path = st.selectbox("CSV file", options=csv_files, index=csv_files.index(default_csv) if default_csv else 0)
    try:
        mtime = os.path.getmtime(csv_path)
        st.caption(f"📄 **{csv_path}** — last modified: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))}")
    except Exception:
        st.caption("📄 File time unknown.")
    if st.button("🔄 Reload data / clear cache"):
        st.cache_data.clear()
        st.experimental_rerun()

@st.cache_data(show_spinner=False)
def load_data(path: str, modified_time: float):
    df = pd.read_csv(path).fillna("")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    return df

try:
    df = load_data(csv_path, os.path.getmtime(csv_path))
except Exception as e:
    st.error(f"⚠️ {e}")
    st.stop()


# ---------------- MODEL ----------------
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
    ("Decision Model", "Data-informed", "Data-driven")
]
DIMENSIONS = [a[0] for a in AXES]


def radar_trace(values01, dims, name, opacity=0.6, fill=True):
    r = list(values01) + [values01[0]]
    t = list(dims) + [dims[0]]
    return go.Scatterpolar(r=r, theta=t, name=name, fill='toself' if fill else None, opacity=opacity)


def ensure_sessions():
    if "_current_scores" not in st.session_state:
        st.session_state["_current_scores"] = {d:50 for d in DIMENSIONS}
    if "_target_scores" not in st.session_state:
        st.session_state["_target_scores"] = {d:50 for d in DIMENSIONS}


# ------------- LENSES EXPLAINER (rich, practical) ----------------
LENSES_EXAMPLES = {
    "Abstraction Level": {
        "left": "Conceptual strategy clarifies principles, outcomes, and direction",
        "right": "Logical/Physical specifies models, standards, platforms",
        "when_left": "When you need sponsorship & shared intent across leaders",
        "when_right": "When delivery is blocked by unclear ownership/architecture",
        "example": "Start with a 1‑page narrative (conceptual) then publish a canonical data model (logical)."
    },
    "Adaptability": {
        "left": "Living strategy that iterates with tech/policy change",
        "right": "Fixed guardrails for consistency and auditability",
        "when_left": "Fast‑changing domains (AI, climate risk, emergencies)",
        "when_right": "Regulated or safety‑critical contexts",
        "example": "Quarterly roadmap reviews with change log; stable retention policy."
    },
    "Ambition": {
        "left": "Essential data management (quality, stewardship, metadata)",
        "right": "Transformational use of AI/automation for services/outcomes",
        "when_left": "Data quality debt, unclear lineage, poor trust",
        "when_right": "Strong foundations and clear value hypotheses",
        "example": "Fix reference data + metadata first → then pilot AI triage on service cases."
    },
    "Coverage": {
        "left": "Horizontal capability across the organisation",
        "right": "Use‑case exemplars to prove value quickly",
        "when_left": "Silo fragmentation and inconsistent methods",
        "when_right": "Need quick wins to unlock sponsorship",
        "example": "Standards + training (horizontal) while delivering 2 flagship use‑cases."
    },
    "Governance Structure": {
        "left": "Ecosystem/Federated with domain ownership",
        "right": "Centralised for coherence and single point of accountability",
        "when_left": "Diverse domains need autonomy within guardrails",
        "when_right": "Crisis, reset, or heavy risk exposure",
        "example": "Domain data owners + central guardrails, catalogue, and design authority."
    },
    "Orientation": {
        "left": "Technology‑focused investments (platforms, pipelines, MDM)",
        "right": "Value‑focused outcomes (policy, service, citizen impact)",
        "when_left": "Missing core capabilities or tooling debt",
        "when_right": "Stakeholders need measurable policy/service wins",
        "example": "Prioritise ‘value slices’ that also uplift platform capability."
    },
    "Motivation": {
        "left": "Compliance‑driven (legal, audit, risk)",
        "right": "Innovation‑driven (opportunity, growth, modernisation)",
        "when_left": "Regulatory deadlines or risk incidents",
        "when_right": "Mature controls; pressure to improve outcomes",
        "example": "DPIAs, privacy‑by‑design; then sandbox new ML approaches."
    },
    "Access Philosophy": {
        "left": "Data‑democratised (open by default where safe)",
        "right": "Controlled access (least privilege)",
        "when_left": "To reduce shadow data and enable reuse",
        "when_right": "Sensitive data and lawful basis constraints",
        "example": "Tiered access: open → internal → restricted → highly sensitive."
    },
    "Delivery Mode": {
        "left": "Incremental, iterative releases",
        "right": "Big Bang step‑change programmes",
        "when_left": "High uncertainty; need feedback loops",
        "when_right": "Mandated deadlines or platform migration",
        "example": "Monthly drops for catalogue improvements; time‑boxed migration cutover."
    },
    "Decision Model": {
        "left": "Data‑informed (human‑in‑the‑loop)",
        "right": "Data‑driven (automation where safe)",
        "when_left": "Complex, value‑laden policy choices",
        "when_right": "High‑volume, repeatable operational decisions",
        "example": "Human policy panels for trade‑offs; automated fraud triage."
    }
}


def render_lenses_explainer():
    st.markdown("### 👁️ Lenses explainer & practical examples")
    st.caption("Each slider represents a **tension to manage**. Use the notes below to decide which way to lean for your context.")
    for dim, left, right in AXES:
        ex = LENSES_EXAMPLES.get(dim, {})
        with st.expander(f"{dim} — {left} ↔ {right}"):
            st.markdown(f"- **{left}:** {ex.get('left','')}")
            st.markdown(f"- **{right}:** {ex.get('right','')}")
            st.markdown(f"- **Lean {left.lower()} when:** {ex.get('when_left','')}")
            st.markdown(f"- **Lean {right.lower()} when:** {ex.get('when_right','')}")
            st.markdown(f"- **Example:** {ex.get('example','')}")


# ---------------- TABS ----------------
tab_explore, tab_types, tab_journey, tab_about = st.tabs(
    ["🔎 Explore", "👁️ Lenses (Set Profiles)", "🧭 Journey (Compare)", "ℹ️ About"]
)


# ====================================================
# 🔎 EXPLORE
# ====================================================
with tab_explore:
    with st.sidebar:
        st.subheader("Filters")
        years = sorted(y for y in df["year"].dropna().unique())
        if years:
            yr = st.slider("Year range", int(min(years)), int(max(years)), (int(min(years)), int(max(years))))
        else:
            yr = None
        org_types = sorted([v for v in df["org_type"].unique() if v != ""])
        org_type_sel = st.multiselect("Org type", org_types, default=org_types)
        countries = sorted([v for v in df["country"].unique() if v != ""])
        country_sel = st.multiselect("Country", countries, default=countries)
        scopes = sorted([v for v in df["scope"].unique() if v != ""])
        scope_sel = st.multiselect("Scope", scopes, default=scopes)
        q = st.text_input("Search title/org/summary")

    def fuzzy(df_in, q, limit=400):
        if not q: return df_in
        text = (df_in["title"] + " " + df_in["organisation"] + " " + df_in["summary"]).fillna("")
        if HAS_RAPIDFUZZ:
            matches = process.extract(q, text.tolist(), scorer=fuzz.WRatio, limit=len(text))
            keep = [i for _, s, i in matches if s >= 60]
            return df_in.iloc[keep].head(limit)
        else:
            mask = text.str.contains(q, case=False, na=False)
            return df_in[mask].head(limit)

    fdf = df.copy()
    if yr: fdf = fdf[fdf["year"].between(yr[0], yr[1])]
    if org_type_sel: fdf = fdf[fdf["org_type"].isin(org_type_sel)]
    if country_sel: fdf = fdf[fdf["country"].isin(country_sel)]
    if scope_sel: fdf = fdf[fdf["scope"].isin(scope_sel)]
    fdf = fuzzy(fdf, q)

    st.info(f"{len(fdf)} strategies shown")
    if not fdf.empty:
        # Quick KPIs
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Strategies", len(fdf))
        k2.metric("Countries", fdf["country"].nunique() if "country" in fdf.columns else 0)
        k3.metric("Org types", fdf["org_type"].nunique() if "org_type" in fdf.columns else 0)
        if "year" in fdf.columns and fdf["year"].notna().any():
            k4.metric("Year span", f"{int(fdf['year'].min())}–{int(fdf['year'].max())}")
        else:
            k4.metric("Year span", "—")

        st.markdown("---")

        # Row 1
        c1, c2 = st.columns(2)
        if "year" in fdf.columns and fdf["year"].notna().any():
            fig_hist = px.histogram(
                fdf[fdf["year"].notna()], x="year",
                color="scope" if "scope" in fdf.columns else None,
                nbins=max(10, min(40, fdf["year"].nunique())),
                title="Strategies by year"
            )
            fig_hist.update_layout(bargap=0.05)
            c1.plotly_chart(fig_hist, use_container_width=True)
        else:
            c1.info("No numeric 'year' values to chart.")

        if "org_type" in fdf.columns and fdf["org_type"].notna().any():
            top_org = fdf.groupby("org_type").size().reset_index(name="count").sort_values("count", ascending=False)
            fig_org = px.bar(top_org, x="org_type", y="count", title="Composition by organisation type")
            fig_org.update_xaxes(title=None, tickangle=20)
            c2.plotly_chart(fig_org, use_container_width=True)
        else:
            c2.info("No 'org_type' values to chart.")

        st.markdown("---")

        # Row 2
        c3, c4 = st.columns(2)
        if all(col in fdf.columns for col in ["country","org_type"]):
            fig_tree = px.treemap(fdf.assign(_value=1), path=["country","org_type","organisation"], values="_value",
                                  title="Landscape by country → org type → organisation")
            c3.plotly_chart(fig_tree, use_container_width=True)
        else:
            c3.info("Need 'country' and 'org_type' for treemap.")

        if "country" in fdf.columns and fdf["country"].notna().any():
            by_ctry = fdf.groupby("country").size().reset_index(name="count")
            fig_map = px.choropleth(by_ctry, locations="country", locationmode="country names", color="count",
                                    title="Global distribution of strategies (by country)", color_continuous_scale="Blues")
            c4.plotly_chart(fig_map, use_container_width=True)
        else:
            c4.info("No 'country' values to map.")

        st.markdown("---")

        # Row 3
        c5, c6 = st.columns(2)
        if all(col in fdf.columns for col in ["country","org_type"]):
            top_ctrys = fdf.groupby("country").size().sort_values(ascending=False).head(12).index.tolist()
            sub = fdf[fdf["country"].isin(top_ctrys)]
            fig_stack = px.bar(sub, x="country", color="org_type", title="Top countries by strategies (stacked by org type)")
            fig_stack.update_xaxes(title=None)
            c5.plotly_chart(fig_stack, use_container_width=True)
        else:
            c5.info("Need 'country' and 'org_type' for stacked bar.")

        need = ["year","organisation","title"]
        if all(col in fdf.columns for col in need) and fdf["year"].notna().any():
            sub = fdf[fdf["year"].notna()].copy()
            fig_scatter = px.scatter(sub, x="year", y="organisation", color="country" if "country" in sub.columns else None,
                                     hover_data=["title","country","scope"] if "scope" in sub.columns else ["title"],
                                     title="Timeline of strategies by organisation")
            c6.plotly_chart(fig_scatter, use_container_width=True)
        else:
            c6.info("Need 'year', 'organisation', 'title' for timeline scatter.")

        st.markdown("---")

        if "scope" in fdf.columns and fdf["scope"].notna().any():
            by_scope = fdf["scope"].value_counts().reset_index()
            by_scope.columns = ["scope","count"]
            fig_scope = px.pie(by_scope, names="scope", values="count", title="Strategy scope breakdown")
            st.plotly_chart(fig_scope, use_container_width=True)

        st.markdown("### Details")
        for _, r in fdf.iterrows():
            with st.expander(f"📄 {r['title']} — {r['organisation']} ({int(r['year']) if pd.notna(r['year']) else '—'})"):
                st.write(r["summary"] or "_No summary provided._")
                meta = st.columns(4)
                meta[0].write(f"**Org type:** {r['org_type']}")
                meta[1].write(f"**Country:** {r['country']}")
                meta[2].write(f"**Scope:** {r['scope']}")
                meta[3].write(f"**Source:** {r['source']}")
                if r["link"]:
                    st.link_button("Open document", r["link"])


# ====================================================
# 👁️ LENSES (SET PROFILES)
# ====================================================
with tab_types:
    st.subheader("👁️ Set your profiles across the Ten Lenses")
    st.caption("0 = left label • 100 = right label. Use the left column for CURRENT, right for TARGET.")

    # Add the explainer right here
    render_lenses_explainer()

    ensure_sessions()
    colL, colR = st.columns(2)

    with colL:
        st.markdown("#### Current profile")
        cols = st.columns(2)
        for i, (dim, left_lbl, right_lbl) in enumerate(AXES):
            with cols[i % 2]:
                st.session_state["_current_scores"][dim] = st.slider(
                    f"{dim} (current)", 0, 100, st.session_state["_current_scores"][dim],
                    format="%d%%", help=f"{left_lbl} ←→ {right_lbl}"
                )
                st.caption(f"{left_lbl} ←── {st.session_state['_current_scores'][dim]}% → {right_lbl}")
        st.download_button("Download current (JSON)",
                           data=json.dumps(st.session_state["_current_scores"], indent=2).encode("utf-8"),
                           file_name="current_profile.json", mime="application/json")

    with colR:
        st.markdown("#### Target profile")
        cols = st.columns(2)
        for i, (dim, left_lbl, right_lbl) in enumerate(AXES):
            with cols[i % 2]:
                st.session_state["_target_scores"][dim] = st.slider(
                    f"{dim} (target)", 0, 100, st.session_state["_target_scores"][dim],
                    format="%d%%", help=f"{left_lbl} ←→ {right_lbl}"
                )
                st.caption(f"{left_lbl} ←── {st.session_state['_target_scores'][dim]}% → {right_lbl}")
        st.download_button("Download target (JSON)",
                           data=json.dumps(st.session_state["_target_scores"], indent=2).encode("utf-8"),
                           file_name="target_profile.json", mime="application/json")

    # Side-by-side radar
    dims = DIMENSIONS
    cur01 = [st.session_state["_current_scores"][d]/100 for d in dims]
    tgt01 = [st.session_state["_target_scores"][d]/100 for d in dims]
    fig = go.Figure()
    fig.add_trace(radar_trace(cur01, dims, "Current", opacity=0.6))
    fig.add_trace(radar_trace(tgt01, dims, "Target", opacity=0.5))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), title="Current vs Target — strategic fingerprints")
    st.plotly_chart(fig, use_container_width=True)


# ====================================================
# 🧭 JOURNEY (COMPARE)
# ====================================================
with tab_journey:
    st.subheader("🧭 Journey — compare current vs target and prioritise")
    st.caption("Signed change: negative = move toward LEFT label; positive = move toward RIGHT label.")

    dims = DIMENSIONS
    current = st.session_state.get("_current_scores", {d:50 for d in dims})
    target = st.session_state.get("_target_scores", {d:50 for d in dims})

    # Gap table
    gap_rows = []
    for d, left_lbl, right_lbl in AXES:
        diff = target[d] - current[d]
        mag = abs(diff)
        direction = f"→ **{right_lbl}**" if diff>0 else (f"→ **{left_lbl}**" if diff<0 else "—")
        gap_rows.append({"Lens": d, "Current": current[d], "Target": target[d], "Change needed": diff, "Magnitude": mag, "Direction": direction})
    gap_df = pd.DataFrame(gap_rows).sort_values("Magnitude", ascending=False)

    st.markdown("#### Gap by lens (largest first)")
    st.dataframe(gap_df[["Lens","Current","Target","Change needed","Direction"]], use_container_width=True)

    bar = px.bar(gap_df.sort_values("Change needed"), x="Change needed", y="Lens", orientation="h",
                 title="Signed change needed (− move left • + move right)")
    st.plotly_chart(bar, use_container_width=True)

    # Top 3 priorities
    TOP_N = 3
    top = gap_df.head(TOP_N)
    if len(top):
        st.markdown(f"#### Priority shifts (top {TOP_N})")
        bullets = []
        for _, row in top.iterrows():
            d = row["Lens"]; diff = row["Change needed"]
            left_lbl, right_lbl = [a[1] for a in AXES if a[0]==d][0], [a[2] for a in AXES if a[0]==d][0]
            if diff > 0:
                bullets.append(f"- **{d}**: shift toward **{right_lbl}** (+{int(diff)} pts)")
            elif diff < 0:
                bullets.append(f"- **{d}**: shift toward **{left_lbl}** ({int(diff)} pts)")
        st.markdown("\n".join(bullets))
    else:
        st.info("Current and target are identical — no change required.")


# ====================================================

# ====================================================

# ====================================================
# ℹ️ ABOUT
# ====================================================
with tab_about:
    render_about_tab_full(tab_about, AXES)
