
# ---------------------------------------------------
# Public Sector Data Strategy Explorer — with Strategic Journey
# ---------------------------------------------------
import os, glob, time, json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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

# ---------------- UI SETUP ----------------
st.set_page_config(page_title="Public Sector Data Strategy Explorer", layout="wide")
st.title("Public Sector Data Strategy Explorer")
st.caption("Presets = target maturity/outcomes • Lenses = tensions to manage • Profile = your chosen balance.")

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

# ---------------- MODEL DEFINITIONS ----------------
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

PRESETS = {
    "Foundational": {"Abstraction Level":100,"Adaptability":100,"Ambition":0,"Coverage":0,"Governance Structure":100,
                     "Orientation":0,"Motivation":0,"Access Philosophy":100,"Delivery Mode":10,"Decision Model":30},
    "Transformational": {"Abstraction Level":20,"Adaptability":10,"Ambition":100,"Coverage":60,"Governance Structure":20,
                         "Orientation":80,"Motivation":100,"Access Philosophy":30,"Delivery Mode":70,"Decision Model":90},
    "Collaborative": {"Abstraction Level":30,"Adaptability":20,"Ambition":50,"Coverage":20,"Governance Structure":10,
                      "Orientation":60,"Motivation":60,"Access Philosophy":20,"Delivery Mode":30,"Decision Model":40},
    "Insight-led": {"Abstraction Level":60,"Adaptability":40,"Ambition":60,"Coverage":40,"Governance Structure":50,
                    "Orientation":70,"Motivation":60,"Access Philosophy":40,"Delivery Mode":40,"Decision Model":70},
    "Citizen-focused": {"Abstraction Level":40,"Adaptability":40,"Ambition":50,"Coverage":40,"Governance Structure":30,
                        "Orientation":100,"Motivation":40,"Access Philosophy":20,"Delivery Mode":40,"Decision Model":40}
}

PRESETS_RATIONALE = {
    "Foundational": {
        "title":"🧱 Foundational — “Build the plumbing”",
        "bullets":[
            "Logical/Physical, Fixed, Essential, Horizontal, Centralised",
            "Tech-focused, Compliance-driven, Controlled, Incremental, Data-informed"
        ]
    },
    "Transformational": {
        "title":"🚀 Transformational — “Accelerate and innovate”",
        "bullets":[
            "Conceptual, Living, Transformational, Use-case-based, Federated",
            "Value-focused, Innovation-driven, Democratised, Big Bang, Data-driven"
        ]
    },
    "Collaborative": {
        "title":"🤝 Collaborative — “Connect the ecosystem”",
        "bullets":[
            "Semi-conceptual, Living, Moderate, Horizontal, Federated",
            "Value-focused, Balanced, Democratised, Incremental, Data-informed"
        ]
    },
    "Insight-led": {
        "title":"📊 Insight-led — “Evidence before action”",
        "bullets":[
            "Logical, Semi-living, Moderate, Targeted, Mixed",
            "Value-focused, Balanced, Semi-open, Incremental, Data-driven"
        ]
    },
    "Citizen-focused": {
        "title":"👥 Citizen-focused — “Ethics, trust, service outcomes”",
        "bullets":[
            "Conceptual, Living, Balanced, Horizontal, Federated",
            "Value-focused, Balanced, Democratised, Incremental, Data-informed"
        ]
    }
}

def render_explainer_block():
    st.markdown("""
### 🧭 How to Use This Framework
- **Presets** describe the *destination* — the kind of data strategy your organisation is aiming for (e.g., Foundational, Transformational).
- **Lenses** describe the *tensions to manage* — the competing forces that shape *how* you’ll get there.
- **Profile** records your *current balance* across those tensions.

Ask yourself:
1. What type of data strategy are we trying to become? *(Preset)*  
2. What tensions define our journey? *(Lenses)*  
3. Where are we now, and how far are we from our target? *(Gap)*  
4. Which tensions require conscious management? *(Priority for action)*
""")

def radar_trace(values01, dims, name, opacity=0.6, fill=True):
    r = list(values01) + [values01[0]]
    t = list(dims) + [dims[0]]
    return go.Scatterpolar(r=r, theta=t, name=name, fill='toself' if fill else None, opacity=opacity)

def get_scores_from_session():
    return {d: st.session_state.get("_ten_scores", {}).get(d, 50) for d in DIMENSIONS}

def ensure_session_scores():
    if "_ten_scores" not in st.session_state:
        st.session_state["_ten_scores"] = {d:50 for d in DIMENSIONS}

tab_explore, tab_types, tab_journey, tab_about = st.tabs(["🔎 Explore", "👁️ Strategy Types", "🧭 Strategic Journey", "ℹ️ About"])

# EXPLORE
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
        col1, col2 = st.columns(2)
        by_year = fdf.groupby("year").size().reset_index(name="count").dropna()
        if not by_year.empty:
            col1.plotly_chart(px.bar(by_year, x="year", y="count", title="Strategies by year"), use_container_width=True)
        else:
            col1.info("No numeric 'year' values to chart.")
        by_country = fdf.groupby("country").size().reset_index(name="count").sort_values("count", ascending=False)
        col2.plotly_chart(px.bar(by_country.head(10), x="country", y="count", title="Top countries"), use_container_width=True)

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

# STRATEGY TYPES
with tab_types:
    st.subheader("👁️ The Ten Lenses of Data Strategy")
    render_explainer_block()

    st.markdown("### 🔗 Presets and what they mean")
    for name, spec in PRESETS_RATIONALE.items():
        with st.expander(spec["title"]):
            for b in spec["bullets"]:
                st.markdown(f"- {b}")

    ensure_session_scores()
    left, right = st.columns([1,1])
    with left:
        st.markdown("#### Self‑assessment sliders")
        cols = st.columns(2)
        for i, (dim, left_lbl, right_lbl) in enumerate(AXES):
            with cols[i % 2]:
                st.session_state["_ten_scores"][dim] = st.slider(
                    f"{dim}", 0, 100, st.session_state["_ten_scores"][dim],
                    format="%d%%", help=f"{left_lbl} ←→ {right_lbl}"
                )
                st.caption(f"{left_lbl} ←── {st.session_state['_ten_scores'][dim]}% → {right_lbl}")

    with right:
        st.markdown("#### Preset overlays")
        chosen = st.multiselect("Overlay presets", list(PRESETS.keys()), default=["Foundational","Transformational"])
        apply_preset = st.selectbox("Apply preset to sliders", ["(none)"] + list(PRESETS.keys()))
        if st.button("Apply preset now") and apply_preset != "(none)":
            st.session_state["_ten_scores"] = PRESETS[apply_preset].copy()

    dims = DIMENSIONS
    fig = go.Figure()
    user01 = [st.session_state["_ten_scores"][d]/100 for d in dims]
    fig.add_trace(radar_trace(user01, dims, "Your profile", opacity=0.7))
    for name in chosen:
        p01 = [PRESETS[name][d]/100 for d in dims]
        fig.add_trace(radar_trace(p01, dims, name, opacity=0.4))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), title="Strategic fingerprint (you vs presets)")
    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Download my self‑assessment (JSON)",
        data=json.dumps(st.session_state["_ten_scores"], indent=2).encode("utf-8"),
        file_name="strategy_ten_lenses_self_assessment.json",
        mime="application/json"
    )

# STRATEGIC JOURNEY
with tab_journey:
    st.subheader("🧭 Strategic Journey — from current to target")
    st.markdown("**Presets** = destination. **Lenses** = tensions to manage. Compare **current** vs **target** and see the gap.")

    ensure_session_scores()
    colA, colB, colC = st.columns([1.2, 1.2, 1])

    with colA:
        st.markdown("#### Current state")
        current_source = st.radio("Use...", ["My sliders", "Preset", "Upload JSON"], horizontal=True)
        if current_source == "My sliders":
            current = get_scores_from_session()
        elif current_source == "Preset":
            c_name = st.selectbox("Current preset", list(PRESETS.keys()), index=0)
            current = PRESETS[c_name]
        else:
            up = st.file_uploader("Upload JSON profile (keys = lenses, values 0–100)", type=["json"])
            if up is not None:
                try:
                    current = json.loads(up.read().decode("utf-8"))
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")
                    current = get_scores_from_session()
            else:
                current = get_scores_from_session()

    with colB:
        st.markdown("#### Target state")
        t_name = st.selectbox("Target preset", list(PRESETS.keys()), index=1)
        target = PRESETS[t_name]

    with colC:
        st.markdown("#### Export")
        st.download_button("Download current (JSON)",
            data=json.dumps(current, indent=2).encode("utf-8"),
            file_name="current_profile.json", mime="application/json")
        st.download_button("Download target (JSON)",
            data=json.dumps(target, indent=2).encode("utf-8"),
            file_name="target_profile.json", mime="application/json")

    dims = DIMENSIONS
    cur01 = [current[d]/100 for d in dims]
    tgt01 = [target[d]/100 for d in dims]
    fig2 = go.Figure()
    fig2.add_trace(radar_trace(cur01, dims, "Current", opacity=0.6))
    fig2.add_trace(radar_trace(tgt01, dims, "Target", opacity=0.5))
    fig2.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), title=f"Current vs Target — {t_name}")
    st.plotly_chart(fig2, use_container_width=True)

    gap_rows = []
    for d in dims:
        diff = target[d] - current[d]
        mag = abs(diff)
        left_lbl, right_lbl = [a[1] for a in AXES if a[0]==d][0], [a[2] for a in AXES if a[0]==d][0]
        direction = f"→ **{right_lbl}**" if diff>0 else (f"→ **{left_lbl}**" if diff<0 else "—")
        gap_rows.append({"Lens": d, "Current": current[d], "Target": target[d], "Change needed": diff, "Magnitude": mag, "Direction": direction})
    gap_df = pd.DataFrame(gap_rows).sort_values("Magnitude", ascending=False)

    st.markdown("#### Gap by lens (largest first)")
    st.dataframe(gap_df[["Lens","Current","Target","Change needed","Direction"]], use_container_width=True)

    bar = px.bar(gap_df.sort_values("Change needed"),
                 x="Change needed", y="Lens",
                 orientation="h",
                 title="Signed change needed (− move left • + move right)")
    st.plotly_chart(bar, use_container_width=True)

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
        if bullets:
            st.markdown("\n".join(bullets))
        else:
            st.info("You're already aligned with the target profile.")

# ABOUT
with tab_about:
    st.subheader("About this Explorer")
    st.markdown("""
This tool helps public bodies understand **how data strategies differ** — and how to move from **current** to **target**.

**Conceptual triad**
- 🎯 **Presets** describe the *destination* (target maturity and outcomes).
- ⚖️ **Lenses** surface the *tensions to manage* on the journey.
- 📊 **Profile** records your chosen balance across the lenses.
""")

    st.markdown("### 👁️ The Ten Lenses (visual overview)")
    fig = go.Figure()
    for i, (dim, left, right) in enumerate(AXES):
        fig.add_trace(go.Bar(x=[50, 50], y=[f"{i+1}. {dim}", f"{i+1}. {dim}"],
                             orientation='h', marker_color=['#70A9FF', '#FFB8B8'],
                             showlegend=False, hovertext=[left, right]))
    fig.update_layout(barmode='stack', xaxis=dict(showticklabels=False, range=[0,100]),
                      height=480, margin=dict(l=20,r=20,t=20,b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
**The Ten Lenses**
1. **Abstraction Level:** Conceptual ↔ Logical/Physical  
2. **Adaptability:** Living ↔ Fixed  
3. **Ambition:** Essential ↔ Transformational  
4. **Coverage:** Horizontal ↔ Use‑case‑based  
5. **Governance Structure:** Ecosystem/Federated ↔ Centralised  
6. **Orientation:** Technology‑focused ↔ Value‑focused  
7. **Motivation:** Compliance‑driven ↔ Innovation‑driven  
8. **Access Philosophy:** Data‑democratised ↔ Controlled access  
9. **Delivery Mode:** Incremental ↔ Big Bang  
10. **Decision Model:** Data‑informed ↔ Data‑driven
""")
