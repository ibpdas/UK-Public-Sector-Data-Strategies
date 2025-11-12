import streamlit as st
import plotly.graph_objects as go
import pandas as pd

def render_about_tab_full(container, AXES):
    with container:
        st.subheader("About this Explorer")

        # --- Purpose & Audience
        st.markdown("""
### 🎯 Purpose

The **Public Sector Data Strategy Explorer** helps data leaders understand **how data strategies differ** — in scope, ambition, and delivery. It combines a searchable dataset of real strategies with tools to make the **key tensions** explicit, compare **current vs target**, and turn gaps into **prioritised actions**.


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
| # | Lens | Description | Public-Sector Example |
|---|------|-------------|----------------------|
| **1** | **Abstraction Level** | **Conceptual** strategies define vision and principles; **Logical / Physical** specify architecture and governance. | A national “Data Vision 2030” is conceptual; a departmental “Data Architecture Blueprint” is logical/physical. |
| **2** | **Adaptability** | **Living** evolves with new tech and policy; **Fixed** provides a stable framework. | The UK’s AI white paper is living; GDPR is fixed. |
| **3** | **Ambition** | **Essential** ensures foundations; **Transformational** drives innovation and automation. | NHS data governance reforms are essential; Estonia’s X-Road is transformational. |
| **4** | **Coverage** | **Horizontal** builds maturity across all functions; **Use-case-based** targets exemplar projects. | A cross-government maturity model vs a sector-specific pilot. |
| **5** | **Governance Structure** | **Ecosystem / Federated** encourages collaboration; **Centralised** ensures uniform control. | UK’s federated CDO network vs Singapore’s Smart Nation. |
| **6** | **Orientation** | **Technology-focused** emphasises platforms; **Value-focused** prioritises outcomes and citizens. | A cloud migration roadmap vs a policy-impact dashboard. |
| **7** | **Motivation** | **Compliance-driven** manages risk; **Innovation-driven** creates opportunity. | GDPR compliance vs data-sharing sandboxes. |
| **8** | **Access Philosophy** | **Democratised** broadens data access; **Controlled** enforces permissions. | Open data portals vs restricted health datasets. |
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
        st.markdown("> **“Every data strategy is a balancing act — between governance and growth, structure and experimentation, control and creativity.”**")
