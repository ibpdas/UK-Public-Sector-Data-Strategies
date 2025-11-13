# resources_tab.py
import streamlit as st


def render_resources() -> None:
    st.subheader("Strategy Frameworks & Further Reading")
    st.caption(
        "These frameworks provide extra lenses for thinking about data strategy. "
        "Each one links to specific parts of the Explorer (maturity, lenses or journey)."
    )

    st.markdown(
        """
<div class="info-panel">
Use these frameworks to deepen discussion, not to add complexity.
Pick one or two that best match the question you are exploring.
</div>
""",
        unsafe_allow_html=True,
    )

    # 1) Playing to Win
    with st.expander("1️⃣ Playing to Win – Strategy Cascade", expanded=True):
        st.markdown(
            """
**What it is**  
A practical set of five linked questions:

1. **Winning aspiration** – what does success look like?  
2. **Where to play?** – which domains, services, users, or problems?  
3. **How to win?** – what advantages or differentiators will matter?  
4. **Capabilities** – what must we be good at to win?  
5. **Management systems** – how will we govern and sustain this?

**How it relates to this Explorer**  
- Aligns with **Ambition** (Essential ↔ Transformational)  
- Supports **Coverage** (Horizontal ↔ Use-case-based)  
- Connects to **Orientation** (Technology-focused ↔ Value-focused)
"""
        )

    # 2) Strategy Diamond
    with st.expander("2️⃣ Strategy Diamond – Arenas, Vehicles, Differentiators, Staging, Economic Logic"):
        st.markdown(
            """
**What it is**  
A holistic view of strategy across five elements:

- **Arenas** – where we will be active (domains, channels, regions)  
- **Vehicles** – how we will get there (build, partner, buy, collaborate)  
- **Differentiators** – why we will succeed (speed, quality, trust, integration)  
- **Staging** – the sequence and speed of moves  
- **Economic logic** – how value is created, protected, and scaled

**How it relates to this Explorer**  
- Helps turn gap analysis in the **Journey** tab into a coherent story  
- Links your **Delivery Mode** and **Governance Structure** choices to staging and vehicles  
- Supports the **Actions & Export** tab by framing a joined-up change plan
"""
        )

    # 3) Good Strategy / Bad Strategy – Strategy Kernel
    with st.expander("3️⃣ Good Strategy / Bad Strategy – The Strategy Kernel"):
        st.markdown(
            """
**What it is**  
A “kernel” of good strategy with three parts:

1. **Diagnosis** – a clear, honest view of the situation or problem  
2. **Guiding policy** – the overall approach you will take  
3. **Coherent actions** – mutually reinforcing actions that implement the policy

**How it relates to this Explorer**  
- Mirrors the Explorer’s flow:  
  - **Maturity** → diagnosis  
  - **Lenses (Current vs Target)** → guiding policy  
  - **Journey & Actions** → coherent actions  
- Many conflict warnings highlight “bad strategy” patterns (for example, high ambition with low maturity)
"""
        )

    # 4) McKinsey Three Horizons
    with st.expander("4️⃣ McKinsey Three Horizons – Pacing Change"):
        st.markdown(
            """
**What it is**  
A way of pacing investment and change across three overlapping horizons:

- **Horizon 1** – strengthen and modernise the core (today’s operations)  
- **Horizon 2** – scale newer capabilities and adjacent services  
- **Horizon 3** – explore future, more experimental bets

**How it relates to this Explorer**  
- Links to **Delivery Mode** (Incremental ↔ Big Bang) and **Ambition**  
- Helps explain why some high-ambition targets should be phased over time  
- Provides language to separate “fix the basics” from “invest in future data/AI capability”
"""
        )

    # 5) DAMA Wheel / DMBOK
    with st.expander("5️⃣ DAMA Wheel – Data Management Functions"):
        st.markdown(
            """
**What it is**  
An industry standard view of data management disciplines such as:

- Data governance  
- Data quality  
- Data architecture  
- Metadata and reference data  
- Security, privacy, and protection  
- Integration, modelling, warehousing, analytics, delivery

**How it relates to this Explorer**  
- Underpins the **six maturity themes** in the Lenses tab  
- Helps unpack what “foundations” actually mean in practice  
- Useful when discussing roles and responsibilities with data management teams
"""
        )

    # 6) TOGAF stack
    with st.expander("6️⃣ TOGAF Architecture Stack – Conceptual, Logical, Physical"):
        st.markdown(
            """
**What it is**  
A simple stack that distinguishes three levels of architectural thinking:

- **Conceptual** – high-level principles, domains, and capabilities  
- **Logical** – services, information flows, integration patterns  
- **Physical** – actual products, platforms, and technical components

**How it relates to this Explorer**  
- Connects directly to the **Abstraction Level** lens (Conceptual ↔ Logical/Physical)  
- Explains why some strategies stay at vision level while others describe detailed platforms  
- Helps align enterprise architects with policy and strategy owners
"""
        )

    st.markdown("### 🧭 How it all fits together")
    st.markdown(
        """
```text
Real strategies  →  Explore tab      →  Patterns & comparators
                  (landscape)

Maturity         →  Lenses (step 1)  →  Where are we now? (Uses, Data, Leadership, Culture, Tools, Skills)

Ten Lenses       →  Lenses (step 2)  →  Where do we want to sit on key tensions?

Gaps & conflicts →  Journey          →  Which shifts matter most? What conflicts with our maturity?

Actions          →  Actions & Export →  Who will do what, by when, and how will we track it?

Frameworks       →  Resources        →  Extra ways of framing choices (Playing to Win, Strategy Diamond, etc.)
