# explore_tab.py
import pandas as pd
import plotly.express as px
import streamlit as st


def _simple_search(df_in: pd.DataFrame, query: str) -> pd.DataFrame:
    if not query:
        return df_in

    cols = [c for c in ["title", "organisation", "summary", "scope"] if c in df_in.columns]
    if not cols:
        return df_in

    text = df_in[cols[0]].astype(str)
    for c in cols[1:]:
        text = text + " " + df_in[c].astype(str)

    mask = text.str.contains(query, case=False, na=False)
    return df_in[mask]


def _render_charts(fdf: pd.DataFrame) -> None:
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
        c1.info("No numeric 'year' values to chart.")

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
            hover_data=["title", "country", "scope"] if "scope" in sub.columns else ["title"],
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
            by_scope,
            names="scope",
            values="count",
            title="Strategy scope breakdown",
        )
        st.plotly_chart(fig_scope, use_container_width=True)


def render_explore(df: pd.DataFrame) -> None:
    st.subheader("Explore strategies")

    st.markdown(
        """
Use filters and search to find strategies by year, country, organisation type, and scope.  
Keyword search scans across title, organisation, summary and scope.
"""
    )

    # --- Filters ---
    with st.expander("Filters & search", expanded=True):
        years = sorted(y for y in df["year"].dropna().unique()) if "year" in df.columns else []
        if years:
            yr = st.slider(
                "Year range",
                int(min(years)),
                int(max(years)),
                (int(min(years)), int(max(years))),
            )
        else:
            yr = None

        org_types = sorted([v for v in df.get("org_type", pd.Series()).unique() if v != ""])
        org_type_sel = st.multiselect("Org type", org_types, default=org_types)

        countries = sorted([v for v in df.get("country", pd.Series()).unique() if v != ""])
        country_sel = st.multiselect("Country", countries, default=countries)

        scopes = sorted([v for v in df.get("scope", pd.Series()).unique() if v != ""])
        scope_sel = st.multiselect("Scope", scopes, default=scopes)

        q = st.text_input(
            "Search (keyword)",
            placeholder="e.g. 'federated data strategy', 'AI ethics', 'national open data'",
        )

    fdf = df.copy()

    if yr and "year" in fdf.columns:
        fdf = fdf[fdf["year"].between(yr[0], yr[1])]
    if org_type_sel and "org_type" in fdf.columns:
        fdf = fdf[fdf["org_type"].isin(org_type_sel)]
    if country_sel and "country" in fdf.columns:
        fdf = fdf[fdf["country"].isin(country_sel)]
    if scope_sel and "scope" in fdf.columns:
        fdf = fdf[fdf["scope"].isin(scope_sel)]

    if q:
        fdf = _simple_search(fdf, q)
        st.caption(f"{len(fdf)} strategies match your query.")

    if fdf.empty:
        st.warning("No strategies match the current filters/search. Try broadening your criteria.")
        return

    _render_charts(fdf)

    st.markdown("### Strategy details")
    for _, r in fdf.iterrows():
        year_str = int(r["year"]) if "year" in r and pd.notna(r["year"]) else "—"
        label = f"{r['title']} — {r['organisation']} ({year_str})"
        with st.expander(label):
            st.write(r.get("summary", "") or "_No summary provided._")
            meta = st.columns(4)
            meta[0].write(f"**Org type:** {r.get('org_type', '')}")
            meta[1].write(f"**Country:** {r.get('country', '')}")
            meta[2].write(f"**Scope:** {r.get('scope', '')}")
            meta[3].write(f"**Source:** {r.get('source', '')}")
            if r.get("link"):
                st.link_button("Open document", r["link"])
