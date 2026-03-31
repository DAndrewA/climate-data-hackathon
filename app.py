import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Optional OpenAI support
USE_OPENAI_AVAILABLE = True
try:
    from openai import OpenAI
except Exception:
    USE_OPENAI_AVAILABLE = False


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Carbon Intelligence Layer",
    page_icon="🌍",
    layout="wide"
)

SUMMARY_PATH = Path("./country_summary_2023_clusters.csv")

# ============================================================
# LOAD
# ============================================================

@st.cache_data
def load_summary():
    df = pd.read_csv(SUMMARY_PATH)

    numeric_cols = [
        "co2_total_mmtco2_2007",
        "co2_total_mmtco2_2023",
        "co2_per_capita_tonnes_2023",
        "co2_per_gdp_tonnes_per_usd_2023",
        "renewables_share_2023",
        "fossil_share_2023",
        "co2_change_2007_2023_pct",
        "gdp_change_2007_2023_pct",
        "co2_per_gdp_change_2007_2023_pct",
        "renewables_share_change_2007_2023_pct_points",
        "fossil_share_change_2007_2023_pct_points",
        "slope_co2_total_mmtco2",
        "slope_renewables_share",
        "co2_vs_gdp_ratio_2007_2023",
        "cluster_id",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


summary = load_summary()

if summary.empty:
    st.error("The summary file is empty or could not be loaded.")
    st.stop()


# ============================================================
# HELPERS
# ============================================================

def fmt_num(x, digits=2):
    if pd.isna(x):
        return "N/A"
    return f"{x:,.{digits}f}"

def fmt_pct(x, digits=1):
    if pd.isna(x):
        return "N/A"
    return f"{x:,.{digits}f}%"

def fmt_pp(x, digits=1):
    if pd.isna(x):
        return "N/A"
    return f"{x:,.{digits}f} percentage points"

def cluster_plain_english(cluster_name: str) -> str:
    mapping = {
        "Rapid emitters": "These are countries where emissions are rising quickly and the energy transition is still limited.",
        "Transitioning systems": "These are countries showing visible movement toward cleaner energy, even if the transition is not complete.",
        "Mixed pathway": "These are countries with no single clear direction yet — some signals improve, while others remain uncertain.",
        "Decoupling leaders": "These are countries managing to grow economically while reducing or stabilising emissions.",
        "Declining emitters": "These are countries where emissions are falling, although the reason may vary from structural transition to slower activity.",
        "Single available group": "Only one group was available in this run, so no meaningful cluster separation was possible."
    }
    return mapping.get(cluster_name, "This cluster represents a transition pattern identified from the data.")

def build_rule_based_explanation(row: pd.Series) -> str:
    parts = []

    co2_change = row.get("co2_change_2007_2023_pct", np.nan)
    gdp_change = row.get("gdp_change_2007_2023_pct", np.nan)
    decouple = row.get("co2_vs_gdp_ratio_2007_2023", np.nan)
    ren_change = row.get("renewables_share_change_2007_2023_pct_points", np.nan)
    fossil_change = row.get("fossil_share_change_2007_2023_pct_points", np.nan)
    co2_slope = row.get("slope_co2_total_mmtco2", np.nan)
    ren_slope = row.get("slope_renewables_share", np.nan)
    cluster = row.get("cluster_name", "Unknown")

    parts.append(f"This country is grouped under **{cluster}**.")
    parts.append(cluster_plain_english(cluster))

    if pd.notna(co2_change) and pd.notna(gdp_change):
        if co2_change < 0 and gdp_change > 0:
            parts.append("The country grew economically while reducing emissions, which is a strong sign of decoupling.")
        elif co2_change > 0 and gdp_change > 0:
            if pd.notna(decouple) and decouple < 1:
                parts.append("Emissions increased, but more slowly than GDP, suggesting relative decoupling rather than fully carbon-intensive growth.")
            else:
                parts.append("Emissions and GDP both increased, and the growth pattern still appears carbon-heavy.")
        elif co2_change < 0:
            parts.append("Emissions declined over the period, reducing the country’s carbon footprint.")
        else:
            parts.append("The emissions-growth relationship is mixed and needs careful interpretation.")

    if pd.notna(ren_change):
        if ren_change > 5:
            parts.append("Renewables grew strongly, which points to a structural shift in the energy system.")
        elif ren_change > 0:
            parts.append("Renewables increased, but the transition remains moderate.")
        else:
            parts.append("There is limited evidence of strong renewable expansion.")

    if pd.notna(fossil_change):
        if fossil_change < 0:
            parts.append("Fossil dependence declined, which is a positive transition signal.")
        elif fossil_change > 0:
            parts.append("Fossil dependence stayed high or increased, which weakens the transition story.")

    if pd.notna(co2_slope) and pd.notna(ren_slope):
        if co2_slope < 0 and ren_slope > 0:
            parts.append("Recent trends are favorable: emissions are trending down while renewables continue to rise.")
        elif co2_slope > 0 and ren_slope <= 0:
            parts.append("Recent trends suggest continued climate risk, with emissions still rising and limited renewable momentum.")
        else:
            parts.append("Recent trends suggest a transition that is still in motion but not yet settled.")

    return " ".join(parts)


def build_feature_evidence(row: pd.Series):
    return {
        "CO₂ change (2007→2023)": fmt_pct(row.get("co2_change_2007_2023_pct")),
        "GDP change (2007→2023)": fmt_pct(row.get("gdp_change_2007_2023_pct")),
        "Renewables shift": fmt_pp(row.get("renewables_share_change_2007_2023_pct_points")),
        "Fossil share shift": fmt_pp(row.get("fossil_share_change_2007_2023_pct_points")),
        "Decoupling ratio": fmt_num(row.get("co2_vs_gdp_ratio_2007_2023"), 2),
        "2023 CO₂ per capita": f"{fmt_num(row.get('co2_per_capita_tonnes_2023'))} tonnes",
        "2023 CO₂ per GDP": fmt_num(row.get("co2_per_gdp_tonnes_per_usd_2023"), 8),
        "Recent CO₂ slope": fmt_num(row.get("slope_co2_total_mmtco2"), 4),
        "Recent renewables slope": fmt_num(row.get("slope_renewables_share"), 4),
    }


def cluster_technical_explanation() -> str:
    return """
**What the clustering is doing technically**

We use **KMeans clustering**, which is an **unsupervised learning** algorithm.  
That means the model is not trained on hand-labeled country categories.  
Instead, each country is represented as a vector of transition features, including:

- emissions level,
- emissions per person,
- emissions per unit of GDP,
- renewable share,
- fossil share,
- emissions change from 2007 to 2023,
- GDP change from 2007 to 2023,
- carbon-efficiency change,
- renewable-share change,
- recent emissions slope,
- recent renewables slope,
- and the decoupling ratio.

Before clustering, the features are standardized so that no single variable dominates because of scale alone.  
KMeans then groups countries around centroids in this normalized feature space.  
So two countries end up in the same cluster when their **transition behavior** is similar — not necessarily because they emit the same absolute amount.
"""


def call_openai_explainer(row: pd.Series) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "OPENAI_API_KEY is not set, so the LLM explainer cannot run."

    if not USE_OPENAI_AVAILABLE:
        return "The OpenAI Python SDK is not installed."

    client = OpenAI(api_key=api_key)

    prompt = f"""
You are helping explain a machine-learning climate clustering result to a hackathon jury.
Write in simple, precise English.
Do not mention code.
Do not mention JSON.
Do not use bullet points.
Keep it to 180-240 words.

Country: {row.get('country_name')}
Cluster: {row.get('cluster_name')}
CO2 change 2007-2023 (%): {row.get('co2_change_2007_2023_pct')}
GDP change 2007-2023 (%): {row.get('gdp_change_2007_2023_pct')}
Renewables share change (percentage points): {row.get('renewables_share_change_2007_2023_pct_points')}
Fossil share change (percentage points): {row.get('fossil_share_change_2007_2023_pct_points')}
Decoupling ratio: {row.get('co2_vs_gdp_ratio_2007_2023')}
CO2 slope: {row.get('slope_co2_total_mmtco2')}
Renewables slope: {row.get('slope_renewables_share')}
CO2 per capita 2023: {row.get('co2_per_capita_tonnes_2023')}
CO2 per GDP 2023: {row.get('co2_per_gdp_tonnes_per_usd_2023')}

Explain:
1. what this cluster means in simple words,
2. why this country belongs there,
3. what the transition story is,
4. what a jury should take away from this result.
"""

    response = client.responses.create(
        model="gpt-5.4",
        input=prompt
    )
    return response.output_text.strip()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Country intelligence")

country_df = summary[["country_name", "country_code"]].drop_duplicates().sort_values("country_name")
country_labels = [f"{r.country_name} ({r.country_code})" for r in country_df.itertuples(index=False)]
country_map = dict(zip(country_labels, country_df["country_code"]))

selected_label = st.sidebar.selectbox("Select a country", country_labels)
selected_code = country_map[selected_label]

use_llm = st.sidebar.toggle("Use OpenAI LLM explanation", value=False)
show_technical = st.sidebar.toggle("Show technical clustering note", value=True)

row_df = summary[summary["country_code"] == selected_code]
if row_df.empty:
    st.error("No country record found.")
    st.stop()

row = row_df.iloc[0]


# ============================================================
# HEADER
# ============================================================

st.title("Country Intelligence Layer")
st.caption("This view translates the machine-learning clustering into understandable climate transition insight.")

st.markdown(f"## {row['country_name']} ({row['country_code']})")

cluster_col, badge_col = st.columns([2, 1])

with cluster_col:
    st.markdown(f"### Cluster: **{row.get('cluster_name', 'Unknown')}**")
    st.markdown(cluster_plain_english(row.get("cluster_name", "Unknown")))

with badge_col:
    dec = row.get("co2_vs_gdp_ratio_2007_2023", np.nan)
    if pd.notna(dec) and dec < 0:
        badge = "Strong decoupling"
    elif pd.notna(dec) and dec < 1:
        badge = "Relative decoupling"
    else:
        badge = "Transition pressure"
    st.metric("Transition signal", badge)


# ============================================================
# MAIN CARDS
# ============================================================

c1, c2, c3, c4 = st.columns(4)
c1.metric("CO₂ change", fmt_pct(row.get("co2_change_2007_2023_pct")))
c2.metric("GDP change", fmt_pct(row.get("gdp_change_2007_2023_pct")))
c3.metric("Renewables shift", fmt_pp(row.get("renewables_share_change_2007_2023_pct_points")))
c4.metric("Decoupling ratio", fmt_num(row.get("co2_vs_gdp_ratio_2007_2023"), 2))

st.markdown("### Why this country sits in this cluster")

if use_llm:
    st.write(call_openai_explainer(row))
else:
    st.write(build_rule_based_explanation(row))

st.markdown("### Evidence used by the model")

evidence = build_feature_evidence(row)
ec1, ec2 = st.columns(2)
items = list(evidence.items())

for k, v in items[: len(items)//2 + len(items)%2]:
    ec1.markdown(f"**{k}:** {v}")

for k, v in items[len(items)//2 + len(items)%2:]:
    ec2.markdown(f"**{k}:** {v}")

st.markdown("### Simple jury translation")
st.info(
    "This country was not grouped by its emissions alone. "
    "It was grouped by how its emissions, economic growth, and energy mix changed together over time. "
    "So the cluster tells us what kind of transition pathway the country is following."
)

if show_technical:
    with st.expander("Technical note: what clustering we used"):
        st.markdown(cluster_technical_explanation())
        st.markdown(
            "For a stronger technical appendix, you can also validate these archetypes with a density-based method such as HDBSCAN to check whether some countries are genuine outliers rather than forced into a centroid-based group."
        )