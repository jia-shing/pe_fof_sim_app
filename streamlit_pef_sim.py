
import streamlit as st
import pandas as pd
import numpy as np
from numpy_financial import irr
import altair as alt

# Load scenario assumptions
@st.cache_data
def load_assumptions():
    df = pd.read_csv("fof_assumptions_template.csv")
    blocks = {}
    current_block = None
    for i, row in df.iterrows():
        if pd.isna(row["Category"]):
            continue
        elif "Quartile" in str(row["Category"]):
            current_block = str(row["Category"]).strip()
            blocks[current_block] = []
        else:
            blocks[current_block].append(row)
    scenarios = {}
    for scenario, rows in blocks.items():
        df = pd.DataFrame(rows).drop(columns="Category").reset_index(drop=True)
        df.index = ["Capital Calls", "Distributions", "Residual NAV"]
        scenarios[scenario] = df.astype(float)
    return scenarios

scenarios = load_assumptions()

# UI Inputs
st.title("PE Fund-of-Funds Return Simulator")
st.markdown("Simulate net returns based on commitment size, fund selection, and scenario.")

commitment = st.number_input("Base Commitment (USD)", min_value=1_000_000, step=100_000, value=10_000_000)
step_up = st.number_input("Commitment Step-Up per Fund (%)", min_value=0.0, value=10.0) / 100
scenario_choice = st.selectbox("Scenario", ["Top Quartile", "Median Quartile", "Bottom Quartile"])
num_funds = 10
fund_years = [2024 + 2*i for i in range(num_funds)]
fund_select = st.multiselect("Select Funds to Invest In", options=[f"Fund {i+1} (Start: {y})" for i,y in enumerate(fund_years)], default=["Fund 1 (Start: 2024)"])

# Map selected funds
selected_funds = [int(s.split()[1]) for s in fund_select]

# Generate cashflows
def simulate_cashflows():
    scenario = scenarios[scenario_choice]
    horizon = 30
    annual_cf = np.zeros(horizon)

    for i, fund_id in enumerate(selected_funds):
        start_year = 2 * (fund_id - 1)
        fund_commitment = commitment * ((1 + step_up) ** (fund_id - 1))
        for j in range(13):
            year = start_year + j
            if year >= horizon:
                break
            calls = scenario.loc["Capital Calls", f"Year {j+1}"]
            dists = scenario.loc["Distributions", f"Year {j+1}"]
            nav = scenario.loc["Residual NAV", f"Year {j+1}"]
            annual_cf[year] += fund_commitment * (dists + nav + calls)

    return annual_cf

cashflows = simulate_cashflows()
cum_cf = np.cumsum(cashflows)

# Compute Metrics
paid_in = -np.sum(cashflows[cashflows < 0])
dists_nav = np.sum(cashflows[cashflows > 0])
tvpi = dists_nav / paid_in if paid_in != 0 else np.nan
dpi = np.sum(cashflows[cashflows > 0]) / paid_in if paid_in != 0 else np.nan
net_irr = irr(cashflows)
max_net_out = np.min(np.cumsum(cashflows))

# Output Metrics
st.header("Results")
col1, col2, col3 = st.columns(3)
col1.metric("Net TVPI", f"{tvpi:.2f}")
col2.metric("Net DPI", f"{dpi:.2f}")
col3.metric("Net IRR", f"{net_irr*100:.2f}%")
st.metric("Max Net Cash Out (Bottom of J Curve)", f"${max_net_out:,.0f}")

# Charts
df_chart = pd.DataFrame({
    "Year": list(range(1, len(cashflows)+1)),
    "Capital Calls": [c if c < 0 else 0 for c in cashflows],
    "Distributions": [c if c > 0 else 0 for c in cashflows],
    "Net Cash Flow": cashflows,
    "Cumulative Net CF": cum_cf
})

st.subheader("Cash Flow Charts")
line = alt.Chart(df_chart.reset_index()).transform_fold(
    ["Capital Calls", "Distributions", "Net Cash Flow"]
).mark_bar().encode(
    x="Year:O",
    y="value:Q",
    color="key:N"
).properties(height=300)

cum_line = alt.Chart(df_chart).mark_line().encode(
    x="Year",
    y="Cumulative Net CF",
).properties(height=300)

st.altair_chart(line, use_container_width=True)
st.altair_chart(cum_line, use_container_width=True)
