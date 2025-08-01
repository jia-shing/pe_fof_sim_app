import streamlit as st
import pandas as pd
import numpy as np
from numpy_financial import irr
import plotly.graph_objects as go

st.set_page_config(layout="wide")

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

# Layout split
st.title("PE Fund-of-Funds Return Simulator")
left, right = st.columns([1, 2])

with left:
    st.subheader("Strategy Inputs")
    commitment = st.number_input("Initial Commitment (USD)", min_value=1_000_000, step=100_000, value=10_000_000)
    step_up = st.slider("Commitment Step-Up (%)", 0.0, 50.0, 10.0) / 100
    num_funds = st.slider("Number of Funds", 1, 15, 10)
    pacing = st.slider("Fund Pacing (Years)", 1, 5, 2)
    scenario_choice = st.radio("Performance Scenario", ["Top Quartile", "Median Quartile", "Bottom Quartile"], index=1)

# Cashflow simulation
scenario = scenarios[scenario_choice]
horizon = (num_funds - 1) * pacing + 13
annual_cf = np.zeros(horizon)

for i in range(num_funds):
    start_year = i * pacing
    fund_commitment = commitment * ((1 + step_up) ** i)
    for j in range(13):
        year = start_year + j
        if year >= horizon:
            break
        calls = scenario.loc["Capital Calls", f"Year {j+1}"]
        dists = scenario.loc["Distributions", f"Year {j+1}"]
        nav = scenario.loc["Residual NAV", f"Year {j+1}"]
        annual_cf[year] += fund_commitment * (dists + nav + calls)

cum_cf = np.cumsum(annual_cf)

# Metrics
paid_in = -np.sum(annual_cf[annual_cf < 0])
dists_nav = np.sum(annual_cf[annual_cf > 0])
tvpi = dists_nav / paid_in if paid_in != 0 else np.nan
dpi = np.sum(annual_cf[annual_cf > 0]) / paid_in if paid_in != 0 else np.nan
net_irr = irr(annual_cf)
max_net_out = np.min(np.cumsum(annual_cf))

# Output panel
with right:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Net TVPI", f"{tvpi:.2f}x")
    col2.metric("Net DPI", f"{dpi:.2f}x")
    col3.metric("Net IRR", f"{net_irr * 100:.1f}%")
    col4.metric("Max Net Cash Out", f"${max_net_out:,.0f}")

    st.subheader("Portfolio Cash Flow Analysis")
    df_chart = pd.DataFrame({
        "Year": list(range(1, len(annual_cf)+1)),
        "Capital Calls": [c if c < 0 else 0 for c in annual_cf],
        "Distributions": [c if c > 0 else 0 for c in annual_cf],
        "Net Cash Flow": annual_cf,
        "Cumulative Net CF": cum_cf
    })

    fig = go.Figure()
    fig.add_bar(x=df_chart["Year"], y=df_chart["Capital Calls"], name="Capital Call", marker_color="#8B0000")
    fig.add_bar(x=df_chart["Year"], y=df_chart["Distributions"], name="Distribution", marker_color="#006400")
    fig.add_trace(go.Scatter(x=df_chart["Year"], y=df_chart["Net Cash Flow"], name="Annual Net Cash Flow", mode="lines+markers", line=dict(color="#FFA500", width=2)))
    fig.add_trace(go.Scatter(x=df_chart["Year"], y=df_chart["Cumulative Net CF"], name="Cumulative Net CF (J-Curve)", mode="lines", line=dict(color="#1E90FF", width=3)))
    fig.update_layout(barmode="relative", xaxis_title="Year", yaxis_title="Cash Flow (USD)", height=500, plot_bgcolor="white")

    st.plotly_chart(fig, use_container_width=True)
