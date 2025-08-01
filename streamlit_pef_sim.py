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
        # Normalize values to decimals (from percentage form)
        scenarios[scenario] = df.astype(float) / 100
    return scenarios

scenarios = load_assumptions()

# Layout
st.title("PE Fund-of-Funds Return Simulator")
left, right = st.columns([1, 2])

with left:
    st.subheader("Strategy Inputs")
    commitment = st.number_input("Initial Commitment (USD)", min_value=1_000_000, step=100_000, value=10_000_000)
    step_up = st.slider("Commitment Step-Up (%)", 0.0, 50.0, 10.0) / 100
    num_funds = st.slider("Number of Funds", 1, 15, 10)
    pacing = st.slider("Fund Pacing (Years)", 1, 5, 2)
    scenario_choice = st.radio("Performance Scenario", ["Top Quartile", "Median Quartile", "Bottom Quartile"], index=1)

# Calculate cashflows
scenario = scenarios[scenario_choice]
horizon = (num_funds - 1) * pacing + 13
capital_calls = np.zeros(horizon)
distributions = np.zeros(horizon)
net_cf = np.zeros(horizon)

for i in range(num_funds):
    start_year = i * pacing
    fund_commitment = commitment * ((1 + step_up) ** i)
    for j in range(13):
        year = start_year + j
        if year >= horizon:
            break
        call_amt = scenario.loc["Capital Calls", f"Year {j+1}"] * fund_commitment
        dist_amt = scenario.loc["Distributions", f"Year {j+1}"] * fund_commitment
        capital_calls[year] += call_amt
        distributions[year] += dist_amt
        net_cf[year] += dist_amt + call_amt

cum_cf = np.cumsum(net_cf)

# Metrics
paid_in = -np.sum(capital_calls[capital_calls < 0])
dists_total = np.sum(distributions)
tvpi = dists_total / paid_in if paid_in else np.nan
dpi = dists_total / paid_in if paid_in else np.nan
net_irr = irr(net_cf)
max_net_out = np.min(cum_cf)
net_cash_moic = (paid_in + cum_cf[-1]) / paid_in if paid_in else np.nan

# Output panel
with right:
    col1, col2 = st.columns(2)
    col1.metric("Net TVPI", f"{tvpi:.2f}x")
    col2.metric("Net DPI", f"{dpi:.2f}x")
    col3, col4 = st.columns(2)
    col3.metric("Net IRR", f"{net_irr * 100:.1f}%")
    col4.metric("Max Net Cash Out", f"${max_net_out:,.0f}")
    st.metric("Net Cash MOIC", f"{net_cash_moic:.2f}x")

    st.subheader("Portfolio Cash Flow Analysis")
    df_chart = pd.DataFrame({
        "Year": list(range(1, len(net_cf)+1)),
        "Capital Calls": capital_calls,
        "Distributions": distributions,
        "Net Cash Flow": net_cf,
        "Cumulative Net CF": cum_cf
    })

    fig = go.Figure()
    fig.add_bar(x=df_chart["Year"], y=df_chart["Capital Calls"], name="Capital Call", marker_color="#8B0000")
    fig.add_bar(x=df_chart["Year"], y=df_chart["Distributions"], name="Distribution", marker_color="#006400")
    fig.add_trace(go.Scatter(x=df_chart["Year"], y=df_chart["Net Cash Flow"], name="Annual Net Cash Flow", mode="lines+markers", line=dict(color="#FFA500", width=2)))
    fig.add_trace(go.Scatter(x=df_chart["Year"], y=df_chart["Cumulative Net CF"], name="Cumulative Net CF (J-Curve)", mode="lines", line=dict(color="#1E90FF", width=3)))
    fig.update_layout(
        barmode="relative",
        xaxis_title="Year",
        yaxis_title="Cash Flow (USD)",
        height=500,
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)
