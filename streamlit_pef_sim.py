import streamlit as st
import pandas as pd
import numpy as np
from numpy_financial import irr
import plotly.graph_objects as go

st.set_page_config(layout="wide")

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
        scenarios[scenario] = df.astype(float) / 100
    return scenarios

scenarios = load_assumptions()

# Style definitions
st.markdown("""
    <style>
    .metric-box {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 16px;
        width: 180px;
        flex: 0 0 auto;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6c757d;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: bold;
        color: #3f51b5;
    }
    </style>
""", unsafe_allow_html=True)

# Layout with Streamlit columns
col1, col2 = st.columns([1, 3], gap="large")

# Left Panel – Inputs
with col1:
    st.subheader("Commitment Inputs")
    commitment_millions = st.number_input("Initial Commitment (USD millions)", min_value=1, max_value=2000, value=100, step=5, format="%d")
    commitment = commitment_millions * 1_000_000
    step_up = st.number_input("Commitment Step-Up (%)", min_value=0, max_value=50, value=0, step=1) / 100
    num_funds = st.slider("Number of Funds", 1, 15, 1)
    scenario_choice = st.radio("Performance Scenario", ["Top Quartile", "Median Quartile", "Bottom Quartile"], index=0, horizontal=False)

# Right Panel – Metrics + Graph
with col2:
    st.markdown("""<div style='font-size: 1.25rem; font-weight: 600; margin-bottom: 0px;'>Key Metrics</div>""", unsafe_allow_html=True)

    scenario = scenarios[scenario_choice]
horizon = (num_funds - 1) * 2 + 13
capital_calls = np.zeros(horizon)
distributions = np.zeros(horizon)
residual_navs = np.zeros(horizon)
net_cf = np.zeros(horizon)

for i in range(num_funds):
    start_year = i * 2
    fund_commitment = commitment * ((1 + step_up) ** i)
    for j in range(13):
        year = start_year + j
        if year >= horizon:
            break
        call_amt = scenario.loc["Capital Calls", f"Year {j+1}"] * fund_commitment
        dist_amt = scenario.loc["Distributions", f"Year {j+1}"] * fund_commitment
        nav_amt = scenario.loc["Residual NAV", f"Year {j+1}"] * fund_commitment
        capital_calls[year] += call_amt
        distributions[year] += dist_amt
        residual_navs[year] += nav_amt
        net_cf[year] += call_amt + dist_amt

cum_cf = np.cumsum(net_cf)

paid_in = -np.sum(capital_calls)
total_dists = np.sum(distributions)
residual_total = residual_navs[-1]
tvpi = (total_dists + residual_total) / paid_in if paid_in else np.nan
dpi = total_dists / paid_in if paid_in else np.nan
net_irr = irr(net_cf)
max_net_out = cum_cf.min()
abs_max_net_out = abs(max_net_out)
cash_on_cash = (cum_cf[-1] + abs_max_net_out) / abs_max_net_out if paid_in else np.nan
net_out_pct = (abs(max_net_out) / commitment) * 100

# Metric Box Rendering
metrics_html = f"""
<div style='display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-start; margin-top: 0; max-width: 640px;'>
  <div class='metric-box' title='Total Value to Paid-In Capital'>
    <div class='metric-label'>Net TVPI</div>
    <div class='metric-value'>{tvpi:.2f}x</div>
  </div>
  <div class='metric-box' title='Distributions to Paid-In Capital'>
    <div class='metric-label'>Net DPI</div>
    <div class='metric-value'>{dpi:.2f}x</div>
  </div>
  <div class='metric-box' title='Internal Rate of Return'>
    <div class='metric-label'>Net IRR</div>
    <div class='metric-value'>{net_irr * 100:.1f}%</div>
  </div>
  <div class='metric-box' title='Cumulative Net Cash Returned'>
    <div class='metric-label'>Cash-on-Cash Multiple</div>
    <div class='metric-value'>{cash_on_cash:.2f}x</div>
  </div>
  <div class='metric-box' title='Maximum Net Capital Outlay'>
    <div class='metric-label'>Max Net Cash Out</div>
    <div class='metric-value'>-${abs(max_net_out)/1e6:.1f}M ({net_out_pct:.0f}%)</div>
  </div>
</div>
"""
st.markdown(metrics_html, unsafe_allow_html=True)

# Graph Section
st.subheader("Illustrative Cashflows and Net Returns to Investor")
df_chart = pd.DataFrame({
    "Year": list(range(1, len(net_cf)+1)),
    "Capital Calls": capital_calls / 1e6,
    "Distributions": distributions / 1e6,
    "Net Cash Flow": net_cf / 1e6,
    "Cumulative Net CF": cum_cf / 1e6
})

fig = go.Figure()
fig.add_bar(x=df_chart["Year"], y=df_chart["Capital Calls"], name="Capital Call", marker_color="#8B0000")
fig.add_bar(x=df_chart["Year"], y=df_chart["Distributions"], name="Distribution", marker_color="#006400")
fig.add_trace(go.Scatter(x=df_chart["Year"], y=df_chart["Net Cash Flow"], name="Annual Net Cash Flow", mode="lines+markers", line=dict(color="#FFA500", width=2)))
fig.add_trace(go.Scatter(x=df_chart["Year"], y=df_chart["Cumulative Net CF"], name="Cumulative Net CF (J-Curve)", mode="lines", line=dict(color="#1E90FF", width=3)))
fig.update_layout(
    barmode="relative",
    xaxis_title="Year",
    yaxis_title="Cash Flow (USD Millions)",
    yaxis_tickformat="$,.0f",
    yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)", zeroline=True),
    legend=dict(x=0.01, y=0.99),
    height=500,
    plot_bgcolor="white",
    margin=dict(l=20, r=20, t=20, b=20),
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# Download CSV with raw values
raw_df = pd.DataFrame({
    "Year": list(range(1, len(net_cf)+1)),
    "Capital Calls": capital_calls,
    "Distributions": distributions,
    "Net Cash Flow": net_cf,
    "Cumulative Net CF": cum_cf
})

st.download_button("Download Cash Flow CSV", raw_df.to_csv(index=False), file_name="cashflows.csv")
