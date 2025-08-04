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
    .metric-container {
        display: flex;
        flex-direction: row;
        justify-content: space-between;
        gap: 16px;
        margin: 0.5rem 0 1rem 0;
    }
    .metric-box {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 16px;
        flex: 1;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-label {
        font-size: 1.1rem;
        color: #6c757d;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: bold;
        color: #3f51b5;
    }
    hr.divider {
        border: none;
        border-top: 1px solid #dee2e6;
        margin: 1.5rem 0 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <h1 style='font-size: 2.2rem; margin-bottom: 1.5rem;'>PE Fund-of-Funds Return Simulator</h1>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 3], gap="large")

with col1:
    st.subheader("Commitment Inputs")
    commitment_millions = st.number_input("Initial Commitment (USD millions)", min_value=1, max_value=2000, value=100, step=5, format="%d")
    commitment = commitment_millions * 1_000_000
    step_up = st.number_input("Commitment Step-Up (%)", min_value=0, max_value=50, value=0, step=1) / 100
    num_funds = st.slider("Number of Funds", 1, 15, 1)
    scenario_choice = st.radio("Performance Scenario", ["Top Quartile", "Median Quartile", "Bottom Quartile"], index=0, horizontal=False)
    enable_compare = st.checkbox("Enable Scenario Comparison")
    toggle_point_in_time = st.checkbox("Show Point-in-Time Metrics (based on year range)", value=True)

    if enable_compare:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.subheader("Comparison Inputs")
        commitment_millions_2 = st.number_input("Comparison Commitment (USD millions)", min_value=1, max_value=2000, value=100, step=5, key="commit2")
        commitment_2 = commitment_millions_2 * 1_000_000
        step_up_2 = st.number_input("Comparison Step-Up (%)", min_value=0, max_value=50, value=0, step=1, key="step2") / 100
        num_funds_2 = st.slider("Comparison Number of Funds", 1, 15, 1, key="fund2")
        scenario_choice_2 = st.radio("Comparison Scenario", ["Top Quartile", "Median Quartile", "Bottom Quartile"], index=1, horizontal=False, key="scen2")

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

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

# Year Filter
min_year = 1
max_year = horizon
year_range = st.slider("Display Year Range", min_value=1, max_value=max_year, value=(1, max_year))
end_index = year_range[1]

if toggle_point_in_time:
    capital_calls_vis = capital_calls[:end_index]
    distributions_vis = distributions[:end_index]
    residual_navs_vis = np.array([0]*(end_index-1) + [residual_navs[end_index-1]]) if end_index <= len(residual_navs) else residual_navs[:end_index]
    net_cf_vis = capital_calls_vis + distributions_vis
    cum_cf_vis = np.cumsum(net_cf_vis)
    paid_in = -np.sum(capital_calls_vis)
    total_dists = np.sum(distributions_vis)
    residual_total = residual_navs_vis[-1]
    net_irr = irr(net_cf_vis)
    max_net_out = cum_cf_vis.min()
    max_net_out_year = np.argmin(cum_cf_vis)
else:
    paid_in = -np.sum(capital_calls)
    total_dists = np.sum(distributions)
    residual_total = residual_navs[-1]
    net_irr = irr(net_cf)
    max_net_out = cum_cf.min()
    max_net_out_year = np.argmin(cum_cf)


def get_committed_capital_until_year(max_year_index, base_commitment, step_up, num_funds):
    total_commit = 0
    for i in range(num_funds):
        fund_start_year = i * 2
        if fund_start_year > max_year_index:
            break
        total_commit += base_commitment * ((1 + step_up) ** i)
    return total_commit

commit_until_max_out = get_committed_capital_until_year(max_net_out_year, commitment, step_up, num_funds)
net_out_pct = (abs(max_net_out) / commit_until_max_out) * 100 if commit_until_max_out else np.nan

cash_on_cash = ((cum_cf_vis[-1] if toggle_point_in_time else cum_cf[-1]) + abs(max_net_out)) / abs(max_net_out) if paid_in else np.nan


# Optional secondary scenario
if enable_compare:
    scenario_2 = scenarios[scenario_choice_2]
    horizon_2 = (num_funds_2 - 1) * 2 + 13
    horizon = max(horizon, horizon_2)
    cap2 = np.zeros(horizon)
    dist2 = np.zeros(horizon)
    nav2 = np.zeros(horizon)
    net2 = np.zeros(horizon)

    for i in range(num_funds_2):
        start_year = i * 2
        fund_commit = commitment_2 * ((1 + step_up_2) ** i)
        for j in range(13):
            year = start_year + j
            if year >= horizon:
                break
            call_amt2 = scenario_2.loc["Capital Calls", f"Year {j+1}"] * fund_commit
            dist_amt2 = scenario_2.loc["Distributions", f"Year {j+1}"] * fund_commit
            cap2[year] += call_amt2
            dist2[year] += dist_amt2
            nav2[year] += scenario_2.loc["Residual NAV", f"Year {j+1}"] * fund_commit
            net2[year] += call_amt2 + dist_amt2
    cum2 = np.cumsum(net2)
    net_flow_compare = net2

# Year Filter
min_year = 1
max_year = horizon
year_range = st.slider("Display Year Range", min_value=1, max_value=max_year, value=(1, max_year))
range_mask = np.arange(horizon) + 1
visible_mask = (range_mask >= year_range[0]) & (range_mask <= year_range[1])

# Right Panel – Chart and Key Metrics
with col2:
    st.subheader("Illustrative Cashflows and Net Returns to Investor")

    metrics_html = f"""
    <div class='metric-container'>
      <div class='metric-box'>
        <div class='metric-label'>Net TVPI</div>
        <div class='metric-value'>{tvpi:.2f}x</div>
      </div>
      <div class='metric-box'>
        <div class='metric-label'>Net DPI</div>
        <div class='metric-value'>{dpi:.2f}x</div>
      </div>
      <div class='metric-box'>
        <div class='metric-label'>Net IRR</div>
        <div class='metric-value'>{net_irr * 100:.1f}%</div>
      </div>
      <div class='metric-box'>
        <div class='metric-label'>Cash-on-Cash Multiple</div>
        <div class='metric-value' title='Compares total net cash returned (including NAV) to maximum capital outlay over time.'>{cash_on_cash:.2f}x</div>
      </div>
      <div class='metric-box'>
        <div class='metric-label'>Max Net Cash Out</div>
        <div class='metric-value' title='The peak negative exposure faced by the investor (i.e. bottom of J-Curve), as a percentage of commitment.'>-${abs(max_net_out)/1e6:.1f}M ({net_out_pct:.0f}%)</div>
      </div>
    </div>
    """
    st.markdown(metrics_html, unsafe_allow_html=True)

    capital_calls_base = np.maximum(capital_calls, cap2) if enable_compare else capital_calls
    capital_calls_delta = (np.minimum(capital_calls, cap2) - capital_calls_base) if enable_compare else np.zeros_like(capital_calls)
    distributions_base = np.minimum(distributions, dist2) if enable_compare else distributions
    distributions_delta = (np.maximum(distributions, dist2) - distributions_base) if enable_compare else np.zeros_like(distributions)

    df_chart = pd.DataFrame({
        "Year": range_mask[visible_mask],
        "Capital Calls": capital_calls_base[visible_mask] / 1e6,
        "Distributions": distributions_base[visible_mask] / 1e6,
        "Capital Calls Delta": capital_calls_delta[visible_mask] / 1e6,
        "Distributions Delta": distributions_delta[visible_mask] / 1e6,
        "Net Cash Flow": net_cf[visible_mask] / 1e6,
        "Net CF Compare": net_flow_compare[visible_mask] / 1e6 if enable_compare else np.nan,
        "Cumulative Net CF": cum_cf[visible_mask] / 1e6,
        "Cumulative Net CF (Compare)": cum2[visible_mask] / 1e6 if enable_compare else np.nan
    })

    fig = go.Figure()
    fig.add_bar(x=df_chart["Year"], y=df_chart["Capital Calls"], name="Capital Call (Base)", marker_color="#8B0000")
    if enable_compare:
        fig.add_bar(x=df_chart["Year"], y=df_chart["Capital Calls Delta"], name="Capital Call (Delta)", marker_color="#FFA07A")
    fig.add_bar(x=df_chart["Year"], y=df_chart["Distributions"], name="Distribution (Base)", marker_color="#006400")
    if enable_compare:
        fig.add_bar(x=df_chart["Year"], y=df_chart["Distributions Delta"], name="Distribution (Delta)", marker_color="#90EE90")

    fig.add_trace(go.Scatter(x=df_chart["Year"], y=df_chart["Net Cash Flow"], name="Annual Net Cash Flow", mode="lines+markers", line=dict(color="#FFA500", width=2)))
    if enable_compare:
        fig.add_trace(go.Scatter(x=df_chart["Year"], y=df_chart["Net CF Compare"], name="Annual Net Cash Flow (Compare)", mode="lines+markers", line=dict(color="#FFD580", width=2, dash="dot")))

    fig.add_trace(go.Scatter(x=df_chart["Year"], y=df_chart["Cumulative Net CF"], name="Cumulative Net CF", mode="lines", line=dict(color="#1E90FF", width=3)))
    if enable_compare:
        fig.add_trace(go.Scatter(x=df_chart["Year"], y=df_chart["Cumulative Net CF (Compare)"], name="Cumulative Net CF (Compare)", mode="lines", line=dict(color="#87CEFA", width=2, dash="dash")))

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

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    with st.expander("Show Year-by-Year Table"):
        table_df = pd.DataFrame({
            "Year": list(range(1, len(net_cf)+1)),
            "Capital Calls": [f"(${x/1e6:.1f})" if x < 0 else f"{x/1e6:.1f}" for x in capital_calls],
            "Distributions": [f"(${x/1e6:.1f})" if x < 0 else f"{x/1e6:.1f}" for x in distributions],
            "Net Cash Flow": [f"(${x/1e6:.1f})" if x < 0 else f"{x/1e6:.1f}" for x in net_cf],
            "Cumulative Net CF": [f"(${x/1e6:.1f})" if x < 0 else f"{x/1e6:.1f}" for x in cum_cf]
        })
        st.dataframe(table_df, use_container_width=True)

    raw_df = pd.DataFrame({
        "Year": list(range(1, len(net_cf)+1)),
        "Capital Calls": capital_calls,
        "Distributions": distributions,
        "Net Cash Flow": net_cf,
        "Cumulative Net CF": cum_cf
    })
    st.download_button("Download Cash Flow CSV", raw_df.to_csv(index=False), file_name="cashflows.csv")
