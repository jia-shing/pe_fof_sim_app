import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy_financial as npf

# Load cash flow scenario data
scenario_data = {
    "Top Quartile": pd.read_excel("FoF Programme Cashflows_Gaia vJul25  (JS).xlsx", sheet_name="Top Quartile", index_col=0),
    "Median Quartile": pd.read_excel("FoF Programme Cashflows_Gaia vJul25  (JS).xlsx", sheet_name="Median Quartile", index_col=0),
    "Bottom Quartile": pd.read_excel("FoF Programme Cashflows_Gaia vJul25  (JS).xlsx", sheet_name="Bottom Quartile", index_col=0),
}

# Streamlit layout
st.set_page_config(layout="wide")
st.title("PE Fund-of-Funds Return Simulator")

col1, col2 = st.columns([1, 3])

# Sidebar Inputs
with col1:
    st.subheader("Strategy Inputs")
    commitment_input = st.number_input("Initial Commitment (USD millions)", min_value=1, max_value=2000, value=10, step=1)
    step_up = st.number_input("Commitment Step-Up (%)", min_value=0, max_value=50, value=0, step=1)
    num_funds = st.slider("Number of Funds", min_value=1, max_value=15, value=1)
    performance = st.radio("Performance Scenario", ["Top Quartile", "Median Quartile", "Bottom Quartile"])

# Core calculations
scenario = scenario_data[performance]
fund_commitment = commitment_input * 1e6
step_up_amount = fund_commitment * (step_up / 100)

data = []
paid_in = 0
net_cf = []
cumulative_net_cf = []

for j in range(scenario.shape[1]):
    if j < 2:
        commitment = fund_commitment
    else:
        commitment = fund_commitment + step_up_amount

    capital_calls = -1 * scenario.loc["Capital Calls", f"Year {j+1}"] * commitment
    distributions = scenario.loc["Distributions", f"Year {j+1}"] * commitment
    residual_nav = scenario.loc["Residual NAV", f"Year {j+1}"] * commitment
    annual_net_cf = capital_calls + distributions

    paid_in += -capital_calls  # Since capital_calls is negative
    net_cf.append(annual_net_cf)
    cumulative_net_cf.append(sum(net_cf))

    data.append({
        "Year": j + 1,
        "Capital Calls": capital_calls,
        "Distributions": distributions,
        "Net Cash Flow": annual_net_cf,
        "Cumulative Net CF": cumulative_net_cf[-1],
    })

cf_df = pd.DataFrame(data)

# Compute metrics
final_cumulative_cf = cumulative_net_cf[-1]
total_distributions = cf_df["Distributions"].sum()
total_paid_in = paid_in
residual_nav_final = scenario.loc["Residual NAV"].values[-1] * (fund_commitment + step_up_amount)

net_cash_moic = (paid_in + final_cumulative_cf) / paid_in
net_tvpi = (total_distributions + residual_nav_final) / total_paid_in
net_dpi = total_distributions / total_paid_in
net_irr = npf.irr(net_cf)
max_net_cash_out = min(cumulative_net_cf)
max_net_cash_out_pct = max_net_cash_out / (fund_commitment + step_up_amount)

# Display metrics
with col2:
    st.subheader("Key Metrics")
    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.metric("Net TVPI", f"{net_tvpi:.2f}x")
    with metric_cols[1]:
        st.metric("Net DPI", f"{net_dpi:.2f}x")
    with metric_cols[2]:
        st.metric("Net IRR", f"{net_irr:.0%}" if not np.isnan(net_irr) else "nan%")

    metric_cols2 = st.columns(2)
    with metric_cols2[0]:
        st.metric("Net Cash MOIC", f"{net_cash_moic:.2f}x")
    with metric_cols2[1]:
        st.metric("Max Net Cash Out", f"${max_net_cash_out/1e6:.1f}M ({abs(max_net_cash_out_pct):.0%})")

# Chart
st.subheader("Portfolio Cash Flow Analysis")
fig, ax = plt.subplots()
years = cf_df["Year"]

ax.bar(years, cf_df["Capital Calls"] / 1e6, width=0.6, color="darkred", label="Capital Call")
ax.bar(years, cf_df["Distributions"] / 1e6, width=0.6, color="darkgreen", label="Distribution", bottom=cf_df["Capital Calls"] / 1e6)
ax.plot(years, cf_df["Net Cash Flow"] / 1e6, marker="o", color="orange", label="Annual Net Cash Flow")
ax.plot(years, cf_df["Cumulative Net CF"] / 1e6, color="dodgerblue", linewidth=2, label="Cumulative Net CF (J-Curve)")

ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:.0f}M"))
ax.set_xlabel("Year")
ax.set_ylabel("Cash Flow (USD Millions)")
ax.legend()

st.pyplot(fig)

# Download CSV
csv = cf_df.to_csv(index=False)
st.download_button("Download Cash Flow CSV", csv, "cashflow.csv")
