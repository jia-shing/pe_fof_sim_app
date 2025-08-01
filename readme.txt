# PE Fund-of-Funds Return Simulator

This Streamlit app simulates the portfolio return journey for a Private Equity Fund-of-Funds program. It helps users analyze capital calls, distributions, and net performance metrics over a 13-year fund lifecycle, supporting multiple fund vintages.

---

## 📂 Files Included

| File                         | Description                                      |
|------------------------------|--------------------------------------------------|
| `streamlit_pef_sim.py`       | Main app code for the simulator                 |
| `fof_assumptions_template.csv` | CSV of scenario assumptions (Top, Median, Bottom) |
| `README.md`                  | This readme file                                |

---

## 🚀 Getting Started on GitHub + Streamlit Cloud

### 1. Upload to GitHub

1. [Create a GitHub account](https://github.com) if you don’t have one.
2. Create a **new public repository** (e.g., `pef-simulator`)
3. Upload the following files:
   - `streamlit_pef_sim.py`
   - `fof_assumptions_template.csv`
   - `README.md`
4. Make sure the files are in the **root** directory (not in subfolders).

---

### 2. Deploy on Streamlit Cloud (Free)

1. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
2. Click **"Get Started"** and sign in with GitHub
3. Select the repository you just created
4. Fill in deployment settings:
   - **Main file path**: `streamlit_pef_sim.py`
   - **Python version**: 3.9+ recommended
   - It will automatically detect packages like `pandas`, `streamlit`, etc.
5. Click **Deploy**

✅ Done! You’ll get a public shareable link like:
```
https://<your-app-name>.streamlit.app/
```

---

## 🧪 How to Use

1. Enter a base commitment amount (e.g., $10M)
2. Define your step-up % (e.g., 10% increase per fund)
3. Select scenario: Top, Median, or Bottom Quartile
4. Choose which fund vintages to include
5. View outputs:
   - Net IRR, TVPI, DPI, Max Net Cash Out
   - Cash flow charts (Annual + Cumulative)

---

## 📥 Customizing the CSV

You can modify the `fof_assumptions_template.csv` to update scenarios. The format is:

```
Category,Year 1,Year 2,...Year 13
Top Quartile,,,
Capital Calls,-21.1,-29.1,...
Distributions,0.09,0.66,...
Residual NAV,20.5,51.1,...

Median Quartile,,,
Capital Calls,...
...
```

> All percentages are **relative to commitment size**.

---

## 📦 Requirements (automatically handled on Streamlit Cloud)
If running locally:
```bash
pip install streamlit pandas numpy numpy-financial altair
streamlit run streamlit_pef_sim.py
```

---

## 🙋 Support
If you run into issues or want to extend the functionality (e.g., fund-level breakdowns, scenario blending, export to Excel), open an issue or pull request!
