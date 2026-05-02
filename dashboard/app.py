import sys
import os
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yaml
import logging

from ingestion.fred_client import FredClient
from ingestion.data_processor import DataProcessor
from pricing.bond_pricer import BondPricer
from pricing.irs_pricer import IRSPricer
from pricing.ledger_simulator import LedgerSimulator
from attribution.pl_decomposer import PLDecomposer

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Fixed Income P&L Engine",
    layout="wide",
)

st.title("Fixed Income P&L Engine")
st.caption("FVTPL & Amortised Cost | Rate / Carry / Residual Attribution | Break Investigation")

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "portfolio.yaml")

@st.cache_data(show_spinner="Loading portfolio config...")
def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

config = load_config()

with st.sidebar:
    st.header("Controls")
    window  = st.slider("Data Window (days)", 10, 60, 30)
    run     = st.button("Fetch & Price", type="primary", use_container_width=True)
    st.divider()

if not run:
    st.info("Enter your FRED API key in the sidebar and click **Fetch & Price** to begin.")
    st.stop()

with st.spinner("Fetching market data from FRED..."):
    try:
        client    = FredClient()
        raw_data  = client.fetch_all(window_days=window)
        processor = DataProcessor(raw_data["yield_curve"], raw_data["sofr"])
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        st.stop()

valuation_date = processor.latest_date()
prev_date      = processor.previous_business_date(valuation_date)
st.success(f"Market data loaded. Valuation date: **{valuation_date.date()}** | T-1: **{prev_date.date()}**")

bond_pricers = {}
irs_pricers  = {}
for asset in config["assets"]:
    if asset["type"] == "bond":
        bond_pricers[asset["id"]] = BondPricer(asset)
    elif asset["type"] == "irs":
        irs_pricers[asset["id"]]  = IRSPricer(asset)

ledger = LedgerSimulator(
    portfolio_config=config,   
    bond_pricers=bond_pricers,
    irs_pricers=irs_pricers,
    data_processor=processor,
    injected_break=0.0         
)

decomposer = PLDecomposer(
    portfolio_config=config,
    bond_pricers=bond_pricers,
    irs_pricers=irs_pricers,
    data_processor=processor,
)

with st.spinner("Running pricing & attribution..."):
    pl_df    = decomposer.decompose(valuation_date)
    recon_df = ledger.reconcile(valuation_date)
    recon_prev = ledger.reconcile(prev_date)

detail_df = pl_df[pl_df["asset_id"] != "TOTAL"]
total_row  = pl_df[pl_df["asset_id"] == "TOTAL"].iloc[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total P&L",   f"${total_row['total_pl_usd']:,.0f}")
col2.metric("Rate Move",   f"${total_row['rate_move_usd']:,.0f}")
col3.metric("Carry",       f"${total_row['carry_usd']:,.0f}")
col4.metric("Residual",    f"${total_row['residual_usd']:,.0f}",
            delta="Model Exception" if total_row["model_exception"] else "✓ Clean",
            delta_color="inverse" if total_row["model_exception"] else "normal")

st.divider()

st.subheader("P&L Attribution Waterfall")

assets    = detail_df["asset_id"].tolist()
rate_vals = detail_df["rate_move_usd"].tolist()
carry_vals = detail_df["carry_usd"].tolist()
resid_vals = detail_df["residual_usd"].tolist()

fig_wf = go.Figure()
fig_wf.add_trace(go.Bar(name="Rate Move", x=assets, y=rate_vals,
                        marker_color=["#EF4444" if v < 0 else "#3B82F6" for v in rate_vals]))
fig_wf.add_trace(go.Bar(name="Carry",     x=assets, y=carry_vals, marker_color="#10B981"))
fig_wf.add_trace(go.Bar(name="Residual",  x=assets, y=resid_vals,
                        marker_color=["#F59E0B" if abs(v) > 500 else "#9CA3AF" for v in resid_vals]))
fig_wf.update_layout(
    barmode="group",
    yaxis_title="P&L (USD)",
    legend=dict(orientation="h", y=1.1),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    height=380,
)
st.plotly_chart(fig_wf, use_container_width=True)

with st.expander("Portfolio-Level Waterfall (Stacked Contribution)"):
    components  = ["Rate Move", "Carry", "Residual", "Total P&L"]
    comp_values = [
        total_row["rate_move_usd"],
        total_row["carry_usd"],
        total_row["residual_usd"],
        total_row["total_pl_usd"],
    ]
    measures = ["relative", "relative", "relative", "total"]
    fig_total = go.Figure(go.Waterfall(
        name="P&L",
        measure=measures,
        x=components,
        y=comp_values,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#10B981"}},
        decreasing={"marker": {"color": "#EF4444"}},
        totals={"marker": {"color": "#3B82F6"}},
    ))
    fig_total.update_layout(height=300, showlegend=False,
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_total, use_container_width=True)

st.divider()

st.subheader("P&L Attribution Table")

def highlight_exception(row):
    color = "background-color: #FEF3C7" if row.get("model_exception") else ""
    return [color] * len(row)

display_pl = detail_df[[
    "asset_id", "book", "ifrs_category",
    "total_pl_usd", "rate_move_usd", "carry_usd", "residual_usd",
    "delta_yield_bps", "dv01_usd", "model_exception"
]].copy()

display_pl.columns = [
    "Asset", "Book", "IFRS Category",
    "Total P&L ($)", "Rate Move ($)", "Carry ($)", "Residual ($)",
    "ΔYield (bps)", "DV01 ($)", "Exception?"
]

fmt_cols = ["Total P&L ($)", "Rate Move ($)", "Carry ($)", "Residual ($)", "DV01 ($)"]
st.dataframe(
    display_pl.style
        .format({c: "{:,.2f}" for c in fmt_cols})
        .format({"ΔYield (bps)": "{:.2f}"})
        .apply(highlight_exception, axis=1),
    use_container_width=True,
    hide_index=True,
)

st.divider()

st.subheader("FO vs Ledger Reconciliation (Break Log)")

recon_t  = recon_df.copy()
recon_t1 = recon_prev.copy()

# Merge T and T-1
merged = recon_t.merge(
    recon_t1[["asset_id", "ledger_value_usd"]].rename(columns={"ledger_value_usd": "ledger_t1_usd"}),
    on="asset_id",
    how="left"
)
merged["ledger_daily_move"] = merged["ledger_value_usd"] - merged["ledger_t1_usd"]

display_recon = merged[[
    "asset_id", "ifrs_category", "book",
    "fo_value_usd", "ledger_value_usd", "break_usd", "break_flag"
]].copy()
display_recon.columns = [
    "Asset", "IFRS Category", "Book",
    "FO Value ($)", "Ledger Value ($)", "Break ($)", "Break Flag"
]

def style_break(row):
    if row["Break Flag"]:
        return ["background-color: #FEE2E2"] * len(row)
    return [""] * len(row)

fmt_recon = ["FO Value ($)", "Ledger Value ($)", "Break ($)"]
st.dataframe(
    display_recon.style
        .format({c: "{:,.2f}" for c in fmt_recon})
        .apply(style_break, axis=1),
    use_container_width=True,
    hide_index=True,
)

# Break summary
breaks = display_recon[display_recon["Break Flag"]]
if not breaks.empty:
    st.warning(f"**{len(breaks)} break(s) detected** exceeding threshold. "
               f"Total break: ${display_recon['Break ($)'].abs().sum():,.2f}")
    with st.expander("Break Details"):
        for _, b in breaks.iterrows():
            st.error(
                f"**{b['Asset']}** | {b['IFRS Category']} | "
                f"FO: ${b['FO Value ($)']:,.2f} | "
                f"Ledger: ${b['Ledger Value ($)']:,.2f} | "
                f"Break: **${b['Break ($)']:,.2f}**"
            )
else:
    st.success("No breaks detected. FO and Ledger values reconcile within threshold.")

st.divider()

with st.expander("Raw Yield Curve Data"):
    yc = processor.yield_curve.tail(10)
    st.dataframe(yc.style.format("{:.4%}"), use_container_width=True)

with st.expander("SOFR History"):
    sofr_df = processor.sofr.tail(10).reset_index()
    sofr_df.columns = ["Date", "SOFR"]
    st.dataframe(sofr_df.style.format({"SOFR": "{:.4%}"}), use_container_width=True)

st.caption("Data sourced from FRED")