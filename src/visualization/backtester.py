"""
BESS Tolling Value Backtester
==============================
Standalone Streamlit app for modelling historical BESS revenue.

Run with:
    streamlit run src/visualization/backtester.py
"""

import sys
from pathlib import Path

# Ensure src/ is on the path when launched from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from src.analysis.revenue_stack import (
    BatterySpec,
    run_backtest,
    sensitivity_table,
    ALL_SERVICES,
    SERVICE_LABELS,
    SERVICE_COLOURS,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="BESS Tolling Value Backtester",
    page_icon="🔋",
    layout="wide",
)

RAW = Path(__file__).parent.parent.parent / "data" / "raw"


# ---------------------------------------------------------------------------
# Data loading (mirrors dashboard.py loaders)
# ---------------------------------------------------------------------------

def _concat_csvs(paths: list, parse_dates: list, dedup_cols: list) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    frames = [pd.read_csv(p, parse_dates=parse_dates) for p in paths]
    merged = pd.concat(frames, ignore_index=True)
    if dedup_cols and all(c in merged.columns for c in dedup_cols):
        merged = merged.drop_duplicates(subset=dedup_cols)
    return merged


@st.cache_data
def load_auctions() -> pd.DataFrame:
    legacy = sorted(RAW.glob("auction_results_*.csv"))
    eac    = sorted(RAW.glob("eac_results_*.csv"))
    df = _concat_csvs(
        legacy + eac,
        parse_dates=["EFA Date", "Delivery Start", "Delivery End"],
        dedup_cols=["Service", "EFA Date", "EFA"],
    )
    return df.sort_values("EFA Date").reset_index(drop=True) if not df.empty else df


@st.cache_data
def load_market_index() -> pd.DataFrame:
    return _concat_csvs(
        sorted(RAW.glob("market_index_*.csv")),
        parse_dates=["settlementDate", "startTime"],
        dedup_cols=["settlementDate", "settlementPeriod", "dataProvider"],
    )


auctions   = load_auctions()
mkt_index  = load_market_index()

# ---------------------------------------------------------------------------
# Sidebar — parameters
# ---------------------------------------------------------------------------

st.sidebar.title("Battery Parameters")

power_mw = st.sidebar.slider("Power (MW)", min_value=1, max_value=500, value=50, step=1)

duration_h = st.sidebar.select_slider(
    "Duration (hours)",
    options=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
    value=2.0,
)

efficiency_pct = st.sidebar.slider(
    "Round-trip efficiency (%)", min_value=80, max_value=98, value=90, step=1
)

cycling_cost = st.sidebar.number_input(
    "Cycling wear cost (£/MWh)",
    min_value=0.0,
    max_value=20.0,
    value=3.0,
    step=0.5,
    help="Cost per MWh of usable energy cycled — accounts for battery degradation.",
)

availability_pct = st.sidebar.slider(
    "Availability factor (%)",
    min_value=80,
    max_value=100,
    value=95,
    step=1,
    help=(
        "Fraction of periods the asset is available. 95% reflects the minimum threshold "
        "specified in NESO's DC and EAC service agreements, and is consistent with observed "
        "GB BESS fleet availability (Modo Energy, 'GB Battery Storage Report', 2024)."
    ),
)

st.sidebar.divider()
st.sidebar.subheader("Services to Include")

selected_services = st.sidebar.multiselect(
    "Frequency response services",
    options=ALL_SERVICES,
    default=ALL_SERVICES,
    format_func=lambda s: f"{s} — {SERVICE_LABELS[s]}",
)

include_imbalance = st.sidebar.checkbox("Include imbalance arbitrage", value=True)

st.sidebar.divider()
st.sidebar.subheader("Date Range")

# Determine available overlap between auction and market index data
if not auctions.empty and not mkt_index.empty:
    data_start = max(auctions["EFA Date"].min(), mkt_index["settlementDate"].min()).date()
    data_end   = min(auctions["EFA Date"].max(), mkt_index["settlementDate"].max()).date()
else:
    import datetime
    data_start = datetime.date(2023, 7, 1)
    data_end   = datetime.date(2026, 2, 19)

date_range = st.sidebar.date_input(
    "Backtest period",
    value=(data_start, data_end),
    min_value=data_start,
    max_value=data_end,
)

start_date = date_range[0] if len(date_range) == 2 else data_start
end_date   = date_range[1] if len(date_range) == 2 else data_end

# ---------------------------------------------------------------------------
# Run backtest
# ---------------------------------------------------------------------------

battery = BatterySpec(
    power_mw=power_mw,
    duration_h=duration_h,
    efficiency_rt=efficiency_pct / 100,
    cycling_cost_per_mwh=cycling_cost,
    availability_factor=availability_pct / 100,
)

mi_input = mkt_index if include_imbalance else pd.DataFrame()

if auctions.empty:
    st.error("No auction data found. Run the data collector first.")
    st.stop()

result  = run_backtest(auctions, mi_input, battery, selected_services, start_date, end_date)
monthly = result["monthly"]
summary = result["summary"]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("BESS Tolling Value Backtester")
st.markdown(
    f"Modelling a **{power_mw} MW / {power_mw * duration_h:.0f} MWh** battery "
    f"({duration_h}h duration, {efficiency_pct}% round-trip efficiency) "
    f"from **{start_date}** to **{end_date}**."
)

# ---------------------------------------------------------------------------
# Top-level metrics
# ---------------------------------------------------------------------------

if summary:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Total Net Revenue",
        f"£{summary['total_net'] / 1_000:,.0f}k",
    )
    c2.metric(
        "Annualised Net Revenue",
        f"£{summary['annualised_net'] / 1_000:,.0f}k / yr",
    )
    c3.metric(
        "Revenue per MW",
        f"£{summary['annualised_per_mw'] / 1_000:,.1f}k / MW / yr",
    )
    c4.metric(
        "Top Revenue Stream",
        SERVICE_LABELS.get(summary.get("top_service", ""), summary.get("top_service", "N/A")),
    )
else:
    st.warning("No results — check that services are selected and data covers the chosen period.")
    st.stop()

st.divider()

# ---------------------------------------------------------------------------
# Chart 1: Monthly stacked bar — revenue by stream + cycling cost deduction
# ---------------------------------------------------------------------------

if not monthly.empty:
    st.subheader("Monthly Revenue Stack")

    fig = go.Figure()

    # Revenue streams (positive bars)
    stream_cols = {
        **{f"{s}_rev": s for s in ALL_SERVICES},
        "imbalance_revenue_gbp": "Imbalance",
    }

    for col, label in stream_cols.items():
        if col not in monthly.columns:
            continue
        if monthly[col].sum() == 0:
            continue
        display_label = SERVICE_LABELS.get(label, label)
        fig.add_trace(go.Bar(
            x=monthly["month_dt"],
            y=monthly[col] / 1_000,
            name=display_label,
            marker_color=SERVICE_COLOURS.get(label, "#888"),
        ))

    # Cycling cost (negative bar)
    if "cycling_cost_gbp" in monthly.columns and monthly["cycling_cost_gbp"].sum() > 0:
        fig.add_trace(go.Bar(
            x=monthly["month_dt"],
            y=-monthly["cycling_cost_gbp"] / 1_000,
            name="Cycling wear cost",
            marker_color=SERVICE_COLOURS["Cycling cost"],
            opacity=0.8,
        ))

    fig.update_layout(
        barmode="relative",
        height=450,
        template="plotly_white",
        yaxis_title="£k",
        xaxis_title="Month",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Charts 2a + 2b: Cumulative revenue | Revenue breakdown pie
# ---------------------------------------------------------------------------

left, right = st.columns([2, 1])

with left:
    st.subheader("Cumulative Net Revenue")
    monthly_sorted = monthly.sort_values("month_dt")
    monthly_sorted["cumulative_net"] = monthly_sorted["net_revenue"].cumsum()

    fig2 = go.Figure(go.Scatter(
        x=monthly_sorted["month_dt"],
        y=monthly_sorted["cumulative_net"] / 1_000,
        mode="lines",
        fill="tozeroy",
        line=dict(color="#1f77b4", width=2),
        fillcolor="rgba(31,119,180,0.15)",
    ))
    fig2.update_layout(
        height=380,
        template="plotly_white",
        yaxis_title="£k",
        xaxis_title="Month",
    )
    st.plotly_chart(fig2, use_container_width=True)

with right:
    st.subheader("Revenue Breakdown")
    bd = summary.get("breakdown", {})
    if bd:
        labels = [SERVICE_LABELS.get(k, k) for k in bd.keys()]
        values = list(bd.values())
        colours = [SERVICE_COLOURS.get(k, "#888") for k in bd.keys()]

        fig3 = go.Figure(go.Pie(
            labels=labels,
            values=values,
            marker_colors=colours,
            hole=0.45,
            textinfo="label+percent",
            hovertemplate="%{label}: £%{value:,.0f}<extra></extra>",
        ))
        fig3.update_layout(height=380, showlegend=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Sensitivity table
# ---------------------------------------------------------------------------

st.subheader("Sensitivity: Revenue by Battery Size")
st.caption("Other parameters held constant at sidebar values.")

sens_df = sensitivity_table(
    auctions,
    mi_input,
    battery,
    power_range=[5, 10, 25, 50, 100, 200],
    start_date=start_date,
    end_date=end_date,
)

st.dataframe(
    sens_df.style.format({
        "Energy (MWh)":             "{:.0f}",
        "Total Net Revenue (£k)":   "{:,.1f}",
        "Ann. Net Revenue (£k/yr)": "{:,.1f}",
        "Revenue / MW (£k/MW/yr)":  "{:,.1f}",
    }),
    use_container_width=True,
    hide_index=True,
)

st.divider()

# ---------------------------------------------------------------------------
# Methodology notes (collapsible)
# ---------------------------------------------------------------------------

with st.expander("Methodology & Assumptions"):
    st.markdown(f"""
**Ancillary service availability revenue**
- Revenue = `clearing_price (£/MW/h) × {power_mw} MW × 4 hours per EFA block`
- Services of different response speeds (DC, DR, DM) can be stacked on the same physical MW
  in the GB market — each earns a separate availability payment.
- High (discharge) and Low (charge) services are modelled as independent and simultaneous,
  assuming the battery maintains sufficient SoC headroom to respond in both directions.
- Clearing prices sourced from NESO Data Portal (legacy DC/DR/DM auctions Sep 2021–Nov 2023,
  EAC service Nov 2023–present).

**Wholesale energy arbitrage revenue**
- One arbitrage cycle per day maximum.
- Charge in the **{int(duration_h * 2)} cheapest** settlement periods; discharge in the
  **{int(duration_h * 2)} most expensive** periods.
- Gross profit = `avg_discharge_price × {power_mw * duration_h:.0f} MWh`
  − `avg_charge_price × {power_mw * duration_h / (efficiency_pct / 100):.1f} MWh`
  (extra input energy needed to account for {100 - efficiency_pct}% round-trip losses).
- Cycle only executed on days where gross profit exceeds the cycling wear cost.
- **Price reference: APXMIDP market index** (APX Power UK) from Elexon Insights
  (Jul 2023–present). This is the actual GB spot settlement reference price, giving a
  materially more realistic daily spread than the imbalance settlement price (SSP),
  which can reach extreme negative values during high-renewable periods and would
  otherwise inflate arbitrage revenue by ~77%.

**Availability factor ({availability_pct}%)**
- Applied as a uniform multiplier to all revenue streams and cycling costs.
- Models periods where the asset is unavailable due to planned maintenance, unplanned
  faults, grid curtailment, or service delivery failures.
- Default of 95% reflects the minimum availability threshold mandated in NESO's Dynamic
  Containment and Enduring Auction Capability service specifications. This is also
  consistent with observed GB BESS fleet performance: Modo Energy's *GB Battery Storage
  Report* (2024) reports median fleet availability of 95–97% across contracted windows.

**Cycling wear cost**
- Applied to imbalance arbitrage trades only: `{cycling_cost} £/MWh × MWh discharged per trade`.
- Ancillary service cycling (energy delivered during frequency events) is not separately
  modelled — it is minor relative to availability payments and is typically compensated
  via the service contract.

**What this model does not capture**
- Intraday / day-ahead market trading (DA price data not yet integrated)
- Balancing Mechanism (BM) direct trading
- Battery degradation over the backtest period
- Real-time dispatch constraints or grid connection limits
- **SoC interaction between FR services and energy arbitrage**: whether the battery can
  participate in arbitrage at any given moment depends continuously on its current state
  of charge, which is itself the product of all prior dispatch decisions across every
  active market. This is a joint optimisation problem — not reducible to a simple
  headroom cap — and is the primary motivation for the LP/MIP dispatch optimiser
  planned as the next analytical module.
    """)
