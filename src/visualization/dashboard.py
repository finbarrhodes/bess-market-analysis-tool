"""
GB BESS Market Dashboard

Launched as a page via app.py (st.navigation).
Can still be run standalone for local development:
    streamlit run src/visualization/dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ---------------------------------------------------------------------------
# Standalone guard — set_page_config only when run directly, not via app.py
# ---------------------------------------------------------------------------
try:
    st.set_page_config(
        page_title="GB BESS Market Dashboard",
        page_icon="⚡",
        layout="wide",
    )
except st.errors.StreamlitAPIException:
    pass  # already set by app.py

PROCESSED = Path(__file__).parent.parent.parent / "data" / "processed"

# EFA block timings (each block = 4 hours; EFA 1 spans midnight)
EFA_BLOCKS = {
    1: "23:00 – 03:00",
    2: "03:00 – 07:00",
    3: "07:00 – 11:00",
    4: "11:00 – 15:00",
    5: "15:00 – 19:00",
    6: "19:00 – 23:00",
}


# ---------------------------------------------------------------------------
# Data loading (cached, reads pre-processed Parquet files)
# ---------------------------------------------------------------------------

@st.cache_data
def load_auctions() -> pd.DataFrame:
    p = PROCESSED / "auctions.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


@st.cache_data
def load_system_prices() -> pd.DataFrame:
    p = PROCESSED / "system_prices.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


@st.cache_data
def load_market_index() -> pd.DataFrame:
    p = PROCESSED / "market_index.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


@st.cache_data
def load_generation() -> pd.DataFrame:
    """
    Returns daily generation totals by fuel group.
    Pre-aggregated by scripts/prepare_data.py — columns: settlementDate, fuelGroup, generation.
    """
    p = PROCESSED / "generation_daily.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


auctions   = load_auctions()
sys_prices = load_system_prices()
mkt_index  = load_market_index()
gen_fuel   = load_generation()  # already has fuelGroup column

# Sidebar filters
st.sidebar.title("Filters")

if not auctions.empty:
    all_services = sorted(auctions["Service"].unique())
    selected_services = st.sidebar.multiselect(
        "DC/DR/DM Services", all_services, default=all_services
    )

    date_min = auctions["EFA Date"].min().date()
    date_max = auctions["EFA Date"].max().date()
    date_range = st.sidebar.date_input(
        "Auction date range",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max,
    )

    auction_filtered = auctions[auctions["Service"].isin(selected_services)]
    if len(date_range) == 2:
        auction_filtered = auction_filtered[
            (auction_filtered["EFA Date"].dt.date >= date_range[0])
            & (auction_filtered["EFA Date"].dt.date <= date_range[1])
        ]
else:
    auction_filtered = auctions
    selected_services = []

# Header
st.title("GB BESS Market Dashboard")
st.markdown("Data from the **Elexon Insights Solution API** and **NESO Data Portal**.")

# Top-level metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Auction Records", f"{len(auction_filtered):,}")
col2.metric("System Price Records", f"{len(sys_prices):,}")
col3.metric("Market Index Records", f"{len(mkt_index):,}")
col4.metric("Generation Records", f"{len(gen_fuel):,}")

# Tab layout
tab_auction, tab_spread, tab_system, tab_gen, tab_cross = st.tabs(
    ["DC/DR/DM Auctions", "H vs L Spread", "System Prices", "Generation Mix", "System Price vs DC High"]
)

# ---------------------------------------------------------------------------
# Tab 1: Auctions
# ---------------------------------------------------------------------------
with tab_auction:
    if auction_filtered.empty:
        st.warning("No auction data loaded. Run scripts/prepare_data.py first.")
    else:
        st.markdown(
            """
            GB frequency response is procured through three **dynamic** services, each split into
            **High** (discharge — activated when frequency falls below 50 Hz) and **Low** (charge —
            activated when frequency rises above 50 Hz) auctions:

            | Service | Frequency band | Role |
            |---------|---------------|------|
            | **DC** – Dynamic Containment | ±0.2–0.5 Hz | Arrests large deviations within ~1 second |
            | **DR** – Dynamic Regulation | ±0.015–0.2 Hz | Maintains frequency in normal operation |
            | **DM** – Dynamic Moderation | ±0.1–0.5 Hz | Moderates frequency during stressed conditions |

            Auctions are run daily for each **EFA block** (six 4-hour windows covering the full day).
            The clearing price is the marginal accepted bid for that block and service.
            """
        )

        with st.expander("EFA Block Timings"):
            efa_df = pd.DataFrame(
                [(k, v) for k, v in EFA_BLOCKS.items()],
                columns=["EFA Block", "Time Window (local clock)"],
            )
            st.dataframe(efa_df, hide_index=True, use_container_width=True)
            st.caption(
                "EFA Block 1 spans midnight (23:00 the previous calendar day to 03:00). "
                "All times are local GB time."
            )

        st.subheader("Clearing Prices — 28-Day Rolling Average by Service")
        st.caption(
            "Individual auction results are first averaged to a daily figure per service, "
            "then smoothed with a 28-day rolling window. This makes the trend for each of "
            "the six services readable without the daily noise obscuring the signal."
        )

        # Daily average per service → 28-day rolling average
        daily_auction = (
            auction_filtered
            .groupby(["EFA Date", "Service"])["Clearing Price"]
            .mean()
            .reset_index()
            .sort_values("EFA Date")
        )
        rolling_parts = []
        for svc, grp in daily_auction.groupby("Service"):
            grp = grp.set_index("EFA Date").sort_index()
            grp["Rolling Avg (£/MW/h)"] = grp["Clearing Price"].rolling("28D").mean()
            grp["Service"] = svc
            rolling_parts.append(grp.reset_index())
        rolling_df = pd.concat(rolling_parts)

        fig = px.line(
            rolling_df,
            x="EFA Date",
            y="Rolling Avg (£/MW/h)",
            color="Service",
            labels={"EFA Date": "Date"},
        )
        fig.update_layout(height=450, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        left, right = st.columns(2)

        with left:
            st.subheader("Price Distribution by Service")
            fig = px.box(
                auction_filtered,
                x="Service",
                y="Clearing Price",
                color="Service",
                labels={"Clearing Price": "£/MW/h"},
            )
            fig.update_layout(height=400, template="plotly_white", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.subheader("Price by EFA Block")
            st.caption(
                "Evening blocks (EFA 5: 15:00–19:00, EFA 6: 19:00–23:00) and the overnight "
                "block (EFA 1: 23:00–03:00) often attract different premia depending on "
                "wind output and demand shape that day."
            )
            fig = px.box(
                auction_filtered,
                x="EFA",
                y="Clearing Price",
                color="Service",
                labels={
                    "Clearing Price": "£/MW/h",
                    "EFA": "EFA Block",
                },
            )
            fig.update_xaxes(
                tickmode="array",
                tickvals=list(EFA_BLOCKS.keys()),
                ticktext=[f"EFA {k}<br>{v}" for k, v in EFA_BLOCKS.items()],
            )
            fig.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Summary Statistics")
        stats = (
            auction_filtered.groupby("Service")
            .agg(
                avg_price=("Clearing Price", "mean"),
                median_price=("Clearing Price", "median"),
                max_price=("Clearing Price", "max"),
                avg_volume=("Cleared Volume", "mean"),
                records=("Clearing Price", "count"),
            )
            .round(2)
        )
        st.dataframe(stats, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2: H vs L Spread
# ---------------------------------------------------------------------------
with tab_spread:
    if auctions.empty:
        st.warning("No auction data loaded. Run scripts/prepare_data.py first.")
    else:
        st.markdown(
            """
            Each frequency response service runs two separate auctions: **High** (responds to
            falling frequency — BESS discharges) and **Low** (responds to rising frequency —
            BESS charges). The clearing prices can differ because the amount of available
            discharge vs charge headroom across the fleet is rarely symmetric.

            **Spread = H clearing price − L clearing price.** Positive = discharge capacity
            scarcer (H > L). Negative = charge capacity scarcer (L > H).
            """
        )

        with st.expander("EFA Block Timings"):
            efa_df = pd.DataFrame(
                [(k, v) for k, v in EFA_BLOCKS.items()],
                columns=["EFA Block", "Time Window (local clock)"],
            )
            st.dataframe(efa_df, hide_index=True, use_container_width=True)

        # Build a wide table: one row per (EFA Date, EFA block), columns H and L per market
        PAIRS = [("DC", "DCH", "DCL"), ("DR", "DRH", "DRL"), ("DM", "DMH", "DML")]

        spread_frames = []
        for market, h_svc, l_svc in PAIRS:
            h = auctions[auctions["Service"] == h_svc][["EFA Date", "EFA", "Clearing Price"]].rename(
                columns={"Clearing Price": "H"}
            )
            l = auctions[auctions["Service"] == l_svc][["EFA Date", "EFA", "Clearing Price"]].rename(
                columns={"Clearing Price": "L"}
            )
            merged = pd.merge(h, l, on=["EFA Date", "EFA"], how="inner")
            merged["Spread"] = merged["H"] - merged["L"]
            merged["Market"] = market
            spread_frames.append(merged)

        if not spread_frames:
            st.warning("Could not compute spreads — check that H and L services are both present.")
        else:
            spread_df = pd.concat(spread_frames, ignore_index=True)

            # ---- Chart 1: daily average spread over time ----
            st.subheader("Daily Average H − L Spread Over Time")
            daily_spread = (
                spread_df.groupby(["EFA Date", "Market"])["Spread"]
                .mean()
                .reset_index()
            )
            fig = px.line(
                daily_spread,
                x="EFA Date",
                y="Spread",
                color="Market",
                labels={"Spread": "£/MW/h", "EFA Date": "Date"},
            )
            fig.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
            fig.update_layout(height=420, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            # DR negative spread explanation (only show if the data supports it)
            dr_spread = spread_df[spread_df["Market"] == "DR"]["Spread"]
            if not dr_spread.empty and dr_spread.mean() < 0:
                st.info(
                    f"**Why is the DR spread consistently negative "
                    f"(avg {dr_spread.mean():.2f} £/MW/h)?**\n\n"
                    "DR (Dynamic Regulation) keeps frequency very close to 50 Hz during normal "
                    "operation, which means providers must maintain a **symmetrical active-power "
                    "window** — an equal amount of discharge headroom (DRH) and charge headroom "
                    "(DRL) must be available at all times.\n\n"
                    "This symmetry constraint pushes operators to target a mid-range state of "
                    "charge, but in practice the fleet's SoC distribution skews high during "
                    "periods of strong renewable output (exactly when frequency tends to rise and "
                    "charge response is most needed). A higher-than-average SoC means **less "
                    "remaining room to charge**, so fewer MW can be offered into the DRL auction "
                    "— reducing supply and driving DRL prices above DRH prices.\n\n"
                    "In short: the charge (DRL) side of the DR market is structurally "
                    "supply-constrained relative to the discharge (DRH) side, which inverts the "
                    "typical spread."
                )

            # ---- Chart 2: spread distribution per market ----
            left, right = st.columns(2)

            with left:
                st.subheader("Spread Distribution by Market")
                fig = px.box(
                    spread_df,
                    x="Market",
                    y="Spread",
                    color="Market",
                    labels={"Spread": "£/MW/h"},
                    points="outliers",
                )
                fig.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
                fig.update_layout(height=400, template="plotly_white", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            with right:
                st.subheader("Average Spread by EFA Block")
                efa_spread = (
                    spread_df.groupby(["EFA", "Market"])["Spread"]
                    .mean()
                    .reset_index()
                )
                fig = px.bar(
                    efa_spread,
                    x="EFA",
                    y="Spread",
                    color="Market",
                    barmode="group",
                    labels={"Spread": "Avg £/MW/h", "EFA": "EFA Block"},
                )
                fig.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
                fig.update_xaxes(
                    tickmode="array",
                    tickvals=list(EFA_BLOCKS.keys()),
                    ticktext=[f"EFA {k}<br>{v}" for k, v in EFA_BLOCKS.items()],
                )
                fig.update_layout(height=400, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

            # ---- Chart 3: heatmap — EFA block × month ----
            st.subheader("H − L Spread Heatmap: EFA Block × Month")
            spread_df["Month"] = spread_df["EFA Date"].dt.to_period("M").astype(str)
            selected_market = st.selectbox("Market", ["DC", "DR", "DM"], key="heatmap_market")
            heatmap_data = (
                spread_df[spread_df["Market"] == selected_market]
                .groupby(["Month", "EFA"])["Spread"]
                .mean()
                .reset_index()
                .pivot(index="Month", columns="EFA", values="Spread")
            )
            # Label columns with EFA block times
            heatmap_data.columns = [
                f"EFA {c} ({EFA_BLOCKS.get(c, '')})" for c in heatmap_data.columns
            ]
            fig = px.imshow(
                heatmap_data,
                color_continuous_scale="RdBu",
                color_continuous_midpoint=0,
                labels={"color": "£/MW/h", "x": "EFA Block", "y": "Month"},
                aspect="auto",
            )
            fig.update_layout(height=420, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            # ---- Summary table ----
            st.subheader("Summary Statistics")
            summary = (
                spread_df.groupby("Market")["Spread"]
                .agg(
                    mean="mean",
                    median="median",
                    std="std",
                    min="min",
                    max="max",
                    pct_positive=lambda s: (s > 0).mean() * 100,
                )
                .round(2)
            )
            summary.columns = ["Mean £/MW/h", "Median", "Std Dev", "Min", "Max", "% Days H > L"]
            st.dataframe(summary, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 3: System Prices
# ---------------------------------------------------------------------------
with tab_system:
    if sys_prices.empty:
        st.warning("No system price data loaded.")
    else:
        st.markdown(
            """
            **System Buy Price (SBP)** and **System Sell Price (SSP)** are the cash-out prices
            used to settle imbalance in the GB Balancing Mechanism. Parties that are *short*
            (consumed more than they contracted for) pay the SBP; parties that are *long* receive
            the SSP. SBP ≥ SSP always — the gap between them incentivises generators and suppliers
            to self-balance rather than rely on the system operator.
            """
        )

        st.subheader("Daily Average System Prices")
        daily_sp = (
            sys_prices.groupby("settlementDate")
            .agg(
                avg_ssp=("systemSellPrice", "mean"),
                avg_sbp=("systemBuyPrice", "mean"),
            )
            .reset_index()
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=daily_sp["settlementDate"],
                y=daily_sp["avg_ssp"],
                name="Avg SSP",
                line=dict(color="#EF553B"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=daily_sp["settlementDate"],
                y=daily_sp["avg_sbp"],
                name="Avg SBP",
                line=dict(color="#636EFA"),
            )
        )
        fig.update_layout(
            height=450,
            template="plotly_white",
            yaxis_title="£/MWh",
        )
        st.plotly_chart(fig, use_container_width=True)

        left, right = st.columns(2)

        with left:
            st.subheader("Intraday SSP Profile (by Settlement Period)")
            st.caption(
                "Each settlement period is 30 minutes. Period 1 = 00:00–00:30, "
                "Period 48 = 23:30–00:00. Evening periods (~34–42, 17:00–21:00) "
                "typically reflect peak-demand price uplift."
            )
            sp_profile = (
                sys_prices.groupby("settlementPeriod")["systemSellPrice"]
                .agg(["mean", "median"])
                .reset_index()
            )
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=sp_profile["settlementPeriod"],
                    y=sp_profile["mean"],
                    name="Mean",
                    mode="lines+markers",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=sp_profile["settlementPeriod"],
                    y=sp_profile["median"],
                    name="Median",
                    line=dict(dash="dash"),
                )
            )
            fig.update_layout(
                height=400,
                template="plotly_white",
                xaxis_title="Settlement Period",
                yaxis_title="£/MWh",
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.subheader("Daily SBP − SSP Imbalance Spread")
            st.caption(
                "A widening spread signals higher system stress — the ESO needed expensive "
                "balancing actions to correct imbalance. Near-zero spread indicates the system "
                "was broadly balanced on that day. Persistent spikes often follow large "
                "unexpected generation or demand changes (e.g. storm events, plant trips)."
            )
            daily_sp["imbalance_spread"] = daily_sp["avg_sbp"] - daily_sp["avg_ssp"]
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=daily_sp["settlementDate"],
                    y=daily_sp["imbalance_spread"],
                    name="SBP − SSP",
                    line=dict(color="#00CC96"),
                    fill="tozeroy",
                    fillcolor="rgba(0,204,150,0.12)",
                )
            )
            fig.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
            fig.update_layout(
                height=400,
                template="plotly_white",
                yaxis_title="£/MWh",
                xaxis_title="Date",
            )
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 4: Generation Mix
# ---------------------------------------------------------------------------
with tab_gen:
    if gen_fuel.empty:
        st.warning("No generation data loaded.")
    else:
        st.markdown(
            """
            GB grid generation broken down by fuel group. The mix matters for BESS because it
            shapes the underlying risk of frequency deviation: high wind + low demand tends to
            push system frequency high (requiring charge / Low-side response), while low wind +
            high demand can cause frequency dips (requiring discharge / High-side response).
            Tracking the long-run shift from fossil to renewable generation gives context for
            why frequency response procurement requirements have grown over time.
            """
        )

        gen_pivot = gen_fuel.pivot_table(
            index="settlementDate",
            columns="fuelGroup",
            values="generation",
            fill_value=0,
        )
        gen_pivot.index = pd.DatetimeIndex(gen_pivot.index)

        st.subheader("Weekly Average Generation by Fuel Group")
        st.caption(
            "Daily generation totals are resampled to weekly averages so that individual "
            "fuel group trends can be read without day-to-day noise. Each line represents "
            "one fuel group; the legend can be used to isolate specific sources."
        )

        gen_weekly = gen_pivot.resample("W").mean()
        col_order = gen_weekly.sum().sort_values(ascending=False).index
        gen_weekly = gen_weekly[col_order]

        melted_weekly = gen_weekly.reset_index().melt(
            id_vars="settlementDate",
            var_name="Fuel Group",
            value_name="Avg Generation (MW)",
        )
        fig = px.line(
            melted_weekly,
            x="settlementDate",
            y="Avg Generation (MW)",
            color="Fuel Group",
            labels={"settlementDate": "Week ending"},
        )
        fig.update_layout(height=500, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Average Share by Fuel Group")
        fuel_share = gen_pivot.mean()
        fuel_share = fuel_share[fuel_share > 0].sort_values(ascending=False)
        fig = px.pie(values=fuel_share.values, names=fuel_share.index)
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 5: System Price vs DC High
# ---------------------------------------------------------------------------
with tab_cross:
    if sys_prices.empty or auctions.empty:
        st.warning("Need both system price and auction data for cross-source analysis.")
    else:
        st.markdown(
            """
            **System Sell Price (SSP)** reflects real-time grid stress: a high SSP means the
            system was short and had to procure expensive balancing energy. **DC High (DCH)**
            clearing prices reflect what the market pays for fast-discharge frequency response
            contracted a day ahead.

            These prices come from different markets — imbalance settlement (real-time) vs.
            contracted frequency response (day-ahead) — but may co-move if they share common
            drivers. For example, tight supply margins could simultaneously push up energy
            prices and increase willingness to pay for frequency response insurance.
            A low correlation, on the other hand, suggests DCH prices are driven mainly by the
            frequency response fleet's own supply/demand dynamics, independent of wholesale
            energy conditions.
            """
        )

        st.subheader("System Price vs DC High Clearing Price")

        daily_sys = (
            sys_prices.groupby("settlementDate")["systemSellPrice"]
            .mean()
            .reset_index()
        )
        daily_sys.columns = ["date", "avg_system_price"]

        dc_high = auctions[auctions["Service"] == "DCH"]
        daily_dc = (
            dc_high.groupby("EFA Date")["Clearing Price"].mean().reset_index()
        )
        daily_dc.columns = ["date", "avg_dc_clearing_price"]

        merged = pd.merge(daily_sys, daily_dc, on="date", how="inner")

        if merged.empty:
            st.info("No overlapping dates between system prices and DC auctions.")
        else:
            corr = merged[["avg_system_price", "avg_dc_clearing_price"]].corr().iloc[0, 1]
            st.metric("Pearson Correlation (SSP vs DCH)", f"{corr:.3f}")

            if abs(corr) < 0.3:
                st.caption(
                    "Low correlation — DCH prices appear to be driven primarily by frequency "
                    "response supply/demand dynamics rather than wholesale energy price levels."
                )
            elif corr > 0.5:
                st.caption(
                    "Moderate-to-strong correlation — common market drivers may be at work, "
                    "such as tight supply conditions lifting both energy and frequency response "
                    "prices simultaneously."
                )

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Scatter(
                    x=merged["date"],
                    y=merged["avg_system_price"],
                    name="Avg SSP",
                    line=dict(color="#EF553B"),
                ),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(
                    x=merged["date"],
                    y=merged["avg_dc_clearing_price"],
                    name="Avg DC High",
                    line=dict(color="#636EFA"),
                ),
                secondary_y=True,
            )
            fig.update_yaxes(title_text="System Price (£/MWh)", secondary_y=False)
            fig.update_yaxes(title_text="DC Clearing (£/MW/h)", secondary_y=True)
            fig.update_layout(height=500, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
