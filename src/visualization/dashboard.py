"""
GB BESS Market Dashboard

Run with:
    streamlit run src/visualization/dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GB BESS Market Dashboard",
    page_icon="\u26a1",
    layout="wide",
)

RAW = Path(__file__).parent.parent.parent / "data" / "raw"


# ---------------------------------------------------------------------------
# Data loading (cached, glob-based — picks up any files in data/raw/)
# ---------------------------------------------------------------------------

def _concat_csvs(paths: list, parse_dates: list, dedup_cols: list) -> pd.DataFrame:
    """Load and merge a list of CSV files, deduplicating on key columns."""
    if not paths:
        return pd.DataFrame()
    frames = [pd.read_csv(p, parse_dates=parse_dates) for p in paths]
    merged = pd.concat(frames, ignore_index=True)
    if dedup_cols and all(c in merged.columns for c in dedup_cols):
        merged = merged.drop_duplicates(subset=dedup_cols)
    return merged


@st.cache_data
def load_auctions() -> pd.DataFrame:
    """
    Merge legacy auction_results_*.csv (Sep 2021 – Nov 2023) with
    eac_results_*.csv (Nov 2023 – present) into a single DataFrame.
    Both share the same column schema so they concatenate directly.
    """
    legacy_paths = sorted(RAW.glob("auction_results_*.csv"))
    eac_paths    = sorted(RAW.glob("eac_results_*.csv"))
    all_paths    = legacy_paths + eac_paths
    if not all_paths:
        return pd.DataFrame()
    df = _concat_csvs(
        all_paths,
        parse_dates=["EFA Date", "Delivery Start", "Delivery End"],
        dedup_cols=["Service", "EFA Date", "EFA"],
    )
    return df.sort_values("EFA Date").reset_index(drop=True)


@st.cache_data
def load_system_prices() -> pd.DataFrame:
    paths = sorted(RAW.glob("system_prices_*.csv"))
    return _concat_csvs(
        paths,
        parse_dates=["settlementDate", "startTime"],
        dedup_cols=["settlementDate", "settlementPeriod"],
    )


@st.cache_data
def load_market_index() -> pd.DataFrame:
    paths = sorted(RAW.glob("market_index_*.csv"))
    return _concat_csvs(
        paths,
        parse_dates=["settlementDate", "startTime"],
        dedup_cols=["settlementDate", "settlementPeriod"],
    )


@st.cache_data
def load_generation() -> pd.DataFrame:
    paths = sorted(RAW.glob("generation_by_fuel_*.csv"))
    return _concat_csvs(
        paths,
        parse_dates=["settlementDate", "startTime", "publishTime"],
        dedup_cols=["settlementDate", "settlementPeriod", "fuelType"],
    )


auctions   = load_auctions()
sys_prices = load_system_prices()
mkt_index  = load_market_index()
gen_fuel   = load_generation()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("GB BESS Market Dashboard")
st.markdown("Data from the **Elexon Insights Solution API** and **NESO Data Portal**.")

# Top-level metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Auction Records", f"{len(auction_filtered):,}")
col2.metric("System Price Records", f"{len(sys_prices):,}")
col3.metric("Market Index Records", f"{len(mkt_index):,}")
col4.metric("Generation Records", f"{len(gen_fuel):,}")

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------
tab_auction, tab_spread, tab_system, tab_gen, tab_cross = st.tabs(
    ["DC/DR/DM Auctions", "H vs L Spread", "System Prices", "Generation Mix", "Cross-Source"]
)

# ---------------------------------------------------------------------------
# Tab 1: Auctions
# ---------------------------------------------------------------------------
with tab_auction:
    if auction_filtered.empty:
        st.warning("No auction data loaded. Run the data collector first.")
    else:
        st.subheader("Clearing Prices Over Time")
        fig = px.line(
            auction_filtered,
            x="EFA Date",
            y="Clearing Price",
            color="Service",
            labels={"Clearing Price": "£/MW/h"},
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
        st.warning("No auction data loaded. Run the data collector first.")
    else:
        st.markdown(
            """
            Each frequency response service runs two separate auctions: **High** (responds to
            falling frequency — BESS discharges) and **Low** (responds to rising frequency —
            BESS charges). The clearing prices can differ because the amount of available
            discharge vs charge headroom across the fleet is rarely symmetric.

            **Spread = H clearing price − L clearing price.** Positive = discharge capacity
            scarcer. Negative = charge capacity scarcer.
            """
        )

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
                st.subheader("Spread by EFA Block")
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
        st.subheader("Daily System Prices")
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
            st.subheader("Intraday Profile (by Settlement Period)")
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
            st.subheader("SBP – SSP Spread Distribution")
            spread = sys_prices["systemBuyPrice"] - sys_prices["systemSellPrice"]
            fig = px.histogram(spread, nbins=80, labels={"value": "£/MWh"})
            fig.update_layout(
                height=400, template="plotly_white", showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 3: Generation Mix
# ---------------------------------------------------------------------------
with tab_gen:
    if gen_fuel.empty:
        st.warning("No generation data loaded.")
    else:
        st.subheader("Daily Generation by Fuel Type")
        daily_gen = (
            gen_fuel.groupby(["settlementDate", "fuelType"])["generation"]
            .sum()
            .reset_index()
        )
        gen_pivot = daily_gen.pivot_table(
            index="settlementDate",
            columns="fuelType",
            values="generation",
            fill_value=0,
        )
        col_order = gen_pivot.sum().sort_values(ascending=False).index
        gen_pivot = gen_pivot[col_order]

        melted = gen_pivot.reset_index().melt(
            id_vars="settlementDate",
            var_name="Fuel Type",
            value_name="Generation",
        )
        fig = px.area(
            melted,
            x="settlementDate",
            y="Generation",
            color="Fuel Type",
            labels={"Generation": "MW", "settlementDate": "Date"},
        )
        fig.update_layout(height=500, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Average Share by Fuel Type")
        fuel_share = gen_pivot.mean()
        fuel_share = fuel_share[fuel_share > 0].sort_values(ascending=False)
        fig = px.pie(values=fuel_share.values, names=fuel_share.index)
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 4: Cross-source
# ---------------------------------------------------------------------------
with tab_cross:
    if sys_prices.empty or auctions.empty:
        st.warning("Need both system price and auction data for cross-source analysis.")
    else:
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
            st.metric("Pearson Correlation", f"{corr:.3f}")

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

            st.subheader("Scatter: Correlation")
            fig = px.scatter(
                merged,
                x="avg_system_price",
                y="avg_dc_clearing_price",
                trendline="ols",
                labels={
                    "avg_system_price": "Avg SSP (£/MWh)",
                    "avg_dc_clearing_price": "Avg DC High (£/MW/h)",
                },
                title=f"r = {corr:.3f}",
            )
            fig.update_layout(height=450, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
