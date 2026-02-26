"""
Home — UK BESS Market Analysis Tool
=====================================
Landing page. Launched as a page via app.py (st.navigation).
Can still be run standalone for local development:
    streamlit run src/visualization/home.py
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Standalone guard — set_page_config only when run directly, not via app.py
# ---------------------------------------------------------------------------
try:
    st.set_page_config(
        page_title="UK BESS Market Analysis",
        page_icon="⚡",
        layout="wide",
    )
except st.errors.StreamlitAPIException:
    pass  # already set by app.py

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("UK BESS Market Analysis Tool")
st.markdown(
    "An interactive analysis platform for GB Battery Energy Storage System (BESS) "
    "market revenue, built on data from the **Elexon Insights Solution API** and "
    "**NESO Data Portal**."
)

st.divider()

# ---------------------------------------------------------------------------
# About — BESS revenue routes & market context
# ---------------------------------------------------------------------------

st.markdown(
    """
    Battery Energy Storage Systems (BESS) do not rely on a single revenue source — they
    **stack** income from multiple markets, often simultaneously:

    | Revenue route | How it works |
    |---|---|
    | **Frequency response** (Dynamic Services) | Paid a £/MW/h availability fee to hold discharge or charge headroom; activated automatically when grid frequency deviates from 50 Hz |
    | **Wholesale arbitrage** | Charge during low-price periods (high wind, low demand); discharge during high-price periods |
    | **Balancing Mechanism (BM)** | Dispatched by NESO in real time as a BM unit to correct short-term supply/demand imbalance |
    | **Capacity Market** | Annual availability payment for committing to generate during periods of system stress |

    **Why this tool focuses on Dynamic Services and wholesale prices:**
    Dynamic services (DC, DM, DR) dominated the GB BESS revenue stack from roughly 2021 through
    early 2023 — at times accounting for over 70% of total asset revenue, with some assets
    earning ~£156k/MW/year at the 2022 peak
    ([Modo Energy](https://modoenergy.com/research/future-of-battery-energy-storage-buildout-in-great-britain);
    [Timera Energy](https://timera-energy.com/blog/battery-investors-confront-revenue-shift-in-2023/)).
    From late 2022, a rapid influx of new BESS capacity saturated the frequency response
    markets and revenues compressed sharply — a trend visible in the clearing price charts in
    the Market Overview. Dynamic services remain a core stack component, but arbitrage and
    Capacity Market income have grown in relative importance since.

    **BESS is not the only participant in these markets.** Pumped-storage hydro (e.g. Dinorwig,
    Cruachan) has provided fast frequency response for decades. Demand-side response and some
    gas peakers also qualify, particularly for the slower DR service. However, BESS's
    sub-second response capability has made it the dominant and marginal price-setting
    technology in DC and DM auctions.
    """
)

st.divider()

# ---------------------------------------------------------------------------
# Navigation guide
# ---------------------------------------------------------------------------

st.subheader("What's in this tool")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Market Overview**")
    st.markdown(
        "Explore GB frequency response auction clearing prices (DC, DR, DM), "
        "High vs Low spread dynamics, system settlement prices (SBP/SSP), "
        "generation mix trends, and the correlation between system prices and "
        "DC High auctions."
    )

with col2:
    st.markdown("**Revenue Backtester**")
    st.markdown(
        "Model the combined FR availability and wholesale arbitrage revenue stack "
        "for a configurable BESS asset. Compare Perfect Foresight, Naive, and ML "
        "dispatch strategies, and run sensitivity analysis across a range of "
        "battery sizes."
    )

with col3:
    st.markdown("**Methodology & Data**")
    st.markdown(
        "Understand the modelling assumptions, data sources, and known limitations "
        "of the backtester. Explains the two-stage participation model, dispatch "
        "strategies, battery cycling cost, and the ML price forecast approach."
    )
