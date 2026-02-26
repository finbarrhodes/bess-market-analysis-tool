"""
Methodology & Data Sources
===========================
Static reference page. Launched as a page via app.py (st.navigation).
Can still be run standalone for local development:
    streamlit run src/visualization/methodology.py
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Standalone guard — set_page_config only when run directly, not via app.py
# ---------------------------------------------------------------------------
try:
    st.set_page_config(
        page_title="Methodology & Data — UK BESS",
        page_icon="⚡",
        layout="wide",
    )
except st.errors.StreamlitAPIException:
    pass  # already set by app.py

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Methodology & Data Sources")
st.markdown(
    "This page documents the modelling approach, assumptions, and data sources "
    "used by the Revenue Backtester."
)

st.divider()

# ---------------------------------------------------------------------------
# Two-stage participation model
# ---------------------------------------------------------------------------

st.header("Two-stage participation model")
st.markdown(
    """
This backtester uses a simplified two-stage model to separate FR availability revenue
from energy arbitrage revenue without double-counting the same physical capacity:

- **Stage 1 — FR commitment:** The MW capacity committed to frequency response
  (configured in the sidebar) holds its state of charge in a headroom band, ready to
  discharge (High services) or charge (Low services) on demand. These units earn
  availability payments but do not trade energy in the wholesale market.
- **Stage 2 — Arbitrage:** The remaining MW runs daily intrinsic arbitrage independently,
  unconstrained by FR headroom requirements.

The trade-off curve in the Revenue Backtester shows how total net revenue varies as the
split changes — the optimal point represents the revenue-maximising balance between the
two strategies at current market prices and battery parameters.

*Remaining simplification:* SoC within the FR band is not explicitly simulated at
half-hourly resolution. A full joint dispatch model (LP/MIP) would track SoC state
continuously and optimise both stages simultaneously.
"""
)

st.divider()

# ---------------------------------------------------------------------------
# Dispatch strategies
# ---------------------------------------------------------------------------

st.header("Dispatch strategies")
st.markdown(
    """
The arbitrage schedule is derived from a price signal for day D. Three signals are available:

- **Perfect Foresight**: the optimiser receives the actual day-D APXMIDP prices and picks
  the true optimal charge/discharge windows. This is the revenue ceiling — it is not
  achievable in practice but provides a useful benchmark.
- **Naive (D-1 prices)**: at the end of day D-1, yesterday's actual prices are used as
  the forecast for day D. The schedule is derived from this forecast, then evaluated
  against actual day-D prices to compute realised revenue. This is the zero-skill floor —
  any useful model should outperform it.
- **ML Model**: a machine learning model trained on historical data predicts day-D prices
  using features available at end of day D-1 (lagged prices, generation mix, seasonality).
  The predicted prices drive the same dispatch logic; realised revenue is computed against
  actual prices.

The **capture rate** (realised revenue ÷ perfect-foresight revenue) summarises how much
of the theoretical ceiling each strategy achieves. The **revenue/cycling tradeoff chart**
in the Strategy Comparison tab shows the efficiency of each strategy: a better strategy
earns more revenue while consuming less cycle life.
"""
)

st.divider()

# ---------------------------------------------------------------------------
# Ancillary service availability revenue
# ---------------------------------------------------------------------------

st.header("Ancillary service availability revenue")
st.markdown(
    """
- Revenue = `clearing_price (£/MW/h) × MW committed to FR × 4 hours per EFA block`
- Services of different response speeds (DC, DR, DM) can be stacked on the same physical MW
  in the GB market — each earns a separate availability payment.
- High (discharge) and Low (charge) services are modelled as independent and simultaneous,
  assuming the battery maintains sufficient SoC headroom to respond in both directions.
- Clearing prices sourced from NESO Data Portal (legacy DC/DR/DM auctions Sep 2021–Nov 2023,
  EAC service Nov 2023–present).
- Ancillary revenue is identical across all three dispatch strategies — it does not depend
  on price forecasting.
"""
)

st.divider()

# ---------------------------------------------------------------------------
# Wholesale energy arbitrage revenue
# ---------------------------------------------------------------------------

st.header("Wholesale energy arbitrage revenue")
st.markdown(
    """
- One arbitrage cycle per day maximum, using the MW available for arbitrage (total power
  minus the MW committed to FR).
- Charge in the N cheapest settlement periods; discharge in the N most expensive periods,
  where N = battery duration (hours) × 2 (i.e. the number of half-hour periods that fill
  or empty the battery at full power).
- Gross profit = `avg_discharge_price × energy capacity (MWh)`
  − `avg_charge_price × (energy capacity ÷ round-trip efficiency)`
  (extra input energy needed to account for round-trip losses).
- Cycle only executed on days where realised gross profit exceeds the cycling wear cost.
- **Price reference: APXMIDP market index** (APX Power UK) from Elexon Insights
  (Jul 2023–present). This is the actual GB spot settlement reference price, giving a
  materially more realistic daily spread than the imbalance settlement price (SSP),
  which can reach extreme negative values during high-renewable periods and would
  otherwise inflate arbitrage revenue.
"""
)

st.divider()

# ---------------------------------------------------------------------------
# Availability factor
# ---------------------------------------------------------------------------

st.header("Availability factor")
st.markdown(
    """
- Applied as a uniform multiplier to all revenue streams and cycling costs.
- Models periods where the asset is unavailable due to planned maintenance, unplanned
  faults, grid curtailment, or service delivery failures.
- The default of 95% reflects the minimum availability threshold mandated in NESO's Dynamic
  Containment and Enduring Auction Capability service specifications. This is also
  consistent with observed GB BESS fleet performance: Modo Energy's *GB Battery Storage
  Report* (2024) reports median fleet availability of 95–97% across contracted windows.
"""
)

st.divider()

# ---------------------------------------------------------------------------
# Cycling wear cost and battery degradation
# ---------------------------------------------------------------------------

st.header("Cycling wear cost and battery degradation")
st.markdown(
    """
- Applied to imbalance arbitrage trades only: `configured cycling wear cost (£/MWh) × MWh discharged per trade`.
- Ancillary service cycling (energy delivered during frequency events) is not separately
  modelled — it is minor relative to availability payments and is typically compensated
  via the service contract.
- *Why cycling matters beyond cost:* lithium-ion cells degrade through two primary
  mechanisms that accelerate with use — SEI (solid electrolyte interphase) layer growth,
  which consumes cyclable lithium irreversibly, and lithium plating at the anode, which
  increases with deeper discharge and higher charge rates. Each MWh cycled consumes a
  small fraction of the cell's finite cycle life. The cycling wear cost parameter is a
  financial proxy for this physical degradation: more aggressive dispatch earns more
  revenue in the short run but consumes cycle life faster, reducing the asset's useful
  life and residual value. This is why the revenue/cycling tradeoff chart is the core
  output of the strategy comparison, not revenue alone.
"""
)

st.divider()

# ---------------------------------------------------------------------------
# ML price forecast model
# ---------------------------------------------------------------------------

st.header("ML price forecast model")
st.markdown(
    """
The ML strategy uses a **tree-based ensemble model (Random Forest or XGBoost)** to predict
the 48 half-hourly APXMIDP prices for day D using features available at the end of day D-1.

*Why tree-based ensembles?* The feature set is tabular (lagged prices, generation mix
ratios, temporal encodings) rather than raw sequences; they require no feature scaling;
they are robust on datasets of this size (~30,000 training rows); and they provide
interpretable feature importances. An LSTM was considered but is likely overkill given
~2 years of training data and would be harder to explain. A naive lag model sets the
zero-skill baseline.

**Features used (all available at end of day D-1):**
- Same-period lagged prices: price at the same settlement period 1, 2, and 7 days prior
- Previous-day price statistics: mean, standard deviation, max, and min across all 48 periods
- Generation mix (daily, from D-1): total generation, renewable fraction, fossil fraction,
  and per-fuel breakdown (gas, wind, nuclear, hydro, etc.)
- Cyclical temporal encodings: settlement period, day-of-week, and day-of-year encoded
  as sin/cos pairs to preserve circularity (e.g. period 48 and period 1 are adjacent)
- Weekend flag

**Train/test split:** strict temporal split — training data ends before 2025-03-01
to prevent any look-ahead bias. The model never sees future prices during training.

**Known limitations:** tree-based models cannot extrapolate beyond the price ranges seen
during training; electricity price forecasting is inherently noisy; and the model improves
dispatch quality on average but does not eliminate forecast error on individual days.

Model performance metrics (RMSE, MAE) and feature importances for the currently selected
model are shown in the **Strategy Comparison** tab of the Revenue Backtester.
"""
)

st.divider()

# ---------------------------------------------------------------------------
# Known limitations
# ---------------------------------------------------------------------------

st.header("Known limitations")
st.markdown(
    """
- Intraday / day-ahead market trading (DA price data not yet integrated)
- Balancing Mechanism (BM) direct trading
- Battery degradation over the backtest period
- Real-time dispatch constraints or grid connection limits
- Half-hourly SoC simulation within the FR headroom band (planned for a future LP/MIP module)
"""
)

st.divider()

# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------

st.header("Data Sources")
st.markdown(
    """
| Dataset | Source | Coverage |
|---|---|---|
| Frequency response auction results (DC/DR/DM) | [NESO Data Portal](https://www.neso.energy/data-portal) | Sep 2021 – present |
| APXMIDP market index price | [Elexon Insights Solution API](https://developer.data.elexon.co.uk/) | Jul 2023 – present |
| System buy/sell prices (SBP/SSP) | [Elexon Insights Solution API](https://developer.data.elexon.co.uk/) | Jul 2023 – present |
| Generation by fuel type (daily) | [Elexon Insights Solution API](https://developer.data.elexon.co.uk/) | Jul 2023 – present |
"""
)
