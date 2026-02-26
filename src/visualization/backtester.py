"""
BESS Tolling Value Backtester
==============================
Launched as a page via app.py (st.navigation).
Can still be run standalone for local development:
    streamlit run src/visualization/backtester.py
"""

import sys
from pathlib import Path

# Ensure src/ is on the path when launched standalone from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from src.analysis.revenue_stack import (
    BatterySpec,
    run_backtest,
    sensitivity_table,
    find_optimal_split,
    ALL_SERVICES,
    SERVICE_LABELS,
    SERVICE_COLOURS,
)
from src.analysis.price_forecast import (
    build_feature_matrix,
    train_forecast_model,
    run_forecast_backtest,
    get_feature_importances,
    DEFAULT_TEST_START,
)

# ---------------------------------------------------------------------------
# Standalone guard — set_page_config only when run directly, not via app.py
# ---------------------------------------------------------------------------
try:
    st.set_page_config(
        page_title="BESS Tolling Value Backtester",
        page_icon="🔋",
        layout="wide",
    )
except st.errors.StreamlitAPIException:
    pass  # already set by app.py

PROCESSED = Path(__file__).parent.parent.parent / "data" / "processed"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_auctions() -> pd.DataFrame:
    p = PROCESSED / "auctions.parquet"
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
def load_generation_daily() -> pd.DataFrame:
    p = PROCESSED / "generation_daily.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


auctions    = load_auctions()
mkt_index   = load_market_index()
gen_daily   = load_generation_daily()

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
st.sidebar.subheader("Market Participation")

fr_mw = st.sidebar.slider(
    "MW committed to FR availability",
    min_value=0,
    max_value=power_mw,
    value=power_mw,
    step=1,
    help=(
        "Capacity committed to frequency response (FR) availability services. "
        "The remaining MW is available for energy arbitrage. "
        "FR-committed units hold their SoC position for frequency response "
        "and cannot simultaneously trade energy."
    ),
)
arb_mw = power_mw - fr_mw

if fr_mw == power_mw:
    st.sidebar.caption("→ All capacity committed to FR; no arbitrage.")
elif fr_mw == 0:
    st.sidebar.caption("→ No FR commitment; all capacity for arbitrage.")
else:
    st.sidebar.caption(f"→ {fr_mw} MW → FR availability | {arb_mw} MW → arbitrage")

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

st.sidebar.divider()
st.sidebar.subheader("Dispatch Strategy")

dispatch_strategy = st.sidebar.radio(
    "Price signal used for arbitrage scheduling",
    options=["Perfect Foresight", "Naive (D-1 prices)", "ML Model"],
    index=0,
    help=(
        "**Perfect Foresight**: schedules dispatch using actual day-D prices — "
        "the revenue ceiling, not achievable in practice.\n\n"
        "**Naive**: uses yesterday's prices as the forecast for today — "
        "a zero-skill baseline.\n\n"
        "**ML Model**: trains on historical features (lagged prices, generation mix, "
        "seasonality) and forecasts day-D prices to drive dispatch."
    ),
)

ml_model_type = "rf"
if dispatch_strategy == "ML Model":
    ml_model_type = st.sidebar.selectbox(
        "Model",
        options=["rf", "xgb"],
        format_func=lambda x: "Random Forest" if x == "rf" else "XGBoost",
        index=0,
    )


@st.cache_resource
def load_forecast_model(model_type: str):
    """Train and cache the ML price forecast model. Runs once per deployment."""
    feature_df = build_feature_matrix(load_market_index(), load_generation_daily())
    model, feature_cols, train_metrics, test_metrics = train_forecast_model(
        feature_df, model_type=model_type, test_start=DEFAULT_TEST_START
    )
    return model, feature_df, feature_cols, train_metrics, test_metrics


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
    st.error("No auction data found. Run scripts/prepare_data.py first.")
    st.stop()

# Compute optimal split (cached — reruns only when parameters change)
@st.cache_data
def _cached_optimal_split(
    power_mw, duration_h, efficiency_rt, cycling_cost_per_mwh, availability_factor,
    services_key, start_date, end_date,
):
    """Thin cache wrapper for find_optimal_split (hashable args only)."""
    _battery = BatterySpec(
        power_mw=power_mw,
        duration_h=duration_h,
        efficiency_rt=efficiency_rt,
        cycling_cost_per_mwh=cycling_cost_per_mwh,
        availability_factor=availability_factor,
    )
    _mi = load_market_index() if include_imbalance else pd.DataFrame()
    return find_optimal_split(
        load_auctions(), _mi, _battery,
        services=list(services_key),
        start_date=start_date,
        end_date=end_date,
    )

optimal_fr_mw, trade_off_df = _cached_optimal_split(
    power_mw, duration_h, efficiency_pct / 100, cycling_cost, availability_pct / 100,
    tuple(sorted(selected_services)), start_date, end_date,
)

# Show optimal marker below the FR slider
st.sidebar.caption(
    f"ℹ Optimal split: **{optimal_fr_mw:.0f} MW** to FR "
    f"(maximises net revenue at current parameters)"
)

# Run the primary backtest for the selected strategy
if dispatch_strategy == "Perfect Foresight":
    result = run_backtest(
        auctions, mi_input, battery, selected_services, start_date, end_date, fr_mw=fr_mw
    )
    # Add total_mwh_cycled to summary for the comparison chart
    if result["monthly"] is not None and not result["monthly"].empty:
        cyc_col = result["monthly"].get("cycling_cost_gbp", result["monthly"].get("cycling_cost", None))
        mwh_cycled = 0.0
        if cyc_col is not None:
            mwh_cycled = float(
                (result["monthly"]["cycling_cost_gbp"] / battery.cycling_cost_per_mwh).sum()
                if "cycling_cost_gbp" in result["monthly"].columns and battery.cycling_cost_per_mwh > 0
                else 0.0
            )
        result["summary"]["total_mwh_cycled"] = round(mwh_cycled, 1)

elif dispatch_strategy == "Naive (D-1 prices)":
    if not include_imbalance:
        result = run_backtest(
            auctions, pd.DataFrame(), battery, selected_services, start_date, end_date, fr_mw=fr_mw
        )
        result["summary"]["total_mwh_cycled"] = 0.0
    else:
        result = run_forecast_backtest(
            strategy="naive",
            market_index=mkt_index,
            auctions=auctions,
            battery=battery,
            services=selected_services,
            start_date=start_date,
            end_date=end_date,
            fr_mw=fr_mw,
        )

else:  # ML Model
    if not include_imbalance:
        result = run_backtest(
            auctions, pd.DataFrame(), battery, selected_services, start_date, end_date, fr_mw=fr_mw
        )
        result["summary"]["total_mwh_cycled"] = 0.0
    else:
        with st.spinner("Training forecast model… (cached after first run)"):
            _model, _feature_df, _feature_cols, _train_m, _test_m = load_forecast_model(ml_model_type)
        result = run_forecast_backtest(
            strategy="ml",
            market_index=mkt_index,
            auctions=auctions,
            battery=battery,
            services=selected_services,
            start_date=start_date,
            end_date=end_date,
            fr_mw=fr_mw,
            model=_model,
            feature_df=_feature_df,
            feature_cols=_feature_cols,
        )

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
# Content tabs
# ---------------------------------------------------------------------------

tab_results, tab_strategy, tab_sensitivity = st.tabs(
    ["Results", "Strategy Comparison", "Sensitivity"]
)

# ---------------------------------------------------------------------------
# Tab: Results
# ---------------------------------------------------------------------------

with tab_results:
    if summary:
        c1, c2, c3, c4, c5 = st.columns(5)
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
        c5.metric(
            "Capacity Split",
            f"{fr_mw} MW FR / {arb_mw} MW arb",
        )
    else:
        st.warning("No results — check that services are selected and data covers the chosen period.")
        st.stop()

    st.divider()

    # Monthly stacked bar — revenue by stream + cycling cost deduction
    if not monthly.empty:
        st.subheader("Monthly Revenue Stack")

        fig = go.Figure()

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

    # FR / Arbitrage trade-off curve
    if not trade_off_df.empty and power_mw > 1:
        with st.expander("FR / Arbitrage trade-off curve", expanded=True):
            st.caption(
                "Total net revenue across the backtest period as a function of how much "
                "capacity is committed to FR availability vs energy arbitrage. "
                "The green marker shows the revenue-maximising split at current parameters."
            )

            fig_tradeoff = go.Figure()

            fig_tradeoff.add_trace(go.Scatter(
                x=trade_off_df["fr_mw"],
                y=trade_off_df["total_net_gbp"] / 1_000,
                mode="lines",
                name="Total net revenue",
                line=dict(color="#1f77b4", width=2.5),
            ))

            fig_tradeoff.add_vline(
                x=fr_mw,
                line_dash="dash",
                line_color="#d62728",
                annotation_text=f"Current: {fr_mw} MW",
                annotation_position="top right",
                annotation_font_color="#d62728",
            )

            optimal_row = trade_off_df.loc[trade_off_df["fr_mw"] == optimal_fr_mw]
            if not optimal_row.empty:
                fig_tradeoff.add_trace(go.Scatter(
                    x=optimal_row["fr_mw"],
                    y=optimal_row["total_net_gbp"] / 1_000,
                    mode="markers",
                    name=f"Optimal: {optimal_fr_mw:.0f} MW to FR",
                    marker=dict(symbol="star", size=14, color="#2ca02c"),
                ))

            fig_tradeoff.update_layout(
                height=340,
                template="plotly_white",
                xaxis_title="MW committed to FR availability",
                yaxis_title="Total net revenue (£k)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                margin=dict(t=40),
            )
            st.plotly_chart(fig_tradeoff, use_container_width=True)

    # Cumulative revenue | Revenue breakdown pie
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

# ---------------------------------------------------------------------------
# Tab: Strategy Comparison
# ---------------------------------------------------------------------------

with tab_strategy:
    st.caption(
        "Each point shows total net revenue against total MWh cycled over the backtest period "
        "for one dispatch strategy. A strategy that sits higher and to the left earns more "
        "revenue while consuming less cycle life — the analytically ideal outcome. "
        "Perfect Foresight is the revenue ceiling; Naive is the zero-skill floor."
    )

    if include_imbalance and not mkt_index.empty:
        with st.spinner("Computing strategy comparison (runs all three strategies)…"):

            # --- Perfect Foresight ---
            pf_result = run_backtest(
                auctions, mi_input, battery, selected_services, start_date, end_date, fr_mw=fr_mw
            )
            pf_summary = pf_result["summary"]
            pf_mwh = 0.0
            if not pf_result["monthly"].empty and "cycling_cost_gbp" in pf_result["monthly"].columns:
                if battery.cycling_cost_per_mwh > 0:
                    pf_mwh = float(
                        (pf_result["monthly"]["cycling_cost_gbp"] / battery.cycling_cost_per_mwh).sum()
                    )

            # --- Naive ---
            naive_result = run_forecast_backtest(
                strategy="naive",
                market_index=mkt_index,
                auctions=auctions,
                battery=battery,
                services=selected_services,
                start_date=start_date,
                end_date=end_date,
                fr_mw=fr_mw,
            )

            # --- ML ---
            with st.spinner("Training ML model for comparison… (cached after first run)"):
                _cmp_model, _cmp_feat_df, _cmp_feat_cols, _, _ = (
                    load_forecast_model(ml_model_type)
                )
            ml_result = run_forecast_backtest(
                strategy="ml",
                market_index=mkt_index,
                auctions=auctions,
                battery=battery,
                services=selected_services,
                start_date=start_date,
                end_date=end_date,
                fr_mw=fr_mw,
                model=_cmp_model,
                feature_df=_cmp_feat_df,
                feature_cols=_cmp_feat_cols,
            )

        pf_net    = pf_summary.get("total_net", 0) / 1_000
        naive_net = naive_result["summary"].get("total_net", 0) / 1_000
        ml_net    = ml_result["summary"].get("total_net", 0) / 1_000

        naive_mwh = naive_result["summary"].get("total_mwh_cycled", 0)
        ml_mwh    = ml_result["summary"].get("total_mwh_cycled", 0)

        pf_capture    = 100.0
        naive_capture = round(naive_net / pf_net * 100, 1) if pf_net > 0 else 0.0
        ml_capture    = round(ml_net    / pf_net * 100, 1) if pf_net > 0 else 0.0

        model_label = "Random Forest" if ml_model_type == "rf" else "XGBoost"

        comparison_df = pd.DataFrame([
            {"Strategy": "Perfect Foresight", "Net Revenue (£k)": pf_net,    "MWh Cycled": round(pf_mwh, 0),    "Capture Rate": f"{pf_capture:.0f}%"},
            {"Strategy": "Naive (D-1)",        "Net Revenue (£k)": naive_net, "MWh Cycled": round(naive_mwh, 0), "Capture Rate": f"{naive_capture:.1f}%"},
            {"Strategy": f"ML ({model_label})", "Net Revenue (£k)": ml_net,   "MWh Cycled": round(ml_mwh, 0),   "Capture Rate": f"{ml_capture:.1f}%"},
        ])

        fig_cmp = go.Figure()
        colours = {"Perfect Foresight": "#2ca02c", "Naive (D-1)": "#ff7f0e", f"ML ({model_label})": "#1f77b4"}
        for _, row in comparison_df.iterrows():
            fig_cmp.add_trace(go.Scatter(
                x=[row["MWh Cycled"]],
                y=[row["Net Revenue (£k)"]],
                mode="markers+text",
                name=row["Strategy"],
                text=[row["Strategy"]],
                textposition="top center",
                marker=dict(size=16, color=colours.get(row["Strategy"], "#888")),
            ))

        fig_cmp.update_layout(
            height=360,
            template="plotly_white",
            xaxis_title="Total MWh Cycled",
            yaxis_title="Total Net Revenue (£k)",
            showlegend=False,
            margin=dict(t=30, b=40),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        st.dataframe(
            comparison_df.style.format({"Net Revenue (£k)": "{:,.1f}", "MWh Cycled": "{:,.0f}"}),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "**Capture rate** = realised net revenue ÷ perfect-foresight net revenue. "
            "Reflects how much of the theoretical revenue ceiling each strategy achieves."
        )

        # ML model detail — shown when ML strategy is selected
        if dispatch_strategy == "ML Model":
            st.divider()
            st.subheader(f"ML Model Detail — {model_label}")
            st.markdown(f"""
The ML strategy uses a **{model_label}** regressor to predict the 48 half-hourly APXMIDP
prices for day D using features available at the end of day D-1.

*Why {model_label}?* Tree-based ensemble methods are well suited to this problem: the
feature set is tabular (lagged prices, generation mix ratios, temporal encodings) rather
than raw sequences; they require no feature scaling; they are robust on datasets of this
size (~30,000 training rows); and they provide interpretable feature importances.
An LSTM was considered but is likely overkill given ~2 years of training data and would
be harder to explain. A naive lag model sets the zero-skill baseline.

**Features used (all available at end of day D-1):**
- Same-period lagged prices: price at the same settlement period 1, 2, and 7 days prior
- Previous-day price statistics: mean, standard deviation, max, and min across all 48 periods
- Generation mix (daily, from D-1): total generation, renewable fraction, fossil fraction,
  and per-fuel breakdown (gas, wind, nuclear, hydro, etc.)
- Cyclical temporal encodings: settlement period, day-of-week, and day-of-year encoded
  as sin/cos pairs to preserve circularity (e.g. period 48 and period 1 are adjacent)
- Weekend flag

**Train/test split:** strict temporal split — training data ends before
`{DEFAULT_TEST_START}` to prevent any look-ahead bias. The model never sees
future prices during training.

**Known limitations:** tree-based models cannot extrapolate beyond the price ranges seen
during training; electricity price forecasting is inherently noisy; and the model improves
dispatch quality on average but does not eliminate forecast error on individual days.
            """)

            with st.spinner("Loading model metrics…"):
                _model_exp, _, _feat_cols_exp, _train_m, _test_m = load_forecast_model(ml_model_type)

            col_a, col_b = st.columns(2)
            col_a.metric("Train RMSE (£/MWh)", f"{_train_m['rmse']:.2f}", help=f"n = {_train_m['n_samples']:,} periods")
            col_b.metric("Test RMSE (£/MWh)",  f"{_test_m['rmse']:.2f}",  help=f"n = {_test_m['n_samples']:,} periods (held-out, after {DEFAULT_TEST_START})")
            col_a.metric("Train MAE (£/MWh)",  f"{_train_m['mae']:.2f}")
            col_b.metric("Test MAE (£/MWh)",   f"{_test_m['mae']:.2f}")

            importances = get_feature_importances(_model_exp, _feat_cols_exp).head(10)
            fig_imp = go.Figure(go.Bar(
                x=importances.values[::-1],
                y=importances.index[::-1],
                orientation="h",
                marker_color="#1f77b4",
            ))
            fig_imp.update_layout(
                height=320,
                template="plotly_white",
                title="Top 10 feature importances",
                xaxis_title="Importance",
                margin=dict(t=40, l=160),
            )
            st.plotly_chart(fig_imp, use_container_width=True)

    else:
        st.info("Enable 'Include imbalance arbitrage' to see the strategy comparison.")

# ---------------------------------------------------------------------------
# Tab: Sensitivity
# ---------------------------------------------------------------------------

with tab_sensitivity:
    st.subheader("Sensitivity: Revenue by Battery Size")
    st.caption("Other parameters held constant at sidebar values.")

    sens_df = sensitivity_table(
        auctions,
        mi_input,
        battery,
        power_range=[5, 10, 25, 50, 100, 200],
        start_date=start_date,
        end_date=end_date,
        fr_fraction=fr_mw / power_mw if power_mw > 0 else 1.0,
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
