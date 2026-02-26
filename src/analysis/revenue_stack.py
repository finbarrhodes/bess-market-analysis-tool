"""
BESS Revenue Stack Backtester
==============================
Models the historical revenue a BESS asset could have earned from:

  1. Ancillary service availability payments (DC High/Low, DR High/Low, DM High/Low)
     Revenue = clearing_price (£/MW/h) × committed_MW × hours_per_EFA_block
     Services are treated as stackable: DCH + DRH + DMH bid the same physical MW
     against different frequency response speeds — this is valid for the GB market.

  2. Wholesale energy arbitrage
     A daily intrinsic model: charge in the N cheapest settlement periods,
     discharge in the N most expensive, constrained by battery duration.
     Uses the APXMIDP market index price as the spot price reference — this
     is the actual GB day-ahead/intraday settlement reference price and a
     significantly more realistic proxy than the imbalance settlement price (SSP),
     which can reach extreme negative values during high-renewable periods.
     Only executes a cycle on days where the net profit exceeds zero after
     accounting for round-trip efficiency losses and cycling wear cost.

Key assumptions:
  - Battery maintains sufficient SoC headroom to provide both H (discharge)
    and L (charge) ancillary services simultaneously.
  - Ancillary cycling cost (energy delivery during frequency events) is minor
    relative to availability payments and is not separately modelled.
  - Cycling wear cost applies to imbalance arbitrage trades only.
  - One arbitrage cycle per day maximum (conservative).
"""

from dataclasses import dataclass
import pandas as pd
import numpy as np

# Each EFA block is 4 hours in GB
EFA_HOURS = 4

SERVICE_LABELS = {
    "DCH": "DC High",
    "DCL": "DC Low",
    "DRH": "DR High",
    "DRL": "DR Low",
    "DMH": "DM High",
    "DML": "DM Low",
}

ALL_SERVICES = list(SERVICE_LABELS.keys())

# Colour map for dashboard charts (revenue streams)
SERVICE_COLOURS = {
    "DCH": "#0D7680",   # FT teal
    "DCL": "#5BA8AE",   # lighter teal
    "DRH": "#4E8A3C",   # dark green
    "DRL": "#8AB87F",   # lighter green
    "DMH": "#7B3FA0",   # purple
    "DML": "#B08FC8",   # lighter purple
    "Imbalance": "#C9400A",  # warm orange-red
    "Cycling cost": "#8B2020",  # dark red
}


@dataclass
class BatterySpec:
    power_mw: float = 50.0
    duration_h: float = 2.0
    efficiency_rt: float = 0.90       # Round-trip, expressed as a fraction (e.g. 0.90)
    cycling_cost_per_mwh: float = 3.0 # £ per MWh of usable energy throughput
    availability_factor: float = 0.95 # Fraction of periods the asset is available (maintenance,
                                      # faults, curtailment). 0.95 reflects the 95% minimum
                                      # availability threshold specified in NESO's Dynamic
                                      # Containment and EAC service agreements, and is consistent
                                      # with observed GB BESS fleet performance (Modo Energy,
                                      # "GB Battery Storage Report", 2024).

    @property
    def energy_mwh(self) -> float:
        return self.power_mw * self.duration_h


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _filter_dates(df: pd.DataFrame, date_col: str, start_date, end_date) -> pd.DataFrame:
    if start_date is not None:
        df = df[df[date_col] >= pd.Timestamp(start_date)]
    if end_date is not None:
        df = df[df[date_col] <= pd.Timestamp(end_date)]
    return df


# ---------------------------------------------------------------------------
# Stream 1 — Ancillary service availability revenue
# ---------------------------------------------------------------------------

def calc_ancillary_revenue(
    auctions: pd.DataFrame,
    battery: BatterySpec,
    services: list,
    start_date=None,
    end_date=None,
    min_price: float = 0.0,
    fr_mw: float = None,
    fr_schedule: "pd.Series | None" = None,
) -> pd.DataFrame:
    """
    Monthly availability revenue from frequency response auctions.

    Negative clearing prices occur in oversupplied GB markets (notably DR High).
    The min_price floor (default 0.0) models an operator who sets a bid price
    floor and opts out of any EFA block clearing below that threshold — a
    rational strategy any real participant would adopt.

    Parameters
    ----------
    fr_mw : float, optional
        Fixed MW committed to FR for all days. Ignored when fr_schedule is
        provided. Defaults to battery.power_mw when both are None.
    fr_schedule : pd.Series, optional
        Per-day FR commitment (index = date, values = fr_mw). When provided,
        each auction record is scaled by the corresponding day's fr_mw value
        rather than a single fixed figure.

    Returns
    -------
    DataFrame with columns: [month (Period), service (str), revenue_gbp (float)]
    """
    df = _filter_dates(auctions.copy(), "EFA Date", start_date, end_date)
    df = df[df["Service"].isin(services)]
    df = df[df["Clearing Price"] >= min_price]

    if df.empty:
        return pd.DataFrame(columns=["month", "service", "revenue_gbp"])

    df = df.copy()

    if fr_schedule is not None:
        sched_map = {pd.Timestamp(k).normalize(): float(v) for k, v in fr_schedule.items()}
        df["fr_mw_d"] = (
            df["EFA Date"].dt.normalize()
            .map(sched_map)
            .fillna(battery.power_mw)
        )
        df["revenue_gbp"] = df["Clearing Price"] * df["fr_mw_d"] * EFA_HOURS
    else:
        if fr_mw is None:
            fr_mw = battery.power_mw
        if fr_mw == 0:
            return pd.DataFrame(columns=["month", "service", "revenue_gbp"])
        df["revenue_gbp"] = df["Clearing Price"] * fr_mw * EFA_HOURS

    df["month"] = df["EFA Date"].dt.to_period("M")

    return (
        df.groupby(["month", "Service"])["revenue_gbp"]
        .sum()
        .reset_index()
        .rename(columns={"Service": "service"})
    )


# ---------------------------------------------------------------------------
# Stream 2 — Imbalance / spot arbitrage revenue
# ---------------------------------------------------------------------------

def calc_imbalance_revenue(
    market_index: pd.DataFrame,
    battery: BatterySpec,
    start_date=None,
    end_date=None,
    arb_mw: float = None,
    arb_energy_mwh: float = None,
    arb_schedule: "pd.Series | None" = None,
) -> pd.DataFrame:
    """
    Monthly wholesale energy arbitrage revenue using a daily intrinsic model.

    Price reference: APXMIDP (APX Power UK Market Index) — the GB spot market
    settlement reference price. Filtered from the market_index DataFrame which
    may contain multiple data providers. N2EXMIDP is excluded as it carries
    near-zero prices across most periods and is not a reliable price signal.

    For each settlement day:
      - Identify the N cheapest half-hour periods (N = duration_h × 2) → charge here
      - Identify the N most expensive half-hour periods → discharge here
      - Gross profit = avg_discharge_price × energy_out
                     − avg_charge_price × energy_in   (energy_in = energy_mwh / efficiency)
      - Deduct cycling wear cost = cycling_cost_per_mwh × energy_out
      - Only execute the cycle if gross profit > cycling wear cost

    Parameters
    ----------
    arb_mw : float, optional
        Fixed MW available for arbitrage. Ignored when arb_schedule is provided.
        Defaults to battery.power_mw when both are None.
    arb_energy_mwh : float, optional
        Fixed usable energy (MWh). Ignored when arb_schedule is provided.
    arb_schedule : pd.Series, optional
        Per-day arb MW (index = date, values = arb_mw). When provided,
        energy capacity is computed per day as arb_mw_d × duration_h.

    Returns
    -------
    DataFrame with columns: [month (Period), imbalance_revenue_gbp (float), cycling_cost_gbp (float)]
    Where imbalance_revenue_gbp is the GROSS arbitrage profit (before deducting cycling wear),
    so that the wear cost can be shown as a separate bar in the stacked chart.
    """
    if arb_schedule is None:
        if arb_mw is None:
            arb_mw = battery.power_mw
        if arb_energy_mwh is None:
            arb_energy_mwh = battery.energy_mwh
        if arb_mw == 0 or arb_energy_mwh == 0:
            return pd.DataFrame(columns=["month", "imbalance_revenue_gbp", "cycling_cost_gbp"])

    if market_index.empty:
        return pd.DataFrame(columns=["month", "imbalance_revenue_gbp", "cycling_cost_gbp"])

    # Filter to APXMIDP only — the reliable GB spot price reference
    df = market_index[market_index["dataProvider"] == "APXMIDP"].copy()
    df = _filter_dates(df, "settlementDate", start_date, end_date)

    if df.empty:
        return pd.DataFrame(columns=["month", "imbalance_revenue_gbp", "cycling_cost_gbp"])

    n_periods = max(1, int(battery.duration_h * 2))

    if arb_schedule is not None:
        arb_sched_map = {pd.Timestamp(k).normalize(): float(v) for k, v in arb_schedule.items()}

    rows = []
    for date, day_df in df.groupby("settlementDate"):
        if len(day_df) < n_periods * 2:
            continue

        if arb_schedule is not None:
            arb_mw_d = arb_sched_map.get(pd.Timestamp(date).normalize(), 0.0)
            if arb_mw_d <= 0:
                continue
            energy_out = arb_mw_d * battery.duration_h
            energy_in  = energy_out / battery.efficiency_rt
        else:
            energy_out = arb_energy_mwh
            energy_in  = arb_energy_mwh / battery.efficiency_rt

        avg_charge    = day_df.nsmallest(n_periods, "price")["price"].mean()
        avg_discharge = day_df.nlargest(n_periods, "price")["price"].mean()

        gross_profit = avg_discharge * energy_out - avg_charge * energy_in
        cycling_wear = battery.cycling_cost_per_mwh * energy_out

        if gross_profit > cycling_wear:
            rows.append({
                "date": date,
                "imbalance_revenue_gbp": gross_profit,
                "cycling_cost_gbp":      cycling_wear,
            })

    if not rows:
        return pd.DataFrame(columns=["month", "imbalance_revenue_gbp", "cycling_cost_gbp"])

    daily = pd.DataFrame(rows)
    daily["month"] = pd.to_datetime(daily["date"]).dt.to_period("M")

    return (
        daily.groupby("month")
        .agg(
            imbalance_revenue_gbp=("imbalance_revenue_gbp", "sum"),
            cycling_cost_gbp=("cycling_cost_gbp", "sum"),
        )
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Full backtest runner
# ---------------------------------------------------------------------------

def run_backtest(
    auctions: pd.DataFrame,
    market_index: pd.DataFrame,
    battery: BatterySpec,
    services: list = None,
    start_date=None,
    end_date=None,
    fr_mw: float = None,
) -> dict:
    """
    Run the full revenue stack backtest.

    Parameters
    ----------
    auctions      : DataFrame from load_auctions()
    market_index  : DataFrame from load_market_index() — used for APXMIDP spot arbitrage.
                    Pass an empty DataFrame to exclude the arbitrage stream.
    battery       : BatterySpec instance
    services      : list of service codes to include (default: all six)
    start_date    : inclusive start date (str or datetime)
    end_date      : inclusive end date (str or datetime)
    fr_mw         : float, optional
        MW committed to FR availability services for the entire period. When
        provided, a fixed split is applied — used internally by find_optimal_split.
        When omitted (default), a day-ahead dynamic allocation is computed for each
        day using the confirmed FR clearing price vs a perfect-foresight shadow arb
        estimate. Falls back to full FR commitment if market_index is empty.

    Returns
    -------
    dict with keys:
      'monthly'  : wide-format DataFrame, one row per month, columns for each revenue stream
      'summary'  : dict of aggregate stats (totals, annualised figures)
    """
    if services is None:
        services = ALL_SERVICES

    # --- Resolve FR/arbitrage split ---
    if fr_mw is not None:
        # Fixed allocation — used by find_optimal_split and sensitivity_table
        fr_mw     = float(np.clip(fr_mw, 0, battery.power_mw))
        fr_sch    = None
        arb_sch   = None
        avg_fr_mw = fr_mw
    else:
        # Dynamic perfect-foresight allocation: use actual prices as the
        # allocation signal (confirmed FR clearing price vs shadow arb value)
        if not market_index.empty:
            apx = market_index[market_index["dataProvider"] == "APXMIDP"].copy()
            apx["settlementDate"] = pd.to_datetime(apx["settlementDate"]).dt.normalize()
            apx_filt = _filter_dates(apx, "settlementDate", start_date, end_date)
            pf_prices = {
                date: grp.set_index("settlementPeriod")["price"]
                for date, grp in apx_filt.groupby("settlementDate")
            }
            fr_sch    = compute_daily_fr_schedule(
                auctions, pf_prices, battery, services, start_date, end_date
            )
            arb_sch   = (battery.power_mw - fr_sch).clip(lower=0)
            avg_fr_mw = float(fr_sch.mean()) if len(fr_sch) > 0 else battery.power_mw
        else:
            # No market data: full FR commitment, no arbitrage
            fr_mw     = battery.power_mw
            fr_sch    = None
            arb_sch   = None
            avg_fr_mw = fr_mw
    avg_arb_mw = battery.power_mw - avg_fr_mw

    # --- Ancillary ---
    anc = calc_ancillary_revenue(
        auctions, battery, services, start_date, end_date,
        fr_schedule=fr_sch, fr_mw=fr_mw if fr_sch is None else None,
    )
    if not anc.empty:
        anc_wide = anc.pivot_table(
            index="month", columns="service", values="revenue_gbp", fill_value=0
        )
        anc_wide.columns = [f"{c}_rev" for c in anc_wide.columns]
    else:
        anc_wide = pd.DataFrame()

    # --- Arbitrage ---
    if arb_sch is not None:
        imb = calc_imbalance_revenue(
            market_index, battery, start_date, end_date, arb_schedule=arb_sch,
        )
    else:
        arb_mw_fixed       = battery.power_mw - (fr_mw if fr_mw is not None else battery.power_mw)
        arb_energy_mwh_fixed = arb_mw_fixed * battery.duration_h
        imb = calc_imbalance_revenue(
            market_index, battery, start_date, end_date,
            arb_mw=arb_mw_fixed, arb_energy_mwh=arb_energy_mwh_fixed,
        )
    if not imb.empty:
        imb_wide = imb.set_index("month")
    else:
        imb_wide = pd.DataFrame()

    # --- Merge streams ---
    frames = [f for f in [anc_wide, imb_wide] if not f.empty]
    if not frames:
        return {"monthly": pd.DataFrame(), "summary": {}}

    monthly = frames[0].join(frames[1:], how="outer").fillna(0).reset_index()
    monthly["month_dt"] = monthly["month"].dt.to_timestamp()

    rev_cols  = [c for c in monthly.columns if c.endswith("_rev") or c == "imbalance_revenue_gbp"]
    cost_cols = [c for c in monthly.columns if c == "cycling_cost_gbp"]

    # Apply availability factor to every revenue stream and cycling cost proportionally.
    # This models the fraction of committed periods/days where the asset is actually
    # available — accounting for planned maintenance, unplanned faults, and curtailment.
    for col in rev_cols + cost_cols:
        monthly[col] = monthly[col] * battery.availability_factor

    monthly["gross_revenue"]  = monthly[rev_cols].sum(axis=1)
    monthly["cycling_cost"]   = monthly[cost_cols].sum(axis=1) if cost_cols else 0.0
    monthly["net_revenue"]    = monthly["gross_revenue"] - monthly["cycling_cost"]

    # --- Summary stats ---
    years     = len(monthly) / 12
    net_total = monthly["net_revenue"].sum()

    breakdown = {}
    for col in rev_cols:
        label = col.replace("_rev", "") if col.endswith("_rev") else "Imbalance"
        breakdown[label] = round(monthly[col].sum(), 0)

    summary = {
        "total_gross":        round(monthly["gross_revenue"].sum(), 0),
        "total_cycling_cost": round(monthly["cycling_cost"].sum(), 0),
        "total_net":          round(net_total, 0),
        "years_covered":      round(years, 2),
        "annualised_net":     round(net_total / years, 0) if years > 0 else 0,
        "annualised_per_mw":  round(net_total / years / battery.power_mw, 0) if years > 0 and battery.power_mw > 0 else 0,
        "breakdown":          breakdown,
        "top_service":        max(breakdown, key=breakdown.get) if breakdown else "N/A",
        "fr_mw":   avg_fr_mw,
        "arb_mw":  avg_arb_mw,
    }

    return {"monthly": monthly, "summary": summary}


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

def sensitivity_table(
    auctions: pd.DataFrame,
    market_index: pd.DataFrame,
    base_spec: BatterySpec,
    power_range: list = None,
    start_date=None,
    end_date=None,
) -> pd.DataFrame:
    """
    Run the backtest across a range of battery sizes, holding other parameters fixed.
    Returns a DataFrame suitable for display as a summary table.

    Each battery size uses the dynamic day-ahead allocation model, so the
    FR/arbitrage split is independently optimised for each size.
    """
    if power_range is None:
        power_range = [10, 25, 50, 100, 200]

    rows = []
    for mw in power_range:
        spec = BatterySpec(
            power_mw=mw,
            duration_h=base_spec.duration_h,
            efficiency_rt=base_spec.efficiency_rt,
            cycling_cost_per_mwh=base_spec.cycling_cost_per_mwh,
        )
        result = run_backtest(
            auctions, market_index, spec,
            start_date=start_date, end_date=end_date,
        )
        s = result["summary"]
        rows.append({
            "Power (MW)":              mw,
            "Energy (MWh)":            round(mw * base_spec.duration_h, 0),
            "Total Net Revenue (£k)":  round(s.get("total_net", 0) / 1_000, 1),
            "Ann. Net Revenue (£k/yr)": round(s.get("annualised_net", 0) / 1_000, 1),
            "Revenue / MW (£k/MW/yr)": round(s.get("annualised_per_mw", 0) / 1_000, 1),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Optimal FR/arbitrage split finder
# ---------------------------------------------------------------------------

def find_optimal_split(
    auctions: pd.DataFrame,
    market_index: pd.DataFrame,
    battery: BatterySpec,
    services: list = None,
    start_date=None,
    end_date=None,
    n_steps: int = 40,
) -> tuple:
    """
    Sweep fr_mw from 0 to battery.power_mw across n_steps evenly-spaced values
    and return the revenue-maximising split alongside the full trade-off curve.

    Parameters
    ----------
    n_steps : int
        Number of fr_mw values to evaluate (default 40). More steps give a
        smoother curve but take proportionally longer to compute.

    Returns
    -------
    (optimal_fr_mw, trade_off_df) where:
      optimal_fr_mw  : float — fr_mw that maximises total net revenue
      trade_off_df   : DataFrame with columns [fr_mw, arb_mw, total_net_gbp, annualised_per_mw]
    """
    if services is None:
        services = ALL_SERVICES

    step = battery.power_mw / n_steps
    fr_values = np.arange(0, battery.power_mw + step * 0.5, step)
    fr_values = np.clip(fr_values, 0, battery.power_mw)

    rows = []
    for fr in fr_values:
        result = run_backtest(
            auctions, market_index, battery, services,
            start_date=start_date, end_date=end_date,
            fr_mw=fr,
        )
        s = result["summary"]
        rows.append({
            "fr_mw":             round(fr, 2),
            "arb_mw":            round(battery.power_mw - fr, 2),
            "total_net_gbp":     s.get("total_net", 0),
            "annualised_per_mw": s.get("annualised_per_mw", 0),
        })

    trade_off_df = pd.DataFrame(rows)

    if trade_off_df.empty:
        return battery.power_mw, trade_off_df

    best_idx = trade_off_df["total_net_gbp"].idxmax()
    optimal_fr_mw = float(trade_off_df.loc[best_idx, "fr_mw"])

    return optimal_fr_mw, trade_off_df


# ---------------------------------------------------------------------------
# Day-ahead FR/arbitrage capacity allocator
# ---------------------------------------------------------------------------

def compute_daily_fr_schedule(
    auctions: pd.DataFrame,
    forecast_prices_by_date: dict,
    battery: BatterySpec,
    services: list = None,
    start_date=None,
    end_date=None,
) -> pd.Series:
    """
    Day-ahead FR/arbitrage capacity allocation optimiser.

    For each day D with an entry in forecast_prices_by_date, compares:

    FR value per MW (£/MW/day):
        Sum of clearing prices × EFA_HOURS across all selected services for
        that EFA date. EAC runs daily day-ahead auctions, so clearing prices
        for day D are known by end of day D-1 — no look-ahead bias.

    Shadow arb value per MW (£/MW/day):
        Estimated net profit from one full arbitrage cycle using forecast prices:
        (avg_discharge − avg_charge / η − cycling_cost) × duration_h.
        This is a per-unit (1 MW) estimate; since arbitrage revenue scales
        linearly with MW, only the per-MW rate is needed for the comparison.

    Allocation rule (proportional):
        fr_fraction = fr_value / (fr_value + arb_value)

    Capacity flows toward whichever stream looks more attractive on that day
    without all-or-nothing switching. Days where arb is unprofitable
    (arb_value = 0) receive full FR commitment.

    Parameters
    ----------
    forecast_prices_by_date : dict
        {date: pd.Series(index=settlementPeriod, values=price)} — the price
        signal for the shadow arb estimate. Pass actual day-D prices for
        perfect-foresight allocation; D-1 prices or ML predictions for
        realistic day-ahead allocation.
    battery : BatterySpec
    services : list of FR service codes to include (default: ALL_SERVICES)
    start_date, end_date : applied to auctions filter only

    Returns
    -------
    pd.Series indexed by pd.Timestamp (normalised to midnight), values = fr_mw.
    """
    if services is None:
        services = ALL_SERVICES

    # Daily FR value per MW from confirmed auction clearing prices.
    # EAC clearing prices are for the service delivery date and are determined
    # in the day-ahead auction — used directly with no lag.
    df_a = _filter_dates(auctions.copy(), "EFA Date", start_date, end_date)
    df_a = df_a[df_a["Service"].isin(services)]
    df_a = df_a[df_a["Clearing Price"] >= 0.0]

    if not df_a.empty:
        daily_fr_value = (
            df_a.groupby("EFA Date")["Clearing Price"]
            .apply(lambda x: (x * EFA_HOURS).sum())
        )
        daily_fr_value.index = pd.DatetimeIndex(daily_fr_value.index).normalize()
    else:
        daily_fr_value = pd.Series(dtype=float)

    n_periods = max(1, int(battery.duration_h * 2))
    schedule  = {}

    for date, forecast_prices in forecast_prices_by_date.items():
        date_ts = pd.Timestamp(date).normalize()

        # FR value per MW: confirmed clearing price for this day
        fr_value = float(daily_fr_value.get(date_ts, 0.0))

        # Shadow arb value per MW: estimated net profit from one cycle
        if len(forecast_prices) >= n_periods * 2:
            avg_discharge = forecast_prices.nlargest(n_periods).mean()
            avg_charge    = forecast_prices.nsmallest(n_periods).mean()
            net_per_mw = (
                avg_discharge
                - avg_charge / battery.efficiency_rt
                - battery.cycling_cost_per_mwh
            ) * battery.duration_h
            arb_value = max(0.0, net_per_mw)
        else:
            arb_value = 0.0

        total       = fr_value + arb_value
        fr_fraction = (fr_value / total) if total > 0 else 1.0
        schedule[date_ts] = fr_fraction * battery.power_mw

    return pd.Series(schedule, name="fr_mw")
