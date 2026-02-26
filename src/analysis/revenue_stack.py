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
        MW committed to FR availability. Defaults to battery.power_mw.
        Use this to model a partial FR commitment where some capacity is
        reserved for energy arbitrage instead.

    Returns
    -------
    DataFrame with columns: [month (Period), service (str), revenue_gbp (float)]
    """
    if fr_mw is None:
        fr_mw = battery.power_mw

    df = _filter_dates(auctions.copy(), "EFA Date", start_date, end_date)
    df = df[df["Service"].isin(services)]
    df = df[df["Clearing Price"] >= min_price]

    if df.empty or fr_mw == 0:
        return pd.DataFrame(columns=["month", "service", "revenue_gbp"])

    df = df.copy()
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
        MW available for arbitrage. Defaults to battery.power_mw.
        Set to battery.power_mw - fr_mw when using a capacity split.
    arb_energy_mwh : float, optional
        Usable energy (MWh) available for arbitrage. Defaults to battery.energy_mwh.
        Set to arb_mw × battery.duration_h when using a capacity split.

    Returns
    -------
    DataFrame with columns: [month (Period), imbalance_revenue_gbp (float), cycling_cost_gbp (float)]
    Where imbalance_revenue_gbp is the GROSS arbitrage profit (before deducting cycling wear),
    so that the wear cost can be shown as a separate bar in the stacked chart.
    """
    if arb_mw is None:
        arb_mw = battery.power_mw
    if arb_energy_mwh is None:
        arb_energy_mwh = battery.energy_mwh

    if market_index.empty or arb_mw == 0 or arb_energy_mwh == 0:
        return pd.DataFrame(columns=["month", "imbalance_revenue_gbp", "cycling_cost_gbp"])

    # Filter to APXMIDP only — the reliable GB spot price reference
    df = market_index[market_index["dataProvider"] == "APXMIDP"].copy()
    df = _filter_dates(df, "settlementDate", start_date, end_date)

    if df.empty:
        return pd.DataFrame(columns=["month", "imbalance_revenue_gbp", "cycling_cost_gbp"])

    n_periods = max(1, int(battery.duration_h * 2))
    energy_out = arb_energy_mwh
    energy_in  = arb_energy_mwh / battery.efficiency_rt  # MWh purchased to store energy_out

    rows = []
    for date, day_df in df.groupby("settlementDate"):
        if len(day_df) < n_periods * 2:
            continue

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
        MW committed to FR availability services. The remainder (battery.power_mw - fr_mw)
        is available for energy arbitrage. Defaults to battery.power_mw (full FR commitment,
        no arbitrage capacity split — same as the original single-stream model).

    Returns
    -------
    dict with keys:
      'monthly'  : wide-format DataFrame, one row per month, columns for each revenue stream
      'summary'  : dict of aggregate stats (totals, annualised figures)
    """
    if services is None:
        services = ALL_SERVICES

    # Resolve FR/arbitrage split
    if fr_mw is None:
        fr_mw = battery.power_mw
    fr_mw = float(np.clip(fr_mw, 0, battery.power_mw))
    arb_mw = battery.power_mw - fr_mw
    arb_energy_mwh = arb_mw * battery.duration_h

    # --- Ancillary ---
    anc = calc_ancillary_revenue(auctions, battery, services, start_date, end_date, fr_mw=fr_mw)
    if not anc.empty:
        anc_wide = anc.pivot_table(
            index="month", columns="service", values="revenue_gbp", fill_value=0
        )
        anc_wide.columns = [f"{c}_rev" for c in anc_wide.columns]
    else:
        anc_wide = pd.DataFrame()

    # --- Arbitrage ---
    imb = calc_imbalance_revenue(
        market_index, battery, start_date, end_date,
        arb_mw=arb_mw, arb_energy_mwh=arb_energy_mwh,
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
        "fr_mw":              fr_mw,
        "arb_mw":             arb_mw,
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
    fr_fraction: float = 1.0,
) -> pd.DataFrame:
    """
    Run the backtest across a range of battery sizes, holding other parameters fixed.
    Returns a DataFrame suitable for display as a summary table.

    Parameters
    ----------
    fr_fraction : float
        FR commitment as a fraction of power_mw (0.0–1.0). Scales with each
        test power level so the participation split is consistent across the table.
        Default 1.0 = full FR commitment (original behaviour).
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
        fr_mw_for_spec = fr_fraction * mw
        result = run_backtest(
            auctions, market_index, spec,
            start_date=start_date, end_date=end_date,
            fr_mw=fr_mw_for_spec,
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
