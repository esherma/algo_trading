"""
Beat the Market (BTM) core algorithm.

Paper: "Beat the Market: An Effective Intraday Momentum Strategy for S&P500 ETF (SPY)"
       Zarattini, Aziz & Barbon.

Algorithm summary
-----------------
At each half-hour mark (10:00, 10:30, … 15:30) compare SPY price to noise bands:
  Long  if price > UpperBound(t, τ)
  Short if price < LowerBound(t, τ)
  Stop for long:  close when price < max(UB, VWAP)   ← "current band + VWAP"
  Stop for short: close when price > min(LB, VWAP)
  All positions close by 15:50.

Noise band construction
-----------------------
  σ(t, τ) = mean over last N days of |Close(day, τ) / Open(day, 09:30) - 1|
  upper_base = max(Open_today, Close_yesterday)
  lower_base = min(Open_today, Close_yesterday)
  UB(t, τ) = upper_base × (1 + vm × σ(t, τ))
  LB(t, τ) = lower_base × (1 - vm × σ(t, τ))

Position sizing
---------------
  σ_SPY = std(daily_returns[-lookback:])
  leverage = min(leverage_cap, target_vol / σ_SPY)
  etf_shares = floor(AUM × leverage / (etf_multiplier × SPY_open))
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NY_TZ = "America/New_York"

# All half-hour decision times within the trading session.
DECISION_TIMES: List[str] = [
    f"{h:02d}:{m:02d}"
    for h in range(10, 16)
    for m in (0, 30)
    if not (h == 15 and m > 30)
]  # ["10:00", "10:30", ..., "15:30"]

CLOSE_POSITIONS_TIME = "15:50"   # force-flat time
MARKET_OPEN_TIME = "09:30"
MARKET_CLOSE_TIME = "16:00"

# ETF leverage multipliers (how many × SPY exposure each share gives)
ETF_MULT: Dict[str, int] = {
    "SPXL": 3,  "SPXS": 3,
    "SPUU": 2,  "SDS":  2,
    "SPY":  1,  "SPDN": 1,
}

# Leverage tiers: (min_leverage, long_etf, short_etf, mult)
# Evaluated top-to-bottom; first matching tier is used.
ETF_TIERS: List[Tuple[float, str, str, int]] = [
    (2.5, "SPXL", "SPXS", 3),
    (1.5, "SPUU", "SDS",  2),
    (0.0, "SPY",  "SPDN", 1),
]

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class BTMConfig:
    """All parameters required to run a backtest or live session."""

    symbol: str = "SPY"
    session: Literal["paper", "live"] = "paper"

    lookback_days: int = 14          # Days used for σ and daily-vol estimation
    vm: float = 1.0                  # Volatility multiplier for band width

    target_daily_vol: float = 0.02   # 2% target – drives position sizing
    leverage_cap: float = 4.0        # Maximum leverage allowed

    # Backtest transaction costs
    commission_per_share: float = 0.0035
    slippage_per_share:   float = 0.001

    initial_aum: float = 100_000.0   # Starting portfolio value ($)


# ---------------------------------------------------------------------------
# Internal helper: pivot of absolute moves from open
# ---------------------------------------------------------------------------

def _build_abs_move_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a (n_days × n_times) pivot table where each cell is
        |Close(day, τ) / Open(day, 09:30) - 1|
    Index: python date objects, sorted ascending.
    Columns: "HH:MM" strings.
    Missing cells (time absent on a given day) are NaN.
    """
    dates = sorted(set(df.index.date))

    # Day opens: first bar of each date
    day_open: Dict = {}
    for d in dates:
        bars = df[df.index.date == d]
        if len(bars) > 0:
            day_open[d] = float(bars["open"].iloc[0])

    # Compute abs moves into a dict keyed by (date, "HH:MM")
    rows: Dict = {}
    for ts, row in df.iterrows():
        d = ts.date()
        if d not in day_open or day_open[d] == 0:
            continue
        t = ts.strftime("%H:%M")
        rows.setdefault(d, {})[t] = abs(float(row["close"]) / day_open[d] - 1.0)

    all_times = sorted({ts.strftime("%H:%M") for ts in df.index})
    pivot = pd.DataFrame(
        {t: [rows.get(d, {}).get(t, np.nan) for d in dates] for t in all_times},
        index=dates,
    )
    return pivot


# ---------------------------------------------------------------------------
# Sigma (noise level) computation
# ---------------------------------------------------------------------------

def compute_sigma_rolling(df: pd.DataFrame, lookback: int = 14) -> pd.Series:
    """
    Backtest-safe, look-ahead-free sigma for every minute in *df*.

    For each day t:
        σ(t, τ) = mean of |Close(t-i, τ)/Open(t-i,9:30) - 1|  for i = 1..lookback

    Implementation: build pivot, rolling(lookback).mean() (uses days [t-N+1..t]),
    then shift(1) so that day-t sigma uses days [t-N..t-1] — no look-ahead.

    Returns a Series with the same DatetimeIndex as *df*.
    """
    pivot = _build_abs_move_pivot(df)

    # rolling(lookback).mean() at row i → mean of rows [i-lookback+1 .. i]
    # .shift(1)                    at row i → that mean for row i-1
    #   ⟹ for date d[i] we use rows [i-lookback .. i-1]  ✓ look-ahead free
    rolling_sigma = (
        pivot
        .rolling(window=lookback, min_periods=1)
        .mean()
        .shift(1)
        .fillna(0.01)  # first day has no prior data → small constant
    )

    # Convert the pivot's date index to a lookup dict for fast row access
    sigma_by_date: Dict = {d: rolling_sigma.loc[d] for d in rolling_sigma.index}

    values: List[float] = []
    for ts in df.index:
        d = ts.date()
        t = ts.strftime("%H:%M")
        row = sigma_by_date.get(d)
        if row is not None and t in row.index:
            values.append(float(row[t]))
        else:
            values.append(0.01)

    return pd.Series(values, index=df.index, name="sigma")


def compute_sigma_for_today(hist_df: pd.DataFrame, lookback: int = 14) -> pd.Series:
    """
    Point-in-time sigma for live trading.

    *hist_df* must contain only data from BEFORE today (no look-ahead).
    Uses the last *lookback* complete trading days in the data.

    Returns a Series indexed by "HH:MM" strings.
    """
    pivot = _build_abs_move_pivot(hist_df)
    if len(pivot) == 0:
        # Fallback: no historical data
        all_times = [f"{h:02d}:{m:02d}" for h in range(9, 16) for m in range(60)]
        return pd.Series(0.01, index=all_times, name="sigma")

    n = min(lookback, len(pivot))
    return pivot.iloc[-n:].mean(skipna=True).fillna(0.01).rename("sigma")


# ---------------------------------------------------------------------------
# Noise band construction
# ---------------------------------------------------------------------------

def compute_bands_for_backtest(
    df: pd.DataFrame,
    sigma: pd.Series,
    vm: float = 1.0,
) -> pd.DataFrame:
    """
    Compute UB and LB for every minute in *df* (backtest use).

    For each day t:
        upper_base = max(Open_t, Close_{t-1})
        lower_base = min(Open_t, Close_{t-1})
        UB(t, τ) = upper_base × (1 + vm × σ(t, τ))
        LB(t, τ) = lower_base × (1 - vm × σ(t, τ))

    Returns a DataFrame with columns {UB, LB, sigma} and the same index as *df*.
    """
    dates = sorted(set(df.index.date))

    day_open_map: Dict = {}
    day_close_map: Dict = {}
    for d in dates:
        bars = df[df.index.date == d]
        if len(bars) > 0:
            day_open_map[d] = float(bars["open"].iloc[0])
            day_close_map[d] = float(bars["close"].iloc[-1])

    day_prev_close: Dict = {}
    for i, d in enumerate(dates):
        if i == 0:
            day_prev_close[d] = day_open_map[d]  # No prior day — use open
        else:
            day_prev_close[d] = day_close_map[dates[i - 1]]

    # Vectorised band computation
    dates_arr = np.array(df.index.date)
    open_arr      = np.vectorize(lambda d: day_open_map.get(d, np.nan))(dates_arr)
    prev_close_arr = np.vectorize(lambda d: day_prev_close.get(d, np.nan))(dates_arr)

    upper_base = np.maximum(open_arr, prev_close_arr)
    lower_base = np.minimum(open_arr, prev_close_arr)

    s = sigma.values * vm
    ub = upper_base * (1.0 + s)
    lb = lower_base * (1.0 - s)

    return pd.DataFrame({"UB": ub, "LB": lb, "sigma": sigma.values}, index=df.index)


def compute_bands_for_today(
    sigma_by_time: pd.Series,
    today_open: float,
    yesterday_close: float,
    vm: float = 1.0,
) -> pd.DataFrame:
    """
    Compute today's noise bands for live trading.

    *sigma_by_time*: Series indexed by "HH:MM" strings (from compute_sigma_for_today).
    Returns a DataFrame with columns {UB, LB, sigma} indexed by "HH:MM".
    """
    upper_base = max(today_open, yesterday_close)
    lower_base = min(today_open, yesterday_close)

    ub = upper_base * (1.0 + sigma_by_time * vm)
    lb = lower_base * (1.0 - sigma_by_time * vm)

    return pd.DataFrame({"UB": ub, "LB": lb, "sigma": sigma_by_time})


# ---------------------------------------------------------------------------
# Intraday VWAP
# ---------------------------------------------------------------------------

def compute_intraday_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Cumulative intraday VWAP using close price × volume, resetting each day.
    Returns a Series with the same index as *df*.
    """
    vwap_vals = np.empty(len(df))
    cum_pv = cum_vol = 0.0
    last_day = None

    for i, (ts, row) in enumerate(df.iterrows()):
        d = ts.date()
        if d != last_day:
            cum_pv = cum_vol = 0.0
            last_day = d
        price = float(row["close"])
        vol   = float(row["volume"]) if pd.notna(row["volume"]) else 0.0
        cum_pv  += price * vol
        cum_vol += vol
        vwap_vals[i] = cum_pv / cum_vol if cum_vol > 0 else price

    return pd.Series(vwap_vals, index=df.index, name="vwap")


# ---------------------------------------------------------------------------
# Position signal logic
# ---------------------------------------------------------------------------

def decide_position(
    price: float,
    ub: float,
    lb: float,
    vwap: float,
    current_pos: int,
) -> int:
    """
    Determine the desired position at a single decision point.

    Implements the "current band + VWAP" stop logic from the paper:
      Long stop:  close if price < max(UB, VWAP)   (price fell back into / below band)
      Short stop: close if price > min(LB, VWAP)   (price rose back into / above band)
      Entry:      long if price > UB,  short if price < LB

    Returns -1 (short), 0 (flat), or 1 (long).
    """
    pos = current_pos

    # 1. Evaluate trailing stop for existing position
    if pos == 1 and price < max(ub, vwap):
        pos = 0
    elif pos == -1 and price > min(lb, vwap):
        pos = 0

    # 2. Enter new position if now flat
    if pos == 0:
        if price > ub:
            pos = 1
        elif price < lb:
            pos = -1

    return pos


# ---------------------------------------------------------------------------
# Daily volatility and position sizing
# ---------------------------------------------------------------------------

def compute_daily_vol(daily_close: pd.Series, lookback: int = 14) -> float:
    """
    Annualised daily volatility estimate: std of last *lookback* daily returns.
    Returns a raw (non-annualised) std value suitable for the sizing formula.
    Falls back to 0.02 (2%) if insufficient data.
    """
    if len(daily_close) < 2:
        return 0.02
    returns = daily_close.pct_change().dropna()
    n = min(lookback, len(returns))
    if n < 2:
        return float(returns.std()) if len(returns) > 0 else 0.02
    return float(returns.iloc[-n:].std())


def select_etfs(leverage: float) -> Tuple[str, str, int]:
    """
    Choose long ETF, short ETF, and ETF multiplier based on *leverage*.
    Short ETF is an inverse fund (bought long to express a short view).
    Returns (long_etf, short_etf, etf_multiplier).
    """
    for min_lev, long_etf, short_etf, mult in ETF_TIERS:
        if leverage >= min_lev:
            return long_etf, short_etf, mult
    return "SPY", "SPDN", 1


def compute_etf_shares(
    aum: float,
    leverage: float,
    spy_open: float,
    etf_mult: int,
) -> int:
    """
    Number of ETF shares to trade for the desired notional SPY exposure:

        shares = floor(AUM × leverage / (etf_mult × spy_open))

    Works regardless of ETF price because we size by SPY-equivalent notional.
    """
    if spy_open <= 0 or etf_mult <= 0:
        return 0
    return int(math.floor(aum * leverage / (etf_mult * spy_open)))


# ---------------------------------------------------------------------------
# Full backtest
# ---------------------------------------------------------------------------

def run_backtest(df: pd.DataFrame, cfg: BTMConfig) -> Tuple[pd.DataFrame, Dict]:
    """
    Run the BTM strategy over *df* (minute bars, NY timezone).

    *df* should span the FULL period including the lookback warm-up window.
    Returns (result_df, summary_dict).

    result_df columns: price, UB, LB, vwap, sigma, position, aum, pnl_net
    """
    # ── 1. Sigma (look-ahead free) and noise bands ───────────────────────
    sigma = compute_sigma_rolling(df, cfg.lookback_days)
    bands = compute_bands_for_backtest(df, sigma, cfg.vm)
    vwap  = compute_intraday_vwap(df)

    # ── 2. Pre-compute daily open prices and close prices ────────────────
    dates = sorted(set(df.index.date))

    day_open_map:  Dict = {}
    day_close_map: Dict = {}
    for d in dates:
        bars = df[df.index.date == d]
        if len(bars) > 0:
            day_open_map[d]  = float(bars["open"].iloc[0])
            day_close_map[d] = float(bars["close"].iloc[-1])

    # ── 3. Extract numpy arrays for tight inner loop ─────────────────────
    price_arr    = df["close"].values.astype(float)
    ub_arr       = bands["UB"].values.astype(float)
    lb_arr       = bands["LB"].values.astype(float)
    vwap_arr     = vwap.values.astype(float)
    index_arr    = df.index  # DatetimeIndex (used for .date() and .strftime)

    decision_set = set(DECISION_TIMES)

    # ── 4. Minute-by-minute simulation ───────────────────────────────────
    n = len(df)
    pos_arr  = np.zeros(n, dtype=np.int8)
    aum_arr  = np.empty(n, dtype=float)
    pnl_arr  = np.zeros(n, dtype=float)

    current_aum  = cfg.initial_aum
    current_pos  = 0
    prev_price   = price_arr[0]
    current_day  = None
    shares_day   = 0

    for i in range(n):
        ts = index_arr[i]
        d  = ts.date()

        # ── Day boundary: reset position and recalculate sizing ──
        if d != current_day:
            current_day = d
            current_pos = 0   # All positions must be flat at EOD

            # Daily vol uses only prior-day closes (look-ahead free)
            prior_closes = pd.Series(
                {pd.Timestamp(dd).tz_localize(NY_TZ): day_close_map[dd]
                 for dd in dates if dd < d and dd in day_close_map}
            )
            vol_day  = compute_daily_vol(prior_closes, cfg.lookback_days)
            leverage = min(cfg.leverage_cap, cfg.target_daily_vol / max(vol_day, 1e-6))
            spy_open = day_open_map.get(d, 1.0)
            shares_day = int(math.floor(current_aum * leverage / max(spy_open, 1e-6)))

            prev_price = price_arr[i]

        # ── Minute PnL from holding prior position ───────────────
        minute_pnl = (price_arr[i] - prev_price) * (shares_day * current_pos)

        # ── Decision point: check stops and entries ───────────────
        t_str = ts.strftime("%H:%M")
        if t_str in decision_set:
            new_pos = decide_position(
                price_arr[i], ub_arr[i], lb_arr[i], vwap_arr[i], current_pos
            )
            delta = abs(new_pos - current_pos)
            if delta > 0:
                traded_shares = delta * shares_day
                minute_pnl -= traded_shares * (
                    cfg.commission_per_share + cfg.slippage_per_share
                )
            current_pos = new_pos

        # ── Force flat at/after close-positions time ──────────────
        # Runs independently of decision times since CLOSE_POSITIONS_TIME
        # (15:50) is intentionally not a decision time.
        elif current_pos != 0 and t_str >= CLOSE_POSITIONS_TIME:
            traded_shares = abs(current_pos) * shares_day
            minute_pnl -= traded_shares * (
                cfg.commission_per_share + cfg.slippage_per_share
            )
            current_pos = 0

        current_aum += minute_pnl
        pos_arr[i]  = current_pos
        aum_arr[i]  = current_aum
        pnl_arr[i]  = minute_pnl
        prev_price  = price_arr[i]

    # ── 5. Assemble result DataFrame ─────────────────────────────────────
    result = pd.DataFrame({
        "price":    price_arr,
        "UB":       ub_arr,
        "LB":       lb_arr,
        "vwap":     vwap_arr,
        "sigma":    sigma.values,
        "position": pos_arr,
        "aum":      aum_arr,
        "pnl_net":  pnl_arr,
    }, index=df.index)

    # ── 6. Summary statistics ─────────────────────────────────────────────
    daily_aum = result["aum"].groupby(result.index.date).last()
    daily_aum.index = pd.to_datetime(daily_aum.index).tz_localize(NY_TZ)
    daily_ret = daily_aum.pct_change().dropna()

    n_days = max(len(daily_ret), 1)
    total_ret = (daily_aum.iloc[-1] / daily_aum.iloc[0] - 1.0) * 100
    irr = ((daily_aum.iloc[-1] / daily_aum.iloc[0]) ** (252.0 / n_days) - 1.0) * 100
    ann_vol = float(daily_ret.std()) * math.sqrt(252) * 100
    sharpe  = (float(daily_ret.mean()) / float(daily_ret.std()) * math.sqrt(252)
               if float(daily_ret.std()) > 0 else 0.0)
    mdd     = float((daily_aum / daily_aum.cummax() - 1.0).min()) * 100

    # Day-level hit ratio (% of trading days with positive P&L)
    day_pnl = pd.Series(pnl_arr, index=df.index).groupby(df.index.date).sum()
    hit_ratio = float((day_pnl > 0).mean()) * 100

    # Trade count = number of times position changes to non-zero
    pos_series   = pd.Series(pos_arr, index=df.index)
    pos_changes  = pos_series.diff().fillna(0)
    trade_entries = (pos_changes != 0) & (pos_series != 0)
    trade_count   = int(trade_entries.sum())

    summary: Dict = {
        "StartDate":        str(daily_aum.index[0].date()),
        "EndDate":          str(daily_aum.index[-1].date()),
        "TotalReturn_Pct":  round(total_ret, 2),
        "IRR_Pct":          round(irr, 2),
        "AnnualVol_Pct":    round(ann_vol, 2),
        "Sharpe":           round(sharpe, 2),
        "MaxDrawdown_Pct":  round(mdd, 2),
        "HitRatio_Pct":     round(hit_ratio, 1),
        "TradeCount":       trade_count,
        "AvgTradesPerDay":  round(trade_count / max(len(dates), 1), 1),
        "FinalAUM":         round(float(daily_aum.iloc[-1]), 2),
        "InitialAUM":       cfg.initial_aum,
    }

    return result, summary
