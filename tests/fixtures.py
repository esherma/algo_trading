"""
Synthetic minute-bar data fixtures used across the test suite.

All fixtures return DataFrames matching the format produced by btm.data.fetch_minute_bars:
  - DatetimeIndex in America/New_York timezone
  - Columns: open, high, low, close, volume, vwap
  - Filtered to 09:30–16:00 market hours
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
import pytz

NY_TZ = pytz.timezone("America/New_York")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trading_minutes(d: date) -> pd.DatetimeIndex:
    """Return a DatetimeIndex of every market minute on *d*."""
    start = pd.Timestamp(d.isoformat() + " 09:30", tz=NY_TZ)
    end   = pd.Timestamp(d.isoformat() + " 16:00", tz=NY_TZ)
    return pd.date_range(start, end, freq="1min")


def _trading_dates(n: int, anchor: date = date(2022, 1, 3)) -> List[date]:
    """Return *n* weekdays starting from *anchor*."""
    days = []
    d = anchor
    while len(days) < n:
        if d.weekday() < 5:   # Mon–Fri
            days.append(d)
        d += timedelta(days=1)
    return days


# ---------------------------------------------------------------------------
# Core synthetic dataset builder
# ---------------------------------------------------------------------------

def make_minute_bars(
    n_days: int = 20,
    base_price: float = 400.0,
    daily_vol: float = 0.01,    # ~1% daily std
    intraday_vol: float = 0.002,
    seed: int = 42,
    anchor: date = date(2022, 1, 3),
) -> pd.DataFrame:
    """
    Build *n_days* × 391 rows of synthetic minute bars with realistic
    price dynamics (geometric Brownian motion, resetting open each day).

    Useful for algorithm unit tests that must not touch the Alpaca API.
    """
    rng   = np.random.default_rng(seed)
    dates = _trading_dates(n_days, anchor)
    rows  = []

    day_price = base_price
    for d in dates:
        # Gap: day opens near previous close (small gap)
        day_open = day_price * np.exp(rng.normal(0, daily_vol * 0.3))
        price    = day_open

        minutes = _trading_minutes(d)
        for ts in minutes:
            ret   = rng.normal(0, intraday_vol)
            close = price * (1 + ret)
            high  = close * (1 + abs(rng.normal(0, intraday_vol * 0.5)))
            low   = close * (1 - abs(rng.normal(0, intraday_vol * 0.5)))
            vol   = int(rng.integers(100_000, 2_000_000))
            rows.append({
                "ts":    ts,
                "open":  float(price),
                "high":  float(high),
                "low":   float(low),
                "close": float(close),
                "volume": vol,
                "vwap":  float((price + close) / 2),
            })
            price = close

        # Day close → next day base
        day_price = price

    df = pd.DataFrame(rows).set_index("ts")
    return df[["open", "high", "low", "close", "volume", "vwap"]]


# ---------------------------------------------------------------------------
# Specialised scenarios
# ---------------------------------------------------------------------------

def make_strong_trend_up(n_days: int = 20, seed: int = 7) -> pd.DataFrame:
    """Bars with a consistent upward drift — should generate long signals."""
    return make_minute_bars(n_days=n_days, daily_vol=0.005, intraday_vol=0.004,
                            seed=seed)


def make_strong_trend_down(n_days: int = 20, seed: int = 99) -> pd.DataFrame:
    """Bars with a consistent downward drift — should generate short signals."""
    rng   = np.random.default_rng(seed)
    dates = _trading_dates(n_days)
    rows  = []
    base  = 400.0
    price = base
    for d in dates:
        day_open = price
        for ts in _trading_minutes(d):
            ret   = rng.normal(-0.004, 0.003)  # negative drift
            close = price * (1 + ret)
            rows.append({
                "ts": ts, "open": price, "high": close * 1.001,
                "low": close * 0.999, "close": close,
                "volume": 500_000, "vwap": (price + close) / 2,
            })
            price = close
    df = pd.DataFrame(rows).set_index("ts")
    return df[["open", "high", "low", "close", "volume", "vwap"]]


def make_flat_market(n_days: int = 20) -> pd.DataFrame:
    """Bars that barely move — price stays inside the noise zone every day."""
    return make_minute_bars(n_days=n_days, daily_vol=0.0001, intraday_vol=0.00005,
                            seed=123)
