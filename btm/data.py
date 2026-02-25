"""
Alpaca data-fetching helpers.

All returned DataFrames use a timezone-aware DatetimeIndex in America/New_York,
filtered to regular session hours (09:30 – 16:00).
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import pytz
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestBarRequest, StockSnapshotRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient

NY_TZ = pytz.timezone("America/New_York")
UTC   = pytz.UTC

# ---------------------------------------------------------------------------
# Client factories
# ---------------------------------------------------------------------------

def _get_keys(session: str) -> tuple[str, str]:
    load_dotenv()
    if session == "paper":
        key    = os.getenv("ALPACA_PAPER_API_KEY", "")
        secret = os.getenv("ALPACA_PAPER_API_SECRET", "")
    else:
        key    = os.getenv("ALPACA_LIVE_API_KEY", "")
        secret = os.getenv("ALPACA_LIVE_API_SECRET", "")
    if not key or not secret:
        raise RuntimeError(
            f"Missing Alpaca credentials for session='{session}'. "
            "Set ALPACA_PAPER_API_KEY / ALPACA_PAPER_API_SECRET (or LIVE equivalents) in .env"
        )
    return key, secret


def make_data_client(session: str = "paper") -> StockHistoricalDataClient:
    key, secret = _get_keys(session)
    return StockHistoricalDataClient(key, secret)


def make_trading_client(session: str = "paper") -> TradingClient:
    key, secret = _get_keys(session)
    return TradingClient(key, secret, paper=(session == "paper"))


# ---------------------------------------------------------------------------
# Internal conversion
# ---------------------------------------------------------------------------

def _bars_to_df(bars_data, symbol: str) -> pd.DataFrame:
    """Convert Alpaca bar objects to a cleaned DataFrame."""
    if symbol not in bars_data.data:
        raise RuntimeError(f"No bar data returned for {symbol!r}")

    records = [
        {
            "ts":          bar.timestamp,
            "open":        bar.open,
            "high":        bar.high,
            "low":         bar.low,
            "close":       bar.close,
            "volume":      bar.volume,
            "trade_count": bar.trade_count,
            "vwap":        bar.vwap,
        }
        for bar in bars_data.data[symbol]
    ]
    if not records:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "vwap"])

    df = pd.DataFrame.from_records(records)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts").sort_index()

    # Convert to New York time
    df.index = df.index.tz_convert(NY_TZ)

    return df[["open", "high", "low", "close", "volume", "vwap"]].copy()


def _filter_session(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only regular session bars: 09:30 ≤ time ≤ 16:00."""
    if df.empty:
        return df
    t = df.index.time
    open_t  = pd.Timestamp("09:30").time()
    close_t = pd.Timestamp("16:00").time()
    return df[(t >= open_t) & (t <= close_t)].copy()


# ---------------------------------------------------------------------------
# Public fetch functions
# ---------------------------------------------------------------------------

def fetch_minute_bars(
    client: StockHistoricalDataClient,
    symbol: str,
    start: str,                     # "YYYY-MM-DD"
    end: Optional[str] = None,      # "YYYY-MM-DD", inclusive; None = yesterday
    adjustment: str = "raw",
) -> pd.DataFrame:
    """
    Fetch 1-minute bars for *symbol* between *start* and *end* (inclusive).

    Uses the *raw* adjustment by default (no split/dividend adjustment).
    Returns a DataFrame with NY-timezone DatetimeIndex, filtered to market hours.
    """
    start_dt = pd.Timestamp(start, tz=UTC)

    if end is None:
        end_date = date.today() - timedelta(days=1)
        end = end_date.strftime("%Y-%m-%d")
    end_dt = pd.Timestamp(end, tz=UTC) + pd.Timedelta(days=1)  # include full end day

    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start_dt,
        end=end_dt,
        adjustment=adjustment,
    )
    bars = client.get_stock_bars(req)
    df   = _bars_to_df(bars, symbol)
    return _filter_session(df)


def fetch_daily_bars(
    client: StockHistoricalDataClient,
    symbol: str,
    start: str,
    end: Optional[str] = None,
    adjustment: str = "raw",
) -> pd.DataFrame:
    """
    Fetch daily bars.  Returns a DataFrame with NY-timezone DatetimeIndex.
    """
    start_dt = pd.Timestamp(start, tz=UTC)

    if end is None:
        end_date = date.today() - timedelta(days=1)
        end = end_date.strftime("%Y-%m-%d")
    end_dt = pd.Timestamp(end, tz=UTC) + pd.Timedelta(days=1)

    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start_dt,
        end=end_dt,
        adjustment=adjustment,
    )
    bars = client.get_stock_bars(req)
    df   = _bars_to_df(bars, symbol)
    return df


def fetch_latest_minute_bars(
    client: StockHistoricalDataClient,
    symbol: str,
    since: str,                    # "YYYY-MM-DD" or "YYYY-MM-DD HH:MM"
) -> pd.DataFrame:
    """
    Fetch all 1-minute bars for *symbol* from *since* up to now.
    Used during live trading to accumulate intraday bars.
    """
    start_dt = pd.Timestamp(since, tz=NY_TZ).tz_convert(UTC)
    now_utc  = pd.Timestamp.utcnow()

    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start_dt,
        end=now_utc,
        adjustment="raw",
    )
    bars = client.get_stock_bars(req)
    try:
        df = _bars_to_df(bars, symbol)
    except RuntimeError:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "vwap"])
    return _filter_session(df)


def fetch_snapshot(
    client: StockHistoricalDataClient,
    symbol: str,
) -> dict:
    """
    Fetch a real-time snapshot for *symbol* via the Alpaca snapshot endpoint.

    Unlike the historical bars endpoint (which reflects only data up to the
    previous day's close during market hours), the snapshot endpoint is served
    in real-time and returns the most recent 1-minute bar and the cumulative
    daily bar for the current session.

    Returns a dict with:
      - price:      close of the most recent 1-minute bar (current market price)
      - vwap:       session VWAP since today's open (daily_bar.vwap)
      - today_open: today's opening price (daily_bar.open)
      - volume:     today's cumulative volume
      - timestamp:  timestamp of the most recent 1-minute bar (UTC-aware)
    """
    snap = client.get_stock_snapshot(StockSnapshotRequest(symbol_or_symbols=symbol))
    s = snap[symbol]
    return {
        "price":      float(s.minute_bar.close),
        "vwap":       float(s.daily_bar.vwap),
        "today_open": float(s.daily_bar.open),
        "volume":     int(s.daily_bar.volume),
        "timestamp":  s.minute_bar.timestamp,
    }


def get_market_clock(trading_client: TradingClient) -> dict:
    """
    Return a dict with keys: is_open, next_open (datetime), next_close (datetime).
    """
    clock = trading_client.get_clock()
    return {
        "is_open":    clock.is_open,
        "next_open":  clock.next_open,
        "next_close": clock.next_close,
        "timestamp":  clock.timestamp,
    }


def is_trading_day(trading_client: TradingClient) -> bool:
    """Return True if today is a market trading day (next_open is tomorrow or later)."""
    clock = trading_client.get_clock()
    today = date.today()
    # next_open is the NEXT time the market opens; if it's today it means market hasn't opened yet
    next_open_date = clock.next_open.date()
    return next_open_date == today or clock.is_open


def get_calendar_trading_days(
    trading_client: TradingClient,
    start: str,
    end: str,
) -> list:
    """Return a list of date objects for trading days in [start, end]."""
    from alpaca.trading.requests import GetCalendarRequest
    calendar = trading_client.get_calendar(GetCalendarRequest(start=start, end=end))
    return [entry.date for entry in calendar]
