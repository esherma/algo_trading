import os
import math
import numpy as np
import pandas as pd
import pytz
from dataclasses import dataclass
from typing import List, Optional, Tuple, Literal
from dotenv import load_dotenv
from threading import Lock, Thread
from alpaca.data.live import StockDataStream
import asyncio

# Alpaca SDK
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


# ============================
# Config and Types
# ============================

SessionType = Literal["paper", "live"]


@dataclass
class TradingConfig:
    symbol: str = "SPY"
    timezone: str = "America/New_York"
    lookback_days: int = 14
    volatility_multiplier: float = 1
    use_gap_adjustment: bool = True
    entry_minutes: Tuple[int, int] = (0, 30)  # HH:00 and HH:30
    start_trading_time: str = "09:59"  # first possible decision time (15 seconds before 10:00)
    close_time: str = "16:00"
    target_daily_volatility: float = 0.02  # 2%
    leverage_cap: float = 3.0
    session: SessionType = "paper"


# ============================
# Utilities
# ============================


def get_alpaca_client(session: SessionType = "paper") -> StockHistoricalDataClient:
    api_key = os.getenv("ALPACA_PAPER_API_KEY") if session == "paper" else os.getenv("ALPACA_LIVE_API_KEY")
    api_secret = (
        os.getenv("ALPACA_PAPER_API_SECRET")
        if session == "paper"
        else os.getenv("ALPACA_LIVE_API_SECRET")
    )
    if not api_key or not api_secret:
        raise RuntimeError(
            "Missing Alpaca API credentials. Ensure .env has ALPACA_PAPER_API_KEY and ALPACA_PAPER_API_SECRET."
        )
    return StockHistoricalDataClient(api_key, api_secret)


def ensure_ny_tz(dtidx: pd.DatetimeIndex, timezone: str) -> pd.DatetimeIndex:
    tz = pytz.timezone(timezone)
    if dtidx.tz is None:
        return dtidx.tz_localize(tz)
    return dtidx.tz_convert(tz)


def as_date_str(dt: pd.Timestamp) -> str:
    return dt.strftime("%Y-%m-%d")


def minute_grid_filter(dtidx: pd.DatetimeIndex, minutes: Tuple[int, int], start_time: str) -> np.ndarray:
    # Select only HH:00 and HH:30 minutes, and not before start_time
    hhmm_start = int(start_time.replace(":", ""))
    flags = []
    for ts in dtidx:
        hhmm = ts.hour * 100 + ts.minute
        flags.append((ts.minute in minutes) and (hhmm >= hhmm_start))
    return np.array(flags, dtype=bool)


# ============================
# Data Fetcher
# ============================

def fetch_intraday_bars(
    client: StockHistoricalDataClient,
    symbol: str,
    start: str,
    end: Optional[str] = None,
) -> pd.DataFrame:
    # Convert start/end dates to UTC, ensuring we get full trading days
    # Market opens at 09:30 EST/EDT, which is 14:30 UTC (EST) or 13:30 UTC (EDT)
    # We'll fetch from 00:00 UTC to ensure we get the full day
    start_dt = pd.Timestamp(start, tz=pytz.UTC)
    end_arg = end if end is not None else pd.Timestamp.today(tz=pytz.UTC).strftime("%Y-%m-%d")
    end_dt = pd.Timestamp(end_arg, tz=pytz.UTC)
    
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start_dt,
        end=end_dt,
        adjustment="raw",
    )
    bars = client.get_stock_bars(req)

    if symbol not in bars.data:
        raise RuntimeError(f"No bar data returned for {symbol}")

    bars_jsonl = []

    for bar in bars.data[symbol]:
        bars_jsonl.append(
            {
                "ts": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "trade_count": bar.trade_count,
                "vwap": bar.vwap,
            }
        )

    df = pd.DataFrame.from_records(bars_jsonl)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts").sort_index()
    df.index = ensure_ny_tz(df.index, "America/New_York")

    # Filter to regular market hours 09:30 - 16:00 ET
    # Use time() method to get time components for more reliable filtering
    df = df[(df.index.time >= pd.Timestamp("09:30").time()) & 
            (df.index.time <= pd.Timestamp("16:00").time())]

    return df[["open", "high", "low", "close", "volume", "vwap"]].copy()


# ============================
# Feature Engineering: Noise Bands and VWAP
# ============================

def compute_daily_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(df.index.date)
    daily = pd.DataFrame({
        "open": g["open"].first(),
        "close": g["close"].last(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "volume": g["volume"].sum(),
    })
    daily.index = pd.to_datetime(daily.index).tz_localize("America/New_York")
    return daily


def compute_sigma_series(df: pd.DataFrame, lookback_days: int) -> pd.Series:
    # sigma_t,9:30-HH:MM = average of abs(move from open) over previous N days at same HH:MM
    # Create columns for HH:MM key
    hhmm = df.index.strftime("%H:%M")
    open_by_day = df.groupby(df.index.date)["open"].first()
    open_map = pd.Series(open_by_day.values, index=pd.to_datetime(open_by_day.index).tz_localize("America/New_York")).reindex(
        df.index.normalize()
    )
    move = (df["close"] / open_map.values - 1.0).abs()

    # For each HH:MM group, compute rolling mean of last N days for that time key
    sigma = []
    # Build a pivot: rows=day, cols=HH:MM, value=abs move
    day_index = df.index.normalize().unique()
    times = sorted(pd.unique(hhmm))
    pivot = pd.DataFrame(index=day_index, columns=times, dtype=float)
    for t in times:
        mask = hhmm == t
        # aggregate by day: use last value at HH:MM for the day (minute close)
        s = move[mask]
        s_by_day = s.groupby(s.index.normalize()).last()
        pivot.loc[s_by_day.index, t] = s_by_day.values

    # rolling mean over past N days per time column
    pivot_sigma = pivot.rolling(window=lookback_days).mean()

    # map back to original minute index
    for idx, ts in enumerate(df.index):
        t = ts.strftime("%H:%M")
        sigma.append(pivot_sigma.loc[ts.normalize(), t])

    return pd.Series(sigma, index=df.index, name="sigma")    


def compute_noise_bands(
    today_open: float,
    yesterday_close: float,
    move_from_open_on_historical_data: pd.Series,
    vm: float = 1.0,
    gap_adjustment: bool = True,
) -> pd.DataFrame:
    if gap_adjustment:
        upper_base = np.maximum(today_open, yesterday_close)
        lower_base = np.minimum(today_open, yesterday_close)
    else:
        upper_base = today_open
        lower_base = today_open
    upper = upper_base * (1.0 + move_from_open_on_historical_data.values * vm)
    lower = lower_base * (1.0 - move_from_open_on_historical_data.values * vm)
    out = pd.DataFrame(index=move_from_open_on_historical_data.index)
    out["UB"] = upper
    out["LB"] = lower
    out["sigma"] = move_from_open_on_historical_data.values
    return out


def compute_intraday_vwap(df: pd.DataFrame) -> pd.Series:
    # Intraday-only VWAP resets daily and uses regular session minutes only
    day_keys = df.index.normalize()
    vwap_vals = np.empty(len(df))
    cum_pv = 0.0
    cum_vol = 0.0
    last_day = None
    for i, (ts, row) in enumerate(df.iterrows()):
        day = ts.normalize()
        if last_day is None or day != last_day:
            cum_pv = 0.0
            cum_vol = 0.0
            last_day = day
        price = row["close"]
        vol = float(row["volume"]) if not math.isnan(row["volume"]) else 0.0
        cum_pv += price * vol
        cum_vol += vol
        vwap_vals[i] = cum_pv / cum_vol if cum_vol > 0 else price
    return pd.Series(vwap_vals, index=df.index, name="VWAP")


# ============================
# Strategy Logic
# ============================

def generate_positions(
    df: pd.DataFrame,
    bands: pd.DataFrame,
    cfg: TradingConfig,
    use_vwap_stop: bool,
) -> pd.DataFrame:
    tz = cfg.timezone
    # Decision times mask: HH:00 and HH:30 and after start_trading_time
    decision_mask = minute_grid_filter(df.index, cfg.entry_minutes, cfg.start_trading_time)

    # Price relative to bands
    price = df["close"]
    ub = bands["UB"]
    lb = bands["LB"]

    vwap = compute_intraday_vwap(df)

    # Signals only at decision times
    signal = pd.Series(0, index=df.index, dtype=int)
    signal[(decision_mask) & (price > ub)] = 1
    signal[(decision_mask) & (price < lb)] = -1

    # Position management with trailing stops evaluated only at decision times
    position = pd.Series(0, index=df.index, dtype=int)
    in_pos = 0

    for i, ts in enumerate(df.index):
        if not decision_mask[i]:
            position.iat[i] = in_pos
            continue

        # Handle stops first
        if in_pos == 1:
            stop_level = lb.iat[i] if not use_vwap_stop else max(ub.iat[i], vwap.iat[i])
            if price.iat[i] < stop_level:
                in_pos = 0
        elif in_pos == -1:
            stop_level = ub.iat[i] if not use_vwap_stop else min(lb.iat[i], vwap.iat[i])
            if price.iat[i] > stop_level:
                in_pos = 0

        # Reverse on opposite boundary cross
        sig = signal.iat[i]
        if sig == 1:
            in_pos = 1
        elif sig == -1:
            in_pos = -1

        # Flat after market close
        if ts.strftime("%H:%M") >= cfg.close_time:
            in_pos = 0

        position.iat[i] = in_pos

    out = pd.DataFrame(index=df.index)
    out["signal"] = signal
    out["position"] = position
    out["VWAP"] = vwap
    return out


# ============================
# Leverage Mapping for Live Trading
# ============================

def get_leverage_ticker(calculated_leverage: float) -> str:
    """
    Map calculated leverage to the appropriate ticker for live trading.
    All positions are long - we use inverse ETFs instead of short selling.
    """
    # Round leverage to nearest integer in range [-3, -2, -1, 1, 2, 3]
    leverage = round(calculated_leverage)
    leverage = max(-3, min(3, leverage))  # Clamp to range
    
    leverage_map = {
        -3: "SPXS",  # 3x inverse leveraged SPY ETF
        -2: "SDS",   # 2x inverse leveraged SPY ETF
        -1: "SPDN",  # 1x inverse SPY ETF
        1: "SPY",    # normal
        2: "SPUU",   # 2x leveraged SPY ETF
        3: "SPXL",   # 3x leveraged SPY ETF
    }
    
    return leverage_map.get(leverage, "SPY")


def calculate_leverage_and_ticker(
    target_volatility: float,
    current_volatility: float,
    leverage_cap: float
) -> Tuple[float, str]:
    """
    Calculate leverage and return the appropriate ticker for live trading.
    """
    if current_volatility <= 0:
        return 1.0, "SPY"
    
    calculated_leverage = target_volatility / current_volatility
    calculated_leverage = min(leverage_cap, calculated_leverage)
    calculated_leverage = max(-leverage_cap, calculated_leverage)
    
    ticker = get_leverage_ticker(calculated_leverage)
    return calculated_leverage, ticker

class SharedQuoteData:
    def __init__(self, api_key: str = None, secret_key: str = None, symbols: List[str] = None, url_override: str = None):
        self.latest_data = {s: None for s in symbols}
        self.lock = Lock()
        self.wss_client = StockDataStream(api_key, secret_key, url_override=url_override)
        self.symbols = symbols

    async def quote_data_handler(self, data):
        self.update(data)

    def update(self, data):
        with self.lock:
            self.latest_data[data["symbol"]] = data

    def get(self, symbol: str):
        with self.lock:
            return self.latest_data[symbol]

    def start(self):
        """Starts the WebSocket stream in a new thread with its own asyncio loop."""
        def run_stream():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.wss_client.subscribe_bars(self.quote_data_handler, *self.symbols)
            self.wss_client.run()

        Thread(target=run_stream, daemon=True).start()