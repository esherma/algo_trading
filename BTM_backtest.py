import os
import math
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple, Literal

import numpy as np
import pandas as pd
import pytz
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dotenv import load_dotenv

# Alpaca SDK
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


# ============================
# Config and Types
# ============================

SessionType = Literal["paper", "live"]


@dataclass
class BacktestConfig:
    symbol: str = "SPY"
    timezone: str = "America/New_York"
    start: str = "2018-01-01"
    end: Optional[str] = None  # inclusive end date in YYYY-MM-DD, None = today
    lookback_days: int = 14
    volatility_multiplier: float = 1.0
    use_gap_adjustment: bool = True
    entry_minutes: Tuple[int, int] = (0, 30)  # HH:00 and HH:30
    start_trading_time: str = "10:00"  # first possible decision time
    close_time: str = "16:00"
    commission_per_share: float = 0.0035
    slippage_per_share: float = 0.001
    target_daily_volatility: float = 0.02  # 2%
    dynamic_sizing: bool = True
    leverage_cap: float = 4.0
    session: SessionType = "paper"


# ============================
# Utilities
# ============================

def load_env() -> None:
    load_dotenv()


def get_alpaca_client(session: SessionType = "paper") -> StockHistoricalDataClient:
    api_key = os.getenv("ALPACA_PAPER_API_KEY") if session == "paper" else os.getenv("ALPACA_LIVE_API_KEY")
    api_secret = (
        os.getenv("ALPACA_PAPER_API_SECRET")
        if session == "paper"
        else os.getenv("ALPACA_LIVE_API_SECRET")
    )
    if not api_key or not api_secret:
        raise RuntimeError(
            "Missing Alpaca API credentials. Ensure .env has ALPACA_PAPER_API_KEY and ALPACA_PAPER_API_KEY_SECRET."
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
    end: Optional[str],
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
    pivot_sigma = pivot.rolling(window=lookback_days, min_periods=1).mean()

    # map back to original minute index
    for idx, ts in enumerate(df.index):
        t = ts.strftime("%H:%M")
        sigma.append(pivot_sigma.loc[ts.normalize(), t])

    return pd.Series(sigma, index=df.index, name="sigma")


def compute_noise_bands(
    df: pd.DataFrame,
    lookback_days: int,
    vm: float,
    gap_adjustment: bool = True,
) -> pd.DataFrame:
    sigma = compute_sigma_series(df, lookback_days) * vm

    # Build reference price per minute: max(Open_t, Close_{t-1}) and min(Open_t, Close_{t-1}) as per paper
    daily = compute_daily_ohlcv(df)
    day_open = df.groupby(df.index.date)["open"].first()
    day_open = pd.Series(day_open.values, index=pd.to_datetime(day_open.index).tz_localize("America/New_York"))
    day_close_prev = daily["close"].shift(1)

    open_ref = day_open.reindex(df.index.normalize()).values
    close_prev_ref = day_close_prev.reindex(df.index.normalize()).values

    if gap_adjustment:
        upper_base = np.maximum(open_ref, close_prev_ref)
        lower_base = np.minimum(open_ref, close_prev_ref)
    else:
        upper_base = open_ref
        lower_base = open_ref 

    upper = upper_base * (1.0 + sigma.values)
    lower = lower_base * (1.0 - sigma.values)

    out = pd.DataFrame(index=df.index)
    out["UB"] = upper
    out["LB"] = lower
    out["sigma"] = sigma.values
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
    cfg: BacktestConfig,
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
# Trade Analysis
# ============================

def calculate_trade_metrics(
    df: pd.DataFrame,
    pos: pd.DataFrame,
    signed_shares_series: pd.Series,
    pnl_net_series: pd.Series,
    cfg: BacktestConfig,
) -> dict:
    """
    Calculate trade-level metrics including hit ratio, number of trades, etc.
    """
    # Identify trades by looking at position changes
    position_changes = pos["position"].diff().fillna(0)
    
    # Group trades by day and calculate trade-level PnL
    trades = []
    current_trade_pnl = 0.0
    current_trade_start = None
    current_position = 0
    
    for i, (ts, pos_val) in enumerate(pos["position"].items()):
        pos_change = position_changes.iloc[i]
        
        # If position changed from non-zero to zero, close the trade
        if pos_change != 0 and current_position != 0 and pos_val == 0:
            trades.append({
                'start_time': current_trade_start,
                'end_time': ts,
                'position': current_position,
                'pnl': current_trade_pnl
            })
            current_position = 0
            current_trade_pnl = 0.0
        
        # If position changed from zero to non-zero, start new trade
        elif pos_change != 0 and current_position == 0 and pos_val != 0:
            current_position = pos_val
            current_trade_start = ts
            current_trade_pnl = 0.0
        
        # Accumulate PnL for current trade
        if current_position != 0:
            current_trade_pnl += pnl_net_series.iloc[i]
    
    # Close final trade if exists
    if current_position != 0 and current_trade_start is not None:
        trades.append({
            'start_time': current_trade_start,
            'end_time': pos.index[-1],
            'position': current_position,
            'pnl': current_trade_pnl
        })
    
    if not trades:
        return {
            "TradeCount": 0,
            "HitRatio": 0.0,
            "AvgTradePnL": 0.0,
            "MaxLossTrade": 0.0,
            "MaxGainTrade": 0.0,
            "AvgTradesPerDay": 0.0
        }
    
    # Calculate trade metrics
    trade_pnls = [trade['pnl'] for trade in trades]
    profitable_trades = [pnl for pnl in trade_pnls if pnl > 0]
    
    hit_ratio = len(profitable_trades) / len(trade_pnls) if trade_pnls else 0.0
    avg_trade_pnl = np.mean(trade_pnls) if trade_pnls else 0.0
    max_loss_trade = min(trade_pnls) if trade_pnls else 0.0
    max_gain_trade = max(trade_pnls) if trade_pnls else 0.0
    
    # Calculate average trades per day
    trading_days = len(df.index.normalize().unique())
    avg_trades_per_day = len(trades) / trading_days if trading_days > 0 else 0.0
    
    return {
        "TradeCount": len(trades),
        "HitRatio": round(hit_ratio * 100, 1),
        "AvgTradePnL": round(avg_trade_pnl, 2),
        "MaxLossTrade": round(max_loss_trade, 2),
        "MaxGainTrade": round(max_gain_trade, 2),
        "AvgTradesPerDay": round(avg_trades_per_day, 1)
    }


# ============================
# Sizing and PnL
# ============================

def compute_dynamic_shares(
    df: pd.DataFrame,
    cfg: BacktestConfig,
    aum_series: pd.Series,
) -> pd.Series:
    # Daily returns for sizing volatility estimate
    daily = compute_daily_ohlcv(df)
    daily_ret = daily["close"].pct_change()
    # 14-day rolling stdev as per paper
    rolling_std = daily_ret.rolling(window=14, min_periods=1).std()
    rolling_mean = daily_ret.rolling(window=14, min_periods=1).mean()
    # Map to minute index
    sigma_spy = rolling_std.reindex(df.index.normalize(), method="ffill").fillna(method="ffill")
    # Shares_t = floor(AUM_{t-1} * min(4, sigma_target/sigma_spy) / Open_{t,9:30}) at day-open
    day_open = df.groupby(df.index.date)["open"].first()
    day_open = pd.Series(day_open.values, index=pd.to_datetime(day_open.index).tz_localize(cfg.timezone))
    open_map = day_open.reindex(df.index.normalize())

    leverage_factor = (cfg.target_daily_volatility / sigma_spy).clip(upper=cfg.leverage_cap)
    leverage_factor = leverage_factor.replace([np.inf, -np.inf], cfg.leverage_cap).fillna(1.0)

    aum_prev = aum_series.shift(1).fillna(method="ffill")
    notional = (aum_prev * leverage_factor)
    shares = np.floor(notional / open_map.values)
    shares = pd.Series(shares).fillna(0)
    shares.index = df.index
    return shares.astype(int)


def backtest_strategy(
    df: pd.DataFrame,
    cfg: BacktestConfig,
    variant: Literal["opp_band", "curr_band_vwap"],
) -> Tuple[pd.DataFrame, dict]:
    bands = compute_noise_bands(
        df=df,
        lookback_days=cfg.lookback_days,
        vm=cfg.volatility_multiplier,
        gap_adjustment=cfg.use_gap_adjustment,
    )
    use_vwap_stop = variant == "curr_band_vwap"
    pos = generate_positions(df, bands, cfg, use_vwap_stop=use_vwap_stop)

    # Prepare daily constants
    daily = compute_daily_ohlcv(df)
    daily_ret_hist = daily["close"].pct_change()
    sigma_spy_day = daily_ret_hist.rolling(window=14, min_periods=1).std().fillna(method="bfill").fillna(0.01)
    day_open = df.groupby(df.index.date)["open"].first()
    day_open = pd.Series(day_open.values, index=pd.to_datetime(day_open.index).tz_localize(cfg.timezone))

    # Iterative day-by-day backtest for proper daily sizing and costs
    aum_series = pd.Series(index=df.index, dtype=float)
    signed_shares_series = pd.Series(0, index=df.index, dtype=int)
    pnl_net_series = pd.Series(0.0, index=df.index, dtype=float)

    current_aum = 100_000.0
    prev_pos = 0

    for day in df.index.normalize().unique():
        mask_day = df.index.normalize() == day
        idxs = np.where(mask_day)[0]
        if len(idxs) == 0:
            continue
        day_open_price = day_open.loc[day]
        vol_day = sigma_spy_day.reindex([day], method=None).iloc[0] if day in sigma_spy_day.index else sigma_spy_day.iloc[-1]
        if cfg.dynamic_sizing:
            lev = min(cfg.leverage_cap, cfg.target_daily_volatility / max(1e-6, float(vol_day)))
        else:
            lev = 1.0
        shares_day = int(np.floor((current_aum * lev) / max(1e-6, float(day_open_price))))

        price_series = df.loc[mask_day, "close"]
        pos_series = pos.loc[mask_day, "position"]

        # Per-minute loop for the day
        prev_price = price_series.iloc[0]
        for j, (ts, px) in enumerate(price_series.items()):
            # PnL from holding previous minute's position
            minute_pnl = (px - prev_price) * (shares_day * prev_pos)

            # Costs if position changes at this minute
            delta_pos = pos_series.loc[ts] - prev_pos
            if delta_pos != 0:
                traded_shares = abs(delta_pos) * shares_day
                minute_pnl -= traded_shares * (cfg.commission_per_share + cfg.slippage_per_share)
            
            current_aum += minute_pnl
            aum_series.loc[ts] = current_aum
            pnl_net_series.loc[ts] = minute_pnl
            signed_shares_series.loc[ts] = shares_day * pos_series.loc[ts]

            prev_pos = int(pos_series.loc[ts])
            prev_price = px

        # ensure flat at end-of-day (already enforced by signals); prev_pos persists to next day (should be 0)

    aum = aum_series.ffill().fillna(current_aum)

    # Daily metrics
    daily_aum = aum.groupby(aum.index.date).last()
    daily_aum = pd.Series(daily_aum.values, index=pd.to_datetime(daily_aum.index).tz_localize(cfg.timezone))
    daily_ret = daily_aum.pct_change().dropna()

    vol_annual = daily_ret.std() * math.sqrt(252)
    irr = (daily_aum.iloc[-1] / daily_aum.iloc[0]) ** (252 / len(daily_ret)) - 1 if len(daily_ret) > 0 else 0.0
    sharpe = (daily_ret.mean() / daily_ret.std()) * math.sqrt(252) if daily_ret.std() > 0 else 0.0
    dd = (daily_aum / daily_aum.cummax() - 1.0).min()

    # Trade-level metrics
    trade_metrics = calculate_trade_metrics(df, pos, signed_shares_series, pnl_net_series, cfg)

    summary = {
        "TotalReturnPct": round((daily_aum.iloc[-1] / daily_aum.iloc[0] - 1) * 100, 2),
        "IRR": round(irr * 100, 2),
        "Vol": round(vol_annual * 100, 2),
        "Sharpe": round(sharpe, 2),
        "HitRatio": trade_metrics["HitRatio"],  # Now using trade-level hit ratio
        "MDDPct": round(dd * 100, 2),
        "FinalAUM": round(float(daily_aum.iloc[-1]), 2),
        "StartDate": as_date_str(daily_aum.index[0]),
        "EndDate": as_date_str(daily_aum.index[-1]),
        "TradeCount": trade_metrics["TradeCount"],
        "AvgTradePnL": trade_metrics["AvgTradePnL"],
        "MaxLossTrade": trade_metrics["MaxLossTrade"],
        "MaxGainTrade": trade_metrics["MaxGainTrade"],
        "AvgTradesPerDay": trade_metrics["AvgTradesPerDay"],
    }

    result = pd.DataFrame(index=df.index)
    result["price"] = df["close"]
    result["UB"] = bands["UB"]
    result["LB"] = bands["LB"]
    result["VWAP"] = pos["VWAP"]
    result["position"] = pos["position"]
    result["shares"] = signed_shares_series
    result["aum"] = aum
    result["pnl_net"] = pnl_net_series

    return result, summary


# ============================
# Plotting Function
# ============================

def plot_noise_bands_for_day(
    df: pd.DataFrame,
    target_date: str,
    volatility_multiplier: float,
    gap_adjustment: bool,
    lookback_days: int,
    cfg: BacktestConfig,
) -> None:
    """
    Plot noise bands for a specific day with historical price movement.
    
    Args:
        df: DataFrame with minute bar data
        target_date: Date string in YYYY-MM-DD format
        volatility_multiplier: Multiplier for noise band calculation
        gap_adjustment: Whether to use gap adjustment
        lookback_days: Number of days for lookback period
        cfg: Backtest configuration
    """
    # Convert target_date to datetime and filter data for that day
    target_dt = pd.to_datetime(target_date).tz_localize(cfg.timezone)
    day_mask = df.index.normalize() == target_dt.normalize()
    day_data = df[day_mask].copy()
    
    if len(day_data) == 0:
        print(f"No data found for date {target_date}")
        return
    
    # Compute noise bands for the entire dataset (needed for historical context)
    bands = compute_noise_bands(
        df=df,
        lookback_days=lookback_days,
        vm=volatility_multiplier,
        gap_adjustment=gap_adjustment,
    )
    
    # Filter bands for the target day
    day_bands = bands[day_mask].copy()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot price movement as heavy black line
    ax.plot(day_data.index, day_data['close'], color='black', linewidth=3, label='Price')
    
    # Shade the noise area with pale yellow
    ax.fill_between(day_data.index, day_bands['LB'], day_bands['UB'], 
                   alpha=0.3, color='yellow', label='Noise Area')
    
    # Plot upper and lower boundaries
    ax.plot(day_data.index, day_bands['UB'], color='red', linewidth=1, alpha=0.7, label='Upper Boundary')
    ax.plot(day_data.index, day_bands['LB'], color='red', linewidth=1, alpha=0.7, label='Lower Boundary')
    
    # Format the plot
    ax.set_title(f'Noise Bands for {target_date} - {cfg.symbol}\n'
                f'Volatility Multiplier: {volatility_multiplier}, '
                f'Gap Adjustment: {gap_adjustment}, '
                f'Lookback: {lookback_days} days', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Format x-axis to show time properly
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=day_data.index.tz))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    plt.xticks(rotation=45)
    
    # Add some padding to y-axis
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
    
    plt.tight_layout()
    plt.show()


def plot_strategy_performance(
    result_df: pd.DataFrame,
    summary: dict,
    variant: str,
    cfg: BacktestConfig,
    save_plot: bool = True
) -> None:
    """
    Plot comprehensive strategy performance over time.
    
    Args:
        result_df: DataFrame with backtest results
        summary: Dictionary with performance summary
        variant: Strategy variant name
        cfg: Backtest configuration
        save_plot: Whether to save the plot to file
    """
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f'Strategy Performance: {variant} - {cfg.symbol}\n'
                f'{summary["StartDate"]} to {summary["EndDate"]}', 
                fontsize=16, fontweight='bold')
    
    # 1. Equity Curve
    ax1 = axes[0, 0]
    daily_aum = result_df['aum'].groupby(result_df.index.date).last()
    daily_aum = pd.Series(daily_aum.values, index=pd.to_datetime(daily_aum.index).tz_localize(cfg.timezone))
    
    ax1.plot(daily_aum.index, daily_aum.values, linewidth=2, color='blue', label='Strategy AUM')
    ax1.set_title('Equity Curve', fontsize=12, fontweight='bold')
    ax1.set_ylabel('AUM ($)', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # 2. Drawdown
    ax2 = axes[0, 1]
    cumulative_max = daily_aum.cummax()
    drawdown = (daily_aum / cumulative_max - 1.0) * 100
    
    ax2.fill_between(daily_aum.index, drawdown.values, 0, alpha=0.3, color='red', label='Drawdown')
    ax2.plot(daily_aum.index, drawdown.values, linewidth=1, color='red')
    ax2.set_title('Drawdown (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Drawdown (%)', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # 3. Daily Returns Distribution
    ax3 = axes[1, 0]
    daily_returns = daily_aum.pct_change().dropna() * 100
    
    ax3.hist(daily_returns, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax3.axvline(daily_returns.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {daily_returns.mean():.2f}%')
    ax3.set_title('Daily Returns Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Daily Return (%)', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Rolling Sharpe Ratio
    ax4 = axes[1, 1]
    rolling_window = 252  # 1 year
    if len(daily_returns) >= rolling_window:
        rolling_sharpe = daily_returns.rolling(window=rolling_window).mean() / daily_returns.rolling(window=rolling_window).std() * np.sqrt(252)
        # Use the same index as rolling_sharpe for plotting
        rolling_sharpe = rolling_sharpe.dropna()
        ax4.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2, color='purple', label='Rolling Sharpe (1Y)')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Sharpe = 1')
        ax4.set_title('Rolling Sharpe Ratio (1 Year)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Sharpe Ratio', fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Format x-axis
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax4.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    else:
        ax4.text(0.5, 0.5, 'Insufficient data for rolling Sharpe', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Rolling Sharpe Ratio (1 Year)', fontsize=12, fontweight='bold')
    
    # 5. Position Distribution
    ax5 = axes[2, 0]
    position_counts = result_df['position'].value_counts()
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    ax5.pie(position_counts.values, labels=['Flat', 'Long', 'Short'], autopct='%1.1f%%', 
            colors=colors, startangle=90)
    ax5.set_title('Position Distribution', fontsize=12, fontweight='bold')
    
    # 6. Performance Metrics Table
    ax6 = axes[2, 1]
    ax6.axis('tight')
    ax6.axis('off')
    
    # Create metrics table
    metrics_data = [
        ['Total Return', f"{summary['TotalReturnPct']}%"],
        ['Annualized Return', f"{summary['IRR']}%"],
        ['Annualized Volatility', f"{summary['Vol']}%"],
        ['Sharpe Ratio', f"{summary['Sharpe']}"],
        ['Hit Ratio', f"{summary['HitRatio']}%"],
        ['Max Drawdown', f"{summary['MDDPct']}%"],
        ['Trade Count', f"{summary['TradeCount']}"],
        ['Avg Trade PnL', f"${summary['AvgTradePnL']}"],
        ['Avg Trades/Day', f"{summary['AvgTradesPerDay']}"],
        ['Final AUM', f"${summary['FinalAUM']:,.0f}"]
    ]
    
    table = ax6.table(cellText=metrics_data, colLabels=['Metric', 'Value'], 
                     cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(metrics_data) + 1):
        for j in range(2):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    ax6.set_title('Performance Summary', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_plot:
        out_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(out_dir, exist_ok=True)
        plot_filename = f"{cfg.symbol}_{variant}_performance.png"
        plt.savefig(os.path.join(out_dir, plot_filename), dpi=300, bbox_inches='tight')
        print(f"Performance plot saved to: {os.path.join(out_dir, plot_filename)}")
    
    plt.show()


def plot_comparison_performance(
    results_dict: dict,
    cfg: BacktestConfig,
    save_plot: bool = True
) -> None:
    """
    Plot comparison of different strategy variants.
    
    Args:
        results_dict: Dictionary with results for each variant
        cfg: Backtest configuration
        save_plot: Whether to save the plot to file
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Strategy Comparison - {cfg.symbol}\n'
                f'{cfg.start} to {cfg.end or "today"}', 
                fontsize=16, fontweight='bold')
    
    colors = ['blue', 'red', 'green', 'orange']
    
    # 1. Equity Curves Comparison
    ax1 = axes[0, 0]
    for i, (variant, (result_df, summary)) in enumerate(results_dict.items()):
        daily_aum = result_df['aum'].groupby(result_df.index.date).last()
        daily_aum = pd.Series(daily_aum.values, index=pd.to_datetime(daily_aum.index).tz_localize(cfg.timezone))
        
        # Normalize to starting value for comparison
        normalized_aum = daily_aum / daily_aum.iloc[0] * 100000
        
        ax1.plot(daily_aum.index, normalized_aum.values, linewidth=2, 
                color=colors[i], label=f'{variant} (Final: ${summary["FinalAUM"]:,.0f})')
    
    ax1.set_title('Equity Curves Comparison', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Normalized AUM ($)', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # 2. Drawdown Comparison
    ax2 = axes[0, 1]
    for i, (variant, (result_df, summary)) in enumerate(results_dict.items()):
        daily_aum = result_df['aum'].groupby(result_df.index.date).last()
        daily_aum = pd.Series(daily_aum.values, index=pd.to_datetime(daily_aum.index).tz_localize(cfg.timezone))
        
        cumulative_max = daily_aum.cummax()
        drawdown = (daily_aum / cumulative_max - 1.0) * 100
        
        ax2.plot(daily_aum.index, drawdown.values, linewidth=1.5, 
                color=colors[i], label=f'{variant} (Max DD: {summary["MDDPct"]}%)')
    
    ax2.set_title('Drawdown Comparison', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Drawdown (%)', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # 3. Performance Metrics Comparison
    ax3 = axes[1, 0]
    ax3.axis('tight')
    ax3.axis('off')
    
    # Prepare comparison table
    metrics = ['TotalReturnPct', 'IRR', 'Vol', 'Sharpe', 'HitRatio', 'MDDPct', 'TradeCount']
    metric_labels = ['Total Return (%)', 'IRR (%)', 'Vol (%)', 'Sharpe', 'Hit Ratio (%)', 'Max DD (%)', 'Trades']
    
    table_data = [metric_labels]
    for variant, (_, summary) in results_dict.items():
        row = [f"{summary[metric]}" for metric in metrics]
        table_data.append(row)
    
    table = ax3.table(cellText=table_data[1:], colLabels=table_data[0], 
                     rowLabels=list(results_dict.keys()),
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Style the table
    for i in range(len(table_data)):
        for j in range(len(metrics)):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    ax3.set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold')
    
    # 4. Monthly Returns Heatmap (for the first variant)
    ax4 = axes[1, 1]
    if results_dict:
        first_variant = list(results_dict.keys())[0]
        result_df, _ = results_dict[first_variant]
        
        daily_aum = result_df['aum'].groupby(result_df.index.date).last()
        daily_aum = pd.Series(daily_aum.values, index=pd.to_datetime(daily_aum.index).tz_localize(cfg.timezone))
        daily_returns = daily_aum.pct_change().dropna()
        
        # Create monthly returns
        monthly_returns = daily_returns.groupby([daily_returns.index.year, daily_returns.index.month]).apply(
            lambda x: (1 + x).prod() - 1
        ) * 100
        
        if len(monthly_returns) > 0:
            # Reshape for heatmap
            monthly_returns.index = pd.MultiIndex.from_tuples(monthly_returns.index, names=['Year', 'Month'])
            monthly_returns = monthly_returns.unstack()
            
            im = ax4.imshow(monthly_returns.values, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)
            ax4.set_title(f'Monthly Returns Heatmap - {first_variant}', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Month', fontsize=10)
            ax4.set_ylabel('Year', fontsize=10)
            
            # Set tick labels
            ax4.set_xticks(range(12))
            ax4.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            ax4.set_yticks(range(len(monthly_returns.index)))
            ax4.set_yticklabels(monthly_returns.index)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax4)
            cbar.set_label('Monthly Return (%)', fontsize=10)
        else:
            ax4.text(0.5, 0.5, 'Insufficient data for heatmap', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title(f'Monthly Returns Heatmap - {first_variant}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_plot:
        out_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(out_dir, exist_ok=True)
        plot_filename = f"{cfg.symbol}_strategy_comparison.png"
        plt.savefig(os.path.join(out_dir, plot_filename), dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {os.path.join(out_dir, plot_filename)}")
    
    plt.show()


# ============================
# Runner
# ============================

def run_backtests(cfg: Optional[BacktestConfig] = None) -> None:
    cfg = cfg or BacktestConfig()
    load_env()
    client = get_alpaca_client(cfg.session)

    print(f"Fetching minute bars for {cfg.symbol} from {cfg.start} to {cfg.end or 'today'}...")
    df = fetch_intraday_bars(client, cfg.symbol, cfg.start, cfg.end)

    # print("Running Opposite Band stop variant...")
    # res_opp, sum_opp = backtest_strategy(df, cfg, variant="opp_band")
    print("Running Current Band + VWAP stop variant...")
    res_vwap, sum_vwap = backtest_strategy(df, cfg, variant="curr_band_vwap")

    # print("\nSummary (Opposite Band stop):")
    # print(json.dumps(sum_opp, indent=2))
    print("\nSummary (Current Band + VWAP stop):")
    print(json.dumps(sum_vwap, indent=2))

    # Generate performance plots
    print("\nGenerating performance plots...")
    
    # Individual strategy performance plots
    # plot_strategy_performance(res_opp, sum_opp, "opp_band", cfg)
    plot_strategy_performance(res_vwap, sum_vwap, "curr_band_vwap", cfg)
    
    # Comparison plot
    results_dict = {
        # "Opposite Band": (res_opp, sum_opp),
        "Current Band + VWAP": (res_vwap, sum_vwap)
    }
    plot_comparison_performance(results_dict, cfg)

    # Save outputs
    out_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    # res_opp.to_csv(os.path.join(out_dir, f"{cfg.symbol}_opp_band.csv"))
    res_vwap.to_csv(os.path.join(out_dir, f"{cfg.symbol}_curr_band_vwap.csv"))
    with open(os.path.join(out_dir, f"{cfg.symbol}_summaries.json"), "w") as f:
        # json.dump({"opp_band": sum_opp, "curr_band_vwap": sum_vwap}, f, indent=2)
        json.dump({"curr_band_vwap": sum_vwap}, f, indent=2)


if __name__ == "__main__":
    cfg = BacktestConfig(symbol='SPY')
    run_backtests(cfg=cfg)
    cfg = BacktestConfig(symbol='HIBL')
    run_backtests(cfg=cfg)
    cfg = BacktestConfig(symbol='TQQQ')
    run_backtests(cfg=cfg)
    cfg = BacktestConfig(symbol='QQQ')
    run_backtests(cfg=cfg)


def demo_plot_noise_bands(cfg: Optional[BacktestConfig] = None) -> None:
    """
    Demonstration function showing how to plot noise bands for a specific day.
    """
    cfg = cfg or BacktestConfig(symbol='SPY', start='2022-04-01', end='2024-05-01', timezone='America/New_York')
    load_env()
    client = get_alpaca_client(cfg.session)

    print(f"Fetching minute bars for {cfg.symbol} from {cfg.start} to {cfg.end or 'today'}...")
    df = fetch_intraday_bars(client, cfg.symbol, cfg.start, cfg.end)

    # Example: Plot noise bands for a specific day
    target_date = "2022-04-29"  # Change this to any date in your data range
    volatility_multiplier = 1.0
    gap_adjustment = True
    lookback_days = 14

    print(f"Plotting noise bands for {target_date}...")
    plot_noise_bands_for_day(
        df=df,
        target_date=target_date,
        volatility_multiplier=volatility_multiplier,
        gap_adjustment=gap_adjustment,
        lookback_days=lookback_days,
        cfg=cfg,
    )



