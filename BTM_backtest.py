import os
import math
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple, Literal

import numpy as np
import pandas as pd
import pytz
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
    start: str = "2020-01-01"
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
    end_arg = end if end is not None else pd.Timestamp.today(tz=pytz.UTC).strftime("%Y-%m-%d")
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=pd.Timestamp(start, tz=pytz.UTC),
        end=pd.Timestamp(end_arg, tz=pytz.UTC),
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
    df = df[(df.index.strftime("%H:%M") >= "09:30") & (df.index.strftime("%H:%M") <= "16:00")]

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
    hit_ratio = (daily_ret > 0).mean() if len(daily_ret) > 0 else 0.0
    dd = (daily_aum / daily_aum.cummax() - 1.0).min()

    summary = {
        "TotalReturnPct": round((daily_aum.iloc[-1] / daily_aum.iloc[0] - 1) * 100, 2),
        "IRR": round(irr * 100, 2),
        "Vol": round(vol_annual * 100, 2),
        "Sharpe": round(sharpe, 2),
        "HitRatio": round(hit_ratio * 100, 1),
        "MDDPct": round(dd * 100, 2),
        "FinalAUM": round(float(daily_aum.iloc[-1]), 2),
        "StartDate": as_date_str(daily_aum.index[0]),
        "EndDate": as_date_str(daily_aum.index[-1]),
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
# Runner
# ============================

def run_backtests(cfg: Optional[BacktestConfig] = None) -> None:
    cfg = cfg or BacktestConfig(symbol='TQQQ',start='2025-01-01')
    load_env()
    client = get_alpaca_client(cfg.session)

    print(f"Fetching minute bars for {cfg.symbol} from {cfg.start} to {cfg.end or 'today'}...")
    df = fetch_intraday_bars(client, cfg.symbol, cfg.start, cfg.end)

    print("Running Opposite Band stop variant...")
    res_opp, sum_opp = backtest_strategy(df, cfg, variant="opp_band")
    print("Running Current Band + VWAP stop variant...")
    res_vwap, sum_vwap = backtest_strategy(df, cfg, variant="curr_band_vwap")

    print("\nSummary (Opposite Band stop):")
    print(json.dumps(sum_opp, indent=2))
    print("\nSummary (Current Band + VWAP stop):")
    print(json.dumps(sum_vwap, indent=2))

    # Save outputs
    out_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    res_opp.to_csv(os.path.join(out_dir, f"{cfg.symbol}_opp_band.csv"))
    res_vwap.to_csv(os.path.join(out_dir, f"{cfg.symbol}_curr_band_vwap.csv"))
    with open(os.path.join(out_dir, f"{cfg.symbol}_summaries.json"), "w") as f:
        json.dump({"opp_band": sum_opp, "curr_band_vwap": sum_vwap}, f, indent=2)


if __name__ == "__main__":
    run_backtests()
