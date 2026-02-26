#!/usr/bin/env python3
"""
Standalone script to calculate SPY volatility over the past 50 and 100 trading days.
Uses the same formula as the main algorithm:
σ_{SPY,t} = sqrt( (1/(K-1)) * Σ_{i=1}^{K} (ret_{t-i} - μ_{SPY,t})^2 )
where μ_{SPY,t} = (1/K) * Σ_{i=1}^{K} (ret_{t-i})
"""

import os
import math
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Alpaca SDK
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


def load_env():
    """Load environment variables."""
    load_dotenv()


def get_alpaca_client():
    """Get Alpaca client for data fetching."""
    api_key = os.getenv("ALPACA_PAPER_API_KEY")
    api_secret = os.getenv("ALPACA_PAPER_API_SECRET")
    
    if not api_key or not api_secret:
        raise RuntimeError(
            "Missing Alpaca API credentials. Ensure .env has ALPACA_PAPER_API_KEY and ALPACA_PAPER_API_SECRET."
        )
    return StockHistoricalDataClient(api_key, api_secret)


def fetch_daily_data(client, symbol, start_date, end_date):
    """Fetch daily OHLCV data for a symbol."""
    start_dt = pd.Timestamp(start_date, tz=pytz.UTC)
    end_dt = pd.Timestamp(end_date, tz=pytz.UTC)
    
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start_dt,
        end=end_dt,
        adjustment="raw",
    )
    
    bars = client.get_stock_bars(req)
    
    if symbol not in bars.data:
        raise RuntimeError(f"No bar data returned for {symbol}")
    
    bars_jsonl = []
    for bar in bars.data[symbol]:
        bars_jsonl.append({
            "date": bar.timestamp.date(),
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
        })
    
    df = pd.DataFrame.from_records(bars_jsonl)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    
    return df


def compute_daily_returns(df):
    """Compute daily returns from close prices."""
    return df["close"].pct_change().dropna()

    # np.log(df['close']) - np.log(df['close'].shift(1))


def calculate_volatility(returns, lookback_days):
    """
    Calculate volatility using the algorithm's formula:
    σ_{SPY,t} = sqrt( (1/(K-1)) * Σ_{i=1}^{K} (ret_{t-i} - μ_{SPY,t})^2 )
    where μ_{SPY,t} = (1/K) * Σ_{i=1}^{K} (ret_{t-i})
    """
    if len(returns) < lookback_days:
        print(f"Warning: Only {len(returns)} trading days of data available, need {lookback_days}")
        return None
    
    # Get the last K daily returns
    last_k_returns = returns.tail(lookback_days)
    
    # Calculate mean return for the K-day period
    mean_return = last_k_returns.mean()
    
    # Calculate standard deviation using the formula from the paper
    # σ = sqrt( (1/(K-1)) * Σ(return - mean_return)^2 )
    variance = ((last_k_returns - mean_return) ** 2).sum() / (len(last_k_returns) - 1)
    volatility = math.sqrt(variance)
    
    return float(volatility)


def main():
    """Main function to calculate and display volatility."""
    print("SPY Volatility Calculator")
    print("=" * 50)
    
    # Load environment and get client
    load_env()
    client = get_alpaca_client()
    
    # Fetch data for a longer period to ensure we have enough trading days
    # Use 150 calendar days to get approximately 100+ trading days
    today = datetime.now()
    start_date = (today - timedelta(days=370)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")
    
    print(f"Fetching SPY data from {start_date} to {end_date}...")
    df = fetch_daily_data(client, "SPY", start_date, end_date)
    
    print(f"Fetched {len(df)} trading days of data")
    print(f"Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    print()
    
    print("=" * 50)
    print(df.iloc[-1])
    print("=" * 50)

    # Calculate daily returns
    returns = compute_daily_returns(df)
    print(f"Calculated {len(returns)} daily returns")
    print()
    
    # Calculate volatility for different lookback periods
    print("Volatility Calculations:")
    print("-" * 30)
    
    # 50-day volatility
    vol_10 = calculate_volatility(returns, 10) * math.sqrt(9)
    if vol_10 is not None:
        print(f"10-day volatility: {vol_10:.4f} ({vol_10*100:.2f}%)")
    
    # 100-day volatility
    vol_90 = calculate_volatility(returns, 90) * math.sqrt(89)
    if vol_90 is not None:
        print(f"90-day volatility: {vol_90:.4f} ({vol_90*100:.2f}%)")

    # 250-day volatility
    vol_250 = calculate_volatility(returns, 250) * math.sqrt(249)
    if vol_250 is not None:
        print(f"250-day volatility: {vol_250:.4f} ({vol_250*100:.2f}%)")
    
    # Also show 14-day volatility for comparison (from the algorithm)
    vol_14 = calculate_volatility(returns, 14) * math.sqrt(13)
    if vol_14 is not None:
        print(f"14-day volatility: {vol_14:.4f} ({vol_14*100:.2f}%)")
    
    print()
    
    # Show the actual date ranges for each lookback period
    print("Date Ranges for Each Lookback Period:")
    print("-" * 40)
    if len(returns) >= 10:
        start_date_10 = returns.tail(10).index[0]
        end_date_10 = returns.tail(10).index[-1]
        print(f"10-day period: {start_date_10.strftime('%Y-%m-%d')} to {end_date_10.strftime('%Y-%m-%d')}")
    
    if len(returns) >= 90:
        start_date_90 = returns.tail(90).index[0]
        end_date_90 = returns.tail(90).index[-1]
        print(f"90-day period: {start_date_90.strftime('%Y-%m-%d')} to {end_date_90.strftime('%Y-%m-%d')}")
    
    if len(returns) >= 250:
        start_date_250 = returns.tail(250).index[0]
        end_date_250 = returns.tail(250).index[-1]
        print(f"250-day period: {start_date_250.strftime('%Y-%m-%d')} to {end_date_250.strftime('%Y-%m-%d')}")
    
    if len(returns) >= 14:
        start_date_14 = returns.tail(14).index[0]
        end_date_14 = returns.tail(14).index[-1]
        print(f"14-day period: {start_date_14.strftime('%Y-%m-%d')} to {end_date_14.strftime('%Y-%m-%d')}")
    
    print()
    
    # Show some recent daily returns for context
    print("Recent Daily Returns (last 10 days):")
    print("-" * 40)
    recent_returns = returns.tail(10)
    for date, ret in recent_returns.items():
        print(f"{date.strftime('%Y-%m-%d')}: {ret:.4f} ({ret*100:.2f}%)")
    
    print()
    print("Note: Volatility is calculated using the algorithm's formula:")
    print("σ = sqrt( (1/(K-1)) * Σ(return - mean_return)² )")
    print("where K is the lookback period and mean_return is the average over K days.")


if __name__ == "__main__":
    main()