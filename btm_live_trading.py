import os
import asyncio
import time
import math
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import pytz
import pandas as pd

# Alpaca SDK
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.live import StockDataStream
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame

# Local imports
from btm_utils import (
    TradingConfig, load_env, get_alpaca_client, fetch_intraday_bars,
    compute_noise_bands, generate_positions, compute_daily_ohlcv,
    calculate_leverage_and_ticker
)


class BTMLiveTrader:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.tz = pytz.timezone(config.timezone)
        
        # Initialize clients
        load_env()
        self.historical_client = get_alpaca_client(config.session)
        self.trading_client = TradingClient(
            os.getenv("ALPACA_PAPER_API_KEY") if config.session == "paper" else os.getenv("ALPACA_LIVE_API_KEY"),
            os.getenv("ALPACA_PAPER_API_SECRET") if config.session == "paper" else os.getenv("ALPACA_LIVE_API_SECRET"),
            paper=config.session == "paper"
        )
        
        # Trading state
        self.current_position = 0  # 0 = flat, 1 = long, -1 = short
        self.current_ticker = "SPY"
        self.current_shares = 0
        self.last_decision_time = None
        self.account_value = 0.0
        
        # Data storage
        self.historical_data = None
        self.noise_bands = None
        self.positions_df = None
        
        # Daily pre-calculated values (fixed for the day)
        self.daily_leverage = 1.0
        self.daily_volatility = 0.02
        self.in_play_tickers = {"long_ticker": "SPY", "short_ticker": "SPDN"}
        
        # Market data stream
        self.data_stream = StockDataStream(
            os.getenv("ALPACA_PAPER_API_KEY") if config.session == "paper" else os.getenv("ALPACA_LIVE_API_KEY"),
            os.getenv("ALPACA_PAPER_API_SECRET") if config.session == "paper" else os.getenv("ALPACA_LIVE_API_SECRET")
        )
        
        print(f"Initialized BTM Live Trader for {config.session} trading")
        print(f"Strategy: Current Band + VWAP")
        print(f"Base Symbol: {config.symbol}")
        print(f"Leverage Cap: {config.leverage_cap}")
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get current account information."""
        account = self.trading_client.get_account()
        return {
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "equity": float(account.equity),
            "portfolio_value": float(account.portfolio_value),
            "pattern_day_trader": account.pattern_day_trader,
            "trading_blocked": account.trading_blocked,
            "account_blocked": account.account_blocked
        }
    
    def fetch_historical_data(self, days_back: int = 30) -> None:
        """Fetch historical data for noise band calculation."""
        end_date = datetime.now(self.tz).strftime("%Y-%m-%d")
        start_date = (datetime.now(self.tz) - timedelta(days=days_back)).strftime("%Y-%m-%d")
        
        print(f"Fetching historical data from {start_date} to {end_date}...")
        
        self.historical_data = fetch_intraday_bars(
            self.historical_client,
            self.config.symbol,
            start_date,
            end_date
        )
        
        print(f"Fetched {len(self.historical_data)} bars of historical data")
    
    def calculate_noise_bands(self) -> None:
        """Calculate noise bands from historical data."""
        if self.historical_data is None:
            raise RuntimeError("Historical data not available. Call fetch_historical_data() first.")
        
        print("Calculating noise bands...")
        
        self.noise_bands = compute_noise_bands(
            df=self.historical_data,
            lookback_days=self.config.lookback_days,
            vm=self.config.volatility_multiplier,
            gap_adjustment=self.config.use_gap_adjustment
        )
        
        print("Noise bands calculated successfully")
    
    def calculate_daily_leverage_and_tickers(self) -> None:
        """
        Calculate daily leverage and determine which two tickers are in play for the day.
        This is fixed for the entire trading day based on pre-market volatility calculation.
        """
        # Calculate daily volatility (fixed for the day)
        self.daily_volatility = self.calculate_daily_volatility()
        
        # Calculate leverage
        if self.daily_volatility <= 0:
            self.daily_leverage = 1.0
        else:
            calculated_leverage = self.config.target_daily_volatility / self.daily_volatility
            self.daily_leverage = min(self.config.leverage_cap, calculated_leverage)
            self.daily_leverage = max(-self.config.leverage_cap, self.daily_leverage)
        
        # Determine which two tickers are in play for the day based on leverage
        if self.daily_leverage >= 2.5:
            self.in_play_tickers = {"long_ticker": "SPXL", "short_ticker": "SPXS"}
        elif self.daily_leverage >= 1.5:
            self.in_play_tickers = {"long_ticker": "SPUU", "short_ticker": "SDS"}
        else:
            self.in_play_tickers = {"long_ticker": "SPY", "short_ticker": "SPDN"}
        
        print(f"Daily pre-calculated values:")
        print(f"  Volatility: {self.daily_volatility:.4f} ({self.daily_volatility*100:.2f}%)")
        print(f"  Leverage: {self.daily_leverage:.2f}")
        print(f"  Long Ticker: {self.in_play_tickers['long_ticker']}")
        print(f"  Short Ticker: {self.in_play_tickers['short_ticker']}")
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quote = self.historical_client.get_stock_latest_quote(request)
            return float(quote[symbol].ask_price)
        except Exception as e:
            print(f"Error getting price for {symbol}: {e}")
            return None
    
    def calculate_daily_volatility(self) -> float:
        """
        Calculate daily volatility based on the preceding 14 days of daily returns.
        This matches the formula: σ_{SPY,t} = sqrt( (1/13) * Σ_{i=1}^{14} (ret_{t-i} - μ_{SPY,t})^2 )
        where μ_{SPY,t} = (1/14) * Σ_{i=1}^{14} (ret_{t-i})
        """
        if self.historical_data is None:
            return 0.02  # Default 2% volatility
        
        daily = compute_daily_ohlcv(self.historical_data)
        daily_returns = daily["close"].pct_change().dropna()
        
        # Need at least 14 days of data
        if len(daily_returns) < 14:
            if len(daily_returns) > 0:
                return float(daily_returns.std())
            else:
                return 0.02  # Default 2% volatility
        
        # Get the last 14 daily returns
        last_14_returns = daily_returns.tail(14)
        
        # Calculate mean return for the 14-day period
        mean_return = last_14_returns.mean()
        
        # Calculate standard deviation using the formula from the paper
        # σ = sqrt( (1/13) * Σ(return - mean_return)^2 )
        variance = ((last_14_returns - mean_return) ** 2).sum() / 13
        volatility = math.sqrt(variance)
        
        return float(volatility)
    
    def should_trade_now(self) -> bool:
        """Check if we should make a trading decision now."""
        now = datetime.now(self.tz)
        
        # Check if it's a weekday
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check if market is open (9:30 AM - 4:00 PM ET)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        if now < market_open or now >= market_close:
            return False
        
        # Check if it's 15 seconds before the half hour
        current_minute = now.minute
        current_second = now.second
        
        # Decision times: 09:59:45, 10:29:45, 10:59:45, etc.
        # Last decision time: 15:29:45
        if current_minute in [59, 29] and current_second == 45:
            # Check if it's after start trading time and before last decision time
            start_time = datetime.strptime(self.config.start_trading_time, "%H:%M").time()
            if now.time() >= start_time and now.time() < datetime.strptime("15:30", "%H:%M").time():
                return True
        
        return False
    
    def should_close_position(self) -> bool:
        """Check if we should close position at end of day."""
        now = datetime.now(self.tz)
        
        # Close position at 15:49:30 if still open
        close_time = now.replace(hour=15, minute=49, second=30, microsecond=0)
        
        return now >= close_time and self.current_position != 0
    
    def get_current_market_data(self) -> Dict[str, Any]:
        """Get current market data for SPY."""
        try:
            # Get latest bar data for SPY
            end_time = datetime.now(self.tz)
            start_time = end_time - timedelta(minutes=5)  # Get last 5 minutes
            
            request = StockBarsRequest(
                symbol_or_symbols=self.config.symbol,
                timeframe=TimeFrame.Minute,
                start=start_time,
                end=end_time
            )
            
            bars = self.historical_client.get_stock_bars(request)
            
            if self.config.symbol in bars.data and len(bars.data[self.config.symbol]) > 0:
                latest_bar = bars.data[self.config.symbol][-1]
                return {
                    "timestamp": latest_bar.timestamp,
                    "open": float(latest_bar.open),
                    "high": float(latest_bar.high),
                    "low": float(latest_bar.low),
                    "close": float(latest_bar.close),
                    "volume": int(latest_bar.volume),
                    "vwap": float(latest_bar.vwap) if latest_bar.vwap else float(latest_bar.close)
                }
        except Exception as e:
            print(f"Error getting current market data: {e}")
        
        return None
    
    def make_trading_decision(self) -> Optional[Dict[str, Any]]:
        """Make a trading decision based on current market conditions."""
        if self.noise_bands is None:
            print("Noise bands not available. Cannot make trading decision.")
            return None
        
        # Get current market data
        market_data = self.get_current_market_data()
        if market_data is None:
            print("Could not get current market data.")
            return None
        
        current_price = market_data["close"]
        current_time = market_data["timestamp"]
        
        # Find the corresponding noise band values
        # We need to find the closest time in our historical data
        time_str = current_time.strftime("%H:%M")
        today = current_time.date()
        
        # Get today's data from historical data
        today_mask = self.historical_data.index.date == today
        today_data = self.historical_data[today_mask]
        
        if len(today_data) == 0:
            print("No data for today available.")
            return None
        
        # Find the closest time to current time
        time_diff = abs(today_data.index - current_time)
        closest_idx = time_diff.argmin()
        closest_time = today_data.index[closest_idx]
        
        # Get noise band values for this time
        if closest_time in self.noise_bands.index:
            ub = self.noise_bands.loc[closest_time, "UB"]
            lb = self.noise_bands.loc[closest_time, "LB"]
        else:
            print(f"No noise band data for time {closest_time}")
            return None
        
        # Calculate VWAP for today
        vwap = self.calculate_intraday_vwap(today_data)
        current_vwap = vwap.iloc[-1] if len(vwap) > 0 else current_price
        
        # Simplified trading logic
        new_position = self.current_position
        ticker = "SPY"  # Default
        shares = 0
        
        # Check if we have an open position
        if self.current_position != 0:
            # Check if position should be closed
            should_close = False
            
            if self.current_position == 1:  # Long position (SPY/SPUU/SPXL)
                # Close if SPY price < upper bound
                if current_price < ub:
                    should_close = True
            elif self.current_position == -1:  # Short position (SPDN/SDS/SPXS)
                # Close if SPY price > lower bound
                if current_price > lb:
                    should_close = True
            
            if should_close:
                new_position = 0
                ticker = "SPY"
        
        # If no position open OR position is about to be closed, check if we should open
        if new_position == 0:
            # Check SPY price vs bounds
            if current_price < lb:
                # Open short position
                new_position = -1
                ticker = self.in_play_tickers["short_ticker"]
            elif current_price > ub:
                # Open long position
                new_position = 1
                ticker = self.in_play_tickers["long_ticker"]
        
        # Calculate shares if we have a position
        if new_position != 0:
            account_info = self.get_account_info()
            aum = account_info["portfolio_value"]
            ticker_price = self.get_current_price(ticker)
            if ticker_price is None:
                print(f"Could not get price for {ticker}")
                return None
            if self.daily_leverage < 1:
                shares = int(
                    math.floor(
                        (aum * self.daily_leverage) / 
                        ticker_price
                    )
                )
            else:
                shares = int(math.floor(aum / ticker_price))
        
        decision = {
            "timestamp": current_time,
            "current_price": current_price,
            "upper_band": ub,
            "lower_band": lb,
            "vwap": current_vwap,
            "current_position": self.current_position,
            "new_position": new_position,
            "daily_volatility": self.daily_volatility,
            "daily_leverage": self.daily_leverage,
            "ticker": ticker,
            "shares": shares,
            "aum": self.get_account_info()["portfolio_value"]
        }
        
        return decision
    
    def calculate_intraday_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate intraday VWAP for a given dataframe."""
        vwap_vals = []
        cum_pv = 0.0
        cum_vol = 0.0
        
        for _, row in df.iterrows():
            price = row["close"]
            vol = float(row["volume"]) if not math.isnan(row["volume"]) else 0.0
            cum_pv += price * vol
            cum_vol += vol
            vwap_vals.append(cum_pv / cum_vol if cum_vol > 0 else price)
        
        return pd.Series(vwap_vals, index=df.index)
    
    def execute_trade(self, decision: Dict[str, Any]) -> bool:
        """Execute a trade based on the decision."""
        try:
            # Close existing position if position changes
            if self.current_position != 0 and self.current_position != decision["new_position"]:
                print(f"Closing position: {self.current_shares} shares of {self.current_ticker}")
                
                close_order = MarketOrderRequest(
                    symbol=self.current_ticker,
                    qty=self.current_shares,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY
                )
                
                close_result = self.trading_client.submit_order(close_order)
                print(f"Close order submitted: {close_result.id}")
                
                # Reset position
                self.current_position = 0
                self.current_ticker = "SPY"
                self.current_shares = 0
            
            # Open new position if needed
            if decision["new_position"] != 0 and decision["shares"] > 0:
                print(f"Opening position: {decision['shares']} shares of {decision['ticker']}")
                
                open_order = MarketOrderRequest(
                    symbol=decision["ticker"],
                    qty=decision["shares"],
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )
                
                open_result = self.trading_client.submit_order(open_order)
                print(f"Open order submitted: {open_result.id}")
                
                # Update position
                self.current_position = decision["new_position"]
                self.current_ticker = decision["ticker"]
                self.current_shares = decision["shares"]
            
            return True
            
        except Exception as e:
            print(f"Error executing trade: {e}")
            return False
    
    def close_all_positions(self) -> bool:
        """Close all open positions."""
        try:
            if self.current_position != 0:
                print(f"Closing all positions: {self.current_shares} shares of {self.current_ticker}")
                
                close_order = MarketOrderRequest(
                    symbol=self.current_ticker,
                    qty=self.current_shares,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY
                )
                
                close_result = self.trading_client.submit_order(close_order)
                print(f"Close order submitted: {close_result.id}")
                
                # Reset position
                self.current_position = 0
                self.current_ticker = "SPY"
                self.current_shares = 0
            
            return True
            
        except Exception as e:
            print(f"Error closing positions: {e}")
            return False
    
    async def run_trading_loop(self) -> None:
        """Main trading loop."""
        print("Starting trading loop...")
        
        # Initial setup
        self.fetch_historical_data()
        self.calculate_noise_bands()
        
        # Calculate daily leverage and tickers (fixed for the day)
        self.calculate_daily_leverage_and_tickers()
        
        # Get account info
        account_info = self.get_account_info()
        print(f"Account Portfolio Value: ${account_info['portfolio_value']:,.2f}")
        print(f"Buying Power: ${account_info['buying_power']:,.2f}")
        
        # Main loop
        while True:
            try:
                now = datetime.now(self.tz)
                
                # Check if we should close positions at end of day
                if self.should_close_position():
                    print("End of day - closing all positions")
                    self.close_all_positions()
                    break
                
                # Check if we should make a trading decision
                if self.should_trade_now():
                    print(f"\n--- Trading Decision at {now.strftime('%H:%M:%S')} ---")
                    
                    decision = self.make_trading_decision()
                    if decision:
                        print(f"Current Price: ${decision['current_price']:.2f}")
                        print(f"Upper Band: ${decision['upper_band']:.2f}")
                        print(f"Lower Band: ${decision['lower_band']:.2f}")
                        print(f"VWAP: ${decision['vwap']:.2f}")
                        print(f"Daily Volatility: {decision['daily_volatility']:.4f} ({decision['daily_volatility']*100:.2f}%)")
                        print(f"Daily Leverage: {decision['daily_leverage']:.2f}")
                        print(f"Current Position: {decision['current_position']}")
                        print(f"New Position: {decision['new_position']}")
                        print(f"Ticker: {decision['ticker']}")
                        print(f"Shares: {decision['shares']}")
                        
                        # Execute trade if position changed
                        if decision['new_position'] != decision['current_position']:
                            success = self.execute_trade(decision)
                            if success:
                                print("Trade executed successfully")
                            else:
                                print("Trade execution failed")
                        else:
                            print("No position change needed")
                    
                    self.last_decision_time = now
                
                # Sleep for 1 second
                await asyncio.sleep(1)
                
            except KeyboardInterrupt:
                print("\nTrading loop interrupted by user")
                break
            except Exception as e:
                print(f"Error in trading loop: {e}")
                await asyncio.sleep(5)  # Wait 5 seconds before retrying
        
        print("Trading loop ended")


def main():
    """Main function to run the live trading strategy."""
    import sys
    
    # Get configuration preset from command line argument
    preset_name = "default"
    if len(sys.argv) > 1:
        preset_name = sys.argv[1]
    
    try:
        from trading_config import get_config, print_config_summary
        config = get_config(preset_name)
        print_config_summary(config)
    except ImportError:
        # Fallback to default configuration
        config = TradingConfig(
            symbol="SPY",
            session="paper",
            leverage_cap=3.0,
            target_daily_volatility=0.02
        )
        print("Using default configuration (trading_config.py not found)")
    
    # Create trader
    trader = BTMLiveTrader(config)
    
    # Run trading loop
    asyncio.run(trader.run_trading_loop())


if __name__ == "__main__":
    main()
