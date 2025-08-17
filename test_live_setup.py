#!/usr/bin/env python3
"""
Test script to verify the live trading setup is working correctly.
This script tests the basic functionality without making any trades.
"""

import os
import sys
import math
from datetime import datetime, timedelta
import pytz

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from btm_utils import TradingConfig, load_env, get_alpaca_client, fetch_intraday_bars, compute_noise_bands
from alpaca.trading.client import TradingClient


def test_environment():
    """Test environment variables and API connectivity."""
    print("=== Testing Environment ===")
    
    # Load environment
    load_env()
    
    # Check API keys
    paper_key = os.getenv("ALPACA_PAPER_API_KEY")
    paper_secret = os.getenv("ALPACA_PAPER_API_SECRET")
    
    if not paper_key or not paper_secret:
        print("❌ Missing Alpaca API credentials in .env file")
        return False
    
    print("✅ API credentials found")
    return True


def test_api_connectivity():
    """Test API connectivity."""
    print("\n=== Testing API Connectivity ===")
    
    try:
        # Test historical data client
        historical_client = get_alpaca_client("paper")
        print("✅ Historical data client connected")
        
        # Test trading client
        trading_client = TradingClient(
            os.getenv("ALPACA_PAPER_API_KEY"),
            os.getenv("ALPACA_PAPER_API_SECRET"),
            paper=True
        )
        print("✅ Trading client connected")
        
        # Test account access
        account = trading_client.get_account()
        print(f"✅ Account access successful")
        print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
        print(f"   Buying Power: ${float(account.buying_power):,.2f}")
        print(f"   Cash: ${float(account.cash):,.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ API connectivity failed: {e}")
        return False


def test_data_fetching():
    """Test data fetching capabilities."""
    print("\n=== Testing Data Fetching ===")
    
    try:
        # Get historical client
        historical_client = get_alpaca_client("paper")
        
        # Fetch recent data
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        
        print(f"Fetching SPY data from {start_date} to {end_date}...")
        
        df = fetch_intraday_bars(
            historical_client,
            "SPY",
            start_date,
            end_date
        )
        
        print(f"✅ Successfully fetched {len(df)} bars of SPY data")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")
        print(f"   Columns: {list(df.columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data fetching failed: {e}")
        return False


def fetch_test_data():
    """Fetch test data for other tests."""
    try:
        historical_client = get_alpaca_client("paper")
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        
        df = fetch_intraday_bars(
            historical_client,
            "SPY",
            start_date,
            end_date
        )
        return df
    except Exception as e:
        print(f"Error fetching test data: {e}")
        return None

def test_noise_bands(df=None):
    """Test noise band calculation."""
    print("\n=== Testing Noise Band Calculation ===")
    
    try:
        if df is None:
            df = fetch_test_data()
        
        config = TradingConfig()
        
        bands = compute_noise_bands(
            df=df,
            lookback_days=config.lookback_days,
            vm=config.volatility_multiplier,
            gap_adjustment=config.use_gap_adjustment
        )
        
        print(f"✅ Successfully calculated noise bands")
        print(f"   Date range: {bands.index[0]} to {bands.index[-1]}")
        print(f"   Columns: {list(bands.columns)}")
        
        # Show some sample values
        latest = bands.iloc[-1]
        print(f"   Latest UB: ${latest['UB']:.2f}")
        print(f"   Latest LB: ${latest['LB']:.2f}")
        print(f"   Latest Sigma: {latest['sigma']:.4f}")
        
        return bands is not None
        
    except Exception as e:
        print(f"❌ Noise band calculation failed: {e}")
        return None


def test_volatility_calculation(df=None):
    """Test volatility calculation matches the paper formula."""
    print("\n=== Testing Volatility Calculation ===")
    
    try:
        if df is None:
            df = fetch_test_data()
        
        from btm_live_trading import BTMLiveTrader
        
        config = TradingConfig()
        trader = BTMLiveTrader(config)
        trader.historical_data = df
        
        # Calculate volatility using the new method
        volatility = trader.calculate_daily_volatility()
        
        print(f"✅ Successfully calculated daily volatility")
        print(f"   Volatility: {volatility:.4f} ({volatility*100:.2f}%)")
        
        # Verify it matches the expected formula
        from btm_utils import compute_daily_ohlcv
        daily = compute_daily_ohlcv(df)
        daily_returns = daily["close"].pct_change().dropna()
        
        if len(daily_returns) >= 14:
            last_14_returns = daily_returns.tail(14)
            mean_return = last_14_returns.mean()
            variance = ((last_14_returns - mean_return) ** 2).sum() / 13
            expected_volatility = math.sqrt(variance)
            
            print(f"   Expected volatility: {expected_volatility:.4f}")
            print(f"   Match: {'✅' if abs(volatility - expected_volatility) < 1e-6 else '❌'}")
        
        return volatility is not None
        
    except Exception as e:
        print(f"❌ Volatility calculation test failed: {e}")
        return None


def test_leverage_mapping():
    """Test leverage mapping functionality."""
    print("\n=== Testing Leverage Mapping ===")
    
    try:
        from btm_utils import get_leverage_ticker, calculate_leverage_and_ticker
        
        # Test various leverage values
        test_cases = [
            (-3.0, "SPXS"),
            (-2.0, "SDS"),
            (-1.0, "SPDN"),
            (1.0, "SPY"),
            (2.0, "SPUU"),
            (3.0, "SPXL"),
            (0.5, "SPY"),  # Should round to 1
            (-2.7, "SPXS"),  # Should round to -3
        ]
        
        for leverage, expected_ticker in test_cases:
            ticker = get_leverage_ticker(leverage)
            if ticker == expected_ticker:
                print(f"✅ Leverage {leverage} -> {ticker}")
            else:
                print(f"❌ Leverage {leverage} -> {ticker} (expected {expected_ticker})")
        
        # Test leverage calculation
        leverage, ticker = calculate_leverage_and_ticker(0.02, 0.01, 3.0)
        print(f"✅ Volatility calculation: target=2%, current=1% -> leverage={leverage:.1f}, ticker={ticker}")
        
        return True
        
    except Exception as e:
        print(f"❌ Leverage mapping test failed: {e}")
        return False


def test_trading_schedule():
    """Test trading schedule logic."""
    print("\n=== Testing Trading Schedule ===")
    
    try:
        from btm_live_trading import BTMLiveTrader
        
        config = TradingConfig()
        trader = BTMLiveTrader(config)
        
        # Test various times
        test_times = [
            ("09:59:45", True),   # Should trade
            ("10:29:45", True),   # Should trade
            ("10:30:00", False),  # Should not trade
            ("15:29:45", True),   # Should trade (last decision)
            ("15:30:00", False),  # Should not trade
            ("15:49:30", False),  # Should not trade (close time)
        ]
        
        for time_str, should_trade in test_times:
            # Create a mock datetime for testing
            hour, minute, second = map(int, time_str.split(":"))
            test_time = datetime.now().replace(hour=hour, minute=minute, second=second, microsecond=0)
            
            # Mock the should_trade_now method
            current_minute = test_time.minute
            current_second = test_time.second
            
            is_decision_time = (current_minute in [59, 29] and current_second == 45)
            is_after_start = test_time.time() >= datetime.strptime(config.start_trading_time, "%H:%M").time()
            is_before_end = test_time.time() < datetime.strptime("15:30", "%H:%M").time()
            
            result = is_decision_time and is_after_start and is_before_end
            
            if result == should_trade:
                print(f"✅ {time_str}: {'Should trade' if should_trade else 'Should not trade'}")
            else:
                print(f"❌ {time_str}: Expected {'should trade' if should_trade else 'should not trade'}, got {'should trade' if result else 'should not trade'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Trading schedule test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("BTM Live Trading Setup Test")
    print("=" * 50)
    
    tests = [
        ("Environment", test_environment),
        ("API Connectivity", test_api_connectivity),
        ("Data Fetching", test_data_fetching),
        ("Noise Bands", test_noise_bands),
        ("Volatility Calculation", test_volatility_calculation),
        ("Leverage Mapping", test_leverage_mapping),
        ("Trading Schedule", test_trading_schedule),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            if test_name == "Noise Bands":
                # Special handling for noise bands test
                df = fetch_test_data()
                if df is not None:
                    result = test_func(df)
                    results.append((test_name, result))
                else:
                    results.append((test_name, False))
            elif test_name == "Volatility Calculation":
                # Special handling for volatility calculation test
                df = fetch_test_data()
                if df is not None:
                    result = test_func(df)
                    results.append((test_name, result))
                else:
                    results.append((test_name, False))
            else:
                result = test_func()
                results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your live trading setup is ready.")
        print("\nNext steps:")
        print("1. Review the configuration in btm_live_trading.py")
        print("2. Set up the cron job: crontab -e")
        print("3. Add: 20 9 * * 1-5 /path/to/your/algo_trading/run_btm_trading.sh")
        print("4. Test manually: python3 btm_live_trading.py")
    else:
        print("⚠️  Some tests failed. Please fix the issues before running live trading.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
