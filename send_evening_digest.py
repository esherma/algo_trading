#!/usr/bin/env python3
"""
Evening Digest Script for BTM Trading Strategy
This script should be run by cron at 16:05 ET (after market close) to send the daily digest.
"""

import os
import sys
import asyncio
from datetime import datetime, timedelta
import pytz
from dotenv import load_dotenv

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from btm_alerts import BTMAlertSystem
from btm_utils import TradingConfig
from trading_config import get_config


def get_daily_trades_from_logs(log_dir: str = "logs") -> list:
    """
    Parse today's trading log to extract trade information.
    This is a fallback method if the alert system doesn't have trade data.
    """
    today = datetime.now().strftime("%Y%m%d")
    log_pattern = f"btm_trading_{today}_*.log"
    
    trades = []
    
    try:
        import glob
        log_files = glob.glob(os.path.join(log_dir, log_pattern))
        
        if not log_files:
            print(f"No log files found for today ({today})")
            return trades
        
        # Use the most recent log file
        latest_log = max(log_files, key=os.path.getctime)
        
        with open(latest_log, 'r') as f:
            for line in f:
                if "Opening position:" in line or "Closing position:" in line:
                    # Parse trade information from log line
                    # This is a simplified parser - you may need to adjust based on actual log format
                    if "Opening position:" in line:
                        action = "Open Position"
                    else:
                        action = "Close Position"
                    
                    # Extract basic trade info (this is a placeholder - adjust based on actual log format)
                    trades.append({
                        'timestamp': datetime.now(),
                        'action': action,
                        'ticker': 'Unknown',
                        'shares': 0,
                        'price': 0.0,
                        'value': 0.0
                    })
        
        print(f"Found {len(trades)} trades in log file")
        
    except Exception as e:
        print(f"Error parsing log files: {e}")
    
    return trades


def main():
    """Main function to send evening digest."""
    load_dotenv()
    
    # Get configuration
    preset_name = "default"
    if len(sys.argv) > 1:
        preset_name = sys.argv[1]
    
    try:
        config = get_config(preset_name)
    except ImportError:
        config = TradingConfig()
        print("Using default configuration")
    
    # Create alert system
    alert_system = BTMAlertSystem(config)
    
    # Get account information (you'll need to implement this based on your setup)
    # For now, we'll use placeholder values
    try:
        from alpaca.trading.client import TradingClient
        from btm_utils import load_env, get_alpaca_client
        
        load_env()
        trading_client = TradingClient(
            os.getenv("ALPACA_PAPER_API_KEY") if config.session == "paper" else os.getenv("ALPACA_LIVE_API_KEY"),
            os.getenv("ALPACA_PAPER_API_SECRET") if config.session == "paper" else os.getenv("ALPACA_LIVE_API_SECRET"),
            paper=config.session == "paper"
        )
        
        account = trading_client.get_account()
        closing_aum = float(account.portfolio_value)
        
        # Try to get opening AUM from a file or use a default
        opening_aum_file = f"opening_aum_{datetime.now().strftime('%Y%m%d')}.txt"
        if os.path.exists(opening_aum_file):
            with open(opening_aum_file, 'r') as f:
                opening_aum = float(f.read().strip())
        else:
            opening_aum = closing_aum  # Fallback
        
        alert_system.set_account_values(opening_aum, closing_aum)
        
    except Exception as e:
        print(f"Error getting account info: {e}")
        # Use placeholder values
        alert_system.set_account_values(100000.0, 100000.0)
    
    # Try to get trade data from logs if not available
    if not alert_system.daily_trades:
        trades = get_daily_trades_from_logs()
        for trade in trades:
            alert_system.add_trade(trade)
    
    # Set placeholder daily metrics (these should ideally be stored from the morning)
    alert_system.set_daily_metrics(0.02, 1.0, {"long_ticker": "SPY", "short_ticker": "SPDN"})
    
    # Send evening digest
    print("Sending evening digest...")
    success = alert_system.send_evening_digest()
    
    if success:
        print("Evening digest sent successfully")
    else:
        print("Failed to send evening digest")
        sys.exit(1)


if __name__ == "__main__":
    main()
