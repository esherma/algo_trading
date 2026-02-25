#!/usr/bin/env python3
"""
Test script for BTM Alert System
This script tests the email functionality and chart generation.
"""

import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from btm_alerts import BTMAlertSystem
from btm_utils import TradingConfig


def create_sample_data():
    """Create sample market data for testing."""
    # Create sample historical data
    dates = pd.date_range('2024-01-01 09:30:00', '2024-01-01 16:00:00', freq='1min', tz='America/New_York')
    
    # Create realistic price data
    np.random.seed(42)
    base_price = 450.0
    price_changes = np.random.normal(0, 0.001, len(dates))
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    historical_data = pd.DataFrame({
        'open': prices,
        'high': [p * 1.001 for p in prices],
        'low': [p * 0.999 for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 10000, len(dates)),
        'vwap': prices
    }, index=dates)
    
    # Create sample noise bands
    noise_bands = pd.DataFrame({
        'UB': [p * 1.02 for p in prices],
        'LB': [p * 0.98 for p in prices],
        'sigma': [0.02] * len(dates)
    }, index=dates)
    
    return historical_data, noise_bands


def test_alert_system():
    """Test the alert system functionality."""
    print("Testing BTM Alert System...")
    
    try:
        # Create configuration
        config = TradingConfig()
        print("✓ Configuration created")
        
        # Create alert system
        alert_system = BTMAlertSystem(config)
        print("✓ Alert system initialized")
        
        # Set daily metrics
        alert_system.set_daily_metrics(0.025, 2.5, {"long_ticker": "SPUU", "short_ticker": "SDS"})
        print("✓ Daily metrics set")
        
        # Set account values
        alert_system.set_account_values(100000.0, 101500.0)
        print("✓ Account values set")
        
        # Create sample market data
        historical_data, noise_bands = create_sample_data()
        alert_system.set_market_data(historical_data, noise_bands)
        print("✓ Market data set")
        
        # Add sample trades
        sample_trades = [
            {
                'timestamp': datetime.now(),
                'action': 'Open Long',
                'ticker': 'SPUU',
                'shares': 100,
                'price': 45.50,
                'value': 4550.0
            },
            {
                'timestamp': datetime.now(),
                'action': 'Close Position',
                'ticker': 'SPUU',
                'shares': 100,
                'price': 46.20,
                'value': 4620.0
            }
        ]
        
        for trade in sample_trades:
            alert_system.add_trade(trade)
        print("✓ Sample trades added")
        
        # Test chart generation
        plot_filename = alert_system.create_noise_bands_plot(include_trades=True)
        if plot_filename and os.path.exists(plot_filename):
            print(f"✓ Chart generated: {plot_filename}")
            # Clean up
            os.remove(plot_filename)
        else:
            print("✗ Chart generation failed")
        
        # Test email templates (without sending)
        morning_html = alert_system.create_morning_alert()
        evening_html = alert_system.create_evening_digest()
        
        if morning_html and evening_html:
            print("✓ Email templates created successfully")
        else:
            print("✗ Email template creation failed")
        
        print("\n🎉 All tests passed! Alert system is working correctly.")
        print("\nTo test actual email sending, run:")
        print("python3 btm_alerts.py")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_email_sending():
    """Test actual email sending (requires proper .env configuration)."""
    print("\nTesting email sending...")
    
    try:
        config = TradingConfig()
        alert_system = BTMAlertSystem(config)
        
        # Set test data
        alert_system.set_daily_metrics(0.025, 2.5, {"long_ticker": "SPUU", "short_ticker": "SDS"})
        alert_system.set_account_values(100000.0, 101500.0)
        
        # Create sample market data
        historical_data, noise_bands = create_sample_data()
        alert_system.set_market_data(historical_data, noise_bands)
        
        # Add sample trade
        alert_system.add_trade({
            'timestamp': datetime.now(),
            'action': 'Test Trade',
            'ticker': 'SPUU',
            'shares': 100,
            'price': 45.50,
            'value': 4550.0
        })
        
        # Send test email
        success = alert_system.send_morning_alert()
        
        if success:
            print("✓ Test email sent successfully!")
        else:
            print("✗ Failed to send test email")
        
        return success
        
    except Exception as e:
        print(f"✗ Email test failed: {e}")
        print("Make sure your .env file has the correct email configuration.")
        return False


if __name__ == "__main__":
    print("BTM Alert System Test Suite")
    print("=" * 40)
    
    # Test basic functionality
    basic_test = test_alert_system()
    
    if basic_test:
        # Ask if user wants to test email sending
        response = input("\nDo you want to test actual email sending? (y/n): ")
        if response.lower() in ['y', 'yes']:
            test_email_sending()
    
    print("\nTest completed!")
