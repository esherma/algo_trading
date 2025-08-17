# BTM Trading Alert System Setup Guide

This guide will help you set up the email alerting system for your BTM trading strategy.

## Overview

The alerting system provides two types of notifications:

1. **Morning Alert** (sent at 9:20 AM ET): Daily setup information including volatility, leverage, trading tickers, and noise bands chart
2. **Evening Digest** (sent at 4:05 PM ET): Daily performance summary including trades, P&L, and annotated noise bands chart

## Prerequisites

1. Python packages: `matplotlib`, `pandas`, `pytz`, `python-dotenv`
2. Email service (Gmail recommended for simplicity)
3. Cron access on your system

## Email Configuration

### 1. Set up Email Credentials

Add the following variables to your `.env` file:

```bash
# Email Configuration
ALERT_EMAIL_ADDRESS=your-email@gmail.com
ALERT_EMAIL_PASSWORD=your-app-password
RECIPIENT_EMAIL=your-email@gmail.com

# Optional: Custom SMTP settings (defaults to Gmail)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
```

### 2. Gmail App Password Setup

If using Gmail, you'll need to create an App Password:

1. Go to your Google Account settings
2. Enable 2-Factor Authentication if not already enabled
3. Go to Security → App passwords
4. Generate a new app password for "Mail"
5. Use this password in your `.env` file (not your regular Gmail password)

## Cron Job Setup

### 1. Morning Trading Script (9:20 AM ET)

Add this to your crontab (`crontab -e`):

```bash
# BTM Trading Strategy - Start at 9:20 AM ET (14:20 UTC in winter, 13:20 UTC in summer)
20 14 * * 1-5 /path/to/your/algo_trading/run_btm_trading.sh
```

### 2. Evening Digest Script (4:05 PM ET)

Add this to your crontab:

```bash
# BTM Evening Digest - Send at 4:05 PM ET (21:05 UTC in winter, 20:05 UTC in summer)
5 21 * * 1-5 /path/to/your/algo_trading/run_evening_digest.sh
```

**Note**: Adjust the UTC times based on daylight saving time in your timezone.

## Testing the Setup

### 1. Test Email Configuration

Run the test script to verify your email setup:

```bash
python3 btm_alerts.py
```

### 2. Test Morning Alert

You can manually trigger a morning alert:

```python
from btm_alerts import BTMAlertSystem
from btm_utils import TradingConfig

config = TradingConfig()
alert_system = BTMAlertSystem(config)

# Set test data
alert_system.set_daily_metrics(0.025, 2.5, {"long_ticker": "SPUU", "short_ticker": "SDS"})
alert_system.set_account_values(100000.0, 100000.0)

# Send test alert
alert_system.send_morning_alert()
```

### 3. Test Evening Digest

You can manually trigger an evening digest:

```bash
python3 send_evening_digest.py
```

## File Structure

```
algo_trading/
├── btm_alerts.py              # Alert system module
├── btm_live_trading.py        # Main trading script (updated with alerts)
├── send_evening_digest.py     # Evening digest script
├── run_btm_trading.sh         # Morning trading script
├── run_evening_digest.sh      # Evening digest script
├── logs/                      # Log files directory
├── opening_aum_YYYYMMDD.txt   # Daily opening AUM files
└── .env                       # Environment variables
```

## Email Content

### Morning Alert Includes:
- Daily volatility and leverage calculations
- Trading tickers for the day (long/short positions)
- Strategy configuration summary
- Noise bands chart (attached as PNG)

### Evening Digest Includes:
- Opening and closing AUM
- Daily P&L calculation
- List of all trades executed
- Daily metrics summary
- Annotated noise bands chart with trade markers

## Troubleshooting

### Common Issues:

1. **Email not sending**: Check SMTP credentials and app password
2. **Charts not generating**: Ensure matplotlib is installed
3. **Cron jobs not running**: Check file permissions and paths
4. **Missing trade data**: Verify log file parsing

### Debug Mode:

Add debug logging by modifying the alert system:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Manual Testing:

You can test individual components:

```bash
# Test email sending
python3 -c "from btm_alerts import test_email_configuration; test_email_configuration()"

# Test chart generation
python3 -c "from btm_alerts import BTMAlertSystem; from btm_utils import TradingConfig; import pandas as pd; alert = BTMAlertSystem(TradingConfig()); alert.create_noise_bands_plot()"
```

## Security Considerations

1. **App Passwords**: Use app-specific passwords, not your main email password
2. **Environment Variables**: Never commit `.env` files to version control
3. **File Permissions**: Ensure sensitive files have appropriate permissions
4. **Network Security**: Use TLS/SSL for email transmission

## Customization

### Modify Email Templates

Edit the HTML templates in `btm_alerts.py`:
- `create_morning_alert()` method
- `create_evening_digest()` method

### Add Custom Metrics

Extend the alert system to include additional metrics:

```python
class BTMAlertSystem:
    def add_custom_metric(self, name: str, value: any):
        # Add custom metrics to email templates
        pass
```

### Change Chart Style

Modify the `create_noise_bands_plot()` method to customize chart appearance.

## Support

If you encounter issues:

1. Check the log files in the `logs/` directory
2. Verify all environment variables are set correctly
3. Test email configuration manually
4. Ensure all required Python packages are installed

## Dependencies

Add these to your `pyproject.toml` or install manually:

```bash
pip install matplotlib pandas pytz python-dotenv
```
