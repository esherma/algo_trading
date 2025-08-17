# BTM Live Trading System

This directory contains the live trading implementation of the "Beat the Market" momentum strategy, adapted for paper trading on Alpaca Markets.

## Overview

The live trading system implements the **Current Band + VWAP** strategy variant, which:
- Uses noise bands calculated from historical SPY data
- Makes trading decisions at 15 seconds before each half-hour (09:59:45, 10:29:45, etc.)
- Uses VWAP as a stop-loss mechanism
- Dynamically sizes positions based on volatility targeting
- Uses leveraged ETFs instead of short selling (since this is not a margin account)

## Files

- `btm_utils.py` - Shared utilities and strategy functions
- `btm_live_trading.py` - Main live trading implementation
- `run_btm_trading.sh` - Shell script for automated execution
- `LIVE_TRADING_README.md` - This documentation file

## Setup

### 1. Environment Variables

Create a `.env` file in the project root with your Alpaca API credentials:

```bash
ALPACA_PAPER_API_KEY=your_paper_api_key_here
ALPACA_PAPER_API_SECRET=your_paper_api_secret_here
ALPACA_LIVE_API_KEY=your_live_api_key_here
ALPACA_LIVE_API_SECRET=your_live_api_secret_here
```

### 2. Dependencies

Install the required Python packages:

```bash
pip install alpaca-py pandas numpy pytz python-dotenv
```

### 3. Cron Job Setup

To run the strategy automatically each weekday at 09:20:00, add this to your crontab:

```bash
# Edit crontab
crontab -e

# Add this line (adjust the path to your project directory)
20 9 * * 1-5 /path/to/your/algo_trading/run_btm_trading.sh
```

## Strategy Details

### Trading Schedule

- **Start Time**: 09:20:00 (15 minutes before market open)
- **Decision Times**: 15 seconds before each half-hour (09:59:45, 10:29:45, 10:59:45, etc.)
- **Last Decision Time**: 15:29:45
- **End of Day**: 15:49:30 (market close order if position still open)

### Leverage Mapping

The strategy uses the following tickers based on calculated leverage:

| Calculated Leverage | Ticker | Description |
|-------------------|--------|-------------|
| -3 | SPXS | 3x inverse leveraged SPY ETF |
| -2 | SDS | 2x inverse leveraged SPY ETF |
| -1 | SPDN | 1x inverse SPY ETF |
| 1 | SPY | Normal SPY |
| 2 | SPUU | 2x leveraged SPY ETF |
| 3 | SPXL | 3x leveraged SPY ETF |

### Position Sizing

- Leverage is calculated as: `min(leverage_cap, target_volatility / current_volatility)`
- Leverage cap is set to 3.0
- Target daily volatility is 2%
- Number of shares = `floor(AUM / ticker_price)`

### Trading Logic

1. **Entry Signals**:
   - Long when SPY price > Upper Band
   - Short (via inverse ETF) when SPY price < Lower Band

2. **Exit Signals**:
   - Stop loss: VWAP level
   - End of day: Market close order at 15:49:30

3. **Position Management**:
   - Only one position at a time
   - All positions are long (use inverse ETFs for short exposure)
   - Close existing position before opening new one

## Usage

### Manual Execution

To run the strategy manually:

```bash
python3 btm_live_trading.py
```

### Automated Execution

The strategy will run automatically via cron job, but you can also run it manually:

```bash
./run_btm_trading.sh
```

### Logs

Logs are stored in the `logs/` directory with timestamps:
- `logs/btm_trading_YYYYMMDD_HHMMSS.log`

## Configuration

You can modify the strategy parameters in `btm_live_trading.py`:

```python
config = TradingConfig(
    symbol="SPY",                    # Base symbol for analysis
    session="paper",                 # "paper" or "live"
    leverage_cap=3.0,               # Maximum leverage
    target_daily_volatility=0.02,   # Target daily volatility (2%)
    lookback_days=14,               # Days for noise band calculation
    volatility_multiplier=1.0,      # Multiplier for noise bands
    use_gap_adjustment=True,        # Use gap adjustment for bands
    start_trading_time="10:00",     # First possible decision time
    close_time="16:00"              # Market close time
)
```

## Risk Management

### Paper Trading Only

This implementation is configured for **paper trading only**. To switch to live trading:

1. Change `session="live"` in the configuration
2. Ensure you have live API credentials
3. Test thoroughly with small amounts first

### Position Limits

- Maximum leverage: 3x
- All positions are long (no short selling)
- Automatic position closure at end of day
- VWAP-based stop losses

### Monitoring

Monitor the strategy through:
- Log files in `logs/` directory
- Alpaca dashboard
- Account statements

## Troubleshooting

### Common Issues

1. **API Connection Errors**:
   - Verify API credentials in `.env` file
   - Check internet connection
   - Ensure Alpaca account is active

2. **No Trading Decisions**:
   - Check if market is open
   - Verify decision time logic
   - Check log files for errors

3. **Order Failures**:
   - Insufficient buying power
   - Invalid ticker symbols
   - Market closed

### Debug Mode

To run with more verbose logging, modify the logging level in the script or add debug prints.

## Performance Tracking

The strategy logs all trading decisions and executions. You can analyze performance by:

1. Reviewing log files
2. Checking Alpaca dashboard
3. Exporting trade history from Alpaca

## Next Steps

After setting up the basic system, consider:

1. **Daily Reporting**: Implement automated daily performance reports
2. **Risk Monitoring**: Add real-time risk metrics
3. **Backtesting**: Compare live performance with backtest results
4. **Optimization**: Fine-tune parameters based on live results

## Support

For issues or questions:
1. Check the log files for error messages
2. Review Alpaca API documentation
3. Test with paper trading first
4. Monitor account status and trading permissions
