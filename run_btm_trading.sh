#!/bin/bash

# BTM Live Trading Strategy Runner
# This script is designed to be run by cron at 09:20:00 on weekdays
# to start the trading strategy before market open

# Set the working directory to the script location
cd "$(dirname "$0")"

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Log file for the trading session
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# Create log filename with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/btm_trading_${TIMESTAMP}.log"

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Start logging
log_message "Starting BTM Live Trading Strategy"

# Check if it's a weekday
DAY_OF_WEEK=$(date +%u)
if [ "$DAY_OF_WEEK" -eq 6 ] || [ "$DAY_OF_WEEK" -eq 7 ]; then
    log_message "Weekend detected. Exiting."
    exit 0
fi

# Check if market is open or will be open soon
CURRENT_HOUR=$(date +%H)
CURRENT_MINUTE=$(date +%M)
CURRENT_TIME=$((10#$CURRENT_HOUR * 60 + 10#$CURRENT_MINUTE))

# Market opens at 9:30 AM ET (14:30 UTC in winter, 13:30 UTC in summer)
# We want to start at 9:20 AM ET to prepare
MARKET_OPEN_MINUTES=570  # 9:30 AM = 9*60 + 30 = 570 minutes from midnight

# Convert current time to minutes from midnight
if [ "$CURRENT_TIME" -lt "$MARKET_OPEN_MINUTES" ]; then
    log_message "Market not yet open. Starting strategy preparation."
else
    log_message "Market is open. Starting strategy."
fi

# Check if Python and required packages are available
if ! command -v python3 &> /dev/null; then
    log_message "ERROR: python3 is not installed or not in PATH"
    exit 1
fi

# Check if the trading script exists
if [ ! -f "btm_live_trading.py" ]; then
    log_message "ERROR: btm_live_trading.py not found in current directory"
    exit 1
fi

# Check if the utils file exists
if [ ! -f "btm_utils.py" ]; then
    log_message "ERROR: btm_utils.py not found in current directory"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    log_message "WARNING: .env file not found. Make sure API keys are set in environment variables."
fi

# Run the trading strategy
log_message "Executing BTM Live Trading Strategy..."
python3 btm_live_trading.py 2>&1 | tee -a "$LOG_FILE"

# Capture exit code
EXIT_CODE=${PIPESTATUS[0]}

# Log completion
if [ "$EXIT_CODE" -eq 0 ]; then
    log_message "BTM Live Trading Strategy completed successfully"
else
    log_message "BTM Live Trading Strategy exited with error code: $EXIT_CODE"
fi

# Clean up old log files (keep last 30 days)
find "$LOG_DIR" -name "btm_trading_*.log" -mtime +30 -delete

log_message "Script execution completed"

exit "$EXIT_CODE"
