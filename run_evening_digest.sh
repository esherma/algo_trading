#!/bin/bash

# BTM Evening Digest Script
# This script should be run by cron at 16:05 ET (after market close)
# to send the daily trading digest

# Set the working directory to the script location
cd "$(dirname "$0")"

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Log file for the digest
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# Create log filename with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/evening_digest_${TIMESTAMP}.log"

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Start logging
log_message "Starting BTM Evening Digest"

# Check if it's a weekday
DAY_OF_WEEK=$(date +%u)
if [ "$DAY_OF_WEEK" -eq 6 ] || [ "$DAY_OF_WEEK" -eq 7 ]; then
    log_message "Weekend detected. Exiting."
    exit 0
fi

# Check if Python and required packages are available
if ! command -v python3 &> /dev/null; then
    log_message "ERROR: python3 is not installed or not in PATH"
    exit 1
fi

# Check if the digest script exists
if [ ! -f "send_evening_digest.py" ]; then
    log_message "ERROR: send_evening_digest.py not found in current directory"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    log_message "WARNING: .env file not found. Make sure API keys are set in environment variables."
fi

# Run the evening digest script
log_message "Executing Evening Digest Script..."
python3 send_evening_digest.py 2>&1 | tee -a "$LOG_FILE"

# Capture exit code
EXIT_CODE=${PIPESTATUS[0]}

# Log completion
if [ "$EXIT_CODE" -eq 0 ]; then
    log_message "Evening Digest completed successfully"
else
    log_message "Evening Digest exited with error code: $EXIT_CODE"
fi

# Clean up old log files (keep last 30 days)
find "$LOG_DIR" -name "evening_digest_*.log" -mtime +30 -delete

# Clean up old opening AUM files (keep last 7 days)
find . -name "opening_aum_*.txt" -mtime +7 -delete

log_message "Script execution completed"

exit "$EXIT_CODE"
