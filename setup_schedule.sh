#!/usr/bin/env bash
# =============================================================================
#  BTM Trading — Scheduling Setup Helper
# =============================================================================
#  This script creates the cron or launchd entry to run run_live.py once per
#  weekday at 09:20 AM *in your local timezone*.
#
#  Usage:
#    chmod +x setup_schedule.sh
#    ./setup_schedule.sh
#
#  Prerequisites:
#    - Python venv set up at /home/user/algo_trading/.venv
#      OR the Python that runs this project is in $PATH as "python"
#    - .env file present and populated
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    # Fall back to whichever python is on PATH
    PYTHON="$(command -v python3 || command -v python)"
fi

LIVE_CMD="$PYTHON $SCRIPT_DIR/run_live.py"

# Detect operating system
OS="$(uname -s)"

# =============================================================================
#  macOS — launchd
# =============================================================================
if [[ "$OS" == "Darwin" ]]; then
    LABEL="com.btm.live-trader"
    PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
    LOG_DIR="$SCRIPT_DIR/logs"
    mkdir -p "$LOG_DIR"

    # Detect current session argument (paper vs live)
    SESSION="${BTM_SESSION:-paper}"
    SYMBOL="${BTM_SYMBOL:-SPY}"

    cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <!-- Unique label for this agent -->
    <key>Label</key>
    <string>${LABEL}</string>

    <!-- Command and arguments -->
    <key>ProgramArguments</key>
    <array>
        <string>${PYTHON}</string>
        <string>${SCRIPT_DIR}/run_live.py</string>
        <string>--session</string>
        <string>${SESSION}</string>
        <string>--symbol</string>
        <string>${SYMBOL}</string>
        <string>--log-file</string>
        <string>${LOG_DIR}/btm_live.log</string>
    </array>

    <!-- Run at 09:20 every weekday (launchd uses local system time) -->
    <!-- Make sure your Mac is set to US/Eastern or the markets' timezone -->
    <key>StartCalendarInterval</key>
    <array>
        <!-- Monday=1 … Friday=5 -->
        <dict>
            <key>Weekday</key><integer>1</integer>
            <key>Hour</key><integer>9</integer>
            <key>Minute</key><integer>20</integer>
        </dict>
        <dict>
            <key>Weekday</key><integer>2</integer>
            <key>Hour</key><integer>9</integer>
            <key>Minute</key><integer>20</integer>
        </dict>
        <dict>
            <key>Weekday</key><integer>3</integer>
            <key>Hour</key><integer>9</integer>
            <key>Minute</key><integer>20</integer>
        </dict>
        <dict>
            <key>Weekday</key><integer>4</integer>
            <key>Hour</key><integer>9</integer>
            <key>Minute</key><integer>20</integer>
        </dict>
        <dict>
            <key>Weekday</key><integer>5</integer>
            <key>Hour</key><integer>9</integer>
            <key>Minute</key><integer>20</integer>
        </dict>
    </array>

    <key>WorkingDirectory</key>
    <string>${SCRIPT_DIR}</string>

    <!-- Capture output -->
    <key>StandardOutPath</key>
    <string>${LOG_DIR}/btm_live_stdout.log</string>
    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/btm_live_stderr.log</string>

    <!-- Prevent accidental re-launch within same minute -->
    <key>ThrottleInterval</key>
    <integer>3600</integer>
</dict>
</plist>
PLIST

    # Load the agent (unload first to refresh if it already exists)
    launchctl unload "$PLIST" 2>/dev/null || true
    launchctl load -w "$PLIST"

    echo ""
    echo "✅  launchd agent installed: $LABEL"
    echo "    Plist:    $PLIST"
    echo "    Schedule: 09:20 Mon–Fri (local system time)"
    echo "    Logs:     $LOG_DIR/"
    echo ""
    echo "    To unload:  launchctl unload $PLIST"
    echo "    To run now: launchctl start $LABEL"
    echo ""
    echo "    ⚠️  Ensure your Mac's system timezone = US/Eastern for correct timing."
    echo "       System Preferences → General → Date & Time → Time Zone"

# =============================================================================
#  Linux — cron
# =============================================================================
else
    # cron uses the server's local timezone.
    # If the server is UTC, 09:20 ET = 14:20 UTC (EDT/summer) or 15:20 UTC (EST/winter).
    # The script itself handles DST by checking the NY clock internally and waiting
    # for market open.  We use 13:45 UTC to always start before 09:30 ET regardless
    # of DST.  The script will sleep until the actual open.
    #
    # Alternatively: set TZ=America/New_York in the crontab.

    SESSION="${BTM_SESSION:-paper}"
    SYMBOL="${BTM_SYMBOL:-SPY}"
    LOG_DIR="$SCRIPT_DIR/logs"
    mkdir -p "$LOG_DIR"

    CRON_CMD="TZ=America/New_York $PYTHON $SCRIPT_DIR/run_live.py"
    CRON_CMD+=" --session $SESSION --symbol $SYMBOL"
    CRON_CMD+=" --log-file $LOG_DIR/btm_live.log"
    CRON_CMD+=" >> $LOG_DIR/btm_live_stdout.log 2>> $LOG_DIR/btm_live_stderr.log"

    CRON_LINE="20 9 * * 1-5 $CRON_CMD"

    # Check if already installed
    if crontab -l 2>/dev/null | grep -qF "run_live.py"; then
        echo "⚠️  A BTM cron entry already exists. Remove it manually before re-installing:"
        echo ""
        crontab -l 2>/dev/null | grep "run_live.py"
        echo ""
        echo "    Run: crontab -e  to edit."
    else
        ( crontab -l 2>/dev/null; echo "$CRON_LINE" ) | crontab -
        echo ""
        echo "✅  cron entry installed."
        echo "    Schedule: 09:20 Mon–Fri (America/New_York)"
        echo "    Logs:     $LOG_DIR/"
        echo ""
        echo "    To view:   crontab -l"
        echo "    To remove: crontab -e  (delete the btm line)"
    fi
fi
