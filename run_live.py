#!/usr/bin/env python3
"""
BTM Live Trading Runner
=======================
Runs the Beat-the-Market intraday momentum strategy for one trading day.
Designed to be invoked once per day by a cron job or launchd task.

Flow
----
  1.  Verify it is a trading day (skip weekends / market holidays).
  2.  Fetch ~25 calendar days of historical minute bars for σ computation.
  3.  Sleep until 09:30 (if started early) and wait for the first bar.
  4.  Compute today's noise bands + daily leverage.
  5.  Generate and email the morning "cone" chart to RECIPIENT_EMAIL.
  6.  At each half-hour decision time (10:00 … 15:30):
        a. Fetch all intraday bars since open.
        b. Compute VWAP from those bars.
        c. Look up UB / LB for this exact minute.
        d. Evaluate stop and entry signals.
        e. Submit market orders if position changes.
        f. Log the decision.
  7.  At 15:50 — close all positions (market order).
  8.  Exit (so the OS scheduler can invoke it again tomorrow).

Usage
-----
  python run_live.py                     # paper trading, SPY
  python run_live.py --session live      # real money!
  python run_live.py --symbol QQQ        # different symbol
  python run_live.py --dry-run           # simulate without submitting orders

Environment (.env)
------------------
  ALPACA_PAPER_API_KEY / ALPACA_PAPER_API_SECRET
  ALPACA_LIVE_API_KEY  / ALPACA_LIVE_API_SECRET
  SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, RECIPIENT_EMAIL
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta
from typing import Optional

import pytz
from dotenv import load_dotenv
import pandas as pd


from btm.core import (
    CLOSE_POSITIONS_TIME,
    DECISION_TIMES,
    NY_TZ as NY_TZ_STR,
    BTMConfig,
    compute_bands_for_today,
    compute_daily_vol,
    compute_sigma_for_today,
    decide_position,
    select_etfs,
    compute_etf_shares,
)
from btm.data import (
    fetch_minute_bars,
    fetch_snapshot,
    get_market_clock,
    is_trading_day,
    make_data_client,
    make_trading_client,
)
from btm.orders import (
    buy,
    close_all_positions,
    get_account_info,
    get_all_positions,
    get_open_position,
    sell,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

NY_TZ = pytz.timezone(NY_TZ_STR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def now_ny() -> datetime:
    return datetime.now(NY_TZ)


def sleep_until(target_hhmm: str, poll_sec: int = 10) -> None:
    """Sleep until *target_hhmm* (NY time) is reached, polling every *poll_sec*."""
    h, m = map(int, target_hhmm.split(":"))
    while True:
        now = now_ny()
        target = now.replace(hour=h, minute=m, second=5, microsecond=0)
        delta = (target - now).total_seconds()
        if delta <= 0:
            return
        sleep_time = min(delta, poll_sec)
        time.sleep(sleep_time)


def sleep_until_next_decision(after_hhmm: Optional[str] = None) -> Optional[str]:
    """
    Sleep until the next decision time >= *after_hhmm* (or the next one after now).
    Returns the decision time string we woke up for, or None if past all decisions.
    """
    now_str = now_ny().strftime("%H:%M")
    remaining = [t for t in DECISION_TIMES if t > (after_hhmm or now_str)]
    if not remaining:
        return None
    next_dt = remaining[0]
    log.info("Sleeping until decision time %s …", next_dt)
    sleep_until(next_dt)
    return next_dt


def hhmm_now() -> str:
    return now_ny().strftime("%H:%M")


# ---------------------------------------------------------------------------
# State dataclass
# ---------------------------------------------------------------------------

class TradingState:
    """Mutable day state for the live trading loop."""

    def __init__(self) -> None:
        self.position: int = 0          # 0=flat, 1=long, -1=short
        self.position_ticker: str = ""  # ETF currently held
        self.position_shares: int = 0   # shares currently held

        # Fixed for the day (computed after open)
        self.long_etf: str  = "SPY"
        self.short_etf: str = "SPDN"
        self.etf_mult: int  = 1
        self.daily_shares: int = 0
        self.leverage: float = 1.0
        self.daily_vol_pct: float = 2.0

        self.today_open: float = 0.0
        self.yesterday_close: float = 0.0

    def __str__(self) -> str:
        return (
            f"pos={self.position:+d}  ticker={self.position_ticker or 'flat'}"
            f"  shares={self.position_shares}"
            f"  leverage={self.leverage:.2f}×"
            f"  long_etf={self.long_etf}  short_etf={self.short_etf}"
        )


# ---------------------------------------------------------------------------
# Core daily setup
# ---------------------------------------------------------------------------

def wait_for_open_bar(data_client, symbol: str) -> float:
    """
    Poll the snapshot endpoint until the 09:30 regular-session bar is available.

    Uses minute_bar.open (not daily_bar.open) to get the true session open.
    On SIP feeds daily_bar.open reflects the 4 AM pre-market first print;
    checking minute_bar.timestamp >= 09:30 guarantees we have the regular-
    session open regardless of feed type.
    Returns today's opening price.
    """
    log.info("Waiting for 09:30 bar …")
    session_open = pd.Timestamp("09:30").time()

    for attempt in range(60):           # give it up to 5 minutes
        try:
            snap = fetch_snapshot(data_client, symbol)
            ts_ny = pd.Timestamp(snap["timestamp"]).tz_convert(NY_TZ)
            if ts_ny.time() >= session_open:
                open_price = snap["minute_open"]
                log.info(
                    "09:30 bar confirmed: %s open = $%.2f  (bar ts %s)",
                    symbol, open_price, ts_ny.strftime("%H:%M"),
                )
                return open_price
        except Exception as exc:
            log.debug("Snapshot attempt %d: %s", attempt + 1, exc)
        time.sleep(5)

    raise RuntimeError(f"09:30 bar for {symbol} not detected after 5 minutes.")


def compute_day_state(
    data_client,
    symbol: str,
    hist_df,
    cfg: BTMConfig,
    state: TradingState,
) -> pd.DataFrame:
    """
    Use *hist_df* (historical data ending yesterday) to compute everything
    that is fixed for the day: sigma, bands, leverage, ETF selection.
    Populates *state* in place.
    Returns the bands DataFrame indexed by "HH:MM".
    """
    # Sigma over last lookback days (historical data only — no today)
    sigma_series = compute_sigma_for_today(hist_df, cfg.lookback_days)

    # Today's bands
    bands_df = compute_bands_for_today(
        sigma_series, state.today_open, state.yesterday_close, cfg.vm
    )

    # Daily volatility → leverage → ETF selection
    daily_close = hist_df.groupby(hist_df.index.date)["close"].last()
    import pandas as pd
    daily_close_ts = pd.Series(
        daily_close.values,
        index=pd.to_datetime(daily_close.index).tz_localize(NY_TZ_STR),
    )
    vol = compute_daily_vol(daily_close_ts, cfg.lookback_days)
    leverage = min(cfg.leverage_cap, cfg.target_daily_vol / max(vol, 1e-6))

    long_etf, short_etf, etf_mult = select_etfs(leverage)

    # Shares — sized on SPY open, scaled by ETF multiplier
    acct = get_account_info(state._trading_client)
    aum  = acct["portfolio_value"]
    shares = compute_etf_shares(aum, leverage, state.today_open, etf_mult)

    # Write into state
    state.leverage       = leverage
    state.long_etf       = long_etf
    state.short_etf      = short_etf
    state.etf_mult       = etf_mult
    state.daily_shares   = shares
    state.daily_vol_pct  = vol * 100

    log.info(
        "Daily setup: vol=%.2f%%  leverage=%.2f×  long=%s  short=%s"
        "  shares=%d  AUM=$%.0f",
        vol * 100, leverage, long_etf, short_etf, shares, aum,
    )
    return bands_df


# ---------------------------------------------------------------------------
# Position execution helpers
# ---------------------------------------------------------------------------

def execute_position_change(
    trading_client,
    state: TradingState,
    desired: int,          # -1, 0, +1
    dry_run: bool,
) -> None:
    """
    Transition *state.position* → *desired* by submitting market orders.
    For long:  buy long_etf
    For short: buy short_etf (inverse ETF)
    For flat:  sell whatever we hold
    """
    if desired == state.position:
        return

    # Close existing position
    if state.position != 0 and state.position_ticker:
        log.info("Closing %d %s", state.position_shares, state.position_ticker)
        if not dry_run:
            sell(trading_client, state.position_ticker, state.position_shares)
        state.position        = 0
        state.position_ticker = ""
        state.position_shares = 0

    # Open new position
    if desired == 1:
        ticker = state.long_etf
        shares = state.daily_shares
        log.info("Opening LONG  %d %s", shares, ticker)
        if not dry_run and shares > 0:
            buy(trading_client, ticker, shares)
        state.position        = 1
        state.position_ticker = ticker
        state.position_shares = shares

    elif desired == -1:
        ticker = state.short_etf
        shares = state.daily_shares
        log.info("Opening SHORT (via %s) %d shares", ticker, shares)
        if not dry_run and shares > 0:
            buy(trading_client, ticker, shares)
        state.position        = -1
        state.position_ticker = ticker
        state.position_shares = shares


# ---------------------------------------------------------------------------
# Decision-time handler
# ---------------------------------------------------------------------------

def handle_decision(
    data_client,
    trading_client,
    state: TradingState,
    bands_df,
    cfg: BTMConfig,
    decision_time: str,
    dry_run: bool,
) -> None:
    """
    Fetch a real-time snapshot, evaluate the BTM signal, and execute any order.

    Uses the Alpaca snapshot endpoint (real-time) rather than the historical
    bars endpoint (which only reflects data up to the previous day during
    market hours).
    """
    import pandas as pd

    try:
        snap = fetch_snapshot(data_client, cfg.symbol)
    except Exception as exc:
        log.warning("Snapshot unavailable at %s (%s) — skipping decision.", decision_time, exc)
        return

    price = snap["price"]
    vwap  = snap["vwap"]

    # Bands for this exact minute
    if decision_time in bands_df.index:
        ub = float(bands_df.loc[decision_time, "UB"])
        lb = float(bands_df.loc[decision_time, "LB"])
    else:
        # Fall back to nearest available time
        times = list(bands_df.index)
        closest = min(times, key=lambda t: abs(
            pd.Timestamp(f"2000-01-01 {t}") - pd.Timestamp(f"2000-01-01 {decision_time}")
        ))
        ub = float(bands_df.loc[closest, "UB"])
        lb = float(bands_df.loc[closest, "LB"])
        log.warning("Exact band time %s not found; using %s", decision_time, closest)

    new_pos = decide_position(price, ub, lb, vwap, state.position)

    log.info(
        "%s │ price=$%.2f  UB=$%.2f  LB=$%.2f  VWAP=$%.2f"
        "  pos %+d→%+d",
        decision_time, price, ub, lb, vwap, state.position, new_pos,
    )

    execute_position_change(trading_client, state, new_pos, dry_run)


# ---------------------------------------------------------------------------
# Preflight check (runs any time, never submits orders)
# ---------------------------------------------------------------------------

def run_preflight(cfg: BTMConfig) -> None:
    """
    Verify end-to-end connectivity and configuration without trading.

    Checks (in order):
      1. Alpaca client authentication (data + trading)
      2. Market clock reachability
      3. Historical minute-bar download
      4. Account info / portfolio value
      5. Real-time snapshot (warns gracefully if market is closed)
      6. Sigma + band computation
      7. Leverage / ETF / share-sizing
      8. Chart generation
      9. Morning email delivery

    Safe to run at any time of day on any day of the week.
    """
    import pandas as pd

    load_dotenv()
    log.info("=== BTM Preflight Check  session=%s  symbol=%s ===",
             cfg.session, cfg.symbol)

    # ── 1. Clients ──────────────────────────────────────────────────────────
    data_client    = make_data_client(cfg.session)
    trading_client = make_trading_client(cfg.session)
    log.info("[OK] Alpaca clients created")

    # ── 2. Market clock ──────────────────────────────────────────────────────
    clock = get_market_clock(trading_client)
    log.info("[OK] Market clock  is_open=%s  next_open=%s",
             clock["is_open"],
             clock["next_open"].strftime("%Y-%m-%d %H:%M %Z"))

    # ── 3. Historical data ───────────────────────────────────────────────────
    hist_start = (date.today() - timedelta(days=30)).strftime("%Y-%m-%d")
    hist_end   = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    log.info("Fetching historical bars %s → %s …", hist_start, hist_end)
    hist_df = fetch_minute_bars(data_client, cfg.symbol, hist_start, hist_end)
    if hist_df.empty:
        raise RuntimeError("No historical data returned — check symbol and credentials.")
    n_days = len(set(hist_df.index.date))
    log.info("[OK] Historical data  %d bars  %d trading days", len(hist_df), n_days)

    # ── 4. Account ───────────────────────────────────────────────────────────
    acct = get_account_info(trading_client)
    log.info("[OK] Account  portfolio_value=$%.2f  cash=$%.2f",
             acct["portfolio_value"], acct["cash"])

    # ── 5. Snapshot ──────────────────────────────────────────────────────────
    snap = None
    try:
        snap = fetch_snapshot(data_client, cfg.symbol)
        ts_ny = pd.Timestamp(snap["timestamp"]).tz_convert(NY_TZ)
        log.info("[OK] Snapshot  price=$%.2f  vwap=$%.2f  bar_ts=%s",
                 snap["price"], snap["vwap"], ts_ny.strftime("%H:%M:%S"))
    except Exception as exc:
        log.warning("[--] Snapshot unavailable (market closed?): %s", exc)

    # ── 6. Bands ─────────────────────────────────────────────────────────────
    yesterday_close = float(hist_df["close"].iloc[-1])
    # Use last snapshot price as a stand-in for today's open when available;
    # fall back to yesterday's close when called outside market hours.
    today_open_est = snap["price"] if snap else yesterday_close

    sigma_series = compute_sigma_for_today(hist_df, cfg.lookback_days)
    bands_df     = compute_bands_for_today(
        sigma_series, today_open_est, yesterday_close, cfg.vm
    )
    log.info("[OK] Bands computed  %d decision times  "
             "UB@10:00=$%.2f  LB@10:00=$%.2f",
             len(bands_df),
             float(bands_df["UB"].iloc[0]),
             float(bands_df["LB"].iloc[0]))

    # ── 7. Sizing ─────────────────────────────────────────────────────────────
    daily_close    = hist_df.groupby(hist_df.index.date)["close"].last()
    daily_close_ts = pd.Series(
        daily_close.values,
        index=pd.to_datetime(daily_close.index).tz_localize(NY_TZ_STR),
    )
    vol      = compute_daily_vol(daily_close_ts, cfg.lookback_days)
    leverage = min(cfg.leverage_cap, cfg.target_daily_vol / max(vol, 1e-6))
    long_etf, short_etf, etf_mult = select_etfs(leverage)
    shares   = compute_etf_shares(
        acct["portfolio_value"], leverage, today_open_est, etf_mult
    )
    log.info("[OK] Sizing  vol=%.2f%%  leverage=%.2f×  long=%s  short=%s  shares=%d",
             vol * 100, leverage, long_etf, short_etf, shares)

    # ── 8. Chart ──────────────────────────────────────────────────────────────
    from btm.chart import plot_morning_bands
    today_str = date.today().strftime("%Y-%m-%d")
    png = plot_morning_bands(
        bands_df       = bands_df,
        today_open     = today_open_est,
        yesterday_close= yesterday_close,
        today_bars     = None,
        date_str       = today_str,
        symbol         = cfg.symbol,
        leverage       = leverage,
        long_etf       = long_etf,
        short_etf      = short_etf,
        daily_vol_pct  = vol * 100,
    )
    log.info("[OK] Chart generated  %d bytes", len(png))

    # ── 9. Email ──────────────────────────────────────────────────────────────
    from btm.email_report import send_morning_email
    send_morning_email(
        chart_png    = png,
        date_str     = today_str,
        symbol       = cfg.symbol,
        leverage     = leverage,
        long_etf     = long_etf,
        short_etf    = short_etf,
        daily_vol_pct= vol * 100,
    )
    recipient = os.getenv("RECIPIENT_EMAIL", "(RECIPIENT_EMAIL not set)")
    log.info("[OK] Email sent → %s", recipient)

    log.info("=== Preflight passed — all systems go ===")


# ---------------------------------------------------------------------------
# Main trading-day loop
# ---------------------------------------------------------------------------

def run_trading_day(cfg: BTMConfig, dry_run: bool = False) -> None:
    load_dotenv()

    data_client    = make_data_client(cfg.session)
    trading_client = make_trading_client(cfg.session)

    # ── 1. Is today a trading day? ────────────────────────────────────────
    if not is_trading_day(trading_client):
        log.info("Today is not a trading day. Exiting.")
        return

    today_str = date.today().strftime("%Y-%m-%d")
    log.info("=== BTM Live Trading  %s  session=%s  symbol=%s%s ===",
             today_str, cfg.session, cfg.symbol, "  [DRY RUN]" if dry_run else "")

    # ── 2. Fetch historical minute bars (last ~25 calendar days) ──────────
    hist_start = (date.today() - timedelta(days=30)).strftime("%Y-%m-%d")
    hist_end   = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    log.info("Fetching historical bars %s → %s …", hist_start, hist_end)
    hist_df = fetch_minute_bars(data_client, cfg.symbol, hist_start, hist_end)

    if hist_df.empty:
        log.error("No historical data returned. Cannot compute bands. Exiting.")
        return

    log.info("  %d bars, %d trading days.", len(hist_df), len(set(hist_df.index.date)))

    # Yesterday's closing price
    yesterday_close = float(hist_df["close"].iloc[-1])

    # ── 3. Wait for 09:30 open ────────────────────────────────────────────
    if hhmm_now() < "09:30":
        log.info("Pre-market — sleeping until 09:30.")
        sleep_until("09:30")

    today_open = wait_for_open_bar(data_client, cfg.symbol)

    # ── 4. Compute day state (bands, leverage, sizing) ────────────────────
    state = TradingState()
    state.today_open      = today_open
    state.yesterday_close = yesterday_close
    state._trading_client = trading_client  # stored for compute_day_state

    bands_df = compute_day_state(data_client, cfg.symbol, hist_df, cfg, state)

    # ── 5. Reconcile with any existing Alpaca positions ───────────────────
    #    (handles restarts mid-day gracefully)
    existing = get_all_positions(trading_client)
    for pos in existing:
        sym = pos["symbol"]
        if sym == state.long_etf:
            state.position        = 1
            state.position_ticker = sym
            state.position_shares = pos["qty"]
            log.info("Detected existing LONG %d %s from Alpaca.", pos["qty"], sym)
        elif sym == state.short_etf:
            state.position        = -1
            state.position_ticker = sym
            state.position_shares = pos["qty"]
            log.info("Detected existing SHORT %d %s from Alpaca.", pos["qty"], sym)

    log.info("Initial state: %s", state)

    # ── 6. Send morning email ─────────────────────────────────────────────
    try:
        from btm.chart import plot_morning_bands
        from btm.email_report import send_morning_email

        png = plot_morning_bands(
            bands_df       = bands_df,
            today_open     = state.today_open,
            yesterday_close= state.yesterday_close,
            today_bars     = None,   # sent at open; no bar history needed yet
            date_str       = today_str,
            symbol         = cfg.symbol,
            leverage       = state.leverage,
            long_etf       = state.long_etf,
            short_etf      = state.short_etf,
            daily_vol_pct  = state.daily_vol_pct,
        )
        send_morning_email(
            chart_png     = png,
            date_str      = today_str,
            symbol        = cfg.symbol,
            leverage      = state.leverage,
            long_etf      = state.long_etf,
            short_etf     = state.short_etf,
            daily_vol_pct = state.daily_vol_pct,
        )
    except Exception as exc:
        log.warning("Morning email failed (non-fatal): %s", exc)

    # ── 7. Main decision loop ─────────────────────────────────────────────
    log.info("Entering decision loop. Decision times: %s", ", ".join(DECISION_TIMES))

    for decision_time in DECISION_TIMES:
        now = hhmm_now()

        # Skip past decision times (e.g. if script started late)
        if now > decision_time:
            log.debug("Skipping past decision time %s (now %s)", decision_time, now)
            continue

        # Sleep until this decision time
        log.info("Next decision: %s  (now %s).", decision_time, now)
        sleep_until(decision_time)

        # Force-flat check
        if decision_time >= CLOSE_POSITIONS_TIME:
            break

        handle_decision(
            data_client, trading_client, state, bands_df, cfg,
            decision_time, dry_run,
        )

    # ── 8. Force-close at 15:50 ───────────────────────────────────────────
    log.info("Approaching close time — sleeping until %s.", CLOSE_POSITIONS_TIME)
    sleep_until(CLOSE_POSITIONS_TIME)

    if state.position != 0:
        log.info("Force-closing position at %s.", CLOSE_POSITIONS_TIME)
        if not dry_run:
            close_all_positions(trading_client)
        state.position        = 0
        state.position_ticker = ""
        state.position_shares = 0
    else:
        log.info("Already flat at close time. Nothing to close.")

    log.info("=== Trading day complete ===")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BTM live trading (one full trading day)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--session", default="paper", choices=["paper", "live"],
                   help="Alpaca account type")
    p.add_argument("--symbol",  default="SPY",
                   help="SPY-equivalent symbol to trade signals on")
    p.add_argument("--vm",      type=float, default=1.0,
                   help="Volatility multiplier")
    p.add_argument("--lookback", type=int,  default=14,
                   help="Lookback days for σ and daily-vol")
    p.add_argument("--leverage-cap", type=float, default=4.0,
                   help="Maximum leverage")
    p.add_argument("--target-vol",   type=float, default=0.02,
                   help="Target daily volatility for sizing")
    p.add_argument("--dry-run", action="store_true",
                   help="Log decisions but do not submit orders")
    p.add_argument("--preflight", action="store_true",
                   help="Verify credentials, data, bands, and email; then exit "
                        "(safe to run any time, never submits orders)")
    p.add_argument("--log-file", default=None,
                   help="Append logs to this file (in addition to stdout)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.log_file:
        file_handler = logging.FileHandler(args.log_file, mode="a")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                               datefmt="%Y-%m-%d %H:%M:%S")
        )
        logging.getLogger().addHandler(file_handler)

    cfg = BTMConfig(
        symbol         = args.symbol,
        session        = args.session,
        lookback_days  = args.lookback,
        vm             = args.vm,
        leverage_cap   = args.leverage_cap,
        target_daily_vol = args.target_vol,
    )

    try:
        if args.preflight:
            run_preflight(cfg)
            return 0
        run_trading_day(cfg, dry_run=args.dry_run)
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
        return 130
    except Exception as exc:
        log.exception("Unhandled exception in trading loop: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
