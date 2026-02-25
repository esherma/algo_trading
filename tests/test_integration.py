"""
Integration tests — exercise real Alpaca API connections.

These tests are SKIPPED unless the environment variables
ALPACA_PAPER_API_KEY and ALPACA_PAPER_API_SECRET are set.

Alpaca provides a free sandbox environment for testing:
  - Historical data: standard API (same endpoint, paper credentials work)
  - Market clock / calendar: standard trading API
  - Test data stream: wss://stream.data.alpaca.markets/v2/test

Run with:
    python -m pytest tests/test_integration.py -v

Or skip in CI with:
    python -m pytest tests/test_core.py -v   # (unit tests only)
"""

from __future__ import annotations

import os
from datetime import date, timedelta

import pandas as pd
import pytest
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Skip condition: no credentials available
# ---------------------------------------------------------------------------

PAPER_KEY    = os.getenv("ALPACA_PAPER_API_KEY", "")
PAPER_SECRET = os.getenv("ALPACA_PAPER_API_SECRET", "")
HAS_CREDS    = bool(PAPER_KEY and PAPER_SECRET)

skip_no_creds = pytest.mark.skipif(
    not HAS_CREDS,
    reason="ALPACA_PAPER_API_KEY / ALPACA_PAPER_API_SECRET not set",
)


# ---------------------------------------------------------------------------
# Data layer integration
# ---------------------------------------------------------------------------

@skip_no_creds
class TestDataIntegration:
    """
    Fetch small slices of real Alpaca data and verify shape / dtypes.
    """

    @pytest.fixture(scope="class")
    def data_client(self):
        from btm.data import make_data_client
        return make_data_client("paper")

    @pytest.fixture(scope="class")
    def trading_client(self):
        from btm.data import make_trading_client
        return make_trading_client("paper")

    def test_fetch_minute_bars_returns_dataframe(self, data_client):
        from btm.data import fetch_minute_bars

        # Use a fixed recent period so the test is deterministic
        df = fetch_minute_bars(data_client, "SPY", "2024-01-02", "2024-01-05")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert set(["open", "high", "low", "close", "volume"]).issubset(df.columns)

    def test_minute_bars_in_market_hours(self, data_client):
        from btm.data import fetch_minute_bars

        df = fetch_minute_bars(data_client, "SPY", "2024-01-02", "2024-01-03")
        if df.empty:
            pytest.skip("No data returned — may be a holiday")

        for ts in df.index:
            t = ts.strftime("%H:%M")
            assert "09:30" <= t <= "16:00", f"Bar outside market hours: {t}"

    def test_minute_bars_timezone(self, data_client):
        from btm.data import fetch_minute_bars

        df = fetch_minute_bars(data_client, "SPY", "2024-01-02", "2024-01-03")
        if df.empty:
            pytest.skip("No data returned")

        assert df.index.tz is not None
        assert "New_York" in str(df.index.tz) or "Eastern" in str(df.index.tz)

    def test_fetch_multiple_days(self, data_client):
        from btm.data import fetch_minute_bars

        df = fetch_minute_bars(data_client, "SPY", "2024-01-02", "2024-01-12")
        days = set(df.index.date)
        assert len(days) >= 5, f"Expected ≥5 trading days, got {len(days)}"

    def test_fetch_daily_bars(self, data_client):
        from btm.data import fetch_daily_bars

        df = fetch_daily_bars(data_client, "SPY", "2024-01-02", "2024-01-12")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "close" in df.columns

    def test_market_clock(self, trading_client):
        from btm.data import get_market_clock

        clock = get_market_clock(trading_client)
        assert "is_open" in clock
        assert "next_open" in clock
        assert isinstance(clock["is_open"], bool)

    def test_is_trading_day_or_not(self, trading_client):
        from btm.data import is_trading_day

        # Should return a bool without raising
        result = is_trading_day(trading_client)
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Algorithm integration: backtest on real SPY data
# ---------------------------------------------------------------------------

@skip_no_creds
class TestBacktestOnRealData:
    """
    Run a short backtest on real Alpaca data and verify all invariants hold.
    """

    @pytest.fixture(scope="class")
    def real_result(self):
        from btm.core import BTMConfig, run_backtest
        from btm.data import fetch_minute_bars, make_data_client

        client = make_data_client("paper")
        df = fetch_minute_bars(client, "SPY", "2024-01-02", "2024-03-29")

        if len(set(df.index.date)) < 16:
            pytest.skip("Insufficient trading days for a meaningful test")

        cfg = BTMConfig(lookback_days=14, vm=1.0, leverage_cap=4.0)
        return run_backtest(df, cfg)

    def test_structure(self, real_result):
        result, summary = real_result
        for col in ("price", "UB", "LB", "vwap", "position", "aum", "pnl_net"):
            assert col in result.columns

    def test_position_valid(self, real_result):
        result, _ = real_result
        assert set(result["position"].unique()).issubset({-1, 0, 1})

    def test_eod_flat(self, real_result):
        from btm.core import DECISION_TIMES
        result, _ = real_result
        for d in set(result.index.date):
            day_pos = result[result.index.date == d]["position"]
            assert int(day_pos.iloc[-1]) == 0, f"Not flat at EOD on {d}"

    def test_aum_positive(self, real_result):
        result, _ = real_result
        assert (result["aum"] > 0).all()

    def test_sharpe_is_finite(self, real_result):
        import math
        _, summary = real_result
        assert math.isfinite(summary["Sharpe"])

    def test_mdd_non_positive(self, real_result):
        _, summary = real_result
        assert summary["MaxDrawdown_Pct"] <= 0


# ---------------------------------------------------------------------------
# Live-data REST polling (uses Alpaca test endpoint)
# ---------------------------------------------------------------------------

@skip_no_creds
class TestLiveDataPolling:
    """
    Test fetch_latest_minute_bars against the IEX test data feed.
    Alpaca provides test data at the standard endpoint; during market hours
    you get real bars, outside hours the response may be empty.
    """

    def test_fetch_latest_returns_dataframe(self):
        from btm.data import fetch_latest_minute_bars, make_data_client

        client = make_data_client("paper")
        yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        df = fetch_latest_minute_bars(client, "SPY", f"{yesterday} 09:30")

        # May be empty if yesterday was a holiday; just check it doesn't crash
        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            assert "close" in df.columns
            assert df.index.tz is not None

    def test_fetch_latest_respects_start(self):
        """Returned bars should all be >= the requested start time."""
        from btm.data import fetch_latest_minute_bars, make_data_client

        client = make_data_client("paper")
        start  = "2024-01-03 10:00"
        df     = fetch_latest_minute_bars(client, "SPY", start)

        if df.empty:
            pytest.skip("No data in that window")

        start_ts = pd.Timestamp(start, tz="America/New_York")
        assert (df.index >= start_ts).all(), "Some bars are before the requested start"


# ---------------------------------------------------------------------------
# Account / order helpers (read-only)
# ---------------------------------------------------------------------------

@skip_no_creds
class TestOrderHelpers:
    """Read-only checks — we never submit live orders in tests."""

    @pytest.fixture(scope="class")
    def trading_client(self):
        from btm.data import make_trading_client
        return make_trading_client("paper")

    def test_get_account_info_structure(self, trading_client):
        from btm.orders import get_account_info

        info = get_account_info(trading_client)
        assert "equity" in info
        assert "buying_power" in info
        assert "portfolio_value" in info
        assert info["portfolio_value"] >= 0

    def test_get_all_positions_list(self, trading_client):
        from btm.orders import get_all_positions

        positions = get_all_positions(trading_client)
        assert isinstance(positions, list)
        for p in positions:
            assert "symbol" in p
            assert "qty" in p
            assert p["qty"] >= 0

    def test_get_open_position_nonexistent(self, trading_client):
        """Querying a position we probably don't hold should return None."""
        from btm.orders import get_open_position

        pos = get_open_position(trading_client, "XYZINVALID123")
        assert pos is None
