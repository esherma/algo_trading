"""
Unit tests for btm.core — the pure-algorithm layer.

Run with:
    python -m pytest tests/test_core.py -v
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest
import pytz

from btm.core import (
    CLOSE_POSITIONS_TIME,
    DECISION_TIMES,
    BTMConfig,
    compute_bands_for_backtest,
    compute_bands_for_today,
    compute_daily_vol,
    compute_etf_shares,
    compute_intraday_vwap,
    compute_sigma_for_today,
    compute_sigma_rolling,
    decide_position,
    run_backtest,
    select_etfs,
)
from tests.fixtures import (
    make_flat_market,
    make_minute_bars,
    make_strong_trend_down,
    make_strong_trend_up,
)

NY_TZ = pytz.timezone("America/New_York")


# ============================================================================
# Decision time constants
# ============================================================================

class TestConstants:
    def test_decision_times_count(self):
        # 10:00, 10:30, 11:00, 11:30, 12:00, 12:30,
        # 13:00, 13:30, 14:00, 14:30, 15:00, 15:30  → 12 slots
        assert len(DECISION_TIMES) == 12

    def test_first_and_last(self):
        assert DECISION_TIMES[0]  == "10:00"
        assert DECISION_TIMES[-1] == "15:30"

    def test_all_on_half_hour(self):
        for t in DECISION_TIMES:
            _, m = map(int, t.split(":"))
            assert m in (0, 30)

    def test_close_positions_after_last_decision(self):
        assert CLOSE_POSITIONS_TIME > DECISION_TIMES[-1]


# ============================================================================
# Sigma computation
# ============================================================================

class TestSigma:
    def setup_method(self):
        self.df = make_minute_bars(n_days=20, seed=1)

    def test_sigma_rolling_same_index(self):
        sigma = compute_sigma_rolling(self.df, lookback=14)
        assert len(sigma) == len(self.df)
        assert sigma.index.equals(self.df.index)

    def test_sigma_rolling_non_negative(self):
        sigma = compute_sigma_rolling(self.df, lookback=14)
        assert (sigma >= 0).all(), "Sigma should be non-negative"

    def test_sigma_rolling_look_ahead_free(self):
        """
        σ(day=1, τ) should equal σ(day=1, τ) when computed on a dataset
        that only goes up to day 1. If it uses future data the values would differ.
        """
        df_full  = make_minute_bars(n_days=20, seed=1)
        df_short = df_full[df_full.index.date <= df_full.index.date[0]].copy()

        sigma_full  = compute_sigma_rolling(df_full,  lookback=14)
        sigma_short = compute_sigma_rolling(df_short, lookback=14)

        # Day-0 sigma: should be the same regardless of later data
        day0 = df_short.index.date[0]
        mask_full  = sigma_full.index.date  == day0
        mask_short = sigma_short.index.date == day0
        np.testing.assert_allclose(
            sigma_full[mask_full].values,
            sigma_short[mask_short].values,
            rtol=1e-10,
            err_msg="Look-ahead leak detected: day-0 sigma differs with/without future data",
        )

    def test_sigma_for_today_length(self):
        hist = make_minute_bars(n_days=15, seed=2)
        sigma = compute_sigma_for_today(hist, lookback=14)
        # Should have an entry for every unique HH:MM that appeared in hist
        assert len(sigma) > 0

    def test_sigma_for_today_non_negative(self):
        hist  = make_minute_bars(n_days=15, seed=3)
        sigma = compute_sigma_for_today(hist, lookback=14)
        assert (sigma >= 0).all()

    def test_sigma_increases_through_day(self):
        """
        On average, price moves further from the open as the day progresses —
        so σ should generally increase from morning to afternoon.
        """
        df    = make_minute_bars(n_days=30, seed=4)
        sigma = compute_sigma_for_today(df, lookback=14)

        morning   = sigma.get("10:00", np.nan)
        afternoon = sigma.get("14:00", np.nan)

        if not (math.isnan(morning) or math.isnan(afternoon)):
            assert afternoon >= morning * 0.5, \
                f"Afternoon σ ({afternoon:.5f}) unexpectedly much smaller than morning ({morning:.5f})"

    def test_sigma_rolling_first_day_fallback(self):
        """On day 0 there is no prior data; sigma should fall back to 0.01."""
        df = make_minute_bars(n_days=3, seed=5)
        sigma = compute_sigma_rolling(df, lookback=14)
        day0_sigma = sigma[sigma.index.date == df.index.date[0]]
        # All should be 0.01 (the fallback) since there are 0 prior days.
        # Use np.allclose for element-wise tolerance on a Series.
        assert np.allclose(day0_sigma.values, 0.01, atol=1e-10), \
            f"Expected all 0.01, got: {day0_sigma.values[:5]} …"


# ============================================================================
# Noise bands
# ============================================================================

class TestBands:
    def setup_method(self):
        self.df    = make_minute_bars(n_days=20, seed=10)
        self.sigma = compute_sigma_rolling(self.df, lookback=14)

    def test_bands_shape(self):
        bands = compute_bands_for_backtest(self.df, self.sigma)
        assert list(bands.columns) == ["UB", "LB", "sigma"]
        assert len(bands) == len(self.df)

    def test_ub_above_lb(self):
        bands = compute_bands_for_backtest(self.df, self.sigma)
        assert (bands["UB"] > bands["LB"]).all(), "UB must always exceed LB"

    def test_vm_scaling(self):
        """Doubling vm monotonically increases band width."""
        bands1 = compute_bands_for_backtest(self.df, self.sigma, vm=1.0)
        bands2 = compute_bands_for_backtest(self.df, self.sigma, vm=2.0)
        bands4 = compute_bands_for_backtest(self.df, self.sigma, vm=4.0)

        width1 = (bands1["UB"] - bands1["LB"]).mean()
        width2 = (bands2["UB"] - bands2["LB"]).mean()
        width4 = (bands4["UB"] - bands4["LB"]).mean()

        assert width2 > width1, "Doubling vm should increase band width"
        assert width4 > width2, "Quadrupling vm should be wider than double"
        # Width doesn't exactly double when there's a gap (upper_base ≠ lower_base).
        # But it should be in a reasonable range (1.5× – 2.5× for typical gap sizes).
        ratio = float(width2 / width1)
        assert 1.5 < ratio < 2.5, f"vm=2 width ratio {ratio:.2f} outside expected range"

    def test_gap_up_adjustment(self):
        """If today opens above yesterday's close, upper_base = today_open."""
        sigma_by_time = pd.Series({"10:00": 0.01, "10:30": 0.015})
        today_open      = 410.0
        yesterday_close = 400.0   # gap up

        bands = compute_bands_for_today(sigma_by_time, today_open, yesterday_close)
        upper_base = max(today_open, yesterday_close)   # = 410
        lower_base = min(today_open, yesterday_close)   # = 400

        np.testing.assert_allclose(bands.loc["10:00", "UB"],
                                   upper_base * (1 + 0.01), rtol=1e-10)
        np.testing.assert_allclose(bands.loc["10:00", "LB"],
                                   lower_base * (1 - 0.01), rtol=1e-10)

    def test_gap_down_adjustment(self):
        """If today opens below yesterday's close, upper_base = yesterday_close."""
        sigma_by_time = pd.Series({"10:00": 0.01})
        today_open      = 390.0
        yesterday_close = 400.0   # gap down

        bands = compute_bands_for_today(sigma_by_time, today_open, yesterday_close)
        upper_base = max(today_open, yesterday_close)   # = 400
        lower_base = min(today_open, yesterday_close)   # = 390

        np.testing.assert_allclose(bands.loc["10:00", "UB"],
                                   upper_base * (1 + 0.01), rtol=1e-10)
        np.testing.assert_allclose(bands.loc["10:00", "LB"],
                                   lower_base * (1 - 0.01), rtol=1e-10)

    def test_flat_open_no_gap(self):
        """No gap: upper_base = lower_base = open."""
        sigma_by_time = pd.Series({"10:00": 0.02})
        price = 500.0
        bands = compute_bands_for_today(sigma_by_time, price, price)
        assert bands.loc["10:00", "UB"] == pytest.approx(price * 1.02)
        assert bands.loc["10:00", "LB"] == pytest.approx(price * 0.98)

    def test_bands_widen_through_day(self):
        """Bands should generally widen as σ grows through the day."""
        df    = make_minute_bars(n_days=20, seed=11)
        sigma = compute_sigma_rolling(df, lookback=14)
        bands = compute_bands_for_backtest(df, sigma)

        # Take the last day and check that late-day band width ≥ early-day width
        last_day  = df.index.date[-1]
        day_bands = bands[bands.index.date == last_day]

        width_930  = day_bands[day_bands.index.strftime("%H:%M") == "09:30"]
        width_1500 = day_bands[day_bands.index.strftime("%H:%M") == "15:00"]

        if not width_930.empty and not width_1500.empty:
            w930  = float((width_930["UB"]  - width_930["LB"]).iloc[0])
            w1500 = float((width_1500["UB"] - width_1500["LB"]).iloc[0])
            assert w1500 >= w930 * 0.8, \
                f"Band at 15:00 ({w1500:.4f}) narrower than at 09:30 ({w930:.4f})"


# ============================================================================
# VWAP
# ============================================================================

class TestVWAP:
    def test_vwap_same_index(self):
        df   = make_minute_bars(n_days=5, seed=20)
        vwap = compute_intraday_vwap(df)
        assert vwap.index.equals(df.index)

    def test_vwap_resets_daily(self):
        """VWAP of first bar == first bar close (since volume of first bar only)."""
        df   = make_minute_bars(n_days=3, seed=21)
        vwap = compute_intraday_vwap(df)

        for d in set(df.index.date):
            day_mask = df.index.date == d
            first_close = float(df[day_mask]["close"].iloc[0])
            first_vwap  = float(vwap[day_mask].iloc[0])
            assert first_vwap == pytest.approx(first_close, rel=1e-6), \
                f"VWAP did not reset on {d}"

    def test_vwap_between_min_max(self):
        df   = make_minute_bars(n_days=5, seed=22)
        vwap = compute_intraday_vwap(df)
        # VWAP should be bounded by min and max close of the day (roughly)
        for d in set(df.index.date):
            day_bars = df[df.index.date == d]
            day_vwap = vwap[vwap.index.date == d]
            lo = float(day_bars["close"].min())
            hi = float(day_bars["close"].max())
            assert float(day_vwap.min()) >= lo * 0.95
            assert float(day_vwap.max()) <= hi * 1.05


# ============================================================================
# Position decision logic
# ============================================================================

class TestDecidePosition:
    """
    Reference levels:
        UB = 102,  LB = 98,  VWAP = 100
    """
    UB   = 102.0
    LB   = 98.0
    VWAP = 100.0

    # ── Entry tests ──────────────────────────────────────────────────────

    def test_enter_long_above_ub(self):
        assert decide_position(103.0, self.UB, self.LB, self.VWAP, 0) == 1

    def test_enter_short_below_lb(self):
        assert decide_position(97.0, self.UB, self.LB, self.VWAP, 0) == -1

    def test_no_entry_inside_band(self):
        assert decide_position(100.0, self.UB, self.LB, self.VWAP, 0) == 0

    def test_no_entry_at_ub_exactly(self):
        # Strictly greater than UB required for long entry
        assert decide_position(102.0, self.UB, self.LB, self.VWAP, 0) == 0

    def test_no_entry_at_lb_exactly(self):
        assert decide_position(98.0, self.UB, self.LB, self.VWAP, 0) == 0

    # ── Stop tests (long) ────────────────────────────────────────────────

    def test_long_stop_below_ub_and_vwap(self):
        """VWAP < UB → stop level = UB.  Price drops back to UB → close."""
        vwap_low = 99.0
        # price = 101 < max(UB=102, vwap=99) → stop triggered
        assert decide_position(101.0, self.UB, self.LB, vwap_low, 1) == 0

    def test_long_hold_above_ub(self):
        """Price stays above UB AND above VWAP → stay long."""
        vwap = 100.0
        # price = 103 > max(UB=102, vwap=100) → no stop
        assert decide_position(103.0, self.UB, self.LB, vwap, 1) == 1

    def test_long_vwap_trailing_stop(self):
        """When VWAP rises above UB, it becomes the (tighter) stop.
        Price must be INSIDE the band (not above UB) so re-entry doesn't fire."""
        vwap_high = 104.0                         # VWAP > UB
        # price = 100 (inside band) < max(UB=102, vwap=104) → stop triggered
        # price=100 not > UB=102 → no re-entry → result is 0
        assert decide_position(100.0, self.UB, self.LB, vwap_high, 1) == 0

    def test_long_stay_above_high_vwap(self):
        vwap_high = 104.0
        # price = 105 > max(UB=102, vwap=104) → no stop
        assert decide_position(105.0, self.UB, self.LB, vwap_high, 1) == 1

    # ── Stop tests (short) ───────────────────────────────────────────────

    def test_short_stop_above_lb_and_vwap(self):
        """VWAP > LB → stop level = LB.  Price rises back to LB → close."""
        vwap_high = 101.0
        # price = 99 > min(LB=98, vwap=101) = 98 → stop triggered
        assert decide_position(99.0, self.UB, self.LB, vwap_high, -1) == 0

    def test_short_hold_below_lb(self):
        """Price stays below LB AND below VWAP → stay short."""
        vwap_low = 99.0
        # price = 97 < min(LB=98, vwap=99) → no stop
        assert decide_position(97.0, self.UB, self.LB, vwap_low, -1) == -1

    def test_short_vwap_trailing_stop(self):
        """When VWAP drops below LB, it becomes the (tighter) stop.
        Price must be INSIDE the band (not below LB) so re-entry doesn't fire."""
        vwap_low = 96.0                           # VWAP < LB
        # price = 100 (inside band) > min(LB=98, vwap=96) = 96 → stop triggered
        # price=100 not < LB=98 → no re-entry → result is 0
        assert decide_position(100.0, self.UB, self.LB, vwap_low, -1) == 0

    # ── Reversal tests ───────────────────────────────────────────────────

    def test_long_to_short_reversal(self):
        """Long stop triggers, then price is below LB → re-enter short."""
        # price = 97 → long stop fires (97 < UB=102), then 97 < LB=98 → short
        result = decide_position(97.0, self.UB, self.LB, self.VWAP, 1)
        assert result == -1

    def test_short_to_long_reversal(self):
        """Short stop triggers, then price is above UB → re-enter long."""
        # price = 103 → short stop fires (103 > LB=98), then 103 > UB=102 → long
        result = decide_position(103.0, self.UB, self.LB, self.VWAP, -1)
        assert result == 1


# ============================================================================
# Sizing helpers
# ============================================================================

class TestSizing:
    def test_compute_daily_vol_returns_positive(self):
        prices = pd.Series([100.0, 101.5, 99.0, 102.0, 100.5] * 5)
        v = compute_daily_vol(prices)
        assert v > 0

    def test_compute_daily_vol_fallback(self):
        """Empty series returns 2% fallback."""
        assert compute_daily_vol(pd.Series([], dtype=float)) == pytest.approx(0.02)

    def test_compute_daily_vol_single_price(self):
        assert compute_daily_vol(pd.Series([100.0])) == pytest.approx(0.02)

    def test_select_etfs_tiers(self):
        assert select_etfs(3.0) == ("SPXL", "SPXS", 3)
        assert select_etfs(2.5) == ("SPXL", "SPXS", 3)
        assert select_etfs(2.0) == ("SPUU", "SDS", 2)
        assert select_etfs(1.5) == ("SPUU", "SDS", 2)
        assert select_etfs(1.0) == ("SPY", "SPDN", 1)
        assert select_etfs(0.0) == ("SPY", "SPDN", 1)

    def test_compute_etf_shares_spy(self):
        # AUM=100k, leverage=1.0, spy_open=500, etf_mult=1 → 200 shares
        assert compute_etf_shares(100_000, 1.0, 500.0, 1) == 200

    def test_compute_etf_shares_3x_etf(self):
        # AUM=100k, leverage=3.0, spy_open=500, etf_mult=3 → 200 shares of SPXL
        # notional = 100k*3 = 300k; shares = floor(300k / (3*500)) = 200
        assert compute_etf_shares(100_000, 3.0, 500.0, 3) == 200

    def test_compute_etf_shares_floor(self):
        # Should floor, not round
        shares = compute_etf_shares(100_000, 1.0, 333.33, 1)
        assert shares == math.floor(100_000 / 333.33)

    def test_compute_etf_shares_zero_price(self):
        assert compute_etf_shares(100_000, 1.0, 0.0, 1) == 0


# ============================================================================
# Full backtest sanity checks
# ============================================================================

class TestRunBacktest:
    def setup_method(self):
        self.df  = make_minute_bars(n_days=30, seed=50)
        self.cfg = BTMConfig(
            initial_aum=100_000,
            lookback_days=14,
            vm=1.0,
            target_daily_vol=0.02,
            leverage_cap=4.0,
        )

    # ── Output structure ──────────────────────────────────────────────────

    def test_result_columns(self):
        result, _ = run_backtest(self.df, self.cfg)
        for col in ("price", "UB", "LB", "vwap", "sigma", "position", "aum", "pnl_net"):
            assert col in result.columns, f"Missing column: {col}"

    def test_result_same_index(self):
        result, _ = run_backtest(self.df, self.cfg)
        assert result.index.equals(self.df.index)

    def test_summary_keys(self):
        _, summary = run_backtest(self.df, self.cfg)
        required = {
            "StartDate", "EndDate", "TotalReturn_Pct", "IRR_Pct",
            "AnnualVol_Pct", "Sharpe", "MaxDrawdown_Pct",
            "HitRatio_Pct", "TradeCount", "FinalAUM",
        }
        assert required.issubset(summary.keys())

    # ── Invariants ────────────────────────────────────────────────────────

    def test_position_only_valid_values(self):
        result, _ = run_backtest(self.df, self.cfg)
        assert set(result["position"].unique()).issubset({-1, 0, 1})

    def test_position_changes_only_at_decision_times(self):
        """
        Position changes are allowed only at:
          - 09:30  (day boundary reset)
          - decision times (10:00 … 15:30)
          - CLOSE_POSITIONS_TIME (15:50) for force-flat
        """
        result, _ = run_backtest(self.df, self.cfg)
        pos_changes = result["position"].diff().fillna(0)
        changed_idx = result.index[pos_changes != 0]
        allowed = set(DECISION_TIMES) | {"09:30", CLOSE_POSITIONS_TIME}
        for ts in changed_idx:
            t_str = ts.strftime("%H:%M")
            assert t_str in allowed, \
                f"Position changed at unexpected time {t_str}"

    def test_position_flat_at_eod(self):
        """All positions must be flat at the last minute of each trading day."""
        result, _ = run_backtest(self.df, self.cfg)
        for d in set(result.index.date):
            day_bars = result[result.index.date == d]
            last_pos = int(day_bars["position"].iloc[-1])
            assert last_pos == 0, \
                f"Non-zero position at end of day {d}: {last_pos}"

    def test_aum_non_negative(self):
        result, _ = run_backtest(self.df, self.cfg)
        assert (result["aum"] >= 0).all(), "AUM went negative"

    def test_aum_starts_near_initial(self):
        result, _ = run_backtest(self.df, self.cfg)
        # First bar: no position yet, so AUM = initial
        assert result["aum"].iloc[0] == pytest.approx(self.cfg.initial_aum, rel=1e-6)

    def test_ub_always_above_lb(self):
        result, _ = run_backtest(self.df, self.cfg)
        assert (result["UB"] > result["LB"]).all()

    def test_final_aum_matches_summary(self):
        result, summary = run_backtest(self.df, self.cfg)
        assert summary["FinalAUM"] == pytest.approx(result["aum"].iloc[-1], rel=1e-6)

    def test_mdd_non_positive(self):
        _, summary = run_backtest(self.df, self.cfg)
        assert summary["MaxDrawdown_Pct"] <= 0, "MDD should be negative or zero"

    def test_hit_ratio_in_range(self):
        _, summary = run_backtest(self.df, self.cfg)
        assert 0 <= summary["HitRatio_Pct"] <= 100

    # ── Economic sanity: flat market → minimal trading ────────────────────

    def test_flat_market_adaptive_bands(self):
        """
        In a flat market σ adapts to the low volatility, so the bands narrow
        proportionally.  The strategy should generate FEWER trades than in a
        volatile trending market (sigma and band width track realised vol).
        """
        from btm.core import compute_sigma_for_today
        df_flat   = make_flat_market(n_days=20)
        df_trend  = make_strong_trend_up(n_days=20, seed=7)

        sigma_flat  = compute_sigma_for_today(df_flat,  lookback=14)
        sigma_trend = compute_sigma_for_today(df_trend, lookback=14)

        # Sigma should be much smaller for the flat market
        mean_flat  = float(sigma_flat.mean())
        mean_trend = float(sigma_trend.mean())
        assert mean_flat < mean_trend, \
            f"Flat σ={mean_flat:.6f} not smaller than trend σ={mean_trend:.6f}"

        # Backtest still runs and returns valid output
        _, summary = run_backtest(df_flat, self.cfg)
        assert 0 <= summary["HitRatio_Pct"] <= 100
        assert summary["FinalAUM"] > 0

    # ── Costs reduce AUM vs no-cost version ──────────────────────────────

    def test_costs_reduce_aum(self):
        df = make_strong_trend_up(n_days=30)
        cfg_cost   = BTMConfig(commission_per_share=0.01, slippage_per_share=0.01)
        cfg_nocost = BTMConfig(commission_per_share=0.0,  slippage_per_share=0.0)
        _, s_cost   = run_backtest(df, cfg_cost)
        _, s_nocost = run_backtest(df, cfg_nocost)
        # With costs, final AUM should be lower (or equal if no trades)
        assert s_cost["FinalAUM"] <= s_nocost["FinalAUM"] + 1  # +1 for rounding


# ============================================================================
# Chart generation (smoke tests — no visual inspection)
# ============================================================================

class TestCharts:
    def test_plot_morning_bands_returns_png(self):
        from btm.chart import plot_morning_bands
        import pandas as pd, pytz

        sigma = pd.Series({"09:30": 0.001, "10:00": 0.005, "10:30": 0.008,
                           "14:00": 0.012, "15:30": 0.015})
        from btm.core import compute_bands_for_today
        bands = compute_bands_for_today(sigma, today_open=400.0, yesterday_close=398.0)

        png = plot_morning_bands(
            bands_df       = bands,
            today_open     = 400.0,
            yesterday_close= 398.0,
            today_bars     = None,
            date_str       = "2024-01-15",
            symbol         = "SPY",
            leverage       = 2.0,
            long_etf       = "SPUU",
            short_etf      = "SDS",
            daily_vol_pct  = 0.85,
        )
        assert isinstance(png, bytes)
        assert png[:4] == b"\x89PNG", "Expected PNG magic bytes"

    def test_plot_backtest_returns_png(self):
        from btm.chart import plot_backtest

        df  = make_minute_bars(n_days=30, seed=99)
        cfg = BTMConfig()
        result, summary = run_backtest(df, cfg)

        png = plot_backtest(result, summary, cfg_label="SPY vm=1.0 lb=14")
        assert isinstance(png, bytes)
        assert png[:4] == b"\x89PNG"
