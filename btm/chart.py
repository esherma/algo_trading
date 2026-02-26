"""
Chart generation for BTM strategy.

Two public functions:
  plot_morning_bands  – noise-band "cone" chart for the morning email.
  plot_backtest       – multi-panel performance summary for a backtest run.

Both return raw PNG bytes so the caller can save or attach them to an email
without touching the filesystem.
"""

from __future__ import annotations

import io
import math
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")          # non-interactive backend, safe in cron/headless
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

NY_TZ = "America/New_York"

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_C = dict(
    price="black",
    ub="#d62728",       # red
    lb="#2ca02c",       # green
    band_fill="#ffffaa",
    vwap="#1f77b4",     # blue
    long_marker="#2ca02c",
    short_marker="#d62728",
    equity="#1f77b4",
    drawdown="#d62728",
    spy="#888888",
)


# ---------------------------------------------------------------------------
# Morning bands chart (emailed before trading starts)
# ---------------------------------------------------------------------------

def plot_morning_bands(
    bands_df: pd.DataFrame,                # index="HH:MM", cols={UB,LB,sigma}
    today_open: float,
    yesterday_close: float,
    today_bars: Optional[pd.DataFrame],    # intraday bars collected so far (may be None)
    date_str: str,                         # "YYYY-MM-DD"
    symbol: str,
    leverage: float,
    long_etf: str,
    short_etf: str,
    daily_vol_pct: float,
) -> bytes:
    """
    Return PNG bytes of a noise-band "cone" chart suitable for the morning email.

    The bands are shown for the full trading day 09:30 – 16:00.
    If *today_bars* is provided (live intraday data so far), the actual price
    is overlaid.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Convert HH:MM index to datetime on today's date for proper x-axis labels
    today = pd.Timestamp(date_str, tz=NY_TZ)

    def _to_dt(hhmm: str) -> pd.Timestamp:
        h, m = map(int, hhmm.split(":"))
        return today.replace(hour=h, minute=m, second=0, microsecond=0)

    times   = [_to_dt(t) for t in bands_df.index]
    ub_vals = bands_df["UB"].values
    lb_vals = bands_df["LB"].values

    # Shaded noise area
    ax.fill_between(times, lb_vals, ub_vals, color=_C["band_fill"], alpha=0.6,
                    zorder=1, label="Noise zone")

    # Band boundary lines
    ax.plot(times, ub_vals, color=_C["ub"], linewidth=1.5, linestyle="--",
            zorder=2, label="Upper band (UB)")
    ax.plot(times, lb_vals, color=_C["lb"], linewidth=1.5, linestyle="--",
            zorder=2, label="Lower band (LB)")

    # Reference lines
    ax.axhline(today_open, color="navy", linewidth=1, linestyle=":",
               alpha=0.7, label=f"Today open  ${today_open:.2f}")
    if abs(yesterday_close - today_open) / today_open > 0.0005:  # show gap only if meaningful
        ax.axhline(yesterday_close, color="purple", linewidth=1, linestyle=":",
                   alpha=0.7, label=f"Prev close  ${yesterday_close:.2f}")

    # Actual intraday price (if available)
    if today_bars is not None and not today_bars.empty:
        ax.plot(today_bars.index, today_bars["close"], color=_C["price"],
                linewidth=2.5, zorder=5, label="Price (actual)")
        # VWAP if we have volume
        if "volume" in today_bars.columns:
            cum_pv  = (today_bars["close"] * today_bars["volume"]).cumsum()
            cum_vol = today_bars["volume"].cumsum()
            vwap    = (cum_pv / cum_vol.replace(0, np.nan)).fillna(today_bars["close"])
            ax.plot(today_bars.index, vwap, color=_C["vwap"], linewidth=1.5,
                    alpha=0.8, zorder=4, label="VWAP")

    # Decision-time vertical grid (faint dashes at each :00/:30)
    for hhmm in bands_df.index:
        h, m = map(int, hhmm.split(":"))
        if m in (0, 30) and h >= 10:
            ax.axvline(_to_dt(hhmm), color="gray", linewidth=0.4,
                       linestyle="--", alpha=0.4, zorder=0)

    # Titles and labels
    gap_pct = (today_open / yesterday_close - 1) * 100
    gap_str = f"  |  Gap: {gap_pct:+.2f}%" if abs(gap_pct) > 0.05 else ""
    ax.set_title(
        f"BTM Trading Plan  —  {symbol}  —  {date_str}{gap_str}\n"
        f"Leverage: {leverage:.2f}×  |  Long: {long_etf}  |  Short: {short_etf}"
        f"  |  σ_SPY (14d): {daily_vol_pct:.2f}%",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Time (NY)", fontsize=11)
    ax.set_ylabel(f"{symbol} Price ($)", fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.85)

    # X-axis formatting: show HH:MM every 30 min
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=9)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Backtest performance chart
# ---------------------------------------------------------------------------

def plot_backtest(
    result_df: pd.DataFrame,
    summary: Dict,
    cfg_label: str,     # short description, e.g. "SPY vm=1.0 lookback=14"
) -> bytes:
    """
    Return PNG bytes of a 4-panel performance chart:
      [0,0] Equity curve vs buy-and-hold
      [0,1] Drawdown
      [1,0] Daily returns distribution
      [1,1] Performance metrics table
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"BTM Strategy — {cfg_label}\n"
        f"{summary['StartDate']} → {summary['EndDate']}",
        fontsize=14, fontweight="bold",
    )

    # ── Daily AUM series ─────────────────────────────────────────────────
    daily_aum = result_df["aum"].groupby(result_df.index.date).last()
    daily_aum.index = pd.to_datetime(daily_aum.index).tz_localize(NY_TZ)
    daily_ret = daily_aum.pct_change().dropna()

    # Buy-and-hold normalised to same initial AUM
    daily_price = result_df["price"].groupby(result_df.index.date).last()
    daily_price.index = pd.to_datetime(daily_price.index).tz_localize(NY_TZ)
    bah = daily_price / daily_price.iloc[0] * daily_aum.iloc[0]

    # ── [0,0] Equity curve ───────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(daily_aum.index, daily_aum.values,  color=_C["equity"], lw=2,
            label=f"BTM (${daily_aum.iloc[-1]:,.0f})")
    ax.plot(bah.index, bah.values, color=_C["spy"], lw=1.5, linestyle="--",
            label=f"SPY B&H (${bah.iloc[-1]:,.0f})")
    ax.set_title("Equity Curve", fontweight="bold")
    ax.set_ylabel("Portfolio Value ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # ── [0,1] Drawdown ───────────────────────────────────────────────────
    ax = axes[0, 1]
    dd = (daily_aum / daily_aum.cummax() - 1.0) * 100
    ax.fill_between(daily_aum.index, dd.values, 0, color=_C["drawdown"], alpha=0.4)
    ax.plot(daily_aum.index, dd.values, color=_C["drawdown"], lw=1)
    ax.set_title("Drawdown", fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # ── [1,0] Daily returns distribution ─────────────────────────────────
    ax = axes[1, 0]
    dr_pct = daily_ret * 100
    ax.hist(dr_pct, bins=60, color=_C["equity"], alpha=0.7, edgecolor="white")
    ax.axvline(float(dr_pct.mean()), color="red", lw=1.5, linestyle="--",
               label=f"Mean {float(dr_pct.mean()):.3f}%")
    ax.axvline(0, color="black", lw=1, alpha=0.5)
    ax.set_title("Daily Returns Distribution", fontweight="bold")
    ax.set_xlabel("Daily Return (%)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # ── [1,1] Metrics table ───────────────────────────────────────────────
    ax = axes[1, 1]
    ax.axis("off")

    rows = [
        ["Total Return",        f"{summary['TotalReturn_Pct']:.1f}%"],
        ["Ann. Return (IRR)",   f"{summary['IRR_Pct']:.1f}%"],
        ["Ann. Volatility",     f"{summary['AnnualVol_Pct']:.1f}%"],
        ["Sharpe Ratio",        f"{summary['Sharpe']:.2f}"],
        ["Max Drawdown",        f"{summary['MaxDrawdown_Pct']:.1f}%"],
        ["Day Hit Ratio",       f"{summary['HitRatio_Pct']:.1f}%"],
        ["Trade Count",         f"{summary['TradeCount']}"],
        ["Avg Trades / Day",    f"{summary['AvgTradesPerDay']:.1f}"],
        ["Final AUM",           f"${summary['FinalAUM']:,.0f}"],
    ]

    tbl = ax.table(
        cellText=rows,
        colLabels=["Metric", "Value"],
        cellLoc="left",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c7bb6")
            cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f0f4f8")
        cell.set_edgecolor("#cccccc")
    ax.set_title("Performance Summary", fontweight="bold", pad=4)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
