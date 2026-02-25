#!/usr/bin/env python3
"""
BTM Backtest CLI
================
Run the Beat-the-Market intraday momentum strategy over any historical period.

Usage examples
--------------
  python run_backtest.py --symbol SPY --start 2020-01-01 --end 2023-12-31
  python run_backtest.py --symbol TQQQ --start 2022-01-01 --vm 1.5
  python run_backtest.py --start 2018-01-01 --no-plot          # skip chart display
  python run_backtest.py --start 2020-01-01 --save-dir outputs  # save CSV + PNG

The script fetches minute bars from Alpaca (requires API credentials in .env),
runs the BTM strategy, prints a summary table, and optionally saves results.
"""

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from btm.core import BTMConfig, run_backtest
from btm.data import fetch_minute_bars, make_data_client


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BTM strategy backtest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol",    default="SPY",   help="Ticker symbol")
    p.add_argument("--start",     required=True,   help="Start date YYYY-MM-DD")
    p.add_argument("--end",       default=None,    help="End date YYYY-MM-DD (default: yesterday)")
    p.add_argument("--session",   default="paper", choices=["paper", "live"],
                   help="Which Alpaca API key set to use")
    p.add_argument("--vm",        type=float, default=1.0,
                   help="Volatility multiplier for band width")
    p.add_argument("--lookback",  type=int,   default=14,
                   help="Days used for σ estimation")
    p.add_argument("--leverage-cap", type=float, default=4.0,
                   help="Maximum leverage")
    p.add_argument("--initial-aum",  type=float, default=100_000.0,
                   help="Starting portfolio value ($)")
    p.add_argument("--commission",   type=float, default=0.0035,
                   help="Commission per share ($)")
    p.add_argument("--slippage",     type=float, default=0.001,
                   help="Slippage per share ($)")
    p.add_argument("--save-dir",  default=None,
                   help="Directory to save CSV, JSON summary, and PNG chart")
    p.add_argument("--no-plot",   action="store_true",
                   help="Skip chart generation (faster; useful in CI/headless)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Pretty summary printer
# ---------------------------------------------------------------------------

def print_summary(summary: dict, symbol: str) -> None:
    print()
    print("=" * 52)
    print(f"  BTM Backtest Results — {symbol}")
    print("=" * 52)
    rows = [
        ("Period",            f"{summary['StartDate']} → {summary['EndDate']}"),
        ("Total Return",      f"{summary['TotalReturn_Pct']:+.2f}%"),
        ("Ann. Return (IRR)", f"{summary['IRR_Pct']:+.2f}%"),
        ("Ann. Volatility",   f"{summary['AnnualVol_Pct']:.2f}%"),
        ("Sharpe Ratio",      f"{summary['Sharpe']:.2f}"),
        ("Max Drawdown",      f"{summary['MaxDrawdown_Pct']:.2f}%"),
        ("Day Hit Ratio",     f"{summary['HitRatio_Pct']:.1f}%"),
        ("Trade Count",       f"{summary['TradeCount']}"),
        ("Avg Trades/Day",    f"{summary['AvgTradesPerDay']:.1f}"),
        ("Final AUM",         f"${summary['FinalAUM']:,.2f}"),
    ]
    col_w = 22
    for label, value in rows:
        print(f"  {label:<{col_w}} {value}")
    print("=" * 52)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    load_dotenv()

    end_date = args.end or (date.today().strftime("%Y-%m-%d"))

    cfg = BTMConfig(
        symbol=args.symbol,
        session=args.session,
        lookback_days=args.lookback,
        vm=args.vm,
        leverage_cap=args.leverage_cap,
        initial_aum=args.initial_aum,
        commission_per_share=args.commission,
        slippage_per_share=args.slippage,
    )

    # ── Fetch data ──────────────────────────────────────────────────────
    print(f"Fetching minute bars for {args.symbol}  {args.start} → {end_date} …")
    client = make_data_client(args.session)
    df = fetch_minute_bars(client, args.symbol, args.start, end_date)

    if df.empty:
        print("ERROR: No data returned. Check symbol and date range.", file=sys.stderr)
        return 1

    n_days = len(set(df.index.date))
    print(f"  Loaded {len(df):,} minute bars across {n_days} trading days.")

    if n_days < cfg.lookback_days + 2:
        print(
            f"WARNING: Only {n_days} trading days — fewer than lookback ({cfg.lookback_days})."
            " Results may be unreliable.",
            file=sys.stderr,
        )

    # ── Run backtest ────────────────────────────────────────────────────
    print("Running backtest …")
    result_df, summary = run_backtest(df, cfg)
    print_summary(summary, args.symbol)

    # ── Save outputs ────────────────────────────────────────────────────
    if args.save_dir:
        out_dir = Path(args.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        tag = f"{args.symbol}_{args.start}_{end_date}_vm{args.vm}_lb{args.lookback}"

        csv_path = out_dir / f"{tag}_result.csv"
        result_df.to_csv(csv_path)
        print(f"Result CSV saved: {csv_path}")

        json_path = out_dir / f"{tag}_summary.json"
        json_path.write_text(json.dumps(summary, indent=2))
        print(f"Summary JSON saved: {json_path}")

    # ── Chart ───────────────────────────────────────────────────────────
    if not args.no_plot:
        from btm.chart import plot_backtest
        cfg_label = f"{args.symbol}  vm={args.vm}  lookback={args.lookback}"
        png = plot_backtest(result_df, summary, cfg_label)

        if args.save_dir:
            tag = f"{args.symbol}_{args.start}_{end_date}_vm{args.vm}_lb{args.lookback}"
            png_path = Path(args.save_dir) / f"{tag}_chart.png"
            png_path.write_bytes(png)
            print(f"Chart saved: {png_path}")
        else:
            # Display inline (only works in environments with a display)
            import tempfile, subprocess, platform
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                f.write(png)
                tmp = f.name
            system = platform.system()
            if system == "Darwin":
                subprocess.run(["open", tmp])
            elif system == "Linux":
                subprocess.run(["xdg-open", tmp], stderr=subprocess.DEVNULL)
            else:
                print(f"Chart written to: {tmp}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
