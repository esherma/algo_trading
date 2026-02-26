# Algorithmic Trading: Paper Replications

A personal project for implementing and backtesting quantitative strategies from academic papers.
Each strategy is replicated with minor modifications to published parameters, run through a
clean historical simulation, and eventually tested in paper trading. One strategy is currently implemented.

---

## Implemented Strategies

### Beat the Market (BTM)

**Paper:** Zarattini, Aziz & Barbon — ["Beat the Market: An Effective Intraday Momentum Strategy for S&P500 ETF (SPY)"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4824172), SSRN 2024.

The paper documents a persistent intraday momentum effect in SPY: the return of the first 30-minute
bar (9:30–10:00 ET) predicts the direction of the last 30 minutes (3:30–4:00 ET). The strategy
compares SPY's current price to volatility-adjusted noise bands at each half-hour mark and enters
a leveraged long or short position when price breaks out of those bands. All positions close by
3:50 PM ET — no overnight exposure.

**Backtest results (2022-01-03 → 2024-12-31, $100k starting AUM):**

| Metric              | BTM Strategy | SPY Buy & Hold |
|---------------------|:------------:|:--------------:|
| Total Return        | +71.7%       | ~22.7%         |
| Ann. Return (IRR)   | +19.9%       | —              |
| Sharpe Ratio        | 1.39         | —              |
| Max Drawdown        | -9.9%        | —              |

Costs assume: $0.0035/share commission, $0.001/share slippage.

![BTM backtest 2022–2024](public_outputs/btm_backtest.png)

Code in this repo does not reflect any private experimentation or parameter tuning.

---

## Architecture

```
algo_trading/
├── btm/
│   ├── core.py          # Signal logic, noise bands, position sizing, full backtest engine
│   ├── data.py          # Alpaca data client: historical minute bars, real-time snapshots
│   ├── orders.py        # Alpaca trading client: buy, sell, account info, positions
│   ├── chart.py         # Backtest chart and morning band visualization
│   └── email_report.py  # Morning email with today's band chart
├── tests/               # Unit tests for core logic
├── run_backtest.py      # Historical simulation CLI
├── run_live.py          # Live/paper trading runner (one full trading day per invocation)
└── setup_schedule.sh    # Installs daily launchd (macOS) or cron (Linux) job
```

---

## Quickstart

**Prerequisites:**
- Python 3.13+
- [uv](https://github.com/astral-sh/uv) (for dependency management)
- [Alpaca](https://alpaca.markets) account (free paper trading account is sufficient)

**Setup:**

```bash
git clone https://github.com/esherma/algo_trading.git
cd algo_trading
cp .env.example .env          # fill in your Alpaca API keys
uv sync
```

**Run a backtest:**

```bash
uv run python run_backtest.py --start 2022-01-01 --end 2024-12-31
```

Optional flags: `--symbol`, `--vm`, `--lookback`, `--save-dir outputs`, `--no-plot`.

**Run live (paper trading):**

```bash
uv run python run_live.py --session paper
```

Run this once per trading day, or let the scheduler handle it:

```bash
chmod +x setup_schedule.sh
./setup_schedule.sh           # installs launchd job (macOS) or cron entry (Linux)
```

**Preflight check** (verifies credentials, data, and email without submitting any orders):

```bash
uv run python run_live.py --preflight
```

---

## Design Notes

- **Replication over invention.** The goal is to implement what the paper describes, not to find
  a better variant. The parameters in `BTMConfig` match the paper's published values.

- **Paper trading first.** No real capital is involved until the strategy has run for a meaningful
  period in paper trading and the execution quality looks reasonable.

- **Parameters are not public.** The exact hyperparameter values (`vm`, `lookback_days`,
  `target_daily_vol`, `leverage_cap`) that might work well with this strategy are left to the reader's experimentation taste.

---

## Disclaimers

Backtested performance is not a guarantee of future results. Historical simulations cannot fully
account for market impact, liquidity constraints, execution delays, or regime changes. Nothing in
this repository constitutes financial advice.

Coded with assistance from Claude Sonnet 4.6 and GPT-5 (poorly! See the archive folder...)

---

## Acknowledgments

- [Alpaca Markets](https://alpaca.markets) for the data and brokerage API.
- Zarattini, Aziz & Barbon for the original research.
