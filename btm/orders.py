"""
Alpaca order execution helpers.

For short exposure we BUY inverse ETFs (SPDN, SDS, SPXS) rather than
short-selling, so no margin account or special permissions are needed.

Position state is tracked externally by run_live.py; these functions are
stateless and purely wrap the Alpaca trading client.
"""

from __future__ import annotations

import logging
from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Account helpers
# ---------------------------------------------------------------------------

def get_account_info(client: TradingClient) -> dict:
    """Return a dict with the key account metrics."""
    acct = client.get_account()
    return {
        "equity":          float(acct.equity),
        "cash":            float(acct.cash),
        "buying_power":    float(acct.buying_power),
        "portfolio_value": float(acct.portfolio_value),
        "trading_blocked": acct.trading_blocked,
        "account_blocked": acct.account_blocked,
    }


def get_open_position(client: TradingClient, symbol: str) -> Optional[dict]:
    """
    Return {symbol, qty, side, market_value} if a position exists, else None.
    *qty* is always positive; use *side* ("long" / "short") to interpret it.
    """
    try:
        pos = client.get_open_position(symbol)
        return {
            "symbol":       pos.symbol,
            "qty":          abs(int(float(pos.qty))),
            "side":         pos.side.value,
            "market_value": float(pos.market_value),
        }
    except Exception:
        return None


def get_all_positions(client: TradingClient) -> list:
    """Return a list of open-position dicts (see get_open_position)."""
    positions = client.get_all_positions()
    return [
        {
            "symbol":       p.symbol,
            "qty":          abs(int(float(p.qty))),
            "side":         p.side.value,
            "market_value": float(p.market_value),
        }
        for p in positions
    ]


# ---------------------------------------------------------------------------
# Order execution
# ---------------------------------------------------------------------------

def _submit(client: TradingClient, symbol: str, qty: int, side: OrderSide) -> dict:
    if qty <= 0:
        log.warning("Skipping order: qty=%d for %s", qty, symbol)
        return {}
    req = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=side,
        time_in_force=TimeInForce.DAY,
    )
    order = client.submit_order(req)
    log.info(
        "Order submitted: %s %d %s → id=%s status=%s",
        side.value, qty, symbol, order.id, order.status,
    )
    return {
        "id":     str(order.id),
        "symbol": order.symbol,
        "qty":    qty,
        "side":   side.value,
        "status": str(order.status),
    }


def buy(client: TradingClient, symbol: str, qty: int) -> dict:
    """Market-buy *qty* shares of *symbol*."""
    return _submit(client, symbol, qty, OrderSide.BUY)


def sell(client: TradingClient, symbol: str, qty: int) -> dict:
    """Market-sell *qty* shares of *symbol*."""
    return _submit(client, symbol, qty, OrderSide.SELL)


def close_position(client: TradingClient, symbol: str) -> bool:
    """
    Close the entire open position in *symbol* (if any) with a market order.
    Returns True if a close order was submitted, False if no position existed.
    """
    pos = get_open_position(client, symbol)
    if pos is None:
        log.info("No position in %s to close.", symbol)
        return False
    qty  = pos["qty"]
    side = OrderSide.SELL if pos["side"] == "long" else OrderSide.BUY
    _submit(client, symbol, qty, side)
    return True


def close_all_positions(client: TradingClient) -> int:
    """
    Close ALL open positions.  Returns the number of positions closed.
    Uses the Alpaca bulk-close endpoint for reliability.
    """
    orders = client.close_all_positions(cancel_orders=True)
    n = len(orders) if orders else 0
    log.info("Closed all positions (%d orders submitted).", n)
    return n
