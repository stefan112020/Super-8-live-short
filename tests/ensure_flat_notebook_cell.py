"""Notebook-friendly version of the ``ensure_flat`` tests.

Copy the entire contents of this file into a new notebook cell that lives
*after* the Super 8 classes (``LiveConfig``, ``SymbolConfig`` and
``Super8LiveRunner``).  Executing the cell and then calling
``run_ensure_flat_tests(Super8LiveRunner, LiveConfig, SymbolConfig)`` will run
the same assertions as ``tests/test_ensure_flat.py`` without importing that
module.
"""

import threading
from unittest import mock


class FakeBroker:
    def __init__(self, position_amt=0.0, mark_price=20000.0):
        self.position_amt = position_amt
        self.mark_price_value = mark_price
        self.cancel_calls = 0
        self.market_orders = []
        self.fallback_orders = []

    def connect(self, cfg):
        pass

    def set_hedge_mode(self, *args, **kwargs):
        pass

    def set_margin_type(self, *args, **kwargs):
        pass

    def set_leverage(self, *args, **kwargs):
        pass

    def exchange_info(self, symbol):
        return {}

    def position_info(self, symbol):
        return {"symbol": symbol, "positionAmt": self.position_amt, "entryPrice": 100.0}

    def mark_price(self, symbol):
        return self.mark_price_value

    def cancel_all(self, symbol):
        self.cancel_calls += 1

    def place_close_all_stop_market(self, symbol, side, stop_price):
        self.fallback_orders.append(("STOP", side, stop_price))
        self.position_amt = 0.0
        return {"orderId": "stop"}

    def place_close_all_take_profit_market(self, symbol, side, tp_price):
        self.fallback_orders.append(("TP", side, tp_price))
        self.position_amt = 0.0
        return {"orderId": "tp"}

    def place_market(self, symbol, side, qty, reduce_only=False):
        self.market_orders.append((symbol, side, qty, reduce_only))
        self.position_amt = 0.0
        return {"status": "FILLED"}

    def open_orders(self, symbol):
        return []

    def stop_stream(self, symbol, interval):
        pass

    def list_positions(self):
        return []

    def list_open_orders(self):
        return []

    def account_equity_usdt(self):
        return 1000.0


def run_ensure_flat_tests(Super8LiveRunner, LiveConfig, SymbolConfig):
    """Execute the ensure_flat scenarios against the provided classes.

    Parameters
    ----------
    Super8LiveRunner, LiveConfig, SymbolConfig: objects
        Definitions from the Super 8 notebook that you imported/executed.
    """

    broker = FakeBroker()
    runner = Super8LiveRunner(
        broker=broker,
        live_cfg=LiveConfig(api_key="", api_secret="", timeframe="1m"),
        sym_cfg=SymbolConfig(symbol="BTCUSDT", usd_fixed=5.0),
        indicator_fn=lambda df: None,
        signal_fn=lambda sym, bar: {},
        sizing_fn=lambda px, cfg, filters: 5.0,
    )
    runner._lock = threading.RLock()
    runner.filters = {
        "stepSize": 0.001,
        "minQty": 0.001,
        "tickSize": 0.1,
        "minNotional": 5.0,
    }
    runner._dbg = lambda *a, **k: None
    runner._err = lambda *a, **k: None

    # 1. already flat returns True without orders
    broker.position_amt = 0.0
    with mock.patch("time.sleep", return_value=None):
        result = runner.ensure_flat("BTCUSDT", reason="test")
    assert result is True
    assert runner._fallback_armed is False
    assert broker.market_orders == []
    assert broker.fallback_orders == []

    # 2. under min notional triggers double trigger
    broker.position_amt = -0.001
    broker.mark_price_value = 50.0
    runner.filters["minNotional"] = 10.0
    with mock.patch("time.sleep", return_value=None):
        result = runner.ensure_flat("BTCUSDT", reason="min-notional")
    assert result is True
    kinds = [k for k, *_ in broker.fallback_orders]
    assert "STOP" in kinds and "TP" in kinds
    assert broker.market_orders == []
    assert broker.position_amt == 0.0

    # 3. market close path executes reduce-only
    broker.position_amt = -0.01
    broker.mark_price_value = 20000.0
    runner.filters["minNotional"] = 10.0
    broker.fallback_orders.clear()
    with mock.patch("time.sleep", return_value=None):
        result = runner.ensure_flat("BTCUSDT", reason="regular")
    assert result is True
    assert len(broker.market_orders) == 1
    symbol, side, qty, reduce_only = broker.market_orders[0]
    assert symbol == "BTCUSDT"
    assert side == "BUY"
    assert reduce_only is True
    assert broker.position_amt == 0.0
    assert broker.fallback_orders == []

    print("ensure_flat tests passed!")


if __name__ == "__main__":
    # Attempt to import notebook definitions if available.
    from tests.test_ensure_flat import load_notebook_module

    module = load_notebook_module()
    run_ensure_flat_tests(
        module.Super8LiveRunner,
        module.LiveConfig,
        module.SymbolConfig,
    )

# Notebook usage example:
# run_ensure_flat_tests(Super8LiveRunner, LiveConfig, SymbolConfig)
