import json
import sys
import types
import unittest
from unittest import mock
import threading


def _install_stubs():
    if 'requests' not in sys.modules:
        requests_stub = types.SimpleNamespace(
            Session=lambda: None,
            HTTPError=Exception,
            RequestException=Exception,
        )
        sys.modules['requests'] = requests_stub
    if 'websocket' not in sys.modules:
        websocket_stub = types.SimpleNamespace(
            WebSocketApp=type('WebSocketApp', (), {})
        )
        sys.modules['websocket'] = websocket_stub
    if 'pandas' not in sys.modules:
        pandas_stub = types.SimpleNamespace(
            Series=object,
            DataFrame=object,
            to_datetime=lambda *args, **kwargs: 0,
        )
        sys.modules['pandas'] = pandas_stub
    if 'numpy' not in sys.modules:
        numpy_stub = types.SimpleNamespace(
            maximum=lambda *args, **kwargs: 0,
            where=lambda *args, **kwargs: [],
            zeros=lambda n: [0] * int(n),
        )
        sys.modules['numpy'] = numpy_stub


_module_cache = None


def load_notebook_module():
    global _module_cache
    if _module_cache is not None:
        return _module_cache

    _install_stubs()
    module = types.ModuleType("super8_notebook")
    module.__dict__['__builtins__'] = __builtins__

    with open("Super 8 Short Live.ipynb", "r", encoding="utf-8") as fh:
        data = json.load(fh)
    for cell in data.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = ''.join(cell.get("source", []))
        if "# === construieste engine" in source:
            before, *_ = source.split("# === construieste engine")
            if before.strip():
                exec(before, module.__dict__)
            break
        exec(source, module.__dict__)

    _module_cache = module
    return module


def _module_from_namespace(namespace=None):
    """Return a module-like object from different namespace inputs."""
    if namespace is None:
        return load_notebook_module()
    if isinstance(namespace, types.ModuleType):
        return namespace
    if isinstance(namespace, dict):
        return types.SimpleNamespace(**namespace)
    return namespace


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


def make_ensure_flat_testcase(namespace=None):
    mod = _module_from_namespace(namespace)

    class EnsureFlatTests(unittest.TestCase):
        def setUp(self):
            LiveConfig = mod.LiveConfig
            SymbolConfig = mod.SymbolConfig
            Super8LiveRunner = mod.Super8LiveRunner

            self.broker = FakeBroker()
            self.runner = Super8LiveRunner(
                broker=self.broker,
                live_cfg=LiveConfig(api_key="", api_secret="", timeframe="1m"),
                sym_cfg=SymbolConfig(symbol="BTCUSDT", usd_fixed=5.0),
                indicator_fn=lambda df: None,
                signal_fn=lambda sym, bar: {},
                sizing_fn=lambda px, cfg, filters: 5.0,
            )
            # Folosim RLock pentru teste ca sa permitem fallback-ul care reia lock-ul.
            self.runner._lock = threading.RLock()
            self.runner.filters = {
                "stepSize": 0.001,
                "minQty": 0.001,
                "tickSize": 0.1,
                "minNotional": 5.0,
            }
            self.runner._dbg = lambda *a, **k: None
            self.runner._err = lambda *a, **k: None

        def test_already_flat_returns_true_without_orders(self):
            self.broker.position_amt = 0.0
            with mock.patch("time.sleep", return_value=None):
                result = self.runner.ensure_flat("BTCUSDT", reason="test")
            self.assertTrue(result)
            self.assertFalse(self.runner._fallback_armed)
            self.assertEqual(self.broker.market_orders, [])
            self.assertEqual(self.broker.fallback_orders, [])

        def test_under_min_notional_triggers_double_trigger(self):
            self.broker.position_amt = -0.001
            self.broker.mark_price_value = 50.0
            self.runner.filters["minNotional"] = 10.0
            with mock.patch("time.sleep", return_value=None):
                result = self.runner.ensure_flat("BTCUSDT", reason="min-notional")
            self.assertTrue(result)
            kinds = [k for k, *_ in self.broker.fallback_orders]
            self.assertIn("STOP", kinds)
            self.assertIn("TP", kinds)
            self.assertEqual(self.broker.market_orders, [])
            self.assertEqual(self.broker.position_amt, 0.0)

        def test_market_close_path_executes_reduce_only(self):
            self.broker.position_amt = -0.01
            self.broker.mark_price_value = 20000.0
            self.runner.filters["minNotional"] = 10.0
            with mock.patch("time.sleep", return_value=None):
                result = self.runner.ensure_flat("BTCUSDT", reason="regular")
            self.assertTrue(result)
            self.assertEqual(len(self.broker.market_orders), 1)
            symbol, side, qty, reduce_only = self.broker.market_orders[0]
            self.assertEqual(symbol, "BTCUSDT")
            self.assertEqual(side, "BUY")
            self.assertTrue(reduce_only)
            self.assertEqual(self.broker.position_amt, 0.0)
            self.assertEqual(self.broker.fallback_orders, [])

    return EnsureFlatTests


def ensure_flat_test_suite(namespace=None):
    testcase = make_ensure_flat_testcase(namespace)
    return unittest.defaultTestLoader.loadTestsFromTestCase(testcase)


def run_ensure_flat_tests(namespace=None, verbosity=2):
    suite = ensure_flat_test_suite(namespace)
    runner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(suite)


if __name__ == "__main__":
    run_ensure_flat_tests()
