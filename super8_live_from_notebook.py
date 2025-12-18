# %% cell 0
# broker_adapter.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any, List
import math
import time, hmac, hashlib, json, threading
from urllib.parse import urlencode
import requests
from websocket import WebSocketApp
import configparser
import json, os
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from pathlib import Path

# ---------- Config ----------
@dataclass
class LiveConfig:
    api_key: str
    api_secret: str
    base_url: str = ""          # ex: "https://fapi.binance.com" / testnet
    timeframe: str = "2h"       # kline interval
    leverage: int = 1
    margin_type: str = "ISOLATED"
    hedge_mode: bool = False    # One-Way = False (Hedge Mode = True)
    testnet: bool = False
    dry_run: bool = False
    log_csv: Optional[str] = "trades_log.csv"
    # ---- RISK (implicit dezactivat) ----
    risk_enabled: bool = False          
    risk_base_pct: float = 0.01         # 1.0% din equity per trade (când e ON)
    risk_min_pct: float = 0.0025        # sub acest prag -> nu intră
    risk_cap_pct: float = 0.073         # plafon global 7.3% equity
    risk_cap_buffer_pct: float = 0.05   # buffer 5% (pentru decizie, anti-race)
    # ---- Trailing DRAWDOWN (global) ----
    dd_stop_pct: float = 0.075          # 7.5% DD => blocăm intrări noi
    # ---- Pyramiding ----
    pyramiding_enabled: bool = True
    max_entry_batches: int = 3          # număr maxim de intrări într-o poziție (inclusiv prima)
    # ---- Reconciliere & persist ----
    reconcile_secs: int = 15            # polling poziții/ordine (sec)
    persist_path_tpl: str = "state_{symbol}.json"
    # ---- Pruning istoric ----
    history_keep_bars: int = 700        # păstrăm ~lookback + buffer

@dataclass
class SymbolConfig:
    symbol: str             # ex: "BTCUSDC"
    usd_fixed: float = 1.0  # sumă fixă USD/ordin (inițial)
    pct_equity: Optional[float] = None  # alternativ (% din equity)
    min_usd: float = 5.0    # prag minim fallback
    max_slip_bps: int = 50  # slippage maxim (bps) - opțional

@dataclass
class PositionState:
    in_pos: bool = False
    qty: float = 0.0
    entry_price: float = math.nan
    atr_stop: float = math.nan   # ultimul SL ATR calculat (stair-step)
    entries: int = 0             # număr de execuții de intrare în poziția curentă
    tp_order_id: Optional[str] = None
    sl_order_id: Optional[str] = None

# ---------- Utils: calcul cantitate cu filtre ----------
def floor_to_step(x: float, step: float) -> float:
    if step <= 0:
        return x
    return math.floor(x / step) * step

def ceil_to_step(x: float, step: float) -> float:
    if step <= 0:
        return x
    return math.ceil(x / step) * step

def calc_qty(price: float, usd_target: float,
             stepSize: float, minQty: float, minNotional: float) -> float:
    # țintim notionalul dorit (cantitate * preț)
    qty = floor_to_step(max(0.0, usd_target / max(price, 1e-12)), stepSize)

    # respectă minQty
    if qty < minQty:
        qty = minQty
        if stepSize > 0:
            qty = ceil_to_step(qty, stepSize)
    # dacă încă suntem sub minNotional, rotunjim în sus la stepSize
    if qty * price < minNotional:
        need = ceil_to_step(minNotional / max(price, 1e-12), stepSize)
        qty = max(qty, need)

    return max(0.0, qty)

def round_to_tick(x: float, tick: float) -> float:
    if tick and tick > 0:
        q = Decimal(str(tick))
        d = (Decimal(str(x)) / q).to_integral_value(rounding=ROUND_DOWN) * q
        return float(d)  # ex.: 113005.4 (fără .00000000001)
    return x

def ceil_to_tick(x: float, tick: float) -> float:
    if tick and tick > 0:
        q = Decimal(str(tick))
        d = (Decimal(str(x)) / q).to_integral_value(rounding=ROUND_UP) * q
        return float(d)
    return x

# Funcții de rotunjire specific SHORT (ordine opuse sensului intrării)
def round_stop_for_short(stop_price: float, tick: float) -> float:
    # SL (BUY) pentru SHORT -> rotunjim în sus la tick
    return ceil_to_tick(stop_price, tick)

def round_tp_for_short(tp_price: float, tick: float) -> float:
    # TP (BUY) pentru SHORT -> rotunjim în jos la tick (să nu depășească nivelul dorit)
    return round_to_tick(tp_price, tick)

import csv, os, datetime as dt

class TradeLogger:
    def __init__(self, path: Optional[str]):
        self.path = path
        # Dacă fișierul nu există, scriem header-ul CSV
        if path and (not os.path.exists(path)):
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["ts", "symbol", "action", "side", "qty", "price", "extra"])

    def log(self, symbol: str, action: str, side: str = "", qty: float = 0.0, price: float = math.nan, extra: str = ""):
        if not self.path:
            return
        with open(self.path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([dt.datetime.utcnow().isoformat(), symbol, action, side, qty, price, extra])

class StateStore:
    def __init__(self, path_tpl: str):
        self.path_tpl = path_tpl

    def _path(self, symbol: str) -> str:
        return self.path_tpl.format(symbol=symbol.upper())

    def load(self, symbol: str) -> dict:
        p = self._path(symbol)
        if not os.path.exists(p):
            return {}
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def save(self, symbol: str, state: dict) -> None:
        p = self._path(symbol)
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

# ---------- Interfața Broker (abstractă) ----------
OnBarClose = Callable[[str, Dict[str, Any]], None]  # tip: funcție (symbol, bar_dict)

class BrokerAdapter(ABC):
    @abstractmethod
    def connect(self, cfg: LiveConfig) -> None: ...
    @abstractmethod
    def exchange_info(self, symbol: str) -> Dict[str, Any]: ...
    @abstractmethod
    def fetch_klines(self, symbol: str, interval: str, limit: int) -> List[Dict[str, Any]]: ...
    @abstractmethod
    def stream_klines(self, symbol: str, interval: str, on_close: OnBarClose) -> None: ...
    @abstractmethod
    def position_info(self, symbol: str) -> Dict[str, Any]: ...
    @abstractmethod
    def set_leverage(self, symbol: str, x: int) -> None: ...
    @abstractmethod
    def set_margin_type(self, symbol: str, isolated: bool) -> None: ...
    @abstractmethod
    def set_hedge_mode(self, on: bool) -> None: ...
    @abstractmethod
    def cancel_all(self, symbol: str) -> None: ...
    @abstractmethod
    def place_market(self, symbol: str, side: str, qty: float, reduce_only: bool = False) -> Dict[str, Any]: ...
    @abstractmethod
    def place_stop_market(self, symbol: str, side: str, qty: float, stop_price: float,
                          reduce_only: bool = True) -> Dict[str, Any]: ...
    @abstractmethod
    def place_take_profit_market(self, symbol: str, side: str, qty: float, tp_price: float,
                                 reduce_only: bool = True) -> Dict[str, Any]: ...

# ==== Motorul de semnal (Super8SignalEngine) ====
import pandas as pd
import numpy as np

def _rma(s: pd.Series, n: int) -> pd.Series:
    # Average mobilă exponențială cu alpha = 1/n (Wilder)
    return s.ewm(alpha=1/float(n), adjust=False).mean()

class Super8SignalEngine:
    def __init__(self, ind_p: dict, sh_p: dict):
        self.p = ind_p.copy()
        self.sp = sh_p.copy()
        self.df = pd.DataFrame()   # istoricul OHLCV curent
        self.keep_bars = 700      # se va seta din Runner (din LiveConfig)
        # lookback minim necesar pentru primele semnale
        self.lookback = int(max(
            self.p["sEma_Length"], self.p["BB_Length"], self.p["DClength"],
            self.p["ADX_len"], self.p["slowLength"], self.sp["atrPeriodSl"], 60
        ))

    def seed(self, bars: List[Dict[str, Any]]):
        """Initializează istoricul intern cu o listă de bare (dict cu open, high, low, close, volume, start, end)."""
        if not bars:
            return
        d = pd.DataFrame(bars)
        d["time"] = pd.to_datetime(d["end"], unit="ms", utc=True)
        d = d.set_index("time")[["open", "high", "low", "close", "volume"]].astype(float)
        d.rename(columns={"close": "Price"}, inplace=True)
        self.df = d

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.p
        out = pd.DataFrame(index=df.index)
        # EMA lungă vs EMA scurtă
        sEMA = df["Price"].ewm(span=int(p["sEma_Length"]), adjust=False).mean()
        fEMA = df["Price"].ewm(span=int(p["fEma_Length"]), adjust=False).mean()
        out["EMA_longCond"] = (fEMA > sEMA) & (sEMA > sEMA.shift(1))
        out["EMA_shortCond"] = (fEMA < sEMA) & (sEMA < sEMA.shift(1))
        # ADX (Wilder)
        tr = np.maximum(df["high"] - df["low"],
                        np.maximum((df["high"] - df["Price"].shift(1)).abs(),
                                   (df["low"] - df["Price"].shift(1)).abs()))
        up = df["high"].diff(); dn = -df["low"].diff()
        plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
        minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
        tr_s = _rma(pd.Series(tr, index=df.index), int(p["ADX_len"]))
        plus_s = _rma(pd.Series(plus_dm, index=df.index), int(p["ADX_len"]))
        minus_s = _rma(pd.Series(minus_dm, index=df.index), int(p["ADX_len"]))
        plus_di = (plus_s / tr_s) * 100.0
        minus_di = (minus_s / tr_s) * 100.0
        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).clip(lower=1e-10)) * 100.0
        adx = _rma(dx, int(p.get("ADX_smo", p["ADX_len"])))
        out["ADX_longCond"] = (plus_di > minus_di) & (adx > float(p["th"]))
        out["ADX_shortCond"] = (plus_di < minus_di) & (adx > float(p["th"]))
        # Parabolic SAR (variantă simplificată)
        h = df["high"].to_numpy(); l = df["low"].to_numpy()
        start, step, smax = float(p["Sst"]), float(p["Sinc"]), float(p["Smax"])
        psar = np.zeros(len(df)); up_trend = True
        if len(df) >= 2:
            up_trend = bool(df["Price"].iloc[1] >= df["Price"].iloc[0])
        af = start; ep = h[0] if up_trend else l[0]
        psar[0] = l[0] if up_trend else h[0]
        for i in range(1, len(df)):
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            if up_trend:
                psar[i] = min(psar[i], l[i-1] if i >= 1 else l[i])
                if h[i] > ep:
                    ep = h[i]
                    af = min(af + step, smax)
                if l[i] < psar[i]:
                    up_trend = False
                    psar[i] = ep
                    ep = l[i]
                    af = start
            else:
                psar[i] = max(psar[i], h[i-1] if i >= 1 else h[i])
                if l[i] < ep:
                    ep = l[i]
                    af = min(af + step, smax)
                if h[i] > psar[i]:
                    up_trend = True
                    psar[i] = ep
                    ep = h[i]
                    af = start
        sar = pd.Series(psar, index=df.index)
        out["SAR_longCond"] = sar < df["Price"]
        out["SAR_shortCond"] = sar > df["Price"]
        # MACD clasic
        fast, slow, sig = int(p["fastLength"]), int(p["slowLength"]), int(p["signalLength"])
        macd_line = df["Price"].ewm(span=fast, adjust=False).mean() - df["Price"].ewm(span=slow, adjust=False).mean()
        signal_line = macd_line.ewm(span=sig, adjust=False).mean()
        hist = macd_line - signal_line
        out["MACD_longCond"] = hist > 0
        out["MACD_shortCond"] = hist < 0
        # Bollinger Bands
        L = int(p["BB_Length"]); m = float(p["BB_mult"])
        mid = df["Price"].rolling(L, min_periods=L).mean()
        std = df["Price"].rolling(L, min_periods=L).std(ddof=0)
        upper = mid + m * std; lower = mid - m * std
        out["BB_upper"] = upper; out["BB_lower"] = lower
        out["BB_middle"] = mid
        out["BB_width"] = (upper - lower) / mid
        # Volum – flag când volumul > SMA(volum) * factor
        vol_sma = df["volume"].rolling(int(p["sma_Length"]), min_periods=1).mean()
        vol_flag = df["volume"] > vol_sma * float(p["volume_f"])
        out["VOL_longCond"] = vol_flag
        out["VOL_shortCond"] = vol_flag
        # Prag minim lățime Bollinger (%)
        out["bbMinWidth01"] = float(p["bbMinWidth01"]) / 100.0
        return out

    def on_bar_close(self, symbol: str, bar: dict) -> dict:
        # Adaugă bara nou închisă în DataFrame-ul intern
        need_warmup = len(self.df) < (self.lookback - 1)
        t = pd.to_datetime(bar["end"], unit="ms", utc=True)
        self.df.loc[t, ["open", "high", "low", "Price", "volume"]] = [
            float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"]), float(bar["volume"])
        ]
        # Dacă nu avem încă suficiente bare pentru lookback, nu generăm semnal
        if need_warmup:
            return {"entry_short": False, "exit_reverse": False, "atr_sl": float("nan"), "tp_level": float("nan")}
        df = self.df.copy()
        ind = self._compute_indicators(df)
        b = ind.iloc[-1]   # indicatorii pe bara curent închisă
        px = df["Price"].iloc[-1]
        # ====== Condiții de semnal ======
        EMA_s = bool(b["EMA_shortCond"]);  EMA_l = bool(b["EMA_longCond"])
        ADX_s = bool(b["ADX_shortCond"]);  ADX_l = bool(b["ADX_longCond"])
        SAR_s = bool(b["SAR_shortCond"]);  SAR_l = bool(b["SAR_longCond"])
        MACD_s = bool(b["MACD_shortCond"]); MACD_l = bool(b["MACD_longCond"])
        VOL_s = bool(b["VOL_shortCond"]);  VOL_l = bool(b["VOL_longCond"])
        # Semnal alternativ SHORT (BB crossing)
        bbw_ok = bool(b["BB_width"] > b["bbMinWidth01"])
        cross_over_upper = (df["high"].shift(1).iloc[-1] <= ind["BB_upper"].shift(1).iloc[-1]) \
                           and (df["high"].iloc[-1] > ind["BB_upper"].iloc[-1])
        BB_short01 = (not ADX_l) and EMA_s and bbw_ok and bool(cross_over_upper)
        shortCond = EMA_s and ADX_s and SAR_s and MACD_s and VOL_s
        entry_short = bool(shortCond or BB_short01)
        # Semnal de ieșire (reverse long)
        exit_reverse = bool(EMA_l or ADX_l or SAR_l or MACD_l)
        # ====== Niveluri TP/SL ======
        # ATR (Wilder) pentru calcul SL brut (Short)
        tr = np.maximum(df["high"] - df["low"],
                        np.maximum((df["high"] - df["Price"].shift(1)).abs(),
                                   (df["low"] - df["Price"].shift(1)).abs()))
        atr = _rma(tr, int(self.sp["atrPeriodSl"]))
        atr_sl_raw = float(df["high"].iloc[-1] + atr.iloc[-1] * float(self.sp["multiplierPeriodSl"]))
        # Donchian Low pentru TP (dacă TP_options include "Both")
        DCl = df["low"].rolling(int(self.p["DClength"]), min_periods=int(self.p["DClength"])).min().iloc[-1]
        tp_normal = px * (1.0 - float(self.sp["tp"]) / 100.0)
        tp_level = min(tp_normal, DCl) if self.sp.get("TP_options", "Both") == "Both" else tp_normal
        # --- Pruning istoric (păstrăm doar `history_keep_bars` bare) ---
        if len(self.df) > int(getattr(self, "keep_bars", 700)):
            self.df = self.df.tail(int(getattr(self, "keep_bars", 700)))
        return {
            "entry_short": entry_short,
            "exit_reverse": exit_reverse,
            "atr_sl": atr_sl_raw,   # SL ATR brut (stair-step se aplică în Runner)
            "tp_level": tp_level
        }

def make_sizing_fn(broker: BrokerAdapter):
    def sizing(px: float, sym_cfg: SymbolConfig, filters: Dict[str, Any]) -> float:
        if sym_cfg.pct_equity is not None and sym_cfg.pct_equity > 0:
            try:
                eq = float(broker.account_equity_usdc())
            except Exception:
                eq = 0.0
            return max(sym_cfg.min_usd, eq * float(sym_cfg.pct_equity))
        return max(sym_cfg.min_usd, float(sym_cfg.usd_fixed))
    return sizing

class BinanceFuturesAdapter(BrokerAdapter):
    """
    Implementare pentru Binance USDⓈ-M Futures (One-Way sau Hedge Mode, ISOLATED margin).
    Funcționează pe testnet sau mainnet, în funcție de LiveConfig.
    """
    def __init__(self):
        self.cfg: Optional[LiveConfig] = None
        self.s: Optional[requests.Session] = None
        self.rest_base: str = ""
        self.ws_base: str = ""
        self._streams: Dict[str, Dict[str, Any]] = {}
        self._t_offset = 0

    # ---------- Internals ----------
    def _ts(self) -> int:
        now = time.time()
        # resincronizare periodică cu serverul ca să prevenim drift-ul
        if (now - getattr(self, "_last_sync", 0.0)) > 300:
            try:
                self._sync_time()
            except Exception:
                pass
        return int(time.time() * 1000)
    
    def _sync_time(self):
        try:
            t0 = time.time()
            r = self.s.get(self.rest_base + "/fapi/v1/time", timeout=5)
            r.raise_for_status()
            srv = int(r.json()["serverTime"])
            t1 = time.time()
            rtt = (t1 - t0) / 2
            self._t_offset = srv - int((t0 + rtt) * 1000)
            self._last_sync = t1
        except Exception:
            self._t_offset = 0
            self._last_sync = time.time()

    def _sign(self, q: dict) -> str:
        query = urlencode(q, doseq=True)
        return hmac.new(self.cfg.api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()

    def _send(self, method: str, path: str, params: dict | None = None, signed: bool = False):
        params = params or {}
        url = self.rest_base + path
        headers = {"X-MBX-APIKEY": self.cfg.api_key}
        base_params = dict(params)
        for i in range(5):  # până la 5 încercări
            try:
                req_params = dict(base_params)
                if signed:
                    req_params["timestamp"] = self._ts() + getattr(self, "_t_offset", 0)
                    req_params.setdefault("recvWindow", 60000)
                    req_params["signature"] = self._sign(req_params)
                if method == "GET":
                    r = self.s.get(url, params=req_params, headers=headers, timeout=10)
                elif method == "POST":
                    r = self.s.post(url, params=req_params, headers=headers, timeout=10)
                elif method == "DELETE":
                    r = self.s.delete(url, params=req_params, headers=headers, timeout=10)
                else:
                    raise ValueError("Metodă HTTP invalidă")
                # retry pe rate-limit sau erori 5xx
                if r.status_code in (429, 418) or 500 <= r.status_code < 600:
                    raise requests.HTTPError(response=r)
                r.raise_for_status()
                return r.json()
            except requests.HTTPError as e:
                try:
                    err_text = e.response.text
                    status = e.response.status_code
                except Exception:
                    err_text = str(e); status = -1
                if i == 4:
                    raise RuntimeError(f"HTTP {status} {path} -> {err_text}") from e
                # dacă eroarea indică drift de timp (-1021), resync și retry
                if signed and isinstance(err_text, str) and "-1021" in err_text:
                    self._sync_time()
                time.sleep(0.4 * (2 ** i))
            except requests.RequestException:
                if i == 4:
                    raise
                time.sleep(0.4 * (2 ** i))

    # ---------- Public ----------
    def connect(self, cfg: LiveConfig) -> None:
        print("[ENV] testnet =", cfg.testnet)
        missing = [name for name, val in (("api_key", cfg.api_key), ("api_secret", cfg.api_secret)) if not val]
        if missing:
            raise ValueError(f"Configurarea Binance lipsește cheile obligatorii: {', '.join(missing)}")
        self.cfg = cfg
        self.s = requests.Session()
        # setează URL-urile în funcție de mediul selectat sau override din config
        if cfg.base_url:
            self.rest_base = cfg.base_url.rstrip("/")
            self.ws_base = "wss://stream.binancefuture.com/stream" if cfg.testnet else "wss://fstream.binance.com/stream"
        elif cfg.testnet:
            self.rest_base = "https://testnet.binancefuture.com"
            self.ws_base = "wss://stream.binancefuture.com/stream"
        else:
            self.rest_base = "https://fapi.binance.com"
            self.ws_base = "wss://fstream.binance.com/stream"
        self._last_sync = 0.0
        self._sync_time()
        print("[BASE]", self.rest_base)
        # setări cont (ordinea apelurilor contează la Binance)
        # Hedge Mode și Margin Type se configurează mai jos în Runner.bootstrap()
        # (Leverage se setează pe simbol tot în Runner.bootstrap)

    def exchange_info(self, symbol: str) -> Dict[str, Any]:
        data = self._send("GET", "/fapi/v1/exchangeInfo", params={"symbol": symbol.upper()}, signed=False)
        sym = data["symbols"][0]
        tickSize = stepSize = minQty = minNotional = 0.0
        for f in sym["filters"]:
            t = f["filterType"]
            if t == "PRICE_FILTER":
                tickSize = float(f["tickSize"])
            elif t == "LOT_SIZE":
                stepSize = float(f["stepSize"]); minQty = float(f["minQty"])
            elif t in ("MIN_NOTIONAL", "NOTIONAL"):
                # Futures folosesc MIN_NOTIONAL (sau NOTIONAL în unele cazuri)
                minNotional = float(f.get("notional", f.get("minNotional", 0.0)))
        return {
            "symbol": sym["symbol"],
            "tickSize": tickSize or 0.0,
            "stepSize": stepSize or 0.0,
            "minQty": minQty or 0.0,
            "minNotional": minNotional or 0.0,
        }

    def fetch_klines(self, symbol: str, interval: str, limit: int) -> List[Dict[str, Any]]:
        resp = self._send("GET", "/fapi/v1/klines",
                          params={"symbol": symbol.upper(), "interval": interval, "limit": limit}, signed=False)
        out = []
        for k in resp:
            out.append({
                "start": int(k[0]), "end": int(k[6]),
                "open": float(k[1]), "high": float(k[2]),
                "low": float(k[3]), "close": float(k[4]),
                "volume": float(k[5])
            })
        return out

    def stream_klines(self, symbol: str, interval: str, on_close: OnBarClose,
                      on_update: Callable[[str, Dict[str, Any]], None] = None) -> None:
        stream = f"{symbol.lower()}@kline_{interval}"
        key = f"{symbol.upper()}_{interval}"
        stop_event = threading.Event()
        self._streams[key] = {"stop": stop_event, "ws": None}

        def _on_msg(ws, msg):
            d = json.loads(msg)
            k = d.get("data", {}).get("k", {}) or d.get("k", {})
            if not k:
                return
            bar = {
                "start": int(k["t"]), "end": int(k["T"]),
                "open": float(k["o"]), "high": float(k["h"]),
                "low": float(k["l"]), "close": float(k["c"]),
                "volume": float(k["v"]),
            }
            if k.get("x", False):
                on_close(symbol.upper(), bar)
            elif on_update is not None:
                on_update(symbol.upper(), bar)

        def _run():
            backoff = 1.0
            failures = 0; max_failures = 10
            while not stop_event.is_set() and failures < max_failures:
                ws = WebSocketApp(f"{self.ws_base}?streams={stream}", on_message=_on_msg)
                info = self._streams.get(key)
                if info is not None:
                    info["ws"] = ws
                else:
                    break  # dacă stream-ul a fost scos din dict între timp
                try:
                    ws.run_forever(ping_interval=15, ping_timeout=10)
                    failures = 0  # reset după o conexiune reușită
                except Exception as e:
                    if not stop_event.is_set():
                        failures += 1
                        print(f"[WARN] WS disconnect {key}: {e}")
                if stop_event.is_set():
                    break
                if failures >= max_failures:
                    print(f"[FATAL] WS {key} failed {failures} times, stopping")
                    break
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
            # Cleanup: marchează ws ca închis
            info = self._streams.get(key)
            if info is not None:
                info["ws"] = None

        threading.Thread(target=_run, daemon=True, name=f"WS-{key}").start()

    def position_info(self, symbol: str) -> Dict[str, Any]:
        data = self._send("GET", "/fapi/v2/positionRisk", params={"symbol": symbol.upper()}, signed=True)
        # API-ul returnează listă (chiar și cu un singur element) pentru USD-M Futures
        p = data[0] if isinstance(data, list) and data else data
        return {
            "symbol": p.get("symbol", symbol.upper()),
            "positionAmt": float(p.get("positionAmt", 0.0)),
            "entryPrice": float(p.get("entryPrice", 0.0)),
            "unRealizedPnL": float(p.get("unRealizedProfit", 0.0)),
        }
    
    def account_equity_usdc(self) -> float:
        # Preferăm totalWalletBalance; fallback la availableBalance (per asset)
        try:
            data = self._send("GET", "/fapi/v2/account", params={}, signed=True)
            return float(data.get("totalWalletBalance", 0.0))
        except Exception:
            pass
        try:
            bal_list = self._send("GET", "/fapi/v2/balance", params={}, signed=True)
            for asset in bal_list:
                if asset.get("asset") == "USDC":
                    return float(asset.get("balance", asset.get("availableBalance", 0.0)))
        except Exception:
            pass
        return 0.0
    
    def stop_stream(self, symbol: str, interval: str) -> None:
        key = f"{symbol.upper()}_{interval}"
        info = self._streams.pop(key, None)
        if not info:
            return
        info["stop"].set()
        try:
            if info["ws"] is not None:
                info["ws"].close()
        except Exception:
            pass

    def set_leverage(self, symbol: str, x: int) -> None:
        self._send("POST", "/fapi/v1/leverage",
                   params={"symbol": symbol.upper(), "leverage": int(x)}, signed=True)

    def set_margin_type(self, symbol: str, isolated: bool) -> None:
        mode = "ISOLATED" if isolated else "CROSSED"
        try:
            if symbol == "ALL":
                return  # (nu schimbăm tipul pe toate, eventual loop extern)
            self._send("POST", "/fapi/v1/marginType",
                       params={"symbol": symbol.upper(), "marginType": mode}, signed=True)
        except Exception as e:
            t = str(e)
            # OK dacă deja e setat sau contul e în modul care nu permite ISOLATED (ex. Credits)
            if ("No need to change margin type" in t) or ("-4046" in t) or ("-4175" in t) or ("credit status" in t):
                print("[INFO] MarginType deja conform cerinței -> continuăm.")
                return
            raise

    def set_hedge_mode(self, on: bool) -> None:
        # activează Hedge Mode dacă on=True (dualSidePosition)
        params = {"dualSidePosition": "true" if on else "false"}
        try:
            self._send("POST", "/fapi/v1/positionSide/dual", params=params, signed=True)
        except Exception as e:
            t = str(e)
            if ("-4059" in t) or ("No need to change position side" in t):
                print("[INFO] Hedge mode deja setat conform solicitării.")
                return
            raise

    def cancel_all(self, symbol: str) -> None:
        self._send("DELETE", "/fapi/v1/allOpenOrders",
                   params={"symbol": symbol.upper()}, signed=True)
        
    def open_orders(self, symbol: str) -> List[dict]:
        return self._send("GET", "/fapi/v1/openOrders",
                          params={"symbol": symbol.upper()}, signed=True)
    
    def list_positions(self) -> List[dict]:
        """Returnează toate pozițiile deschise (USD-M Futures)."""
        try:
            data = self._send("GET", "/fapi/v2/positionRisk", params={}, signed=True)
            return data if isinstance(data, list) else [data]
        except Exception:
            return []

    def list_open_orders(self) -> List[dict]:
        """Returnează toate ordinele deschise pe toate simbolurile."""
        try:
            data = self._send("GET", "/fapi/v1/openOrders", params={}, signed=True)
            return data if isinstance(data, list) else [data]
        except Exception:
            return []

    def place_market(self, symbol: str, side: str, qty: float, reduce_only: bool = False) -> Dict[str, Any]:
        params = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": "MARKET",
            "quantity": qty,
            "reduceOnly": "true" if reduce_only else "false",
            "newOrderRespType": "RESULT",
        }
        # One-Way vs Hedge: setează positionSide corespunzător
        if not getattr(self.cfg, "hedge_mode", False):
            params["positionSide"] = "BOTH"
        else:
            if reduce_only:
                params["positionSide"] = "SHORT" if side.upper() == "BUY" else "LONG"
            else:
                params["positionSide"] = "LONG" if side.upper() == "BUY" else "SHORT"
        return self._send("POST", "/fapi/v1/order", params=params, signed=True)

    def place_stop_market(self, symbol: str, side: str, qty: float, stop_price: float,
                          reduce_only: bool = True) -> Dict[str, Any]:
        params = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": "STOP_MARKET",
            "stopPrice": f"{stop_price:.8f}",  # trebuie string la tick exact
            "closePosition": "false",
            "quantity": qty,
            "reduceOnly": "true" if reduce_only else "false",
            "workingType": "MARK_PRICE",
            "priceProtect": "true",
            "newOrderRespType": "RESULT",
        }
        if not getattr(self.cfg, "hedge_mode", False):
            params["positionSide"] = "BOTH"
        else:
            if reduce_only:
                params["positionSide"] = "SHORT" if side.upper() == "BUY" else "LONG"
            else:
                params["positionSide"] = "LONG" if side.upper() == "BUY" else "SHORT"
        return self._send("POST", "/fapi/v1/order", params=params, signed=True)

    def place_take_profit_market(self, symbol: str, side: str, qty: float, tp_price: float,
                                 reduce_only: bool = True) -> Dict[str, Any]:
        params = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": f"{tp_price:.8f}",
            "closePosition": "false",
            "quantity": qty,
            "reduceOnly": "true" if reduce_only else "false",
            "workingType": "MARK_PRICE",
            "priceProtect": "true",
            "newOrderRespType": "RESULT",
        }
        if not getattr(self.cfg, "hedge_mode", False):
            params["positionSide"] = "BOTH"
        else:
            if reduce_only:
                params["positionSide"] = "SHORT" if side.upper() == "BUY" else "LONG"
            else:
                params["positionSide"] = "LONG" if side.upper() == "BUY" else "SHORT"
        return self._send("POST", "/fapi/v1/order", params=params, signed=True)

    def mark_price(self, symbol: str) -> float:
        """Întoarce Mark Price (pentru declanșarea SL/TP pe futures)."""
        r = self._send("GET", "/fapi/v1/premiumIndex", params={"symbol": symbol.upper()}, signed=False)
        return float(r.get("markPrice", 0.0))

    def last_price(self, symbol: str) -> float:
        """Întoarce ultimul preț tranzacționat (fallback pentru date rapide)."""
        r = self._send("GET", "/fapi/v1/ticker/price", params={"symbol": symbol.upper()}, signed=False)
        return float(r.get("price", 0.0))

    def place_close_all_stop_market(self, symbol: str, side: str, stop_price: float) -> Dict[str, Any]:
        """Plasează un STOP_MARKET cu closePosition=true (închide tot fără qty specificată)."""
        params = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": "STOP_MARKET",
            "stopPrice": f"{stop_price:.8f}",
            "closePosition": "true",
            "workingType": "MARK_PRICE",
            "priceProtect": "true",
            "newOrderRespType": "RESULT",
        }
        if not getattr(self.cfg, "hedge_mode", False):
            params["positionSide"] = "BOTH"
        else:
            # În modul Hedge, specificăm ce parte închidem (BUY->SHORT, SELL->LONG)
            params["positionSide"] = "SHORT" if side.upper() == "BUY" else "LONG"
        return self._send("POST", "/fapi/v1/order", params=params, signed=True)

    def place_close_all_take_profit_market(self, symbol: str, side: str, tp_price: float) -> Dict[str, Any]:
        """Plasează un TAKE_PROFIT_MARKET cu closePosition=true (închide tot fără qty)."""
        params = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": f"{tp_price:.8f}",
            "closePosition": "true",
            "workingType": "MARK_PRICE",
            "priceProtect": "true",
            "newOrderRespType": "RESULT",
        }
        if not getattr(self.cfg, "hedge_mode", False):
            params["positionSide"] = "BOTH"
        else:
            params["positionSide"] = "SHORT" if side.upper() == "BUY" else "LONG"
        return self._send("POST", "/fapi/v1/order", params=params, signed=True)

# === Parametri stratgie (exemplu) ===
ind_params = {
    "fEma_Length": 58, "sEma_Length": 426,
    "ADX_len": 10, "ADX_smo": 8, "th": 11.53,
    "fastLength": 22, "slowLength": 39, "signalLength": 13,
    "BB_Length": 156, "BB_mult": 13.18,
    "sma_Length": 81, "volume_f": 0.87,
    "DClength": 76,
    "Sst": 0.10, "Sinc": 0.04, "Smax": 0.40,
    "bbMinWidth01": 9.3, "bbMinWidth02": 0.0
}
short_params = {
    "TP_options": "Both", "SL_options": "Both",
    "tp": 3.6, "sl": 8.0, "atrPeriodSl": 100,
    "multiplierPeriodSl": 95.77, "trailOffset": 0.38,
    "reverse_exit": True, "start_time": None
}

# ---------- Runner (logica de execuție live, cu delay de 1 bară) ----------
class Super8LiveRunner:
    def __init__(self, broker: BrokerAdapter, live_cfg: LiveConfig, sym_cfg: SymbolConfig,
                 indicator_fn: Callable, signal_fn: Callable, sizing_fn: Optional[Callable] = None,
                 short_params: Optional[Dict[str, Any]] = None):
        self.broker = broker
        self.live_cfg = live_cfg
        self.sym_cfg = sym_cfg
        self.state = PositionState()
        self._last_live_amt = 0.0
        self.filters: Dict[str, Any] = {}
        self.indicator_fn = indicator_fn  # funcție indicatori (df -> indic.)
        self.signal_fn = signal_fn        # funcție semnal (symbol, bar) -> dict
        self.sizing_fn = sizing_fn        # funcție opțională dimensionare
        self.exit_pending = False         # dacă avem o ieșire în așteptare de executat
        self.dry_run = getattr(self.live_cfg, "dry_run", True)
        self.log = TradeLogger(getattr(self.live_cfg, "log_csv", None))
        self.pending_entry = False
        self._next_bar_start = None
        self._pending_levels = {"sl": math.nan, "tp": math.nan}
        self._bar_updates = 0
        self.short_params = short_params or {}
        self.start_time = self._parse_start_time(self.short_params.get("start_time"))
        self.reverse_exit = bool(self.short_params.get("reverse_exit", True))
        # -- Variabile pentru log OANDA-like --
        self._cum_pl = 0.0
        self._last_entry = {"qty": 0.0, "price": math.nan, "side": None}
        # -- Persistență stare & thread sync --
        self.store = StateStore(self.live_cfg.persist_path_tpl)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._last_sync = 0.0
        # -- Semafor de ieșire (blochează re-armarea SL/TP cât timp ieșim) --
        self._exiting = False
        # -- Flag fallback double-trigger armat (evită dublă anulare) --
        self._fallback_armed = False

    def _dbg(self, msg: str):
        print(f"[DBG] {msg}")

    def _err(self, msg: str):
        print(f"[ERR] {msg}")

    def _tick_decimals(self) -> int:
        t = float(self.filters.get("tickSize", 0.0))
        if t <= 0:
            return 0
        s = f"{t:.10f}".rstrip("0").rstrip(".")
        return len(s.split(".")[1]) if "." in s else 0

    def _fmt_px(self, px: float) -> str:
        # Formatăm prețul cu numărul de zecimale permis de tick size
        dec = self._tick_decimals()
        return f"{px:.{dec}f}"

    def _parse_start_time(self, value: Any) -> Optional[pd.Timestamp]:
        if value is None:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        if isinstance(value, str) and not value.strip():
            return None
        try:
            ts = pd.to_datetime(value)
        except Exception:
            self._err(f"Nu am putut interpreta start_time: {value}")
            return None
        if pd.isna(ts):
            return None
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts

    def bootstrap(self):
        # Conectează adaptorul broker și aplică setările inițiale
        self.broker.connect(self.live_cfg)
        # Setăm modul de poziții (One-Way sau Hedge) conform configurării
        self.broker.set_hedge_mode(self.live_cfg.hedge_mode)
        # Setăm tipul de marjă (ISOLATED/CROSSED)
        self.broker.set_margin_type(self.sym_cfg.symbol, True if self.live_cfg.margin_type.upper() == "ISOLATED" else False)
        # Leverage per simbol
        self.broker.set_leverage(self.sym_cfg.symbol, self.live_cfg.leverage)
        # Obținem filtrele de exchange (tickSize, lot size etc.)
        self.filters = self.broker.exchange_info(self.sym_cfg.symbol)
        # Setăm numărul de bare de istoric de păstrat în engine (pentru memorie)
        try:
            if hasattr(self, "engine"):
                self.engine.keep_bars = int(self.live_cfg.history_keep_bars)
        except Exception:
            pass
        # Încărcăm starea persistentă (dacă există) și sincronizăm cu piața actuală
        st = self.store.load(self.sym_cfg.symbol) or {}
        try:
            p = self.broker.position_info(self.sym_cfg.symbol)
            live_amt = float(p.get("positionAmt", 0.0))
        except Exception:
            live_amt = 0.0
        if live_amt == 0:
            # nicio poziție pe bursă -> reset local
            self.state = PositionState()
            self._last_live_amt = 0.0
        else:
            # există poziție deschisă pe bursă: rehidratează starea locală
            self.state.in_pos = True
            self.state.qty = abs(live_amt)
            live_entry_price = 0.0
            try:
                live_entry_price = float(p.get("entryPrice", 0.0))
            except Exception:
                live_entry_price = 0.0
            # folosim prețul de intrare din fișier dacă există, altfel pe cel de la bursă
            self.state.entry_price = float(st.get("entry_price", live_entry_price if live_entry_price > 0 else self.state.entry_price))
            self.state.atr_stop = float(st.get("atr_stop", self.state.atr_stop))
            self.state.entries = int(st.get("entries", max(1, self.state.entries or 0)))
            # setează _last_entry pentru calcul P&L la ieșire
            if self.state.entry_price and not math.isnan(self.state.entry_price):
                self._last_entry = {
                    "qty": self.state.qty,
                    "price": self.state.entry_price,
                    "side": "SHORT" if live_amt < 0 else "LONG"
                }
        # Pornește thread-ul de reconciliere periodică a stării
        threading.Thread(target=self._reconcile_loop, daemon=True).start()

    def ensure_flat(self, symbol: str, reason: str = "reverse", max_retries: int = 10, sleep_s: float = 0.6) -> bool:
        """
        Închide poziția (dacă există) cu retry + debounce. 
        Dacă ordinul MARKET eșuează sau cantitatea e sub minNotional, folosește mecanismul 
        fallback 'double-trigger' (două ordine STOP/TPS opuse) pentru închidere.
        """
        step = self.filters.get("stepSize", 0.0)
        minq = self.filters.get("minQty", 0.0)
        tick = self.filters.get("tickSize", 0.0)
        min_notional = self.filters.get("minNotional", 0.0)
        self._exiting = True
        self._dbg(f"ensure_flat start (reason={reason})")

        def _double_trigger_close(side: str) -> bool:
            # Plasează două ordine (SL și TP) closePosition=true în jurul prețului curent (+/-0.15%)
            try:
                px = float(self.broker.mark_price(symbol))
            except Exception:
                px = float(self.state.entry_price if self.state.entry_price and not math.isnan(self.state.entry_price) else 0.0)
            if px <= 0 or tick <= 0:
                self._err("double-trigger: nu am găsit prețul sau tick-ul pentru fallback")
                return False
            # Calculăm nivelele offsetate ±0.15% din px
            if side.upper() == "BUY":   # închidere SHORT (vom cumpăra)
                sl_px = ceil_to_tick(px * 1.0015, tick)   # SL mai sus puțin
                tp_px = round_to_tick(px * 0.9985, tick)  # TP puțin sub prețul curent
            else:                      # închidere LONG (vom vinde)
                sl_px = round_to_tick(px * 0.9985, tick)
                tp_px = ceil_to_tick(px * 1.0015, tick)
            sl_s = self._fmt_px(sl_px)
            tp_s = self._fmt_px(tp_px)
            with self._lock:
                # Anulează ordinele existente doar dacă fallback-ul nu era deja armat
                if not getattr(self, "_fallback_armed", False):
                    try:
                        self.broker.cancel_all(symbol)
                    except Exception as e:
                        self._err(f"eroare cancel_all înainte de fallback: {e}")
                try:
                    self.broker.place_close_all_stop_market(symbol, side=side, stop_price=sl_px)
                    self.broker.place_close_all_take_profit_market(symbol, side=side, tp_price=tp_px)
                    self._dbg(f"fallback double-trigger armat sl={sl_s} tp={tp_s}")
                    self._fallback_armed = True
                except Exception as e:
                    self._err(f"eroare la plasarea double-trigger: {e}")
                    return False
            # Așteaptă câteva secunde să vadă dacă poziția se închide
            for _ in range(20):
                time.sleep(0.4)
                try:
                    p_check = self.broker.position_info(symbol)
                    if abs(float(p_check.get("positionAmt", 0.0))) < max(minq, 0.0):
                        self._dbg("double-trigger: poziția a ajuns la zero")
                        try:
                            self.broker.cancel_all(symbol)
                        except Exception:
                            pass
                        self._fallback_armed = False
                        self._last_live_amt = 0.0
                        return True
                except Exception:
                    continue
            return False

        for attempt in range(max_retries):
            placed_order = False
            unknown = False
            with self._lock:
                try:
                    pos = self.broker.position_info(symbol)
                    amt = float(pos.get("positionAmt", 0.0))
                except Exception as e:
                    self._err(f"position_info error: {e}")
                    # Dacă nu putem obține poziția, marcăm situația ca necunoscută
                    try:
                        all_positions = self.broker.list_positions()
                        found_amt = None
                        for pos in all_positions:
                            if pos.get("symbol", "").upper() == symbol.upper():
                                found_amt = float(pos.get("positionAmt", 0.0))
                                break
                        if found_amt is None:
                            unknown = True
                        else:
                            amt = found_amt
                    except Exception:
                        unknown = True
                    if unknown:
                        amt = getattr(self, "_last_live_amt", 0.0)
                self._last_live_amt = float(amt)
                qty = abs(float(amt))
                self._dbg(f"ensure_flat: verific qty={qty}, minQty={minq}")
                # Dacă știm sigur poziția și qty e practic zero (< minQty), considerăm închis
                if not unknown and qty < max(minq, 0.0):
                    self._dbg("ensure_flat: deja flat (<= minQty)")
                    self._fallback_armed = False
                    self._exiting = False
                    self._last_live_amt = 0.0
                    return True
                # Dacă nu cunoaștem situația poziției, trecem la următoarea încercare
                if unknown:
                    # ieșim din blocul lock pentru a reîncerca
                    pass
                else:
                    try:
                        last_px = float(self.broker.mark_price(symbol))
                    except Exception:
                        last_px = float(self.state.entry_price if hasattr(self.state, "entry_price") and self.state.entry_price else 0.0)
                    notional = qty * last_px if last_px > 0 else float("inf")
                    side = "BUY" if amt < 0 else "SELL"
                    # Verifică dacă suntem sub minNotional
                    min_qty_for_notional = None
                    if last_px > 0 and min_notional > 0:
                        min_qty_for_notional = math.ceil(min_notional / last_px)
                        if step and step > 0:
                            min_qty_for_notional = ceil_to_step(min_qty_for_notional, step)
                    if last_px > 0 and min_qty_for_notional and qty < min_qty_for_notional:
                        self._dbg(f"under minNotional (qty*px={notional} < {min_notional}) -> double-trigger fallback")
                        ok = _double_trigger_close(side)
                        if ok:
                            self._exiting = False
                            self._last_live_amt = 0.0
                            return True
                        # dacă fallback-ul nu reușește imediat, continuăm încercările
                    # Anulează protecțiile doar dacă fallback-ul nu e deja armat
                    if not getattr(self, "_fallback_armed", False):
                        try:
                            self.broker.cancel_all(symbol)
                        except Exception as e:
                            self._err(f"cancel_all error: {e}")
                    # Plasează ordin MARKET reduceOnly (qty rotunjită la pas; fallback la minQty dacă rotunjirea dă 0)
                    q = floor_to_step(qty, step)
                    if q <= 0:
                        q = ceil_to_step(minq, step) if step and step > 0 else minq
                        self._dbg(f"close qty floor_to_step=0; fallback q={q}")
                    try:
                        self._dbg(f"close MARKET reduceOnly side={side} q={q}")
                        resp = self.broker.place_market(symbol, side=side, qty=q, reduce_only=True)
                        self._dbg(f"close MARKET resp: {resp}")
                        placed_order = True
                    except Exception as e:
                        self._err(f"close MARKET error: {e}")
            # end with (lock)
            if unknown:
                time.sleep(0.4 * (2 ** attempt))
                continue
            # Debounce: așteaptă puțin și verifică din nou poziția
            time.sleep(max(0.3, sleep_s if placed_order else 0.3))
            try:
                pos_check = self.broker.position_info(symbol)
                amt_check = float(pos_check.get("positionAmt", 0.0))
            except Exception as e:
                self._err(f"position_info (post-close) error: {e}")
                amt_check = 0.0
            self._last_live_amt = float(amt_check)
            if abs(amt_check) < max(minq, 0.0):
                self._dbg("ensure_flat: flat confirmat după debounce")
                if getattr(self, "_fallback_armed", False):
                    try:
                        self.broker.cancel_all(symbol)
                    except Exception as e:
                        self._err(f"cancel_all error: {e}")
                self._fallback_armed = False
                self._exiting = False
                self._last_live_amt = 0.0
                return True
            self._dbg(f"ensure_flat retry {attempt+1}/{max_retries} (încă qty={amt_check})")
            time.sleep(sleep_s)
        # end for retrieri
        self._err("ensure_flat: retries epuizate; aplicăm fallback final")
        fallback_amt = getattr(self, "_last_live_amt", 0.0)
        if abs(fallback_amt) < max(minq, 0.0) and isinstance(getattr(self, "_last_entry", None), dict):
            last_side = (self._last_entry.get("side") or "").upper()
            if last_side == "SELL":
                fallback_amt = -abs(self._last_entry.get("qty", 0.0))
            elif last_side == "BUY":
                fallback_amt = abs(self._last_entry.get("qty", 0.0))
        fallback_side = "BUY"
        if fallback_amt > 0:
            fallback_side = "SELL"
        elif fallback_amt < 0:
            fallback_side = "BUY"
        ok = _double_trigger_close(fallback_side)
        if ok:
            self._exiting = False
            self._last_live_amt = 0.0
            return True
        self._err("double-trigger: tot nu e flat; rămânem în modul EXITING")
        return False
    
    def force_flat_now(self) -> bool:
        """Închidere forțată imediată (market + fallback) cu log de debug."""
        self._exiting = True
        ok = self.ensure_flat(self.sym_cfg.symbol, reason="force-flat")
        self._exiting = False
        return ok

    def verify_protections(self, symbol: str, want_sl: Optional[float], want_tp: Optional[float]):
        # Nu arma protecții cât timp suntem în proces de EXITING
        if getattr(self, "_exiting", False) or not self.state.in_pos:
            return
        try:
            oo = self.broker.open_orders(symbol)
        except Exception:
            oo = []
        # În raspunsurile Binance, reduceOnly poate fi bool sau string "true"/"false"
        def _is_true(v):
            return (v is True) or (isinstance(v, str) and v.lower() == "true")
        have_sl = any(o.get("type") == "STOP_MARKET" and _is_true(o.get("reduceOnly")) for o in oo)
        have_tp = any(o.get("type") == "TAKE_PROFIT_MARKET" and _is_true(o.get("reduceOnly")) for o in oo)
        tick = self.filters.get("tickSize", 0.0)
        if (not have_sl) and (want_sl is not None) and not math.isnan(want_sl):
            sl_px = round_stop_for_short(want_sl, tick)
            self._sl(symbol, side="BUY", qty=self.state.qty, stop_price=sl_px)
            self.state.atr_stop = sl_px
        if (not have_tp) and (want_tp is not None) and not math.isnan(want_tp):
            tp_px = round_tp_for_short(want_tp, tick)
            self._tp(symbol, side="BUY", qty=self.state.qty, tp_price=tp_px)

    def _report_trade_oanda(self, action: str, units: float, price: float, pl: float = 0.0):
        """Log stil OANDA pentru tranzacții (units negative = short)."""
        self._cum_pl += float(pl)
        ts = dt.datetime.utcnow().isoformat() + "Z"
        print("\n" + "-" * 100)
        print(f"{ts} | {action}")
        print(f"{ts} | units = {units} | price = {round(float(price), 5)} | P&L = {round(float(pl), 4)} | Cum P&L = {round(self._cum_pl, 4)}")
        print("-" * 100 + "\n")

    def stop(self, flatten: bool = False):
        """
        Oprire grațioasă a strategiei: oprește stream-ul; 
        opțional închide poziția și anulează ordinele în așteptare.
        """
        try:
            self._stop_event.set()
        except Exception:
            pass
        if flatten and self.state.in_pos:
            try:
                self.ensure_flat(self.sym_cfg.symbol, reason="user-stop")
            except Exception:
                pass
            self.state = PositionState()
            self._last_live_amt = 0.0
        try:
            self.broker.cancel_all(self.sym_cfg.symbol)
        except Exception:
            pass
        try:
            self.broker.stop_stream(self.sym_cfg.symbol, self.live_cfg.timeframe)
        except Exception:
            pass

    # Metode interne pentru plasarea de ordine (folosesc logger și broker)
    def _mkt(self, symbol: str, side: str, qty: float, reduce_only: bool = False):
        if self.dry_run:
            self.log.log(symbol, "MARKET(DRY)", side, qty, extra=f"reduceOnly={reduce_only}")
            return {"status": "DRY"}
        self.log.log(symbol, "MARKET", side, qty)
        return self.broker.place_market(symbol, side=side, qty=qty, reduce_only=reduce_only)

    def _sl(self, symbol: str, side: str, qty: float, stop_price: float):
        if self.dry_run:
            self.log.log(symbol, "STOP_MARKET(DRY)", side, qty, price=stop_price)
            return {"status": "DRY"}
        self.log.log(symbol, "STOP_MARKET", side, qty, stop_price)
        price_str = self._fmt_px(stop_price)
        return self.broker.place_stop_market(symbol, side=side, qty=qty, stop_price=float(price_str), reduce_only=True)

    def _tp(self, symbol: str, side: str, qty: float, tp_price: float):
        if self.dry_run:
            self.log.log(symbol, "TP_MARKET(DRY)", side, qty, price=tp_price)
            return {"status": "DRY"}
        self.log.log(symbol, "TP_MARKET", side, qty, tp_price)
        price_str = self._fmt_px(tp_price)
        return self.broker.place_take_profit_market(symbol, side=side, qty=qty, tp_price=float(price_str), reduce_only=True)

    def _reconcile_loop(self):
        """Thread daemon: interoghează periodic contul pentru actualizări (poziții închise extern, re-armare SL)."""
        while not self._stop_event.is_set():
            try:
                self._reconcile_once()
            except Exception:
                pass
            self._stop_event.wait(float(self.live_cfg.reconcile_secs))

    def _reconcile_once(self):
        symbol = self.sym_cfg.symbol
        with self._lock:
            try:
                p = self.broker.position_info(symbol)
                amt = float(p.get("positionAmt", 0.0))
            except Exception:
                amt = None
                return  # dacă nu putem obține poziția, ieșim (vom reîncerca la următorul ciclu)
            if self.state.in_pos and (amt is not None) and abs(amt) < max(self.filters.get("minQty", 0.0), 0.0):
                # Poziția s-a închis (TP/SL sau manual) -> marcăm flat local, anulăm ordinele rămase, persistăm starea
                try:
                    self.broker.cancel_all(symbol)
                except Exception:
                    pass
                self.state = PositionState()
                self._last_live_amt = 0.0
                self.store.save(symbol, {"in_pos": False, "entries": 0})
                self.log.log(symbol, "RECONCILE_FLAT")
            # Re-armare protecții dacă suntem în poziție și nu ieșim acum
            if self.state.in_pos and not self._exiting:
                self.verify_protections(
                    symbol,
                    want_sl=self.state.atr_stop if not math.isnan(self.state.atr_stop) else None,
                    want_tp=None  # trailing TP opțional
                )
            # Persistă periodic starea curentă (pentru recovery)
            self.store.save(symbol, {
                "in_pos": self.state.in_pos,
                "qty": self.state.qty,
                "entry_price": self.state.entry_price,
                "atr_stop": self.state.atr_stop,
                "entries": self.state.entries
            })

    def _account_equity(self) -> float:
        try:
            return float(self.broker.account_equity_usdc())
        except Exception:
            return 0.0

    def _compute_open_risk_pct(self) -> (float, bool):
        """
        Calculează riscul deschis (Open Risk % din equity) pe tot contul, 
        pe baza pozițiilor existente și a SL-urilor aferente.
        Returnează (open_risk_pct, has_missing_sl).
        """
        equity = self._account_equity()
        if equity <= 0:
            return 0.0, False
        positions = self.broker.list_positions()
        orders = self.broker.list_open_orders()
        # Mapare SL (reduceOnly STOP) per simbol
        sl_map: Dict[str, float] = {}
        for o in orders:
            try:
                if o.get("type") == "STOP_MARKET" and (o.get("reduceOnly") is True or str(o.get("reduceOnly", "")).lower() == "true"):
                    sym = o.get("symbol", "").upper()
                    sl_px = float(o.get("stopPrice", 0.0))
                    if sl_px > 0:
                        sl_map[sym] = sl_px  # dacă există mai multe fragmente, păstrăm ultimul setat
            except Exception:
                continue
        total_risk = 0.0
        missing_sl = False
        for pos in positions:
            try:
                sym = pos.get("symbol", "").upper()
                amt = float(pos.get("positionAmt", 0.0))
                if amt == 0:
                    continue
                entry = float(pos.get("entryPrice", 0.0))
                if entry <= 0:
                    continue
                has_sl = sym in sl_map
                if not has_sl:
                    missing_sl = True
                    # conservator: blocăm intrări noi dacă există vreo poziție fără SL
                    continue
                sl_px = sl_map[sym]
                qty = abs(amt)
                # calculăm pierderea potențială dacă s-ar atinge SL (diferența între entry și SL)
                dist = (sl_px - entry) if amt < 0 else (entry - sl_px)
                risk_usd = max(0.0, dist) * qty
                total_risk += (risk_usd / equity)
            except Exception:
                continue
        return float(total_risk), bool(missing_sl)

    def _dd_blocked(self) -> bool:
        """
        Trailing drawdown global: 
        dacă equity curent a scăzut cu >= dd_stop_pct față de peak-ul salvat, 
        blocăm intrările noi (până equity depășește din nou acel peak).
        """
        sym = self.sym_cfg.symbol
        st = self.store.load(sym) or {}
        eq = self._account_equity()
        if eq <= 0:
            return False
        peak = float(st.get("peak_equity", eq))
        if eq > peak:
            # actualizăm peak-ul dacă am atins equity nou maxim
            st["peak_equity"] = eq
            self.store.save(sym, st)
            return False
        drop = (peak - eq) / peak
        return drop >= float(self.live_cfg.dd_stop_pct)

    def _risk_size(self, px_entry: float, sl_price_raw: float) -> float:
        """
        Calculează cantitatea (qty) pentru o poziție nouă, conform managementului de risc:
          - r_avail = cap_eff - OpenRisk (riscul disponibil până la plafon global)
          - r_new = min(r_base, r_avail); dacă r_new < r_min => return 0 (skip trade)
          - qty = floor( (r_new * equity) / dist_to_SL, stepSize ), respectând minQty/minNotional
        """
        equity = self._account_equity()
        if equity <= 0 or px_entry <= 0 or math.isnan(sl_price_raw):
            return 0.0
        open_risk, missing_sl = self._compute_open_risk_pct()
        cap_eff = float(self.live_cfg.risk_cap_pct) * (1.0 - float(self.live_cfg.risk_cap_buffer_pct))
        if missing_sl or open_risk >= cap_eff:
            # există poziții fără SL sau risc deja la plafon -> nu deschidem nimic nou
            return 0.0
        r_base = float(self.live_cfg.risk_base_pct)
        r_min = float(self.live_cfg.risk_min_pct)
        r_avail = max(0.0, cap_eff - open_risk)
        r_new = min(r_base, r_avail)
        if r_new < r_min:
            return 0.0
        # Distanța în $ până la SL (pentru SHORT)
        tick = self.filters.get("tickSize", 0.0)
        sl_px = round_stop_for_short(sl_price_raw, tick)
        dist = max(0.0, sl_px - px_entry)
        if dist <= 0:
            return 0.0
        # Cantitatea teoretică (notional * r_new / dist)
        qty_raw = (r_new * equity) / dist
        # Aplicăm filtre lot size, minQty, minNotional
        step = self.filters.get("stepSize", 0.0)
        minq = self.filters.get("minQty", 0.0)
        min_not = self.filters.get("minNotional", 0.0)
        qty = floor_to_step(max(qty_raw, minq), step)
        if qty <= 0 or qty * px_entry < min_not:
            return 0.0
        return float(qty)

    def on_bar_close(self, symbol: str, bar: Dict[str, Any]):
        # Dacă aveam o ieșire în așteptare (exit_pending), încercăm din nou să închidem și ieșim
        if self.exit_pending:
            if self.ensure_flat(symbol, reason="pending"):
                self.state = PositionState()
                self._last_live_amt = 0.0
                self.exit_pending = False
            return
        # Calculăm semnalul pe bara închisă
        sig = self.signal_fn(symbol, bar)
        # Log de heartbeat la închiderea barei
        bar_time = pd.to_datetime(bar["end"], unit="ms", utc=True)
        print(f"\n[BAR CLOSE] {symbol} {bar_time.strftime('%Y-%m-%d %H:%M')} | Close: {bar['close']:.4f}")
        if sig.get("entry_short", False) or (self.reverse_exit and self.state.in_pos and sig.get("exit_reverse", False)):
            print("="*80)
            if sig.get("entry_short"):
                print(f"🔴 SHORT SIGNAL | SL: {sig.get('atr_sl'):.4f} | TP: {sig.get('tp_level'):.4f}")
            if self.reverse_exit and sig.get("exit_reverse"):
                print(f"🔵 EXIT SIGNAL (reverse)")
            print("="*80)
        # Resetăm contorul de update-uri intrabar (pentru bara nouă)
        self._bar_updates = 0
        # Notă: execuția ieșirii pe semnal de reverse se face intrabar (în on_bar_update)
        # Dacă avem semnal de intrare short, marcăm pending_entry (permitând pyramiding)
        entry_signal = bool(sig.get("entry_short", False))
        allow_pyr = bool(getattr(self.live_cfg, "pyramiding_enabled", False))
        max_batches = max(1, int(getattr(self.live_cfg, "max_entry_batches", 1)))
        start_time_ok = (self.start_time is None) or (bar_time >= self.start_time)
        with self._lock:
            already_in_pos = self.state.in_pos
            current_batches = self.state.entries if already_in_pos else 0
        self.pending_entry = entry_signal and start_time_ok and (
            (not already_in_pos) or (allow_pyr and current_batches < max_batches)
        )
        self._pending_levels["sl"] = sig.get("atr_sl", float("nan"))
        self._pending_levels["tp"] = sig.get("tp_level", float("nan"))
        self._next_bar_start = bar["end"]  # startul următoarei bare = end-ul barei curente
        # Re-aranjăm protecțiile (SL) dacă suntem în poziție și lipsesc, pentru siguranță
        self.verify_protections(
            symbol,
            want_sl=sig.get("atr_sl", float("nan")),
            want_tp=sig.get("tp_level", float("nan")),
        )
        
    def on_bar_update(self, symbol: str, bar: Dict[str, Any]):
        # Este apelat la fiecare update tick al barei curente (parțial)
        with self._lock:
            self._bar_updates += 1
            ready_to_enter = (
                self.pending_entry
                and (self._next_bar_start is not None)
                and (bar["start"] >= self._next_bar_start)
            )
            pending_levels = dict(self._pending_levels)
        print(self._bar_updates, end="\r", flush=True)
        # 1) Execută intrarea dacă avem semnal pending (la deschiderea barei curente)
        if ready_to_enter:
            px_open = float(bar["open"])
            with self._lock:
                prev_qty = self.state.qty if self.state.in_pos else 0.0
                prev_price = self.state.entry_price
                prev_entries = self.state.entries if self.state.in_pos else 0
            qty: float
            # Sizing pe bază de risc dacă e activat, altfel fix
            if getattr(self.live_cfg, "risk_enabled", False):
                if self._dd_blocked():
                    print("**Entry blocked (DD stop active)**")
                    with self._lock:
                        self.pending_entry = False
                    return
                open_risk, missing_sl = self._compute_open_risk_pct()
                if missing_sl:
                    print("**Entry blocked (existing position without SL)**")
                    with self._lock:
                        self.pending_entry = False
                    return
                proposed_sl = pending_levels["sl"]
                risk_qty = self._risk_size(px_open, proposed_sl)
                if risk_qty <= 0:
                    print("**Skipping entry (risk sizing returned 0)**")
                    with self._lock:
                        self.pending_entry = False
                    return
                qty = risk_qty
            else:
                usd = self.sym_cfg.usd_fixed if self.sizing_fn is None else self.sizing_fn(px_open, self.sym_cfg, self.filters)
                qty = calc_qty(px_open, usd, self.filters["stepSize"], self.filters["minQty"], self.filters["minNotional"])
            if qty * px_open >= self.filters.get("minNotional", 0.0) and qty > 0:
                try:
                    self.broker.cancel_all(symbol)
                except Exception:
                    pass
                self._mkt(symbol, side="SELL", qty=qty, reduce_only=False)
                with self._lock:
                    total_qty = prev_qty + qty
                    base_price = prev_price if (prev_qty > 0 and not math.isnan(prev_price)) else px_open
                    avg_price = ((base_price * prev_qty) + (px_open * qty)) / max(total_qty, 1e-12)
                    self.state.in_pos = True
                    self.state.qty = total_qty
                    self.state.entry_price = avg_price
                    self.state.entries = (prev_entries + 1) if prev_entries else 1
                    self._last_entry = {"qty": total_qty, "price": avg_price, "side": "SHORT"}
                self._report_trade_oanda("GOING SHORT", units=-qty, price=px_open, pl=0.0)
                # Plasăm imediat SL și TP server-side după intrare (folosind Mark Price)
                total_for_protection = (prev_qty + qty)
                if not math.isnan(pending_levels["sl"]):
                    sl_px = round_stop_for_short(pending_levels["sl"], self.filters.get("tickSize", 0.0))
                    with self._lock:
                        self.state.atr_stop = sl_px
                    self._sl(symbol, side="BUY", qty=total_for_protection, stop_price=sl_px)
                if not math.isnan(pending_levels["tp"]):
                    tp_px = round_tp_for_short(pending_levels["tp"], self.filters.get("tickSize", 0.0))
                    self._tp(symbol, side="BUY", qty=total_for_protection, tp_price=tp_px)
            with self._lock:
                self.pending_entry = False  # resetăm flag-ul de intrare (o singură execuție)
        # 2) Semnal de reverse intrabar: dacă suntem în poziție și apare semnal de ieșire contrară
        sig_now = self.signal_fn(symbol, bar)
        with self._lock:
            in_pos = self.state.in_pos
        if in_pos and self.reverse_exit and sig_now.get("exit_reverse", False):
            self._dbg("reverse signal -> calling ensure_flat()")
            with self._lock:
                self._exiting = True
            if self.ensure_flat(symbol, reason="reverse-intrabar"):
                exit_px = float(bar["close"])
                qty = float(self._last_entry.get("qty", 0.0))
                ent_px = float(self._last_entry.get("price", exit_px))
                pl = (ent_px - exit_px) * qty  # SHORT P&L
                self._report_trade_oanda("GOING NEUTRAL", units=qty, price=exit_px, pl=pl)
                with self._lock:
                    self.state = PositionState()
                    self._last_live_amt = 0.0
                    self.exit_pending = False
                    self._last_entry = {"qty": 0.0, "price": math.nan, "side": None}
                    self._exiting = False
            else:
                self._dbg("reverse signal -> ensure_flat() pending/failed (will retry)")
                with self._lock:
                    self.exit_pending = True
            return
        # 3) Ajustare dinamică SL/TP intrabar (trailing) – doar dacă avem poziție deschisă
        if self.state.in_pos:
            new_sl = sig_now.get("atr_sl", float("nan"))
            new_tp = sig_now.get("tp_level", float("nan"))
            tick = self.filters.get("tickSize, ", 0.0)
            sl_px = round_stop_for_short(new_sl, tick) if not math.isnan(new_sl) else float("nan")
            tp_px = round_tp_for_short(new_tp, tick)   if not math.isnan(new_tp) else float("nan")
            # Histerezis: nu re-setăm dacă modificarea < 0.2%
            def moved(a, b, thr=0.002):
                return (not math.isnan(a)) and (not math.isnan(b)) and (abs(a - b) / max(b, 1e-12) > thr)
            need_reset = False
            # Stair-step: la short, SL poate doar să scadă (preț mai mic)
            if not math.isnan(sl_px):
                if math.isnan(self.state.atr_stop) or sl_px < self.state.atr_stop or moved(sl_px, self.state.atr_stop):
                    need_reset = True
            if not math.isnan(tp_px):
                need_reset = True  # TP poate fi modificat oricând
            if need_reset:
                # Protejăm secvența de re-ordonare cu lock (evită conflicte cu reconcilierea)
                with self._lock:
                    try:
                        self.broker.cancel_all(symbol)
                    except Exception:
                        pass
                    if not math.isnan(sl_px):
                        if math.isnan(self.state.atr_stop) or sl_px < self.state.atr_stop:
                            self.state.atr_stop = sl_px
                            self._sl(symbol, side="BUY", qty=self.state.qty, stop_price=sl_px)
                    if not math.isnan(tp_px):
                        self._tp(symbol, side="BUY", qty=self.state.qty, tp_price=tp_px)
        # 4) Asigurare: protecții la loc (re-armare dacă lipsesc)
        self.verify_protections(
            symbol,
            want_sl=sig_now.get("atr_sl", float("nan")),
            want_tp=sig_now.get("tp_level", float("nan")),
        )
        # 5) Verificare ușoară: dacă un SL/TP a închis poziția (în fundal), logăm ieșirea
        if self.state.in_pos and (self._bar_updates % 12 == 0):
            try:
                p = self.broker.position_info(symbol)
                amt = abs(float(p.get("positionAmt", 0.0)))
            except Exception:
                amt = None
            if amt is not None and amt <= max(self.filters.get("minQty", 0.0), 0.0):
                exit_px = float(bar["close"])
                qty = float(self._last_entry.get("qty", 0.0))
                ent_px = float(self._last_entry.get("price", exit_px))
                pl = (ent_px - exit_px) * qty  # P&L SHORT
                self._report_trade_oanda("GOING NEUTRAL", units=qty, price=exit_px, pl=pl)
                try:
                    self.broker.cancel_all(symbol)
                except Exception:
                    pass
                self.state = PositionState()
                self._last_live_amt = 0.0
                self._last_entry = {"qty": 0.0, "price": math.nan, "side": None}


def load_binance_credentials(cfg_path: str, use_testnet: bool) -> tuple[str, str, str]:
    """Încărcare sigură a credentialelor Binance cu verificări explicite."""
    cfg = configparser.ConfigParser()
    cfg_file = Path(cfg_path)
    if not cfg_file.exists():
        raise FileNotFoundError(f"Fișierul de configurare {cfg_path} nu există.")
    if not cfg.read(cfg_path):
        raise FileNotFoundError(f"Fișierul de configurare {cfg_path} nu a putut fi citit.")
    section = "binance_testnet" if use_testnet else "binance"
    if section not in cfg:
        raise KeyError(f"Secțiunea {section} lipsește din {cfg_path}.")
    def _get_required(key: str) -> str:
        val = cfg[section].get(key, "").strip()
        if not val:
            raise ValueError(f"Cheia '{key}' lipsește sau este goală în secțiunea {section}.")
        return val
    api_key = _get_required("api_key")
    api_secret = _get_required("secret_key")
    base_url = cfg[section].get("base_url", "").strip()
    return api_key, api_secret, base_url

# ===== Inițializare motor de semnal și Runner (exemplu de utilizare) =====
engine = Super8SignalEngine(ind_params, short_params)
symbol = "AVAXUSDC"
broker = BinanceFuturesAdapter()
USE_TESTNET = False  # folosește testnet (True) sau mainnet (False)
# Curățare eventual runner existent (în mediul interactiv)
try:
    del runner
except NameError:
    pass

API_KEY = API_SECRET = BASE_URL = ""
try:
    API_KEY, API_SECRET, BASE_URL = load_binance_credentials("binance.cfg", USE_TESTNET)
except Exception as exc:
    raise RuntimeError(f"[CONFIG] Eroare la citirea credentialelor Binance: {exc}") from exc

runner = Super8LiveRunner(
    broker=broker,
    live_cfg=LiveConfig(
        api_key=API_KEY,
        api_secret=API_SECRET,
        base_url=BASE_URL,
        timeframe="2h",
        leverage=1,
        margin_type="ISOLATED",
        hedge_mode=False,
        testnet=USE_TESTNET,
        dry_run=False,
        # Setări RISK (dacă se activează)
        risk_enabled=False,
        risk_base_pct=0.010,      # 1.0% per tranzacție
        risk_cap_pct=0.073,
        risk_cap_buffer_pct=0.05,
        risk_min_pct=0.002,
        dd_stop_pct=0.075,
        reconcile_secs=30,
        persist_path_tpl="state_{symbol}.json",
        history_keep_bars=700
    ),
    sym_cfg=SymbolConfig(symbol="AVAXUSDC", usd_fixed=5, pct_equity=None),
    indicator_fn=lambda df: None,  # (Poate fi înlocuit cu un indicator custom, dacă e cazul)
    signal_fn=lambda sym, bar: engine.on_bar_close(sym, bar),
    sizing_fn=make_sizing_fn(broker),
    short_params=engine.sp,
)

# Configurări de mediu și bootstrap runner
print("[CFG]", "testnet=", runner.live_cfg.testnet, "dry_run=", runner.live_cfg.dry_run)
if __name__ == "__main__":
    try:
        runner.bootstrap()
        # Seed istoric inițial (lookback + câteva bare extra)
        need_bars = engine.lookback + 10
        seed_bars = runner.broker.fetch_klines(symbol, runner.live_cfg.timeframe, need_bars)
        engine.seed(seed_bars)
        engine.keep_bars = runner.live_cfg.history_keep_bars
        print(f"[SEED] {symbol} {runner.live_cfg.timeframe} bars={len(seed_bars)} lookback={engine.lookback}")
        # Pornește stream-ul de date live (websocket)
        runner.broker.stream_klines(
            symbol,
            runner.live_cfg.timeframe,
            on_close=runner.on_bar_close,
            on_update=runner.on_bar_update
        )
        # Menține scriptul activ până la întrerupere manuală (Ctrl+C)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("KeyboardInterrupt -> Oprire runner (flatten pozitie dacă este cazul)...")
            runner.stop(flatten=True)
            print("Runner oprit și pozițiile închise (dacă existau).")
    except Exception as exc:
        print(f"[ERR] Execuție oprită din cauza unei erori neașteptate: {exc}")
        try:
            runner.stop(flatten=True)
        except Exception as stop_err:
            print(f"[WARN] Oprire runner a eșuat: {stop_err}")
        raise


# %% cell 1
# Rulează asta o singură dată ca să fii sigur că e totul oprit.
try:
    runner.stop(flatten=True)
    print("Runner oprit + flatten încercat.")
except Exception:
    print("Runner nu era definit sau deja oprit. Continuăm.")


# %% cell 2
