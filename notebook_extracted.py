# CELL 0
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



# ---------- Config ----------
@dataclass
class LiveConfig:
    api_key: str
    api_secret: str
    base_url: str = ""          # ex: "https://fapi.binance.com" / testnet
    timeframe: str = "2h"   # kline interval
    leverage: int = 1
    margin_type: str = "ISOLATED"
    hedge_mode: bool = False   # One-Way = False
    testnet: bool = True
    dry_run: bool = False                 # << nou
    log_csv: Optional[str] = "trades_log.csv"  # << nou
    # ---- RISK (disabled implicit) ----
    risk_enabled: bool = False          # << OFF by default
    risk_base_pct: float = 0.01         # 1.0% per-trade (cand e ON)
    risk_min_pct: float = 0.0025        # sub acest prag -> skip trade
    risk_cap_pct: float = 0.073         # cap global 7.3%
    risk_cap_buffer_pct: float = 0.05   # buffer 5% la decizie (anti-race)

    # ---- DRAWdown (global trailing) ----
    dd_stop_pct: float = 0.075          # 7.5% DD => blocam intrari noi

    # ---- Reconciliere & persist ----
    reconcile_secs: int = 15            # polling pozitii/ordere (sec)
    persist_path_tpl: str = "state_{symbol}.json"

    # ---- Pruning istoric ----
    history_keep_bars: int = 700        # pastram ~lookback + buffer

@dataclass
class SymbolConfig:
    symbol: str             # ex: "BTCUSDT"
    usd_fixed: float = 1.0  # suma fixa USD/ordine (initial)
    pct_equity: Optional[float] = None  # alternativ (% din equity) - folosit daca e setat
    min_usd: float = 5.0    # prag fallback
    max_slip_bps: int = 50  # optional safety

@dataclass
class PositionState:
    in_pos: bool = False
    qty: float = 0.0
    entry_price: float = math.nan
    atr_stop: float = math.nan   # stair-step memorat
    tp_order_id: Optional[str] = None
    sl_order_id: Optional[str] = None

# ---------- Utils: qty calc cu filtre ----------
def floor_to_step(x: float, step: float) -> float:
    if step <= 0: return x
    return math.floor(x / step) * step



def calc_qty(price: float, usd_target: float,
             stepSize: float, minQty: float, minNotional: float) -> float:
    # tintim notionalul dorit
    qty = floor_to_step(max(0.0, usd_target / max(price, 1e-12)), stepSize)

    # respecta minQty
    if qty < minQty:
        qty = ceil_to_step(minQty, stepSize)

    # daca inca suntem sub minNotional, ROTUNJIM IN SUS la stepSize
    if qty * price < minNotional:
        need = ceil_to_step(minNotional / max(price, 1e-12), stepSize)
        qty = max(qty, need)

    return max(0.0, qty)


def ceil_to_step(x: float, step: float) -> float:
    if step <= 0: return x
    return math.ceil(x / step) * step

def round_to_tick(x: float, tick: float) -> float:
    if tick and tick > 0:
        q = Decimal(str(tick))
        d = (Decimal(str(x)) / q).to_integral_value(rounding=ROUND_DOWN) * q
        return float(d)  # ex.: 113005.4, fara .00000000001
    return x

def ceil_to_tick(x: float, tick: float) -> float:
    if tick and tick > 0:
        q = Decimal(str(tick))
        d = (Decimal(str(x)) / q).to_integral_value(rounding=ROUND_UP) * q
        return float(d)
    return x


# Pentru SHORT:
def round_stop_for_short(stop_price: float, tick: float) -> float:
    # SL (BUY) trebuie rotunjit in SUS
    return ceil_to_tick(stop_price, tick)

def round_tp_for_short(tp_price: float, tick: float) -> float:
    # TP (BUY) poate fi rotunjit in JOS (sa nu fie peste nivelul dorit)
    return round_to_tick(tp_price, tick)


import csv, os, datetime as dt

class TradeLogger:
    def __init__(self, path: Optional[str]):
        self.path = path
        if path and (not os.path.exists(path)):
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["ts","symbol","action","side","qty","price","extra"])

    def log(self, symbol, action, side="", qty=0.0, price=math.nan, extra=""):
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
        if not os.path.exists(p): return {}
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



# ---------- Broker interface ----------
OnBarClose = Callable[[str, Dict[str, Any]], None]  # (symbol, bar_dict)

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

# CELL 1
# ==== Super8SignalEngine (stateless pe bara, cu istoric intern) ====
import pandas as pd
import numpy as np

def _rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1/float(n), adjust=False).mean()

class Super8SignalEngine:
    def __init__(self, ind_p: dict, sh_p: dict):
        self.p = ind_p.copy()
        self.sp = sh_p.copy()
        self.df = pd.DataFrame()   # ne tinem istoricul OHLCV
        self.keep_bars = 700  # va fi setat din runner din LiveConfig
        # lookback minim pt. primele semnale
        self.lookback = int(max(
            self.p["sEma_Length"], self.p["BB_Length"], self.p["DClength"],
            self.p["ADX_len"], self.p["slowLength"], self.sp["atrPeriodSl"], 60
        ))

    def seed(self, bars: list[dict]):
        """bars: lista de dicturi {'open','high','low','close','volume', 'start','end'}"""
        if not bars: return
        d = pd.DataFrame(bars)
        d["time"] = pd.to_datetime(d["end"], unit="ms", utc=True)
        d = d.set_index("time")[["open","high","low","close","volume"]].astype(float)
        d.rename(columns={"close":"Price"}, inplace=True)
        self.df = d

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.p
        out = pd.DataFrame(index=df.index)

        # EMA
        sEMA = df["Price"].ewm(span=int(p["sEma_Length"]), adjust=False).mean()
        fEMA = df["Price"].ewm(span=int(p["fEma_Length"]), adjust=False).mean()
        out["EMA_longCond"]  = (fEMA > sEMA) & (sEMA > sEMA.shift(1))
        out["EMA_shortCond"] = (fEMA < sEMA) & (sEMA < sEMA.shift(1))

        # ADX (Wilder)
        tr = np.maximum(df["high"] - df["low"],
                        np.maximum((df["high"]-df["Price"].shift(1)).abs(),
                                   (df["low"] -df["Price"].shift(1)).abs()))
        up = df["high"].diff(); dn = -df["low"].diff()
        plus_dm  = np.where((up>dn)&(up>0), up, 0.0)
        minus_dm = np.where((dn>up)&(dn>0), dn, 0.0)
        tr_s    = _rma(pd.Series(tr, index=df.index), int(p["ADX_len"]))
        plus_s  = _rma(pd.Series(plus_dm, index=df.index), int(p["ADX_len"]))
        minus_s = _rma(pd.Series(minus_dm, index=df.index), int(p["ADX_len"]))
        plus_di  = (plus_s/tr_s)*100.0; minus_di = (minus_s/tr_s)*100.0
        dx  = ((plus_di - minus_di).abs() / (plus_di + minus_di).clip(lower=1e-10))*100.0
        adx = _rma(dx, int(p.get("ADX_smo", p["ADX_len"])))
        out["ADX_longCond"]  = (plus_di > minus_di) & (adx > float(p["th"]))
        out["ADX_shortCond"] = (plus_di < minus_di) & (adx > float(p["th"]))

        # SAR (versiune TV scurta)
        h = df["high"].to_numpy(); l = df["low"].to_numpy()
        start, step, smax = float(p["Sst"]), float(p["Sinc"]), float(p["Smax"])
        psar = np.zeros(len(df)); up = True
        if len(df)>=2: up = bool(df["Price"].iloc[1] >= df["Price"].iloc[0])
        af = start; ep = h[0] if up else l[0]; psar[0] = l[0] if up else h[0]
        for i in range(1,len(df)):
            psar[i] = psar[i-1] + af*(ep-psar[i-1])
            if up:
                psar[i] = min(psar[i], l[i-1] if i>=1 else l[i])
                if h[i] > ep: ep = h[i]; af = min(af+step, smax)
                if l[i] < psar[i]: up=False; psar[i]=ep; ep=l[i]; af=start
            else:
                psar[i] = max(psar[i], h[i-1] if i>=1 else h[i])
                if l[i] < ep: ep = l[i]; af = min(af+step, smax)
                if h[i] > psar[i]: up=True; psar[i]=ep; ep=h[i]; af=start
        sar = pd.Series(psar, index=df.index)
        out["SAR_longCond"]  = sar < df["Price"]
        out["SAR_shortCond"] = sar > df["Price"]

        # MACD clasic
        fast, slow, sig = int(p["fastLength"]), int(p["slowLength"]), int(p["signalLength"])
        lMA = df["Price"].ewm(span=fast, adjust=False).mean() - df["Price"].ewm(span=slow, adjust=False).mean()
        sMA = lMA.ewm(span=sig, adjust=False).mean()
        hist = lMA - sMA
        out["MACD_longCond"]  = hist > 0
        out["MACD_shortCond"] = hist < 0

        # Bollinger
        L = int(p["BB_Length"]); m = float(p["BB_mult"])
        mid = df["Price"].rolling(L, min_periods=L).mean()
        std = df["Price"].rolling(L, min_periods=L).std(ddof=0)
        upper = mid + m*std; lower = mid - m*std
        out["BB_upper"]  = upper; out["BB_lower"] = lower
        out["BB_middle"] = mid
        out["BB_width"]  = (upper - lower) / mid

        # Volum
        vol_sma = df["volume"].rolling(int(p["sma_Length"]), min_periods=1).mean()
        vol_flag = df["volume"] > vol_sma * float(p["volume_f"])
        out["VOL_longCond"] = vol_flag; out["VOL_shortCond"] = vol_flag

        # praguri
        out["bbMinWidth01"] = float(p["bbMinWidth01"]) / 100.0
        return out

    def on_bar_close(self, symbol: str, bar: dict) -> dict:
        # adauga bara noua in istoric
        t = pd.to_datetime(bar["end"], unit="ms", utc=True)
        self.df.loc[t, ["open","high","low","Price","volume"]] = [
            float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"]), float(bar["volume"])
        ]
        df = self.df.copy()

        # daca nu avem inca lookback suficient -> nu semnalam nimic
        if len(df) < self.lookback:
            return {"entry_short": False, "exit_reverse": False, "atr_sl": float("nan"), "tp_level": float("nan")}

        ind = self._compute_indicators(df)
        b  = ind.iloc[-1]     # bara curent inchisa
        px = df["Price"].iloc[-1]

        # ====== conditii ca in backtest ======
        EMA_s  = bool(b["EMA_shortCond"]);  EMA_l  = bool(b["EMA_longCond"])
        ADX_s  = bool(b["ADX_shortCond"]);  ADX_l  = bool(b["ADX_longCond"])
        SAR_s  = bool(b["SAR_shortCond"]);  SAR_l  = bool(b["SAR_longCond"])
        MACD_s = bool(b["MACD_shortCond"]); MACD_l = bool(b["MACD_longCond"])
        VOL_s  = bool(b["VOL_shortCond"]);  VOL_l  = bool(b["VOL_longCond"])

        # BB_short01
        bbw_ok = bool(b["BB_width"] > b["bbMinWidth01"])
        cross_over_upper = (df["high"].shift(1).iloc[-1] <= ind["BB_upper"].shift(1).iloc[-1]) and \
                           (df["high"].iloc[-1]        >  ind["BB_upper"].iloc[-1])
        BB_short01 = (not ADX_l) and EMA_s and bbw_ok and bool(cross_over_upper)

        shortCond = EMA_s and ADX_s and SAR_s and MACD_s and VOL_s
        entry_short = bool(shortCond or BB_short01)

        # reverse exit
        exit_reverse = bool(EMA_l or ADX_l or SAR_l or MACD_l)

        # ====== nivele TP/SL ======
        # ATR (Wilder) + SL brut pentru SHORT
        tr = np.maximum(df["high"] - df["low"],
                        np.maximum((df["high"]-df["Price"].shift(1)).abs(),
                                   (df["low"] -df["Price"].shift(1)).abs()))
        atr = _rma(tr, int(self.sp["atrPeriodSl"]))
        atr_sl_raw = float(df["high"].iloc[-1] + atr.iloc[-1] * float(self.sp["multiplierPeriodSl"]))

        # Donchian pentru TP (daca sp['TP_options'] include Donchian)
        DCl = df["low"].rolling(int(self.p["DClength"]), min_periods=int(self.p["DClength"])).min().iloc[-1]
        tp_normal = px * (1.0 - float(self.sp["tp"])/100.0)
        tp_level  = min(tp_normal, DCl) if self.sp.get("TP_options","Both")=="Both" else tp_normal
        # --- PRUNING istoric ---
        if len(self.df) > int(getattr(self, "keep_bars", 700)):
            self.df = self.df.tail(int(getattr(self, "keep_bars", 700)))

        return {
            "entry_short": entry_short,
            "exit_reverse": exit_reverse,
            "atr_sl": atr_sl_raw,     # runner aplica "stair-step only down"
            "tp_level": tp_level
        }


# CELL 2
def make_sizing_fn(broker):
    def sizing(px: float, sym_cfg: SymbolConfig, filters: Dict[str, Any]) -> float:
        if sym_cfg.pct_equity is not None and sym_cfg.pct_equity > 0:
            try:
                eq = float(broker.account_equity_usdt())
            except Exception:
                eq = 0.0
            return max(sym_cfg.min_usd, eq * float(sym_cfg.pct_equity))
        return max(sym_cfg.min_usd, float(sym_cfg.usd_fixed))
    return sizing


# CELL 3
class BinanceFuturesAdapter(BrokerAdapter):
    """
    Implementare pentru Binance USD-M Futures (One-Way, ISOLATED).
    Functioneaza pe testnet sau mainnet in functie de `LiveConfig`.
    """
    def __init__(self):
        self.cfg: Optional[LiveConfig] = None
        self.s: Optional[requests.Session] = None
        self.rest_base = None
        self.ws_base = None
        self._streams = {}  # key: "SYMBOL_INTERVAL" -> {"stop": Event, "ws": WebSocketApp | None}

    # ---------- internals ----------
    def _ts(self) -> int:
        return int(time.time() * 1000)
    
    def _sync_time(self):
        try:
            r = self.s.get(self.rest_base + "/fapi/v1/time", timeout=5)
            r.raise_for_status()
            srv = int(r.json()["serverTime"])
            loc = int(time.time() * 1000)
            self._t_offset = srv - loc
        except Exception:
            self._t_offset = 0


    def _sign(self, q: dict) -> str:
        query = urlencode(q, doseq=True)
        return hmac.new(self.cfg.api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()

    def _send(self, method: str, path: str, params: dict | None = None, signed: bool = False):
        params = params or {}
        base_params = params.copy()
        url = self.rest_base + path
        headers = {"X-MBX-APIKEY": self.cfg.api_key}

        for i in range(5):  # pana la 5 incercari
            try:
                req_params = base_params.copy()
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
                    raise ValueError("method invalid")

                # retry pe 429/5xx
                if r.status_code in (429, 418) or 500 <= r.status_code < 600:
                    raise requests.HTTPError(response=r)
                r.raise_for_status()
                return r.json()
            
            except requests.HTTPError as e:
                # --- LOG BODY PENTRU DIAGNOSTIC ---
                try:
                    err_text = e.response.text
                    status = e.response.status_code
                except Exception:
                    err_text = str(e)
                    status = -1
                if i == 4:
                    raise RuntimeError(f"HTTP {status} {path} -> {err_text}") from e
                time.sleep(0.4 * (2 ** i))

            except requests.RequestException:
                if i == 4:
                    raise
                time.sleep(0.4 * (2 ** i))


    # ---------- public ----------
    def connect(self, cfg: LiveConfig) -> None:
        print("[ENV] testnet =", cfg.testnet)
        self.cfg = cfg
        self.s = requests.Session()
        # baze URL testnet/mainnet
        if cfg.testnet:
            self.rest_base = "https://testnet.binancefuture.com"
            self.ws_base = "wss://stream.binancefuture.com/stream"
        else:
            self.rest_base = "https://fapi.binance.com"
            self.ws_base = "wss://fstream.binance.com/stream"

        self._t_offset = 0
        self._sync_time()
        print("[BASE]", self.rest_base)
        # setari cont (ordine conteaza)
        #self.set_hedge_mode(False)  # One-Way
        #self.set_margin_type("ALL", cfg.margin_type.upper() == "ISOLATED")  # default "ALL": incercam pe toate perechile
        # leverage il setam per simbol in bootstrap-ul runner-ului (ai deja apel acolo)

    def exchange_info(self, symbol: str) -> Dict[str, Any]:
        data = self._send("GET", "/fapi/v1/exchangeInfo", params={"symbol": symbol.upper()}, signed=False)
        sym = data["symbols"][0]
        tickSize = stepSize = minQty = minNotional = None
        for f in sym["filters"]:
            t = f["filterType"]
            if t == "PRICE_FILTER":
                tickSize = float(f["tickSize"])
            elif t == "LOT_SIZE":
                stepSize = float(f["stepSize"]); minQty = float(f["minQty"])
            elif t in ("MIN_NOTIONAL", "NOTIONAL"):
                # futures folosesc adesea MIN_NOTIONAL
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
                "open": float(k[1]), "high": float(k[2]), "low": float(k[3]), "close": float(k[4]),
                "volume": float(k[5])
            })
        return out

    def stream_klines(self, symbol: str, interval: str, on_close: OnBarClose, on_update: Callable[[str, Dict[str, Any]], None] = None) -> None:
        stream = f"{symbol.lower()}@kline_{interval}"
        key = f"{symbol.upper()}_{interval}"
        stop = threading.Event()
        self._streams[key] = {"stop": stop, "ws": None}

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
            while not stop.is_set():
                ws = WebSocketApp(f"{self.ws_base}?streams={stream}", on_message=_on_msg)
                self._streams[key]["ws"] = ws
                try:
                    ws.run_forever(ping_interval=15, ping_timeout=10)
                except Exception:
                    pass
                if stop.is_set():
                    break
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
            self._streams[key]["ws"] = None

        threading.Thread(target=_run, daemon=True).start()



    def position_info(self, symbol: str) -> Dict[str, Any]:
        data = self._send("GET", "/fapi/v2/positionRisk", params={"symbol": symbol.upper()}, signed=True)
        if isinstance(data, list) and data:
            p = data[0]
        else:
            p = data
        return {
            "symbol": p.get("symbol", symbol.upper()),
            "positionAmt": float(p.get("positionAmt", 0.0)),
            "entryPrice": float(p.get("entryPrice", 0.0)),
            "unRealizedPnL": float(p.get("unRealizedProfit", 0.0)),
        }
    
    def account_equity_usdt(self) -> float:
        # Prefera equity total (wallet) ; fallback la availableBalance.
        try:
            data = self._send("GET", "/fapi/v2/account", params={}, signed=True)
            return float(data.get("totalWalletBalance", 0.0))
        except Exception:
            pass
        try:
            bal = self._send("GET", "/fapi/v2/balance", params={}, signed=True)
            for a in bal:
                if a.get("asset") == "USDT":
                    return float(a.get("balance", a.get("availableBalance", 0.0)))
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
        self._send("POST", "/fapi/v1/leverage", params={"symbol": symbol.upper(), "leverage": int(x)}, signed=True)

    def set_margin_type(self, symbol: str, isolated: bool) -> None:
        mode = "ISOLATED" if isolated else "CROSSED"
        try:
            if symbol == "ALL":
                return
            self._send(
                "POST", "/fapi/v1/marginType",
                params={"symbol": symbol.upper(), "marginType": mode},
                signed=True
            )
        except Exception as e:
            t = str(e)
            # OK daca deja e setat sau esti in Credits Mode (nu permite ISOLATED)
            if ("No need to change margin type" in t) or ("-4046" in t) or ("-4175" in t) or ("credit status" in t):
                print("[INFO] MarginType fortat CROSS / Credits Mode -> continui.")
                return
            raise


    def set_hedge_mode(self, on: bool) -> None:
        # One-Way: dualSidePosition = false
        params = {"dualSidePosition": "true" if on else "false"}
        try:
            self._send("POST", "/fapi/v1/positionSide/dual", params=params, signed=True)
        except Exception as e:
            t = str(e)
            # daca e deja in modul dorit, tratam ca SUCCES
            if ("-4059" in t) or ("No need to change position side" in t):
                print("[INFO] Hedge mode deja setat -> OK")
                return
            raise

    def cancel_all(self, symbol: str) -> None:
        self._send("DELETE", "/fapi/v1/allOpenOrders", params={"symbol": symbol.upper()}, signed=True)
        
    def open_orders(self, symbol: str) -> list[dict]:
        return self._send(
            "GET", "/fapi/v1/openOrders",
            params={"symbol": symbol.upper()},
            signed=True
        )
    
    
    def list_positions(self) -> list[dict]:
        """Toate pozitiile (USD-M)."""
        try:
            data = self._send("GET", "/fapi/v2/positionRisk", params={}, signed=True)
            return data if isinstance(data, list) else [data]
        except Exception:
            return []

    def list_open_orders(self) -> list[dict]:
        """Toate ordinele deschise (optional pe tot contul)."""
        try:
            # fara 'symbol' -> toate simbolurile
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
        # One-Way -> fii explicit
        if not getattr(self.cfg, "hedge_mode", False):
            params["positionSide"] = "BOTH"
        return self._send("POST", "/fapi/v1/order", params=params, signed=True)

    def place_stop_market(self, symbol: str, side: str, qty: float, stop_price: str,
                          reduce_only: bool = True) -> Dict[str, Any]:
        params = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": "STOP_MARKET",
            "stopPrice": stop_price,            # STRING la tick!
            "closePosition": "false",
            "quantity": qty,
            "reduceOnly": "true" if reduce_only else "false",
            "workingType": "MARK_PRICE",
            "priceProtect": "true",
            "newOrderRespType": "RESULT",
        }
        if not getattr(self.cfg, "hedge_mode", False):
            params["positionSide"] = "BOTH"
        return self._send("POST", "/fapi/v1/order", params=params, signed=True)

    def place_take_profit_market(self, symbol: str, side: str, qty: float, tp_price: str,
                                 reduce_only: bool = True) -> Dict[str, Any]:
        params = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": tp_price,              # STRING la tick!
            "closePosition": "false",
            "quantity": qty,
            "reduceOnly": "true" if reduce_only else "false",
            "workingType": "MARK_PRICE",
            "priceProtect": "true",
            "newOrderRespType": "RESULT",
        }
        if not getattr(self.cfg, "hedge_mode", False):
            params["positionSide"] = "BOTH"
        return self._send("POST", "/fapi/v1/order", params=params, signed=True)

    def mark_price(self, symbol: str) -> float:
        """Mark price (pentru triggere)."""
        r = self._send("GET", "/fapi/v1/premiumIndex", params={"symbol": symbol.upper()}, signed=False)
        return float(r.get("markPrice", 0.0))

    def last_price(self, symbol: str) -> float:
        """Ultimul pret tranzactionat (fallback daca vrei)."""
        r = self._send("GET", "/fapi/v1/ticker/price", params={"symbol": symbol.upper()}, signed=False)
        return float(r.get("price", 0.0))

    def place_close_all_stop_market(self, symbol: str, side: str, stop_price: str) -> Dict[str, Any]:
        """STOP_MARKET closePosition=true (inchide TOT fara qty)."""
        params = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": "STOP_MARKET",
            "stopPrice": stop_price,           # STRING la tick!
            "closePosition": "true",
            "workingType": "MARK_PRICE",
            "priceProtect": "true",
            "newOrderRespType": "RESULT",
        }
        if not getattr(self.cfg, "hedge_mode", False):
            params["positionSide"] = "BOTH"
        return self._send("POST", "/fapi/v1/order", params=params, signed=True)

    def place_close_all_take_profit_market(self, symbol: str, side: str, tp_price: str) -> Dict[str, Any]:
        """TP_MARKET closePosition=true (inchide TOT fara qty)."""
        params = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": tp_price,             # STRING la tick!
            "closePosition": "true",
            "workingType": "MARK_PRICE",
            "priceProtect": "true",
            "newOrderRespType": "RESULT",
        }
        if not getattr(self.cfg, "hedge_mode", False):
            params["positionSide"] = "BOTH"
        return self._send("POST", "/fapi/v1/order", params=params, signed=True)



# === parametri (poti pune ai tai din backtest) ===
ind_params = dict(
    fEma_Length=61, sEma_Length=444,
    ADX_len=15, ADX_smo=10, th=5.47,
    fastLength=20, slowLength=43, signalLength=12,
    BB_Length=89, BB_mult=6.281,
    sma_Length=81, volume_f=0.87,
    DClength=79,
    Sst=0.10, Sinc=0.04, Smax=0.40,
    bbMinWidth01=9.3, bbMinWidth02=0.0
)
short_params = dict(
    TP_options="Both", SL_options="Both",
    tp=3.6, sl=8.0, atrPeriodSl=50, multiplierPeriodSl=36.84, trailOffset=0.38
)

# ---------- Runner (logica "lag 1 bara" & ordine server-side) ----------
class Super8LiveRunner:
    def __init__(self, broker: BrokerAdapter, live_cfg: LiveConfig, sym_cfg: SymbolConfig,
                 indicator_fn, signal_fn, sizing_fn=None):
        self.broker = broker
        self.live_cfg = live_cfg
        self.sym_cfg = sym_cfg
        self.state = PositionState()
        self.filters: Dict[str, Any] = {}
        self.indicator_fn = indicator_fn  # (df)->ind; reusezi din backtest
        self.signal_fn = signal_fn        # (df,ind)-> {entry_short, exit_reverse, atr_sl, tp_level}
        self.sizing_fn = sizing_fn        # optional override
        self.exit_pending = False  # tinem minte ca trebuie sa iesim cu orice pret
        self.dry_run = getattr(self.live_cfg, "dry_run", True)
        self.log = TradeLogger(getattr(self.live_cfg, "log_csv", None))
        self.pending_entry = False
        self._next_bar_start = None
        self._pending_levels = {"sl": math.nan, "tp": math.nan}
        self._bar_updates = 0
        # --- noi (pentru log OANDA) ---
        self._cum_pl = 0.0                 # P&L cumulat
        self._last_entry = {"qty": 0.0, "price": math.nan, "side": None}
        # --- NEW: persist & sync ---
        self.store = StateStore(self.live_cfg.persist_path_tpl)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        # --- anti-race: semafor de iesire (suprima re-armarea SL/TP) ---
        self._exiting = False
        # fallback dublu armat (nu mai dam cancel_all peste el)
        self._fallback_armed = False

        
    # --- Helpers mici pentru debug/erori (dupa __init__) ---
    def _dbg(self, msg: str):
        print(f"[DBG] {msg}")

    def _err(self, msg: str):
        print(f"[ERR] {msg}")
        
    def _tick_decimals(self) -> int:
        t = float(self.filters.get("tickSize", 0.0))
        if t <= 0: return 0
        s = f"{t:.10f}".rstrip("0").rstrip(".")
        return len(s.split(".")[1]) if "." in s else 0

    def _fmt_px(self, px: float) -> str:
        # formateaza in sir cu exact decimalele permise de tick
        dec = self._tick_decimals()
        return f"{px:.{dec}f}"


    def bootstrap(self):
        self.broker.connect(self.live_cfg)
        self.broker.set_hedge_mode(False)            # One-Way
        self.broker.set_margin_type(self.sym_cfg.symbol, True)  # ISOLATED
        self.broker.set_leverage(self.sym_cfg.symbol, self.live_cfg.leverage)
        self.filters = self.broker.exchange_info(self.sym_cfg.symbol)
        # --- NEW: load persisted state & set engine.keep_bars ---
        try:
            if hasattr(self, "engine"):
                self.engine.keep_bars = int(self.live_cfg.history_keep_bars)
        except Exception:
            pass

        st = self.store.load(self.sym_cfg.symbol) or {}
        # reconciliere rapida cu bursa
        try:
            p = self.broker.position_info(self.sym_cfg.symbol)
            live_amt = float(p.get("positionAmt", 0.0))
        except Exception:
            live_amt = 0.0

        if live_amt == 0:
            # pozitie nu exista in piata -> reset local
            self.state = PositionState()
        else:
            # exista pozitie in piata: rehidrateaza din fisier, cu fallback la bursa
            self.state.in_pos = True
            self.state.qty = abs(live_amt)
            live_entry = 0.0
            try:
                live_entry = float(p.get("entryPrice", 0.0))
            except Exception:
                pass
            self.state.entry_price = float(st.get("entry_price", live_entry if live_entry > 0 else self.state.entry_price))
            self.state.atr_stop = float(st.get("atr_stop", self.state.atr_stop))
            # seteaza si _last_entry pentru P&L corect la iesire
            if self.state.entry_price and not math.isnan(self.state.entry_price):
                self._last_entry = {
                    "qty": self.state.qty,
                    "price": self.state.entry_price,
                    "side": "SHORT" if live_amt < 0 else "LONG"
                }


        # --- NEW: porneste thread de reconciliere ---
        t = threading.Thread(target=self._reconcile_loop, daemon=True)
        t.start()

        
        
    def ensure_flat(self, symbol: str, reason: str = "reverse", max_retries: int = 10, sleep_s: float = 0.6) -> bool:
        """
        Inchide pozitia cu retry + debounce; daca MARKET nu merge sau e sub notional,
        foloseste fallback-ul 'double-trigger' (closePosition=True).
        """
        step = self.filters.get("stepSize", 0.0)
        minq = self.filters.get("minQty", 0.0)
        tick = self.filters.get("tickSize", 0.0)
        min_notional = self.filters.get("minNotional", 0.0)

        self._exiting = True
        self._dbg(f"ensure_flat start (reason={reason})")

        def _double_trigger_close(side: str) -> bool:
            # SL/TP closePosition=True imediat in jurul MARK price (+/-0.15%)
            try:
                px = float(self.broker.mark_price(symbol))
            except Exception:
                px = float(self.state.entry_price if self.state.entry_price and not math.isnan(self.state.entry_price) else 0.0)
            if px <= 0 or tick <= 0:
                self._err("double-trigger: no price/tick")
                return False

            # offsets mai late: +/-0.15% (1.0015 / 0.9985)
            if side.upper() == "BUY":      # inchidere SHORT
                sl_px = ceil_to_tick(px * 1.0015, tick)
                tp_px = round_to_tick(px * 0.9985, tick)
            else:                          # inchidere LONG
                sl_px = round_to_tick(px * 0.9985, tick)
                tp_px = ceil_to_tick(px * 1.0015, tick)

            sl_s = self._fmt_px(sl_px)
            tp_s = self._fmt_px(tp_px)

            with self._lock:
                # nu sterge ordinele daca fallback-ul e deja armat
                if not getattr(self, "_fallback_armed", False):
                    try:
                        self.broker.cancel_all(symbol)
                    except Exception as e:
                        self._err(f"cancel_all before fallback error: {e}")
                try:
                    self.broker.place_close_all_stop_market(symbol, side=side, stop_price=sl_s)
                    self.broker.place_close_all_take_profit_market(symbol, side=side, tp_price=tp_s)
                    self._dbg(f"armed double-trigger sl={sl_s} tp={tp_s}")
                    self._fallback_armed = True
                except Exception as e:
                    self._err(f"double-trigger place error: {e}")
                    return False

            # asteapta inchidere
            for _ in range(20):
                time.sleep(0.4)
                try:
                    p3 = self.broker.position_info(symbol)
                    if abs(float(p3.get("positionAmt", 0.0))) < max(minq, 0.0):
                        self._dbg("double-trigger: flat confirmed")
                        try:
                            self.broker.cancel_all(symbol)  # curatenie finala
                        except Exception:
                            pass
                        self._fallback_armed = False
                        return True
                except Exception:
                    pass
            return False

        last_amt = 0.0

        for tr in range(max_retries):
            placed_order = False

            with self._lock:  # - SECTIUNE ATOMICA -
                # 1) pozitia curenta
                try:
                    p = self.broker.position_info(symbol)
                    amt = float(p.get("positionAmt", 0.0))
                except Exception as e:
                    self._err(f"position_info error: {e}")
                    amt = 0.0

                last_amt = amt

                qty = abs(amt)
                self._dbg(f"ensure_flat check qty={qty}, minQty={minq}")

                # 2) flat daca sub minQty (STRICT '<')
                if qty < max(minq, 0.0):
                    self._dbg("ensure_flat: already flat (<=minQty)")
                    self._fallback_armed = False
                    self._exiting = False
                    return True

                # 3) daca sub minNotional la MARK -> fallback direct
                try:
                    last_px = float(self.broker.mark_price(symbol))
                except Exception:
                    last_px = 0.0
                notional = qty * last_px if last_px > 0 else float("inf")
                side = "BUY" if amt < 0 else "SELL"

                if last_px > 0 and notional < max(min_notional, 0.0):
                    self._dbg(f"under minNotional (qty*px={notional} < {min_notional}) -> double-trigger fallback")
                    ok = _double_trigger_close(side)
                    if ok:
                        self._exiting = False
                        return True
                    # daca a esuat, continuam cu retry-urile clasice

                # 4) curata protectiile doar daca NU e armat fallback-ul
                if not getattr(self, "_fallback_armed", False):
                    try:
                        self.broker.cancel_all(symbol)
                    except Exception as e:
                        self._err(f"cancel_all error: {e}")

                # 5) MARKET reduceOnly (cu qty rotunjit; fallback pe minQty daca iese 0)
                q = floor_to_step(qty, step)
                if q <= 0:
                    q = ceil_to_step(minq, step) if step > 0 else minq
                    self._dbg(f"close qty floor_to_step=0; fallback q={q}")

                try:
                    self._dbg(f"close MARKET reduceOnly side={side} q={q}")
                    resp = self._mkt(symbol, side=side, qty=q, reduce_only=True)
                    self._dbg(f"close MARKET resp: {resp}")
                    placed_order = True
                except Exception as e:
                    self._err(f"close MARKET error: {e}")

            # 6) debounce + reconfirmare
            time.sleep(max(0.3, sleep_s if placed_order else 0.3))
            try:
                p2 = self.broker.position_info(symbol)
                amt2 = float(p2.get("positionAmt", 0.0))
            except Exception as e:
                self._err(f"position_info (post-close) error: {e}")
                amt2 = 0.0

            last_amt = amt2

            if abs(amt2) < max(minq, 0.0):
                self._dbg("ensure_flat: flat confirmed after debounce")
                self._fallback_armed = False
                self._exiting = False
                return True

            self._dbg(f"ensure_flat retry {tr+1}/{max_retries} (still amt={amt2})")
            time.sleep(sleep_s)

        # 7) fallback final (ramane ca la tine; 2B nu schimba aici directia)
        self._err("ensure_flat: retries exhausted; final double-trigger fallback")
        fallback_side = "BUY" if last_amt < 0 else "SELL"
        ok = _double_trigger_close(fallback_side)
        if ok:
            self._exiting = False
            return True

        self._err("double-trigger: still not flat; staying in EXITING")
        return False
    
    def force_flat_now(self):
        """Inchidere fortata (market + fallback) cu log clar."""
        sym = self.sym_cfg.symbol
        self._exiting = True
        ok = self.ensure_flat(sym, reason="force-flat")
        self._exiting = False
        return ok

    
    def verify_protections(self, symbol: str, want_sl: float | None, want_tp: float | None):
        # nu arma protectii cat timp suntem in EXITING
        if getattr(self, "_exiting", False):
            return
        if not self.state.in_pos:
            return
        try:
            oo = self.broker.open_orders(symbol)
        except Exception:
            oo = []

        # atentie: pe Binance reduceOnly vine ca boolean (nu string) in multe raspunsuri
        def _is_true(v): 
            return (v is True) or (isinstance(v, str) and v.lower() == "true")

        have_sl = any(o.get("type") == "STOP_MARKET" and _is_true(o.get("reduceOnly")) for o in oo)
        have_tp = any(o.get("type") == "TAKE_PROFIT_MARKET" and _is_true(o.get("reduceOnly")) for o in oo)

        if (not have_sl) and (want_sl is not None) and not math.isnan(want_sl):
            sl_px = round_to_tick(want_sl, self.filters.get("tickSize", 0.0))
            self._sl(symbol, side="BUY", qty=self.state.qty, stop_price=sl_px)
            self.state.atr_stop = sl_px
        if (not have_tp) and (want_tp is not None) and not math.isnan(want_tp):
            tp_px = round_to_tick(want_tp, self.filters.get("tickSize", 0.0))
            self._tp(symbol, side="BUY", qty=self.state.qty,tp_price=tp_px)
            
    def _report_trade_oanda(self, going: str, units: float, price: float, pl: float = 0.0):
        """
        Print OANDA-style logs + tine evidenta P&L cumulat (aprox. pe pretul primit).
        For USDT linear futures: PnL = (entry - exit) * qty pentru SHORT.
        """
        self._cum_pl += float(pl)
        ts = dt.datetime.utcnow().isoformat() + "Z"
        print("\n" + "-" * 100)
        print(f"{ts} | {going}")
        print(f"{ts} | units = {units} | price = {round(float(price), 5)} | P&L = {round(float(pl), 4)} | Cum P&L = {round(self._cum_pl, 4)}")
        print("-" * 100 + "\n")

            
    def stop(self, flatten: bool = False):
        """
        Oprire gratioasa: opreste streamul; optional inchide pozitia si anuleaza ordinele.
        """
        # anunta thread-ul de reconciliere sa se opreasca
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
        try:
            self.broker.cancel_all(self.sym_cfg.symbol)
        except Exception:
            pass
        try:
            self.broker.stop_stream(self.sym_cfg.symbol, self.live_cfg.timeframe)
        except Exception:
            pass


            
    def _mkt(self, symbol, side, qty, reduce_only=False):
        if self.dry_run:
            self.log.log(symbol, "MARKET(DRY)", side, qty, extra=f"reduceOnly={reduce_only}")
            return {"status": "DRY"}
        self.log.log(symbol, "MARKET", side, qty)
        return self.broker.place_market(symbol, side=side, qty=qty, reduce_only=reduce_only)

    def _sl(self, symbol, side, qty, stop_price):
        if self.dry_run:
            self.log.log(symbol, "STOP_MARKET(DRY)", side, qty, price=stop_price)
            return {"status": "DRY"}
        self.log.log(symbol, "STOP_MARKET", side, qty, stop_price)
        price_str = self._fmt_px(float(stop_price))
        return self.broker.place_stop_market(symbol, side=side, qty=qty, stop_price=price_str, reduce_only=True)


    def _tp(self, symbol, side, qty, tp_price):
        if self.dry_run:
            self.log.log(symbol, "TP_MARKET(DRY)", side, qty, price=tp_price)
            return {"status": "DRY"}
        self.log.log(symbol, "TP_MARKET", side, qty, tp_price)
        price_str = self._fmt_px(float(tp_price))
        return self.broker.place_take_profit_market(symbol, side=side, qty=qty, tp_price=price_str, reduce_only=True)


    def _reconcile_loop(self):
        """Poll account & simbol la fiecare X sec pentru: close TP/SL/manual, protectii lipsa, cap/DD info."""
        while not self._stop_event.is_set():
            try:
                self._reconcile_once()
            except Exception:
                pass
            self._stop_event.wait(float(self.live_cfg.reconcile_secs))

    def _reconcile_once(self):
        symbol = self.sym_cfg.symbol
        with self._lock:
            # 1) daca pozitia s-a inchis la bursa (TP/SL/manual)
            try:
                p = self.broker.position_info(symbol)
                amt = float(p.get("positionAmt", 0.0))
            except Exception:
                amt = 0.0

            if self.state.in_pos and abs(amt) < max(self.filters.get("minQty", 0.0), 0.0):
                # marchez flat, curat ordine, persist
                try:
                    self.broker.cancel_all(symbol)
                except Exception:
                    pass
                self.state = PositionState()
                self.store.save(symbol, {"in_pos": False})
                self.log.log(symbol, "RECONCILE_FLAT")

            # 2) re-armare protectii (doar daca NU suntem in EXITING)
            if self.state.in_pos and not self._exiting:
                self.verify_protections(
                    symbol,
                    want_sl=self.state.atr_stop if not math.isnan(self.state.atr_stop) else None,
                    want_tp=None
                )


            # 3) persist periodic
            self.store.save(symbol, {
                "in_pos": self.state.in_pos,
                "qty": self.state.qty,
                "entry_price": self.state.entry_price,
                "atr_stop": self.state.atr_stop
            })
            
    def _account_equity(self) -> float:
        try:
            return float(self.broker.account_equity_usdt())
        except Exception:
            return 0.0

    def _compute_open_risk_pct(self) -> tuple[float, bool]:
        """
        Calculeaza OpenRisk (% din equity) pe intreg contul, folosind pozitii + STOP reduceOnly (SL).
        Returneaza (open_risk_pct, has_missing_sl).
        """
        equity = self._account_equity()
        if equity <= 0:
            return 0.0, False

        positions = self.broker.list_positions()
        orders = self.broker.list_open_orders()

        # map SL per simbol (reduceOnly STOP)
        sl_map: Dict[str, float] = {}
        for o in orders:
            try:
                if o.get("type") == "STOP_MARKET" and (o.get("reduceOnly") is True or str(o.get("reduceOnly","")).lower()=="true"):
                    sym = o.get("symbol", "").upper()
                    sl_px = float(o.get("stopPrice", 0.0))
                    if sl_px > 0:
                        # pe futures poti avea mai multe fragmente; pastreaza cel mai aproape de entry (nu complicam acum)
                        sl_map[sym] = sl_px
            except Exception:
                continue

        total = 0.0
        missing_sl = False
        for p in positions:
            try:
                sym = p.get("symbol", "").upper()
                amt = float(p.get("positionAmt", 0.0))
                if amt == 0:
                    continue
                entry = float(p.get("entryPrice", 0.0))
                if entry <= 0:
                    continue

                has_sl = sym in sl_map
                if not has_sl:
                    missing_sl = True
                    # conservator: blocam intrari noi pana fixam SL la toate pozitiile
                    continue

                sl_px = sl_map[sym]
                qty = abs(amt)
                if amt < 0:  # SHORT
                    dist = max(0.0, sl_px - entry)
                else:        # LONG
                    dist = max(0.0, entry - sl_px)

                risk_usdt = qty * dist
                total += (risk_usdt / equity)
            except Exception:
                continue

        return float(total), bool(missing_sl)

    def _dd_blocked(self) -> bool:
        """DD trailing simplu: daca equity a cazut >= dd_stop fata de peak salvat -> blocam intrari noi."""
        sym = self.sym_cfg.symbol
        st = self.store.load(sym) or {}
        eq = self._account_equity()
        if eq <= 0:
            return False
        peak = float(st.get("peak_equity", eq))
        if eq > peak:
            # update peak
            st["peak_equity"] = eq
            self.store.save(sym, st)
            return False
        drop = (peak - eq) / peak
        return drop >= float(self.live_cfg.dd_stop_pct)

    def _risk_size(self, px_entry: float, sl_price_raw: float) -> float:
        """
        Baza+Cap:
          r_avail = cap_eff - OpenRisk
          r_new   = min(r_base, r_avail); daca < r_min => 0 (skip)
          qty     = floor( (r_new * equity) / dist_SL, step ), cu filtre
        """
        equity = self._account_equity()
        if equity <= 0 or px_entry <= 0 or math.isnan(sl_price_raw):
            return 0.0

        open_risk, missing_sl = self._compute_open_risk_pct()
        cap_eff = float(self.live_cfg.risk_cap_pct) * (1.0 - float(self.live_cfg.risk_cap_buffer_pct))
        if missing_sl:
            # exista pozitii fara SL => blocam intrari noi pana le fixam
            return 0.0
        if open_risk >= cap_eff:
            return 0.0

        r_base = float(self.live_cfg.risk_base_pct)
        r_min  = float(self.live_cfg.risk_min_pct)
        r_avail = max(0.0, cap_eff - open_risk)
        r_new = min(r_base, r_avail)
        if r_new < r_min:
            return 0.0

        # distanta pana la SL (SHORT)
        tick = self.filters.get("tickSize", 0.0)
        sl_px = round_stop_for_short(sl_price_raw, tick)
        dist = max(0.0, sl_px - px_entry)
        if dist <= 0:
            return 0.0

        # qty teoretica pe risc
        qty_raw = (r_new * equity) / dist

        # filtre
        step = self.filters.get("stepSize", 0.0)
        minq = self.filters.get("minQty", 0.0)
        minNotional = self.filters.get("minNotional", 0.0)

        qty = floor_to_step(max(qty_raw, minq), step)
        if qty <= 0:
            return 0.0
        if qty * px_entry < minNotional:
            return 0.0

        return float(qty)

    

    def on_bar_close(self, symbol: str, bar: Dict[str, Any]):
        # daca aveam o iesire restanta, incearca din nou si iesi din functie
        if self.exit_pending:
            if self.ensure_flat(symbol, reason="pending"):
                self.state = PositionState()
                self.exit_pending = False
            return

        # 1) Semnale pe bara tocmai INCHISA (echiv. decizie la close(t-1) -> exec la open(t))
        sig = self.signal_fn(symbol, bar)   # dict: entry_short, exit_reverse, atr_sl, tp_level
        # Afiseaza semnalul doar cand conteaza:
        # - intrare noua (entry_short True), SAU
        # - suntem in pozitie si exista semnal de reverse
        if sig.get("entry_short", False) or (self.state.in_pos and sig.get("exit_reverse", False)):
            print(
                f"[SIG] {symbol} @{pd.to_datetime(bar['end'], unit='ms', utc=True)} | "
                f"entry={sig.get('entry_short')} exit={sig.get('exit_reverse')} "
                f"sl={sig.get('atr_sl'):.4f} tp={sig.get('tp_level'):.4f} "
                f"in_pos={self.state.in_pos} qty={self.state.qty}"
            )
        # resetam contorul pentru bara noua (va creste la on_bar_update)
        self._bar_updates = 0

        # NOTA: reverse il tratam INTRABAR pe bara urmatoare (nu inchidem aici).
        # Pregatim intrarea pentru open(t+1) daca avem semnal si suntem flat.
        self.pending_entry = (not self.state.in_pos) and sig.get("entry_short", False)
        self._pending_levels["sl"] = sig.get("atr_sl", float("nan"))
        self._pending_levels["tp"] = sig.get("tp_level", float("nan"))
        self._next_bar_start = bar["end"]  # startul barei viitoare = end-ul barei curente

        # Re-aranjeaza protectiile daca suntem in pozitie si lipsesc (server-side safety).
        self.verify_protections(
            symbol,
            want_sl=sig.get("atr_sl", float("nan")),
            want_tp=sig.get("tp_level", float("nan")),
        )
        
    def on_bar_update(self, symbol: str, bar: Dict[str, Any]):
        # ^ counter live ca in OANDA
        self._bar_updates += 1
        print(self._bar_updates, end="\r", flush=True)
        if self.pending_entry and (self._next_bar_start is not None) and (bar["start"] >= self._next_bar_start):
            px_open = float(bar["open"])

            # --- Risk gating & sizing (activ doar daca risk_enabled=True) ---
            qty = None
            if getattr(self.live_cfg, "risk_enabled", False):
                # 1) blocare pe drawdown
                if self._dd_blocked():
                    print("**Entry blocked (DD stop active)**")
                    self.pending_entry = False
                    return
                # 2) verifica OpenRisk + SL-uri lipsa la pozitii existente
                open_risk, missing_sl = self._compute_open_risk_pct()
                if missing_sl:
                    print("**Entry blocked (existing position without SL)**")
                    self.pending_entry = False
                    return
                # 3) dimensionare pe risc
                proposed_sl = self._pending_levels["sl"]
                risk_qty = self._risk_size(px_open, proposed_sl)
                if risk_qty <= 0:
                    print("**Skipping entry (risk sizing returned 0)**")
                    self.pending_entry = False
                    return
                qty = risk_qty
            else:
                # sizing standard USD/pct_equity
                usd = self.sym_cfg.usd_fixed if self.sizing_fn is None else self.sizing_fn(px_open, self.sym_cfg, self.filters)
                qty = calc_qty(px_open, usd, self.filters["stepSize"], self.filters["minQty"], self.filters["minNotional"])

            if qty * px_open >= self.filters.get("minNotional", 0.0) and qty > 0:
                try:
                    self.broker.cancel_all(symbol)
                except Exception:
                    pass
                self._mkt(symbol, side="SELL", qty=qty, reduce_only=False)
                self.state.in_pos = True
                self.state.qty = qty
                self.state.entry_price = px_open
                # memoreaza ultima intrare (pentru P&L la iesire) si log OANDA
                self._last_entry = {"qty": qty, "price": px_open, "side": "SHORT"}
                self._report_trade_oanda("GOING SHORT", units=-qty, price=px_open, pl=0.0)

                # setam TP/SL imediat, MARK_PRICE (server-side)
                if not math.isnan(self._pending_levels["sl"]):
                    sl_px = round_stop_for_short(self._pending_levels["sl"], self.filters.get("tickSize", 0.0))
                    self.state.atr_stop = sl_px
                    self._sl(symbol, side="BUY", qty=qty, stop_price=sl_px)
                if not math.isnan(self._pending_levels["tp"]):
                    tp_px = round_tp_for_short(self._pending_levels["tp"], self.filters.get("tickSize", 0.0))
                    self._tp(symbol, side="BUY", qty=qty, tp_price=tp_px)

            self.pending_entry = False  # consumat o singura data


        # 2) REVERSE intrabar (inchidem imediat daca semnalul contra e activ pe bara in curs)
        #    Refolosim engine-ul tau pe bara partiala: returneaza conditii bazate pe OHLC curent.
        sig_now = self.signal_fn(symbol, bar)
        if self.state.in_pos and sig_now.get("exit_reverse", False):
            self._dbg("reverse signal -> calling ensure_flat()")
            self._exiting = True
            if self.ensure_flat(symbol, reason="reverse-intrabar"):
                exit_px = float(bar["close"])
                qty     = float(self._last_entry.get("qty", 0.0))
                ent_px  = float(self._last_entry.get("price", exit_px))
                pl      = (ent_px - exit_px) * qty  # SHORT
                self._report_trade_oanda("GOING NEUTRAL", units=qty, price=exit_px, pl=pl)
                self.state = PositionState()
                self.exit_pending = False
                self._last_entry = {"qty": 0.0, "price": math.nan, "side": None}
                self._exiting = False
            else:
                self._dbg("reverse signal -> ensure_flat() pending/failed (will retry)")
                self.exit_pending = True
            return




        # 3) SL/TP dinamice intrabar - doar daca suntem in pozitie
        if self.state.in_pos:
            new_sl = sig_now.get("atr_sl", float("nan"))
            new_tp = sig_now.get("tp_level", float("nan"))
            tick = self.filters.get("tickSize", 0.0)
            # rotunjiri corecte pentru short
            sl_px = round_stop_for_short(new_sl, tick) if not math.isnan(new_sl) else float("nan")
            tp_px = round_tp_for_short(new_tp, tick)   if not math.isnan(new_tp) else float("nan")

            # histerezis: nu re-setam daca schimbarea < 0.2%
            def moved(a, b, thr=0.002):
                return (not math.isnan(a)) and (not math.isnan(b)) and (abs(a - b) / max(b, 1e-12) > thr)

            need_reset = False
            # stair-step: la short SL poate doar sa coboare (numar mai mic)
            if not math.isnan(sl_px):
                if math.isnan(self.state.atr_stop) or sl_px < self.state.atr_stop or moved(sl_px, self.state.atr_stop):
                    need_reset = True

            if not math.isnan(tp_px):
                need_reset = need_reset or True  # TP poate fi miscat oricand

            if need_reset:
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

        # 4) Asigura-te ca protectiile exista la server (re-armare daca lipsesc)
        self.verify_protections(
            symbol,
            want_sl=sig_now.get("atr_sl", float("nan")),
            want_tp=sig_now.get("tp_level", float("nan")),
        )
        
        # --- Reconciliere usoara: daca TP/SL a inchis pozitia la bursa, logam iesirea ---
        if self.state.in_pos and (self._bar_updates % 12 == 0):  # ajusteaza frecventa daca vrei
            try:
                p = self.broker.position_info(symbol)
                amt = abs(float(p.get("positionAmt", 0.0)))
            except Exception:
                amt = None
            if amt is not None and amt <= max(self.filters.get("minQty", 0.0), 0.0):
                exit_px = float(bar["close"])
                qty     = float(self._last_entry.get("qty", 0.0))
                ent_px  = float(self._last_entry.get("price", exit_px))
                pl      = (ent_px - exit_px) * qty  # SHORT
                self._report_trade_oanda("GOING NEUTRAL", units=qty, price=exit_px, pl=pl)
                try:
                    self.broker.cancel_all(symbol)
                except Exception:
                    pass
                self.state = PositionState()
                self._last_entry = {"qty": 0.0, "price": math.nan, "side": None}



# === construieste engine + runner ===
engine = Super8SignalEngine(ind_params, short_params)

symbol = "BTCUSDT"
broker=BinanceFuturesAdapter()
cfg = configparser.ConfigParser()
cfg.read("binance.cfg")

USE_TESTNET = True  # << sandbox
section = "binance_testnet" if USE_TESTNET else "binance"

API_KEY    = cfg[section]["api_key"]
API_SECRET = cfg[section]["secret_key"]


# (optional) curatare runner vechi - PUNE INAINTE de a crea unul nou
try:
    del runner
except NameError:
    pass

runner = Super8LiveRunner(
    broker=broker,
    live_cfg=LiveConfig(
        api_key=API_KEY,
        api_secret=API_SECRET,
        timeframe="1m",
        leverage=1,
        margin_type="ISOLATED",
        hedge_mode=False,
        testnet=USE_TESTNET,   # << acelasi flag
        dry_run=False,          # << pe testnet vrem ordine reale in sandbox
        # --- RISK ---
        risk_enabled=True,          # <- acesta e "ON/OFF"
        risk_base_pct=0.010,        # 1.0% pe tranzactie (baza)
        risk_cap_pct=0.073,         # plafon global ~7.3% equity
        risk_cap_buffer_pct=0.05,   # buffer 5% (protejeaza sa nu lovim plafonul exact)
        risk_min_pct=0.002,         # prag minim 0.2% (altfel nu intra)
        dd_stop_pct=0.075,          # DD trailing 7.5% -> blocheaza intrari noi
        reconcile_secs=5,           # pentru _reconcile_loop
        persist_path_tpl="state_{symbol}.json",
        history_keep_bars=700

    ),
    sym_cfg = SymbolConfig(symbol="BTCUSDT", usd_fixed=5.0, pct_equity=0.01),  # 1% din equity, min 5 USDT,
    indicator_fn=lambda df: None,
    signal_fn=lambda sym, bar: engine.on_bar_close(sym, bar),
    sizing_fn=make_sizing_fn(broker),
)


# === bootstrap broker + leverage + filtre ===
print("[CFG]", "testnet=", runner.live_cfg.testnet, "dry_run=", runner.live_cfg.dry_run)
runner.bootstrap()

# === seed istoric minim necesar (luam ~lookback+10 bare) ===
need = engine.lookback + 10
seed_bars = runner.broker.fetch_klines(symbol, runner.live_cfg.timeframe, need)
engine.seed(seed_bars)
engine.keep_bars = runner.live_cfg.history_keep_bars
print(f"[SEED] {symbol} {runner.live_cfg.timeframe} bars={len(seed_bars)} lookback={engine.lookback}")

runner.broker.stream_klines(
    symbol,
    runner.live_cfg.timeframe,
    on_close=runner.on_bar_close,
    on_update=runner.on_bar_update
)


# CELL 4


# CELL 5
runner.stop(flatten=True)   # inchide pozitia (daca exista), anuleaza ordinele, opreste streamul
# runner.stop(flatten=False)  # doar opreste streamul + anuleaza ordinele


# CELL 6
runner.ensure_flat(symbol)


# CELL 7


# CELL 8
# --- Quick open SHORT + armare TP/SL (pentru test) ---
px = seed_bars[-1]["close"]
f  = runner.filters
step, tick, min_notional = f["stepSize"], f["tickSize"], f["minNotional"]

qty = ceil_to_step(min_notional / px, step)   # cea mai mica cantitate valida

runner.broker.cancel_all(symbol)
runner._mkt(symbol, "SELL", qty)              # intra SHORT market

# TP/SL apropiate ca sa vezi repede miscari intrabar
sl_px = ceil_to_tick(px * 1.002, tick)        # ~+0.2% peste entry (SHORT)
tp_px = round_to_tick(px * 0.998, tick)       # ~-0.2% sub entry (SHORT)

runner._sl(symbol, "BUY", qty, sl_px)
runner._tp(symbol, "BUY", qty, tp_px)

print("OPENED SHORT:", "qty=", qty, "sl=", sl_px, "tp=", tp_px)
print("open orders:", runner.broker.open_orders(symbol))
print("pos:", runner.broker.position_info(symbol))


# CELL 9
runner.force_flat_now()
print("post:", runner.broker.position_info(symbol))
print("open orders:", runner.broker.open_orders(symbol))


# CELL 10
# --- FORCE reverse intrabar pe urmatorul update (numai pentru test) ---
_saved = runner.signal_fn
runner.signal_fn = lambda s, b: {"entry_short": False, "exit_reverse": True, "atr_sl": math.nan, "tp_level": math.nan}

# fabricam un "update" pe bara curenta ca sa declansam imediat on_bar_update
last = seed_bars[-1]
fake = {
    "start": last["end"], "end": last["end"],  # aceeasi bara "in curs"
    "open": px, "high": px, "low": px, "close": px, "volume": 0.0
}
runner.on_bar_update(symbol, fake)

# revenim la semnalele tale reale
runner.signal_fn = _saved

print("open orders dupa reverse:", runner.broker.open_orders(symbol))
print("pos dupa reverse:", runner.broker.position_info(symbol))


# CELL 11


