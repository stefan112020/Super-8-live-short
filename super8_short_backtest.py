import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Optional
from super8_indicators import Super8Indicators

# === DEBUG switch ===
DEBUG = True
def _dbg(msg: str = ""):
    if DEBUG:
        print(msg)


def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure essential columns
    if "Price" not in df.columns:
        if "close" in df.columns:
            df["Price"] = df["close"]
        else:
            raise ValueError("Missing 'Price' or 'close'.")
    for c in ["high", "low", "volume"]:
        if c not in df.columns:
            raise ValueError(f"Missing '{c}' column.")
    if "Return" not in df.columns:
        df["Return"] = np.log(df["Price"] / df["Price"].shift(1))
    if "Spread" not in df.columns:
        df["Spread"] = (df["high"] - df["low"]) / df["Price"]
    return df


def _crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    a1 = a.shift(1)
    b1 = b.shift(1)
    x = (a1 <= b1) & (a > b)
    return x.fillna(False)


def _crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    a1 = a.shift(1)
    b1 = b.shift(1)
    x = (a1 >= b1) & (a < b)
    return x.fillna(False)


def _rma(series: pd.Series, period: int) -> pd.Series:
    """
    Wilder's RMA = EMA cu alpha=1/period.
    Pentru backtest este suficientă aproximarea cu ewm(alpha=1/period).
    """
    alpha = 1.0 / float(period)
    return series.ewm(alpha=alpha, adjust=False).mean()


@dataclass
class ShortParams:
    Position: str = "Short"
    TP_options: str = "Both"
    SL_options: str = "Both"
    tp: float = 3.6
    sl: float = 5.0
    atrPeriodSl: int = 116
    multiplierPeriodSl: float = 107.0
    trailOffset: float = 0.38
    reverse_exit: bool = True
    ignore_additional_entries: bool = True
    exit_on_entry_loss: bool = False
    start_time: Optional[pd.Timestamp] = None

    # === Pasul 3: setări intrabar (ca în Pine) ===
    intrabar_touch: bool = True            # activează emularea intrabar cu OHLC
    bar_path: str = "OLHC"                 # “Open-Low-High-Close” → pentru SHORT favorizează TP-first
                                           # alternativ: "OHLC" (Open-High-Low-Close) → SL-first
    no_same_bar_exit: bool = True          # NU permite TP/SL în aceeași bară cu intrarea (ca în Pine)


@dataclass
class RiskSettings:
    enabled: bool = False
    base_pct: float = 0.01
    min_pct: float = 0.0025
    cap_pct: float = 0.073
    cap_buffer_pct: float = 0.05
    dd_stop_pct: float = 0.075
    equity_start: float = 1.0


def _extract_trade_returns(bt: pd.DataFrame, log_returns: bool = True):
    pos = bt.get("position", pd.Series(dtype=float)).fillna(0.0)
    strat_ret = bt.get("strategy", pd.Series(dtype=float)).fillna(0.0)
    trades = []
    current = 0.0
    in_trade = False
    prev_pos = pos.shift(1).fillna(0.0)
    for r, p, prev in zip(strat_ret, pos, prev_pos):
        opened = (prev == 0) and (p != 0)
        closed = (p == 0) and (prev != 0)
        if opened:
            current = 0.0
            in_trade = True
        if in_trade:
            current += r
        if in_trade and closed:
            trades.append(current)
            in_trade = False
    return trades


def compute_performance_metrics(bt: pd.DataFrame, initial_equity: float = 1.0, log_returns: bool = True) -> dict:
    if bt is None or bt.empty:
        return {
            "total_pnl": 0.0,
            "total_pnl_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "profit_factor": 0.0,
            "win_rate_pct": 0.0,
            "total_trades": 0,
            "max_trade_duration_days": 0.0,
        }

    equity_curve = bt.get("equity_curve")
    if equity_curve is None or equity_curve.isna().all():
        equity_curve = bt.get("cstrategy")
    equity_curve = equity_curve.astype(float)
    total_equity = float(equity_curve.iloc[-1])
    total_pnl = total_equity - initial_equity
    total_pnl_pct = (total_equity / initial_equity - 1.0) * 100

    running_max = equity_curve.cummax()
    dd = (running_max - equity_curve) / running_max.replace(0, np.nan)
    max_dd_pct = float(dd.max() * 100) if len(dd) else 0.0

    trade_logs = _extract_trade_returns(bt, log_returns=log_returns)
    trade_pnls = [np.exp(x) - 1 if log_returns else x for x in trade_logs]
    gross_profit = sum(v for v in trade_pnls if v > 0)
    gross_loss = sum(v for v in trade_pnls if v < 0)
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss < 0 else float("inf")

    total_trades = len(trade_pnls)
    wins = sum(1 for v in trade_pnls if v > 0)
    win_rate_pct = (wins / total_trades * 100) if total_trades else 0.0

    pos = bt.get("position", pd.Series(dtype=float)).fillna(0.0)
    idx = bt.index
    durations = []
    start_idx = None
    for cur, prev, ts in zip(pos, pos.shift(1).fillna(0.0), idx):
        if prev == 0 and cur != 0:
            start_idx = ts
        if prev != 0 and cur == 0 and start_idx is not None:
            dur_days = (ts - start_idx).total_seconds() / 86400 if hasattr(ts, 'to_pydatetime') else 0.0
            durations.append(dur_days)
            start_idx = None
    max_dur = max(durations) if durations else 0.0

    return {
        "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl_pct,
        "max_drawdown_pct": max_dd_pct,
        "profit_factor": profit_factor,
        "win_rate_pct": win_rate_pct,
        "total_trades": total_trades,
        "max_trade_duration_days": max_dur,
    }


def print_performance_summary(symbol: str, metrics: dict):
    print(f"Rezumat performanță pentru {symbol}:")
    print(f"  Total P&L: {metrics['total_pnl']:.2f} ({metrics['total_pnl_pct']:.2f}%)")
    print(f"  Max drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Profit factor: {metrics['profit_factor']:.2f}")
    print(f"  Win rate: {metrics['win_rate_pct']:.2f}% din {metrics['total_trades']} tranzacții")
    print(f"  Durată maximă tranzacție: {metrics['max_trade_duration_days']:.2f} zile")


@dataclass
class Super8ShortBacktester:
    indicator_params: Dict[str, Any]
    short_params: ShortParams
    use_spread: bool = True
    log_return: bool = True
    risk: Optional[RiskSettings] = None

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
            df = df.dropna(subset=["time"]).set_index("time").sort_index()
        df = _ensure_features(df).dropna()
        _dbg(f"[BT] df rows={len(df)} | cols={list(df.columns)}")
        try:
            _dbg(f"[BT] time: {df.index.min()} → {df.index.max()} | dup_time={int(df.index.duplicated().sum())}")
        except Exception:
            _dbg("[BT] index nu e datetime încă (nu e problemă dacă îl setezi mai jos)")

        # Compute indicators via external module
        ind = Super8Indicators(self.indicator_params).compute(df)
        _dbg(f"[BT] type(ind)={type(ind)} | isNone={ind is None}")
        try:
            _dbg(f"[BT] ind rows={(0 if ind is None else len(ind))}")
        except Exception as e:
            _dbg(f"[BT] len(ind) error: {e}")

        # Fallback robust
        if ind is None or not isinstance(ind, pd.DataFrame):
            _dbg("[BT] compute() a întors None / non-DataFrame → folosesc DataFrame gol cu același index.")
            ind = pd.DataFrame(index=df.index)
        elif len(ind) == 0 and not ind.index.equals(df.index):
            _dbg("[BT] compute() a întors DF gol; îi setez același index ca df (ca join-ul să nu taie nimic).")
            ind = ind.reindex(df.index)

        # join left — baza rămâne toate rândurile
        base = df.join(ind, how="left")
        _dbg(f"[BT] base rows={len(base)} | same_index={base.index.equals(df.index)}")

        print("len(df)=", len(df), " | len(ind)=", len(ind), " | len(base)=", len(base))

        # --- condiții cu fallback & NaN-safe (citim din `base`) ---
        EMA_shortCond = base.get("EMA_shortCond", pd.Series(False, index=base.index)).fillna(False)
        ADX_shortCond = base.get("ADX_shortCond", pd.Series(False, index=base.index)).fillna(False)
        ADX_longCond  = base.get("ADX_longCond",  pd.Series(False, index=base.index)).fillna(False)
        SAR_shortCond = base.get("SAR_shortCond", pd.Series(False, index=base.index)).fillna(False)
        VOL_shortCond = base.get("VOL_shortCond", pd.Series(False, index=base.index)).fillna(False)

        # MACD vs MAC-Z
        if self.indicator_params.get("MACD_options") == "MAC-Z":
            h = base.get("histmacz")
        else:
            h = base.get("hist")
        MACD_shortCond = (h.lt(0).fillna(False)) if isinstance(h, pd.Series) else pd.Series(False, index=base.index)

        BB_upper  = base.get("BB_upper",  pd.Series(np.nan, index=base.index))
        BB_middle = base.get("BB_middle", pd.Series(np.nan, index=base.index))
        BB_width  = base.get("BB_width",  pd.Series(np.nan, index=base.index))
        bbMin01   = base.get("bbMinWidth01", pd.Series(0.05, index=base.index))  # 5% fallback

        allow_short = (self.short_params.Position != "Long")

        BB_short01 = (allow_short) & (~ADX_longCond) & _crossover(base["high"], BB_upper) \
                     & EMA_shortCond & (BB_width > bbMin01)
        BB_short01 = BB_short01.fillna(False)

        # condiția principală short
        shortCond  = (allow_short) & EMA_shortCond & ADX_shortCond & SAR_shortCond & MACD_shortCond & VOL_shortCond
        entry_cond = (shortCond | BB_short01).fillna(False)

                # === EXIT COND (numai pentru a ieși din short; NU intrăm long) ===
        EMA_longCond = base.get("EMA_longCond", pd.Series(False, index=base.index)).fillna(False)
        SAR_longCond = base.get("SAR_longCond", pd.Series(False, index=base.index)).fillna(False)

        if str(self.indicator_params.get("MACD_options", "MACD")) == "MAC-Z":
            h_long = base.get("histmacz")
        else:
            h_long = base.get("hist")
        MACD_longCond = (h_long.gt(0).fillna(False)) if isinstance(h_long, pd.Series) else pd.Series(False, index=base.index)

        ADX_longCond = base.get("ADX_longCond", pd.Series(False, index=base.index)).fillna(False)

        # ieșim când oricare dintre condițiile CONTRA devine activă (permisiv, ca în varianta performantă)
        exit_cond = (EMA_longCond | ADX_longCond | SAR_longCond | MACD_longCond).fillna(False)
        
        entry_sig = entry_cond.shift(1).fillna(False).astype(bool)  # intrăm next-bar
        exit_sig  = exit_cond.fillna(False).astype(bool)            # ieșim same-bar


        # (opțional) coloane de debug – DOAR pentru inspecție
        base["DBG_exit_cond"]     = exit_cond.astype(int)
        base["DBG_EMA_longCond"]  = EMA_longCond.astype(int)
        base["DBG_SAR_longCond"]  = SAR_longCond.astype(int)
        base["DBG_MACD_longCond"] = MACD_longCond.astype(int)

        # === REVERSE-EXIT (longCond sau BB_long01) — doar pentru a ÎNCHIDE shortul ===
        VOL_longCond = base.get("VOL_longCond", pd.Series(False, index=base.index)).fillna(False)
        BB_lower     = base.get("BB_lower",  pd.Series(np.nan, index=base.index))

        # BB_long01 DOAR pt. debug (NU îl includem în exit_cond în această variantă)
        BB_long01 = (~ADX_shortCond) & _crossunder(base["low"], BB_lower) & EMA_longCond & (BB_width > bbMin01)
        BB_long01 = BB_long01.fillna(False)

        # longCond raw (strict ca în Pine pentru semnal de long) – doar debug
        longCond_raw = EMA_longCond & ADX_longCond & SAR_longCond & MACD_longCond & VOL_longCond
        reverse_trigger = (longCond_raw | BB_long01).fillna(False)

        base["DBG_longCond_raw"] = longCond_raw.astype(int)
        base["DBG_BB_long01"]    = BB_long01.astype(int)
        base["DBG_reverse_trig"] = reverse_trigger.astype(int)


        # --- diag scurt ---
        def _rate(s: pd.Series) -> float:
            return float(s.mean()) if len(s) else 0.0

        _dbg("[BT] rates → "
             f"EMA_short={_rate(EMA_shortCond):.3f}, "
             f"ADX_short={_rate(ADX_shortCond):.3f}, "
             f"SAR_short={_rate(SAR_shortCond):.3f}, "
             f"MACD_short={_rate(MACD_shortCond):.3f}, "
             f"VOL_short={_rate(VOL_shortCond):.3f}, "
             f"BB_short01={_rate(BB_short01):.3f}, "
             f"entry={_rate(entry_cond):.3f}")
        for name, s in [("EMA_short", EMA_shortCond), ("ADX_short", ADX_shortCond),
                        ("SAR_short", SAR_shortCond), ("MACD_short", MACD_shortCond),
                        ("VOL_short", VOL_shortCond), ("BB_short01", BB_short01), ("entry", entry_cond)]:
            if s.any():
                _dbg(f"[BT] first TRUE {name}: {s[s].index[0]}")

        # === ATR (Wilder/RMA) + "stair-step" pentru SHORT (simetric Pine) ===
        tr = np.maximum(
            base["high"] - base["low"],
            np.maximum((base["high"] - base["Price"].shift(1)).abs(),
                       (base["low"]  - base["Price"].shift(1)).abs())
        )
        atr_rma = _rma(pd.Series(tr, index=base.index), self.short_params.atrPeriodSl)

        # Stop ATR brut (SHORT): high + ATR * multiplier
        atr_sl_short_raw = base["high"] + atr_rma * self.short_params.multiplierPeriodSl

        # Aproximăm 'open': dacă nu există coloană 'open', folosim close anterior (Price.shift(1))
        _open = base["open"] if "open" in base.columns else base["Price"].shift(1).fillna(base["Price"])

        # "Stair-step" SHORT (oglindă față de Pine pe long):
        # dacă open < stop_prev  => clamp downward (stop nu are voie să crească; îl aducem la min(raw, prev))
        # altfel                  => reset la valoarea brută
        atr_sl_short_series = pd.Series(index=base.index, dtype=float)
        prev_stop = np.nan
        for i in range(len(base.index)):
            raw = float(atr_sl_short_raw.iloc[i]) if not pd.isna(atr_sl_short_raw.iloc[i]) else np.nan
            o   = float(_open.iloc[i])            if not pd.isna(_open.iloc[i]) else raw
            if i == 0 or np.isnan(prev_stop) or np.isnan(raw) or np.isnan(o):
                val = raw
            else:
                val = min(raw, prev_stop) if o < prev_stop else raw
            atr_sl_short_series.iloc[i] = val
            prev_stop = val

        base["DBG_ATR_SL_Short"] = atr_sl_short_series  # pentru inspecție

        # Precompute Donchian for TP if needed
        if self.short_params.TP_options in ("Both", "Donchian"):
            DClower = base["low"].rolling(window=self.indicator_params["DClength"],
                                          min_periods=self.indicator_params["DClength"]).min()
        else:
            DClower = pd.Series(np.nan, index=base.index)

        # 8) Bucla de backtest (single-position, next-bar entries & reverse-exits; TP/SL same-bar)
        risk_cfg = self.risk or RiskSettings()
        risk_enabled = bool(risk_cfg.enabled)
        equity = float(risk_cfg.equity_start if risk_enabled else 1.0)
        peak_equity = equity
        entry_tol = 1e-9

        positions = []
        strategy_vals = []
        trade_costs = []
        weight_changes = []
        equity_series = []
        risk_records = []

        prev_pos = 0.0
        qty_open = 0.0
        entry_price = float('nan')
        sl_active = float('nan')

        cap_eff = float(risk_cfg.cap_pct) * (1.0 - float(risk_cfg.cap_buffer_pct)) if risk_enabled else 0.0
        start_time = self.short_params.start_time

        for t, row in base.iterrows():
            price = float(row["Price"])
            bar_return_raw = row.get("Return", 0.0)
            spread_raw = row.get("Spread", 0.0)
            bar_return = float(0.0 if pd.isna(bar_return_raw) else bar_return_raw)
            spread = float(0.0 if pd.isna(spread_raw) else spread_raw)

            pos_weight = prev_pos
            equity_start_bar = equity
            drawdown_pct = ((peak_equity - equity_start_bar) / peak_equity) if peak_equity > 0 else 0.0
            dd_blocked = risk_enabled and (drawdown_pct >= float(risk_cfg.dd_stop_pct))

            if risk_enabled and qty_open > 0 and equity_start_bar > 0 and not np.isnan(sl_active) and not np.isnan(entry_price):
                dist_curr = max(0.0, sl_active - entry_price)
                open_risk_before = (qty_open * dist_curr / equity_start_bar) if dist_curr > 0 else 0.0
            else:
                open_risk_before = 0.0

            allow_trading = (start_time is None) or (t >= start_time)
            entry_signal = bool(entry_sig.loc[t])
            exit_signal = bool(exit_sig.loc[t])

            avg_price = entry_price if (not np.isnan(entry_price) and abs(pos_weight) > entry_tol) else price

            if self.short_params.TP_options == "Both":
                dc_val = DClower.loc[t] if 'DClower' in locals() and t in DClower.index else np.nan
                tp_level = np.nanmin([dc_val if not np.isnan(dc_val) else np.inf,
                                      (1.0 - self.short_params.tp/100.0) * avg_price])
            elif self.short_params.TP_options == "Normal":
                tp_level = (1.0 - self.short_params.tp/100.0) * avg_price
            elif self.short_params.TP_options == "Donchian":
                dc_val = DClower.loc[t] if 'DClower' in locals() and t in DClower.index else np.nan
                tp_level = dc_val if not np.isnan(dc_val) else avg_price
            else:
                tp_level = np.nan

            atr_sl_val = atr_sl_short_series.loc[t] if t in atr_sl_short_series.index else np.nan
            if self.short_params.SL_options == "Both":
                # În Pine, când avem atât ATR cât și SL procentual, se folosește nivelul MAI APROPIAT (minim pentru short)
                sl_level = min(atr_sl_val, (1.0 + self.short_params.sl/100.0) * avg_price)
            elif self.short_params.SL_options == "Normal":
                sl_level = (1.0 + self.short_params.sl/100.0) * avg_price
            elif self.short_params.SL_options == "ATR":
                sl_level = atr_sl_val
            else:
                sl_level = np.nan

            risk_info = {
                "risk_equity_start": float(equity_start_bar),
                "risk_peak_equity_before": float(peak_equity),
                "risk_drawdown_pct": float(drawdown_pct),
                "risk_dd_blocked": bool(dd_blocked),
                "risk_open_risk_pct_before": float(open_risk_before),
                "risk_entry_signal": bool(entry_signal),
                "risk_exit_signal": bool(exit_signal),
                "risk_allow_trading": bool(allow_trading),
                "risk_cap_eff": float(cap_eff) if risk_enabled else np.nan,
                "risk_r_base": float(risk_cfg.base_pct) if risk_enabled else np.nan,
                "risk_r_min": float(risk_cfg.min_pct) if risk_enabled else np.nan,
                "risk_r_new": np.nan,
                "risk_entry_allowed": None,
                "risk_entry_reason": "",
                "risk_entry_qty": np.nan,
                "risk_entry_weight": 0.0,
                "risk_entry_dist": np.nan,
                "risk_sl_level": float(sl_level) if not np.isnan(sl_level) else np.nan,
                "risk_tp_level": float(tp_level) if not np.isnan(tp_level) else np.nan,
                "risk_sl_active": float(sl_active) if not np.isnan(sl_active) else np.nan,
                "risk_qty_open": float(qty_open) if risk_enabled else np.nan,
                "risk_position_weight_prev": float(prev_pos),
                "risk_tp_hit": False,
                "risk_sl_hit": False,
                "risk_reverse_exit": False,
                "risk_exit_reason": "",
            }

            executed_entry = False
            reason = ""
            if abs(pos_weight) <= entry_tol and entry_signal:
                if not allow_trading:
                    reason = "start_time"
                elif dd_blocked:
                    reason = "dd_block"
                elif risk_enabled:
                    if np.isnan(sl_level):
                        reason = "missing_sl"
                    else:
                        dist = max(0.0, sl_level - price)
                        if dist <= 0:
                            reason = "invalid_sl"
                        else:
                            if open_risk_before >= cap_eff:
                                reason = "cap_reached"
                            else:
                                r_avail = max(0.0, cap_eff - open_risk_before)
                                r_new = min(float(risk_cfg.base_pct), r_avail)
                                risk_info["risk_r_new"] = float(r_new)
                                if r_new < float(risk_cfg.min_pct):
                                    reason = "risk_min"
                                else:
                                    qty_new = (r_new * equity_start_bar) / dist if dist > 0 else 0.0
                                    weight = (qty_new * price / equity_start_bar) if equity_start_bar > 0 else 0.0
                                    if weight <= 0:
                                        reason = "zero_weight"
                                    else:
                                        pos_weight = -float(weight)
                                        qty_open = float(qty_new)
                                        entry_price = price
                                        sl_active = float(sl_level) if not np.isnan(sl_level) else float('nan')
                                        executed_entry = True
                                        risk_info["risk_entry_qty"] = float(qty_new)
                                        risk_info["risk_entry_weight"] = float(weight)
                                        risk_info["risk_entry_dist"] = float(dist)
                                        risk_info["risk_entry_allowed"] = True
                                        risk_info["risk_entry_reason"] = ""
                else:
                    pos_weight = -1.0
                    qty_open = 1.0
                    entry_price = price
                    sl_active = float(sl_level) if not np.isnan(sl_level) else float('nan')
                    executed_entry = True
                    risk_info["risk_entry_allowed"] = True
                    risk_info["risk_entry_weight"] = 1.0
                    risk_info["risk_entry_dist"] = float(sl_level - price) if not np.isnan(sl_level) else np.nan

                if not executed_entry and reason:
                    risk_info["risk_entry_allowed"] = False
                    risk_info["risk_entry_reason"] = reason

            if abs(pos_weight) > entry_tol and not np.isnan(sl_level):
                if np.isnan(sl_active) or sl_level < sl_active:
                    sl_active = float(sl_level)

            tp_hit = (not np.isnan(tp_level)) and (price <= tp_level) and (abs(pos_weight) > entry_tol)
            sl_hit = (not np.isnan(sl_level)) and (price >= sl_level) and (abs(pos_weight) > entry_tol)
            rev_hit = bool(self.short_params.reverse_exit and exit_signal and (abs(pos_weight) > entry_tol))

            if tp_hit or sl_hit or rev_hit:
                if tp_hit:
                    risk_info["risk_exit_reason"] = "tp"
                elif sl_hit:
                    risk_info["risk_exit_reason"] = "sl"
                else:
                    risk_info["risk_exit_reason"] = "reverse"
                risk_info["risk_tp_hit"] = bool(tp_hit)
                risk_info["risk_sl_hit"] = bool(sl_hit)
                risk_info["risk_reverse_exit"] = bool(rev_hit)
                pos_weight = 0.0
                qty_open = 0.0
                entry_price = float('nan')
                sl_active = float('nan')

            if abs(pos_weight) <= entry_tol:
                qty_open = 0.0

            if risk_enabled and qty_open > 0 and equity_start_bar > 0 and not np.isnan(sl_active) and not np.isnan(entry_price):
                dist_after = max(0.0, sl_active - entry_price)
                open_risk_after = (qty_open * dist_after / equity_start_bar) if dist_after > 0 else 0.0
            else:
                open_risk_after = 0.0

            weight_change = abs(pos_weight - prev_pos)
            trade_cost = weight_change * spread if self.use_spread else 0.0
            strategy_ret = pos_weight * bar_return - trade_cost

            if self.log_return:
                equity = float(equity * np.exp(strategy_ret))
            else:
                equity = float(equity * (1.0 + strategy_ret))

            peak_equity = max(peak_equity, equity)

            risk_info.update({
                "risk_open_risk_pct_after": float(open_risk_after),
                "risk_sl_active": float(sl_active) if not np.isnan(sl_active) else np.nan,
                "risk_qty_open": float(qty_open) if risk_enabled else np.nan,
                "risk_position_weight": float(pos_weight),
                "risk_entry_price": float(entry_price) if not np.isnan(entry_price) else np.nan,
                "risk_strategy_return": float(strategy_ret),
                "risk_trade_cost": float(trade_cost),
                "risk_weight_change": float(weight_change),
                "risk_equity_end": float(equity),
                "risk_peak_equity_after": float(peak_equity),
                "risk_drawdown_pct_after": float(((peak_equity - equity) / peak_equity) if peak_equity > 0 else 0.0),
                "risk_equity_multiple": float(equity / (risk_cfg.equity_start if risk_enabled else 1.0)),
            })

            positions.append(float(pos_weight))
            strategy_vals.append(float(strategy_ret))
            trade_costs.append(float(trade_cost))
            weight_changes.append(float(weight_change))
            equity_series.append(float(equity))
            risk_records.append(risk_info)

            prev_pos = pos_weight

        positions_series = pd.Series(positions, index=base.index, name="position")
        strategy_series = pd.Series(strategy_vals, index=base.index, name="strategy")
        trade_cost_series = pd.Series(trade_costs, index=base.index, name="trade_cost")
        weight_change_series = pd.Series(weight_changes, index=base.index, name="weight_change")
        equity_series = pd.Series(equity_series, index=base.index, name="equity_curve")
        risk_df = pd.DataFrame(risk_records, index=base.index)

        bt = base.copy()
        if bt.empty:
            raise ValueError("Backtest frame is empty (după join). Verifică datele/parametrii.")

        bt["position"] = positions_series
        bt["Return"] = bt["Return"].fillna(0.0)
        bt["Spread"] = bt["Spread"].fillna(0.0)
        bt["strategy"] = strategy_series
        bt["trade_cost"] = trade_cost_series
        bt["weight_change"] = weight_change_series
        bt["equity_curve"] = equity_series
        bt = bt.join(risk_df)

        for col in ("DBG_reverse_trig", "DBG_longCond_raw", "DBG_BB_long01", "DBG_ATR_SL_Short",
                    "DBG_exit_cond", "DBG_EMA_longCond", "DBG_SAR_longCond", "DBG_MACD_longCond"):
            if col in base.columns:
                bt[col] = base[col]

        changes = (weight_change_series > entry_tol).astype(int)
        if not changes.empty:
            changes.iloc[0] = int(abs(positions_series.iloc[0]) > entry_tol)
        bt["trades"] = changes

        if self.log_return:
            bt["creturn"]   = bt["Return"].cumsum().apply(np.exp)
            bt["cstrategy"] = bt["strategy"].cumsum().apply(np.exp)
        else:
            bt["creturn"]   = (1.0 + bt["Return"]).cumprod()
            bt["cstrategy"] = (1.0 + bt["strategy"]).cumprod()

        _dbg(
            f"[BT] summary → rows={len(bt)}, trades={int(bt['trades'].sum())}, "
            f"pos_rate={(bt['position'].abs() > entry_tol).mean():.3f}, "
            f"perf={bt['cstrategy'].iloc[-1]:.4f}, bh={bt['creturn'].iloc[-1]:.4f}"
        )

        self._last_bt = bt
        return bt