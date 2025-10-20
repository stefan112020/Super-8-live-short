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
    Position: str = "Both"
    TP_options: str = "Both"
    SL_options: str = "Both"
    tp: float = 1.8
    sl: float = 8.0
    atrPeriodSl: int = 14
    multiplierPeriodSl: float = 15.0
    trailOffset: float = 0.0
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
class Super8ShortBacktester:
    indicator_params: Dict[str, Any]
    short_params: ShortParams
    use_spread: bool = True
    log_return: bool = True

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
        position = []
        pos = 0
        entry_price = np.nan

        for t, row in base.iterrows():
            # respect start_time
            if self.short_params.start_time is not None and t < self.short_params.start_time:
                position.append(0)
                continue

            price = row["Price"]

            # 2) aplică intrările semnalate pe bara anterioară (entry next-bar)
            if pos == 0 and entry_sig.loc[t]:
                pos = -1
                entry_price = price  # intrare la close-ul barei curente

            # 3) calculează niveluri TP/SL pentru starea curentă
            avg_price = entry_price if (pos == -1 and not np.isnan(entry_price)) else price

            # --- TP (short) ---
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

            # --- SL (short) ATR stair-step ---
            atr_sl_val = atr_sl_short_series.loc[t] if t in atr_sl_short_series.index else np.nan
            if self.short_params.SL_options == "Both":
                sl_level = max(atr_sl_val, (1.0 + self.short_params.sl/100.0) * avg_price)
            elif self.short_params.SL_options == "Normal":
                sl_level = (1.0 + self.short_params.sl/100.0) * avg_price
            elif self.short_params.SL_options == "ATR":
                sl_level = atr_sl_val
            else:
                sl_level = np.nan
            

            # 5) dacă suntem în poziție, evaluează ieșirile SAME-BAR (TP/SL și reverse-exit)
            if pos == -1:
                tp_hit = (not np.isnan(tp_level)) and (price <= tp_level)
                sl_hit = (not np.isnan(sl_level)) and (price >= sl_level)
                rev_hit = bool(self.short_params.reverse_exit and exit_sig.loc[t])

                if tp_hit or sl_hit or rev_hit:
                    # ieșim la finalul barei curente; P&L pe bară rămâne prins (position deja salvat)
                    pos = 0
                    entry_price = np.nan

            position.append(pos)
            
        positions = pd.Series(position, index=base.index, name="position")

        # Backtest using returns + spread
        bt = base.copy()

        
        bt["position"] = positions
        bt["Return"]   = bt["Return"].fillna(0.0)
        bt["Spread"]   = bt["Spread"].fillna(0.0)
        bt["strategy"] = bt["position"] * bt["Return"]

        if bt.empty:
            raise ValueError("Backtest frame is empty (după join). Verifică datele/parametrii.")

        # adu în bt flag-urile de debug (dacă există în base)
        for col in ("DBG_reverse_trig", "DBG_longCond_raw", "DBG_BB_long01", "DBG_ATR_SL_Short",
                    "DBG_exit_cond", "DBG_EMA_longCond", "DBG_SAR_longCond", "DBG_MACD_longCond"):
            if col in base.columns:
                bt[col] = base[col]

        # trades: 1 la schimbare de poziție; primul bar e trade dacă intri direct în -1
        pos_series = bt["position"].astype(int)
        changes = pos_series.ne(pos_series.shift(1)).astype(int)
        changes.iloc[0] = int(pos_series.iloc[0] != 0)
        bt["trades"] = changes

        # cost de spread (dacă e cazul)
        if self.use_spread:
            bt["strategy"] = bt["strategy"] - bt["trades"] * bt["Spread"]

        # cumul
        if self.log_return:
            bt["creturn"]   = bt["Return"].cumsum().apply(np.exp)
            bt["cstrategy"] = bt["strategy"].cumsum().apply(np.exp)
        else:
            bt["creturn"]   = (1.0 + bt["Return"]).cumprod()
            bt["cstrategy"] = (1.0 + bt["strategy"]).cumprod()

        _dbg(
            f"[BT] summary → rows={len(bt)}, trades={int(bt['trades'].sum())}, "
            f"pos_rate={(bt['position'] == -1).mean():.3f}, "
            f"perf={bt['cstrategy'].iloc[-1]:.4f}, bh={bt['creturn'].iloc[-1]:.4f}"
        )

        self._last_bt = bt
        return bt
