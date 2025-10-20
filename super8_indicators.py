
import numpy as np
import pandas as pd
from typing import Dict, Any

def _rma(series: pd.Series, period: int) -> pd.Series:
    alpha = 1.0 / period
    return series.ewm(alpha=alpha, adjust=False).mean()

class Super8Indicators:
    def __init__(self, params: Dict[str, Any]):
        self.p = params
        self.results: pd.DataFrame | None = None

    def _ema(self, s: pd.Series, span: int) -> pd.Series:
        return s.ewm(span=span, adjust=False).mean()

    def _adx_block(self, df: pd.DataFrame, length: int, smoothing: int, threshold: float, suffix: str = "") -> pd.DataFrame:
        tr = np.maximum(
            df["high"] - df["low"],
            np.maximum((df["high"] - df["Price"].shift(1)).abs(), (df["low"] - df["Price"].shift(1)).abs())
        )
        up = df["high"].diff()
        dn = -df["low"].diff()
        plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
        minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)

        tr_s    = _rma(pd.Series(tr, index=df.index), length)
        plus_s  = _rma(pd.Series(plus_dm, index=df.index), length)
        minus_s = _rma(pd.Series(minus_dm, index=df.index), length)

        plus_di  = (plus_s / tr_s) * 100.0
        minus_di = (minus_s / tr_s) * 100.0
        denom    = (plus_di + minus_di).clip(lower=1e-10)  # protecție ca în Pine (max 1e-10)
        dx       = ((plus_di - minus_di).abs() / denom) * 100.0

        # ADX cu smoothing separat (ca ta.dmi(len, smo))
        adx = _rma(dx, smoothing)

        out = pd.DataFrame(index=df.index)
        out[f"adx{suffix}"] = adx
        out[f"di_plus{suffix}"] = plus_di
        out[f"di_minus{suffix}"] = minus_di
        out[f"ADX_longCond{suffix}"] = (plus_di > minus_di) & (adx > threshold)
        out[f"ADX_shortCond{suffix}"] = (plus_di < minus_di) & (adx > threshold)
        return out

    def _sar(self, df: pd.DataFrame, start: float, step: float, smax: float) -> pd.Series:
        sar_vals = []
        af = start
        is_long = True
        ep = df["low"].iloc[0] if is_long else df["high"].iloc[0]
        sar_vals.append(df["high"].iloc[0] if is_long else df["low"].iloc[0])
        for i in range(1, len(df)):
            prev_sar = sar_vals[-1]
            current_sar = prev_sar + af * (ep - prev_sar)
            if is_long:
                current_sar = min(current_sar, df["low"].iloc[i-1], df["low"].iloc[i])
            else:
                current_sar = max(current_sar, df["high"].iloc[i-1], df["high"].iloc[i])

            if is_long:
                if df["high"].iloc[i] > ep:
                    ep = df["high"].iloc[i]
                    af = min(af + step, smax)
            else:
                if df["low"].iloc[i] < ep:
                    ep = df["low"].iloc[i]
                    af = min(af + step, smax)

            if (is_long and df["low"].iloc[i] < current_sar) or (not is_long and df["high"].iloc[i] > current_sar):
                is_long = not is_long
                af = start
                current_sar = ep
                ep = df["high"].iloc[i] if is_long else df["low"].iloc[i]

            sar_vals.append(current_sar)
        return pd.Series(sar_vals, index=df.index, name="SAR")
    
    def _sar_tv(self,
                high: pd.Series,
                low: pd.Series,
                start: float,
                step: float,
                smax: float,
                price: pd.Series | None = None) -> pd.Series:
        """
        Parabolic SAR în stil TradingView/Wilder (compatibil cu ta.sar).
        - Clamping: pentru uptrend, PSAR ≤ min(low[i-1], low[i-2]); pentru downtrend, PSAR ≥ max(high[i-1], high[i-2])
        - Flip: PSAR devine EP, AF reset la 'start', EP devine extrema direcției noi
        """
        h = high.to_numpy()
        l = low.to_numpy()
        n = len(h)
        if n == 0:
            return pd.Series([], index=high.index, dtype=float)
        if n == 1:
            # fallback trivial
            init = l[0] if (price is None or n == 1) else (l[0] if np.isnan(price.iloc[0]) else price.iloc[0])
            return pd.Series([init], index=high.index, dtype=float)

        psar = np.zeros(n, dtype=float)

        # 1) Direcție inițială: dacă avem preț (close/Price), folosește-l;
        #    altfel, o euristică robustă din high/low.
        if price is not None and len(price) >= 2 and not (pd.isna(price.iloc[0]) or pd.isna(price.iloc[1])):
            up = bool(price.iloc[1] >= price.iloc[0])
        else:
            up = bool((h[1] + l[1]) >= (h[0] + l[0]))

        # 2) Inițializări Wilder
        af = float(start)
        ep = float(h[0] if up else l[0])
        psar[0] = float(l[0] if up else h[0])

        for i in range(1, n):
            # PSAR de bază
            psar[i] = psar[i-1] + af * (ep - psar[i-1])

            if up:
                # clamping sub ultimele 1-2 minime
                if i >= 2:
                    psar[i] = min(psar[i], l[i-1], l[i-2])
                else:
                    psar[i] = min(psar[i], l[i-1])

                # nou EP?
                if h[i] > ep:
                    ep = h[i]
                    af = min(af + step, smax)

                # flip?
                if l[i] < psar[i]:
                    up = False
                    psar[i] = ep
                    ep = l[i]
                    af = start
            else:
                # clamping peste ultimele 1-2 maxime
                if i >= 2:
                    psar[i] = max(psar[i], h[i-1], h[i-2])
                else:
                    psar[i] = max(psar[i], h[i-1])

                # nou EP?
                if l[i] < ep:
                    ep = l[i]
                    af = min(af + step, smax)

                # flip?
                if h[i] > psar[i]:
                    up = True
                    psar[i] = ep
                    ep = h[i]
                    af = start

        return pd.Series(psar, index=high.index, name="SAR")


    def _macd(self, s: pd.Series, fast: int, slow: int, signal: int) -> pd.DataFrame:
        fast_ma = s.ewm(span=fast, adjust=False).mean()
        slow_ma = s.ewm(span=slow, adjust=False).mean()
        macd_line = fast_ma - slow_ma
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        hist = macd_line - signal_line
        out = pd.DataFrame(index=s.index)
        out["hist"] = hist
        out["lMACD"] = macd_line
        out["sMACD"] = signal_line
        return out

    def _macz(self, df: pd.DataFrame) -> pd.DataFrame:
        # MAC-Z per Pine
        lengthz = self.p["lengthz"]
        lengthStdev = self.p["lengthStdev"]
        A = self.p["A"]
        B = self.p["B"]
        signalLength = self.p["signalLength"]
        # z-vwap
        vol = df["volume"]
        px = df["Price"]
        vw_mean = (vol * px).rolling(window=lengthz, min_periods=lengthz).sum() / vol.rolling(window=lengthz, min_periods=lengthz).sum()
        vw_sd = (px - vw_mean).pow(2).rolling(window=lengthz, min_periods=lengthz).mean().pow(0.5)
        zscore = (px - vw_mean) / vw_sd
        # macz
        macd_std = px.rolling(window=lengthStdev, min_periods=lengthStdev).std(ddof=0)
        macd = self._macd(px, self.p["fastLength"], self.p["slowLength"], self.p["signalLength"])
        macz = (zscore * A) + (macd["lMACD"] / (macd_std * B))
        signal = macz.rolling(window=signalLength, min_periods=signalLength).mean()
        histmacz = macz - signal
        out = pd.DataFrame(index=df.index)
        out["histmacz"] = histmacz
        return out

    def _bbands(self, s: pd.Series, length: int, mult: float) -> pd.DataFrame:
        mid = s.rolling(window=length, min_periods=length).mean()
        std = s.rolling(window=length, min_periods=length).std(ddof=0)
        upper = mid + mult * std
        lower = mid - mult * std
        width = (upper - lower) / mid
        out = pd.DataFrame(index=s.index)
        out["BB_middle"] = mid
        out["BB_upper"] = upper
        out["BB_lower"] = lower
        out["BB_width"] = width
        return out

    def _volume_flags(self, vol: pd.Series, length: int, factor: float) -> pd.DataFrame:
        sma = vol.rolling(window=length, min_periods=1).mean()
        cond = vol > (sma * factor)
        out = pd.DataFrame(index=vol.index)
        out["VOL_shortCond"] = cond
        out["VOL_longCond"] = cond
        out["sma_volume"] = sma
        return out

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.p
        out = pd.DataFrame(index=df.index)

        # EMAs
        sEMA = self._ema(df["Price"], p["sEma_Length"])
        fEMA = self._ema(df["Price"], p["fEma_Length"])
        out["sEMA"] = sEMA
        out["fEMA"] = fEMA
        out["EMA_longCond"] = (fEMA > sEMA) & (sEMA > sEMA.shift(1))
        out["EMA_shortCond"] = (fEMA < sEMA) & (sEMA < sEMA.shift(1))

        # ADX
        adx_main = self._adx_block(df, p["ADX_len"], p.get("ADX_smo", p["ADX_len"]), p["th"], suffix="")
        out = out.join(adx_main)

        sar = self._sar_tv(df["high"], df["low"], p["Sst"], p["Sinc"], p["Smax"], df["Price"])
        out["SAR"] = sar
        out["SAR_longCond"]  = (out["SAR"] < df["Price"]).astype(bool)
        out["SAR_shortCond"] = (out["SAR"] > df["Price"]).astype(bool)

        # MACD
        macd = self._macd(df["Price"], p["fastLength"], p["slowLength"], p["signalLength"])
        out = out.join(macd)

        # MAC-Z (optional; keep histmacz only)
        use_macz = (p.get('MACD_options', 'MACD') == 'MAC-Z')
        if use_macz:
            macz = self._macz(df)
            out = out.join(macz)
        else:
            out['histmacz'] = out['hist']

        # MACD option flags
        use_macz = (p.get("MACD_options", "MACD") == "MAC-Z")
        out["MACD_longCond"] = (out["histmacz"] > 0) if use_macz else (out["hist"] > 0)
        out["MACD_shortCond"] = (out["histmacz"] < 0) if use_macz else (out["hist"] < 0)

        # BB
        bb = self._bbands(df["Price"], p["BB_Length"], p["BB_mult"])
        out = out.join(bb)

        # Volume
        volf = self._volume_flags(df["volume"], p["sma_Length"], p["volume_f"])
        out = out.join(volf)

        # Min widths as decimals (e.g., 5% -> 0.05)
        out["bbMinWidth01"] = p["bbMinWidth01"] / 100.0
        out["bbMinWidth02"] = p["bbMinWidth02"] / 100.0

        # Cross helpers (used later in strategy, but keep here for reuse)
        # No direct Pine ta.crossover here; implement in strategy layer.

        # Clean
        #self.results = out.dropna(subset=["BB_middle", "BB_upper", "BB_lower"]).copy()
        
        print(f"[IND] rows={len(out)} | cols={list(out.columns)[:8]}... "
              f"| has_BB={all(c in out.columns for c in ['BB_middle','BB_upper','BB_lower'])} "
              f"| has_hist={'hist' in out.columns} | has_histmacz={'histmacz' in out.columns}")
        self.results = out.copy() 
        return self.results
