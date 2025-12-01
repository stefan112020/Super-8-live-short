# Modificări introduse recent

## super8_short_backtest.py
- Am sincronizat parametrii impliciți ai strategiei short (TP/SL, ATR) cu cei folosiți în Versiune-2.ipynb.
- Am implementat calculul și afișarea metrilor de performanță (profit factor, max drawdown, P&L, win rate, durată maximă a tranzacției) după logica PineScript.
- Am corectat modul de combinare a SL ATR cu SL procentual pentru opțiunea `Both`, folosind nivelul mai apropiat de preț (minimul pentru short), exact ca în PineScript.
- Am păstrat coloanele de diagnostic și comportamentul de intrare/ieșire (TP, SL, reverse) conform condițiilor din PineScript.

## Super 8 parametri - 1.ipynb
- Notebook-ul folosește parametrii actualizați și afișează metricele de performanță calculate în backtest.
- Exportul Excel conține atât datele de backtest, cât și o foaie suplimentară cu metricele (profit factor, max DD etc.).
- Am menținut logica de evaluare intrabar și prezentarea rezultatelor ca să rămână consistente cu rulările din Versiune-2.ipynb.
