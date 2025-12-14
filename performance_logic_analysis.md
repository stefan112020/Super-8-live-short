# Analiză comparație metrici de performanță

## Concluzie scurtă
Logica din `super8_short_backtest.py` este mai corectă pentru evaluarea performanței deoarece agregă aritmetic randamentele pe tranzacție prin compunere (în mod non-log), aliniind profit factor-ul și win rate-ul cu curba `cstrategy`. Versiunea din `Versiune 2.ipynb` însumează randamentele aritmetice fără compunere, ceea ce subestimează pierderile și poate supraestima profit factor-ul și rata de câștig.

## Argumente detaliate
- **Agregare randamente pe tranzacție**: în `Versiune 2.ipynb`, `_extract_trade_returns` adună pur și simplu randamentele bară cu bară indiferent de mod (`log_returns=True/False`), ceea ce tratează randamentele aritmetice ca și cum ar fi log-return-uri. Rezultatul este că, atunci când rulezi backtestul în modul aritmetic, profitul/pierderea pe tranzacție nu reflectă efectul de compunere și poate devia de la curba `cstrategy`.【F:Versiune 2.ipynb†L18-L109】
- **Compunere corectă în modul aritmetic**: în `super8_short_backtest.py`, `_extract_trade_returns` multiplică randamentele aritmetice în interiorul unei tranzacții și scade 1 la închidere, astfel încât profit factor-ul și win rate-ul se bazează pe același P&L ca `cstrategy`. În modul log se păstrează suma log-return-urilor, deci ambele moduri sunt consistente.【F:super8_short_backtest.py†L156-L227】
- **Aliniere metrici**: ambele versiuni calculează P&L final, drawdown, profit factor, win rate și durate pe baza `cstrategy`/`equity_curve`, însă doar implementarea din `super8_short_backtest.py` garantează că profit factor-ul și win rate-ul folosesc aceleași randamente compuse ca performanța raportată, evitând discrepanțe între optimizare și backtest.【F:super8_short_backtest.py†L185-L239】
