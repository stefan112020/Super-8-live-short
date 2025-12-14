# Note despre evaluarea performanței (Strategiile 1 și 2)

## Situația de **dinainte** (ambele strategii)
- **Profit factor**: formula era `gross_profit / abs(gross_loss)` și, dacă `gross_loss == 0`, era setată direct la `inf`, indiferent dacă existau sau nu tranzacții. Asta făcea ca backtestele fără tranzacții sau cu o singură tranzacție câștigătoare să raporteze PF = ∞, deși nu exista risc consumat sau eșantion relevant.
- **Conversia randamentelor log**: lista de randamente pe tranzacție era construită în log-return (suma bar-urilor), apoi convertită cu `np.exp(x) - 1`. Dacă nu exista tranzacție, lista era goală, dar PF era tot ∞.
- **Win rate**: era calculat ca wins / total_trades, dar pentru PF nu exista protecție pentru cazul „zero tranzacții”.

### Exemple de ce nu era realist
- **Exemplu A — zero tranzacții**: dacă strategia nu deschidea poziții (de pildă, toate condițiile de intrare erau false), `trade_logs = []` ⇒ `gross_profit = 0`, `gross_loss = 0`, iar PF devenea ∞. Un PF infinit fără tranzacții creează impresia falsă că strategia ar fi „perfectă”.
- **Exemplu B — 1 tranzacție câștigătoare, fără pierderi**: cu un singur trade de +2% (`trade_logs = [np.log(1.02)]`), conversia dă `trade_pnls = [0.02]`, `gross_profit = 0.02`, `gross_loss = 0`, PF = ∞. În practică, un singur câștig nu justifică un PF infinit; lipsesc pierderile pentru a măsura raportul risc/recompensă.
- **Exemplu C — 1 tranzacție pierzătoare, fără câștiguri**: `trade_logs = [np.log(0.98)]` ⇒ `trade_pnls = [-0.02]`, `gross_profit = 0`, `gross_loss = -0.02`, PF = 0. Aici PF 0 e ok, dar problema de mai sus (∞ la zero trades sau doar wins) rămânea.

## Situația **după** corectări (aliniat la Strategia 1 validată)
- **Profit factor**: se calculează la fel, dar acum dacă nu există tranzacții (`total_trades == 0`) PF devine `0.0`, iar dacă există doar câștiguri PF este `inf` (și e clar vizibil că există trades pentru care raportul e valid). Protecția „zero trades” elimină raportările înșelătoare.
- **Win rate**: păstrează formula wins / total_trades, dar împreună cu PF protejat (0 la zero trades) nu mai apare perechea „PF infinit, win rate 0%, trades 0”.
- **Conversie randamente**: `trade_pnls = [np.exp(x) - 1.0 for x in trade_logs]` când log_returns=True, altfel se folosesc valorile brute. Se păstrează compatibilitatea cu log-return.
- **Raportare**: sumarul de performanță afișează explicit `total_trades`, astfel încât PF-ul infinit apare doar când există trades câștigătoare și zero pierderi; cazul fără trades apare cu PF = 0.

### Exemple de ce e mai corect
- **Exemplu A (zero tranzacții)**: `trade_logs = []` ⇒ `total_trades = 0`, PF = 0.0, win rate = 0%. Output-ul indică limpede că nu au existat tranzacții, deci nu există un raport PF relevant.
- **Exemplu B (1 câștig)**: `trade_logs = [np.log(1.02)]` ⇒ `trade_pnls = [0.02]`, `gross_profit = 0.02`, `gross_loss = 0`, `total_trades = 1`, PF = ∞. Afișarea include „din 1 tranzacție”, deci devine evident că PF-ul infinit provine dintr-un singur eveniment, nu dintr-un istoric robust.
- **Exemplu C (mix câștig + pierdere)**: `trade_logs = [np.log(1.02), np.log(0.98)]` ⇒ `trade_pnls = [0.02, -0.02]`, PF = 1.0, win rate 50%, total_trades = 2. Raportul este finitul așteptat și reflectă balansul câștig/pierdere.

## Referințe în cod
- Strategia 2 (backtest scurt) folosește acum protecția pentru zero tranzacții în `profit_factor` și win rate, plus conversia safe a randamentelor log.【F:super8_short_backtest.py†L108-L146】
- Exemplele de mai sus reproduc exact scenariile pe care vechea formulă le raporta greșit (PF = ∞ fără trades) și pe care noua formulă le tratează explicit (PF = 0 la zero trades).【F:performance_notes.md†L6-L47】
