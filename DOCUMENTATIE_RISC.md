# Documentație în limba română pentru ajustările de risk

## Context
Modificările aduse fișierului `super8_short_backtest.py` urmăresc să copieze fidel mecanismul de gestionare a riscului din notebook-ul live `Super_8_Short_live_CODEX.ipynb`, astfel încât testele din varianta Python (`super8_short_backtest.py` + `super8_indicators.py`) să răspundă identic la aceleași setări de risk.

## Ce s-a implementat
1. **Structură dedicată pentru configurarea riscului** – am introdus un `dataclass` numit `RiskSettings` care grupează toți parametrii folosiți în execuția live (baza de risc, limita minimă/maximă, bufferul de cap, oprirea la drawdown și capitalul inițial).
2. **Buclă de backtest conștientă de risc** – logica de poziționare calculează distanța până la stop-loss și dimensionează poziția astfel încât riscul per tranzacție să respecte procentul setat și să țină cont de riscul deja deschis.
3. **Blocaj la drawdown** – dacă echitatea scade sub pragul `dd_stop_pct`, sistemul refuză noi intrări până când echitatea se recuperează, la fel ca în varianta live.
4. **Diagrame și coloane suplimentare** – fiecare bară primește valori detaliate despre risc (`risk_*`), echitate, costuri de tranzacționare și motivele pentru care o intrare a fost acceptată sau refuzată.
5. **Compatibilitate cu logica existentă** – am păstrat toate condițiile de intrare/ieșire și modul de calcul ATR, astfel încât diferența față de versiunea anterioară constă exclusiv în modul în care se dimensionează pozițiile și se raportează diagnosticele.

## Cum am gândit modificările
- **Paritate cu rularea live**: am analizat fluxul din notebook-ul live și am reprodus etapele-cheie (verificări de semnal, distanțe până la SL, limite de risc) într-o formă curată și deterministă pentru backtest.
- **Observabilitate maximă**: fiecare decizie (intrare refuzată, stop ajustat, TP/SL atins) este logată într-un set de coloane `risk_...`, astfel încât să vezi rapid ce s-a întâmplat pe fiecare bară.
- **Protecție împotriva erorilor**: toate ramurile au fallback-uri (de exemplu, dacă indicatorii lipsesc, dacă distanța SL este nevalidă), ceea ce permite testarea scenariilor extreme fără a opri rularea.

## Cum aplici în practică
1. **Inițializează configurarea riscului**:
   ```python
   risk = RiskSettings(
       enabled=True,
       base_pct=0.01,
       min_pct=0.0025,
       cap_pct=0.073,
       cap_buffer_pct=0.05,
       dd_stop_pct=0.075,
       equity_start=1_000_000,
   )
   ```
2. **Rulează backtestul cu aceleași setări ca în live**:
   ```python
   bt = Super8ShortBacktester(indicator_params, short_params, risk=risk).run(dataframe)
   ```
3. **Analizează diagnosticele**: în `bt` vei găsi coloane precum `risk_entry_reason`, `risk_open_risk_pct_before`, `risk_equity_multiple`. Acestea te ajută să vezi imediat de ce o intrare a fost blocată, ce expunere aveai și cum evoluează capitalul.
4. **Iterează parametrii**: modifică `base_pct`, `cap_pct` sau `dd_stop_pct` și re-rulează backtestul; diferențele în poziții (`position`), profit (`cstrategy`) și risc (`risk_*`) îți arată impactul exact al fiecărei ajustări.

## De ce există trei profiluri de risk (din răspunsul anterior)
- **Conservator** – plafonează expunerea pentru validarea inițială.
- **Balanced live mirror** – replică exact comportamentul live pentru comparație corectă.
- **Agresiv** – împinge limitele pentru optimizare, dar menține raportările de risc ca să observi rapid când devine prea periculos.

## Recomandare
Folosește profilul „Balanced live mirror” ca variantă de bază. Astfel, orice experiment în backtest se va corela direct cu modul în care strategia se comportă în execuția live, iar coloanele de diagnostic îți oferă transparență completă asupra efectului fiecărui parametru de risk.
