# Stare curentă a verificărilor și testelor

Această listă reflectă ce este acoperit (sau nu) în codul actual pentru cerințele de testare și operare.

| Aspect | Observații din cod/repositoriu | Stare curentă |
| --- | --- | --- |
| Testare extensivă (unit tests) | Nu există directoare de teste sau fișiere de test; repo-ul conține doar modulele principale și notebook-uri. | **Neacoperit** |
| Simulare de erori API | Nu am găsit teste automate sau hooks dedicate pentru a simula răspunsuri de eroare din API-urile Binance; fluxul live se bazează pe apeluri directe fără harness de test. | **Neacoperit** |
| Backtesting vs live (comparare rezultate) | Există cod de backtest (`super8_short_backtest.py`) și notebook-uri de parametrare, dar nu există un script/raport care să compare sistematic rezultatele live cu cele din backtest. | **Neacoperit** |
| Paper trading (`dry_run=True`) | Parametrul `dry_run` există în configurația live, dar valoarea implicită este `False`, iar repo-ul nu include loguri sau teste care să ateste rulări extinse în mod paper. | **Neacoperit** |
| Monitoring 24/7 | Nu este documentat niciun mecanism de health-check/alertare; există doar thread-ul de reconciliere din codul live. | **Neacoperit** |
| Capital mic inițial | Configurația simbolului acceptă un prag minim (`min_usd`), dar nu există scenarii/teste dedicate pentru rulări cu capital redus. | **Neacoperit** |
| Kill switch / oprire rapidă | Codul live expune metoda `stop(flatten=True)` care setează un eveniment de oprire, închide pozițiile și oprește stream-urile. | **Parțial acoperit (funcționalitate există, fără teste)** |
| Respectarea rate limiting-ului Binance | Nu există logică de throttling sau tracking al numărului de request-uri în utilitățile de broker; nici teste pentru acest comportament. | **Neacoperit** |
| Backup / jurnalizare trades | Configurația live permite scrierea într-un CSV (`log_csv`), însă nu există persistență într-o bază de date sau testare a backup-urilor. | **Parțial acoperit (CSV opțional)** |
| Version control pentru rulările live | Repozitoriul nu conține etichete (`git tag` nu returnează rezultate) sau versiuni marcate pentru build-urile rulate live. | **Neacoperit** |
