# Bateria de teste

Fișierele de test sunt în directorul `tests/`.

- `test_ensure_flat.py` – conține cele trei scenarii unitare pentru metoda `ensure_flat` din notebook:
  1. `test_already_flat_returns_true_without_orders` verifică situația în care runner-ul este deja flat.
  2. `test_under_min_notional_triggers_double_trigger` verifică activarea fallback-ului sub pragul de notional minim.
  3. `test_market_close_path_executes_reduce_only` confirmă că ieșirea principală MARKET reduce-only funcționează.

Execută bateria de teste cu:

```bash
python -m unittest discover -v
```

Acestă comandă scanează directorul `tests/` și rulează toate testele definite în fișierele `test_*.py`.
