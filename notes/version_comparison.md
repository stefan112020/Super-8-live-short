# Version comparison: USDT vs USDC risk handling

## Version 1 (USDT)
- `SymbolConfig.symbol` examples and defaults used USDT-margined pairs such as `BTCUSDT`.
- The broker adapter retrieved equity by inspecting the `USDT` balance via `account_equity_usdt`.
- Risk aggregation variables (for example `risk_usdt`) and PnL annotations assumed all risk and reporting were denominated in USDT.
- Minimum sizing documentation mentioned thresholds in USDT.

## Version 2 (USDC)
- All symbol examples now reference USDC-margined markets such as `AVAXUSDC` or `BTCUSDC`.
- The broker adapter exposes `account_equity_usdc`, pulling the account's `USDC` balance and feeding it through sizing helpers.
- Risk tracking variables (e.g., `risk_usdc`) and PnL documentation clarify that calculations run in USDC.
- Minimum order sizing guidance is rewritten for USDC thresholds.

## Recommendation
For traders who operate exclusively on Binance Futures USDC-margined contracts, Version 2 is the safer choice because the equity lookups, risk limits, and documentation all align with the actual collateral currency. Using Version 1 in that environment would make the bot inspect the wrong wallet (`USDT`), potentially reporting zero equity and disabling entries or mis-sizing orders. Conversely, if your account and instruments are still USDT-margined, Version 1 should be retained so that risk controls continue to measure the correct balance.
