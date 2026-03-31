# Forex Quant Bot

Professional-grade, forex-only, systematic trading bot for Interactive Brokers paper trading and deterministic CSV-based backtesting.

## Why Dukascopy for historical data

Dukascopy is the default historical provider in this project because it offers broad forex coverage, high-quality free data, and a practical Python download path through `dukascopy-python`, while still fitting a simple local CSV cache workflow. The bot downloads bars into `data_cache/`, reuses cached files when available, and replays them in strict chronological order for backtests with no live API calls.

## Architecture

```text
forex_quant_bot/
├── backtest/
├── core/
├── data/
├── live/
├── logs/
├── strategies/
├── utils/
├── models.py
├── settings.py
└── cli.py
```

## Features

- Forex-only execution and analytics.
- Three strategies: trend-following, mean reversion, breakout.
- Regime detection with ADX, ATR, and Hurst exponent.
- Multi-strategy allocator with performance, confidence, regime fit, and diversification scoring.
- Risk overlay with ATR-sized stops, 1% default risk sizing, 5% max open risk, and a 10% drawdown circuit breaker.
- CSV outputs for trades, transactions, strategy metrics, regime metrics, risk metrics, and equity curve.
- Human-readable text summary matching the requested metrics layout.
- IB paper-only live mode guarded by paper ports `7497` and `4002`.

## Installation

```bash
pip install -r requirements.txt
```

## Interactive Brokers paper setup

1. Start TWS or IB Gateway in paper trading mode.
2. Enable API access.
3. Use paper ports only: `7497` for TWS paper or `4002` for Gateway paper.
4. Run the bot with `--mode live`.

## Backtest examples

```bash
python main.py --mode backtest --pair EURUSD --start 2018-01-01 --end 2024-12-31 --timeframe 1h
python main.py --mode backtest --pair GBPUSD --start 2015-01-01 --end 2025-01-01 --timeframe 15m --slippage-bps 1.0
```

If you want to test the full pipeline without downloading Dukascopy history first:

```bash
python main.py --mode backtest --pair EURUSD --start 2024-01-01 --end 2024-12-31 --timeframe 1h --use-sample-data
```

## Live paper trading example

```bash
python main.py --mode live --pairs EURUSD,GBPUSD --timeframe 5m --ib-port 7497 --client-id 17
```

## Output artifacts

Each run creates a timestamped folder under `output/` containing:

- `trades.csv`
- `transactions.csv`
- `strategy_metrics.csv`
- `regime_metrics.csv`
- `risk_overlay_metrics.csv`
- `equity_curve.csv`
- `returns.csv`
- `details.csv`
- `summary.txt`

## Notes

- Backtests never call Interactive Brokers.
- Live mode only uses Interactive Brokers through `ib_insync` or `ib_async` as a compatibility fallback.
- CSV metrics are computed from actual trade history only; there is no look-ahead logic.
- For non-USD quote currencies, the system attempts to fetch the necessary USD conversion pair so P&L stays in USD.
