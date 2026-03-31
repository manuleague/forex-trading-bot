from __future__ import annotations

import argparse
import asyncio
import logging

from forex_quant_bot.backtest.engine import BacktestEngine
from forex_quant_bot.live.live_runner import LiveRunner
from forex_quant_bot.settings import build_config_from_args



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Forex-only quant bot for IB paper trading and CSV-backed backtesting.")
    parser.add_argument("--mode", choices=["backtest", "live"], required=True)
    parser.add_argument("--pair", dest="pair", help="Single forex pair, e.g. EURUSD")
    parser.add_argument("--pairs", dest="pairs", help="Comma-separated forex pairs, e.g. EURUSD,GBPUSD")
    parser.add_argument("--timeframe", required=True, choices=["1m", "5m", "15m", "30m", "1h", "4h", "D", "W"])
    parser.add_argument("--start", help="UTC start time, e.g. 2018-01-01 or 2018-01-01T00:00:00+00:00")
    parser.add_argument("--end", help="UTC end time, e.g. 2025-12-31 or 2025-12-31T00:00:00+00:00")
    parser.add_argument("--strategy-config", default="config/strategy_config.yaml")
    parser.add_argument("--pairs-config", default="config/pairs.yaml")
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--provider", default="dukascopy")
    parser.add_argument("--use-sample-data", action="store_true")
    parser.add_argument("--sample-rows", type=int, default=4000)
    parser.add_argument("--no-auto-download", action="store_true")
    parser.add_argument("--initial-capital", type=float)
    parser.add_argument("--risk-per-trade", type=float)
    parser.add_argument("--max-total-exposure", type=float)
    parser.add_argument("--max-notional-leverage", type=float)
    parser.add_argument("--atr-stop-multiplier", type=float)
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument("--ib-host", default="127.0.0.1")
    parser.add_argument("--ib-port", type=int, default=7497)
    parser.add_argument("--client-id", type=int, default=7)
    parser.add_argument("--account")
    parser.add_argument("--quiet", action="store_true")
    return parser



def configure_logging() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(name)s:%(message)s", force=True)
    for logger_name in ["DUKASCRIPT", "dukascopy_python", "dukascopy"]:
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.propagate = False
        logger.setLevel(logging.WARNING)
        logger.addHandler(logging.NullHandler())



def main() -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()
    config = build_config_from_args(args)

    if config.mode == "backtest":
        report = BacktestEngine(config).run()
        if config.summary_to_stdout:
            print(report["summary_text"])
            print(f"\nArtifacts saved to: {report['output_dir']}")
        return

    if config.mode == "live":
        asyncio.run(LiveRunner(config).run())
        return

    raise ValueError(f"Unsupported mode: {config.mode}")
