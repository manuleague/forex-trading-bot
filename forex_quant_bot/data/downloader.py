from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from forex_quant_bot.utils.time_utils import dukascopy_interval_name, normalize_pair, pandas_freq_for_timeframe

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DukascopyDownloader:
    cache_dir: Path
    provider_name: str = "dukascopy"

    def download(self, pair: str, start: datetime, end: datetime, timeframe: str, output_path: Path) -> pd.DataFrame:
        pair = normalize_pair(pair)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        start = start.astimezone(timezone.utc)
        end = end.astimezone(timezone.utc)

        try:
            dataframe = self._download_with_dukascopy_python(pair, start, end, timeframe)
        except Exception as exc:  # pragma: no cover - defensive runtime fallback
            raise RuntimeError(
                "Dukascopy download failed. Install dukascopy-python and ensure network access is available, "
                "or pre-populate the CSV cache."
            ) from exc

        dataframe.to_csv(output_path, index=False)
        logger.info("Saved %s bars to %s", pair, output_path)
        return dataframe

    def generate_sample_data(self, pair: str, start: datetime, end: datetime, timeframe: str, rows: int, output_path: Path) -> pd.DataFrame:
        pair = normalize_pair(pair)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        freq = pandas_freq_for_timeframe(timeframe)
        if start >= end:
            end = start + pd.Timedelta(minutes=max(rows, 2))
        index = pd.date_range(start=start, end=end, freq=freq, tz="UTC")
        if len(index) < rows:
            index = pd.date_range(end=end, periods=rows, freq=freq, tz="UTC")

        rng = np.random.default_rng(abs(hash((pair, timeframe))) % (2**32))
        base_price = 1.1 if pair.endswith("USD") else 140.0 if pair.endswith("JPY") else 0.85
        shocks = rng.normal(loc=0.00005, scale=0.002, size=len(index))
        close = base_price * np.exp(np.cumsum(shocks))
        open_ = np.concatenate([[close[0]], close[:-1]])
        high = np.maximum(open_, close) * (1 + rng.uniform(0.0001, 0.0015, size=len(index)))
        low = np.minimum(open_, close) * (1 - rng.uniform(0.0001, 0.0015, size=len(index)))
        volume = rng.integers(low=50, high=500, size=len(index))
        dataframe = pd.DataFrame(
            {
                "timestamp": index,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )
        dataframe.to_csv(output_path, index=False)
        logger.warning("Generated sample data for %s at %s because live historical data was not used.", pair, output_path)
        return dataframe

    def _download_with_dukascopy_python(self, pair: str, start: datetime, end: datetime, timeframe: str) -> pd.DataFrame:
        dukascopy_python, instruments_module = self._resolve_dukascopy_modules()
        instrument = self._resolve_instrument(pair, instruments_module)
        interval = getattr(dukascopy_python, dukascopy_interval_name(timeframe))
        offer_side = getattr(dukascopy_python, "OFFER_SIDE_BID")
        dataframe = dukascopy_python.fetch(instrument, interval, offer_side, start, end)
        return self._normalize_frame(dataframe)

    @staticmethod
    def _resolve_dukascopy_modules() -> tuple[Any, Any]:
        import dukascopy_python  # type: ignore
        from dukascopy_python import instruments as instruments_module  # type: ignore

        for logger_name in ("DUKASCRIPT", "dukascopy_python", "dukascopy"):
            lib_logger = logging.getLogger(logger_name)
            lib_logger.handlers.clear()
            lib_logger.propagate = False
            lib_logger.setLevel(logging.WARNING)
            lib_logger.addHandler(logging.NullHandler())

        return dukascopy_python, instruments_module

    @staticmethod
    def _resolve_instrument(pair: str, instruments_module: Any) -> Any:
        direct_patterns = [
            f"INSTRUMENT_FX_MAJORS_{pair[:3]}_{pair[3:]}",
            f"INSTRUMENT_FX_MINORS_{pair[:3]}_{pair[3:]}",
            f"INSTRUMENT_{pair[:3]}_{pair[3:]}",
        ]
        for candidate in direct_patterns:
            if hasattr(instruments_module, candidate):
                return getattr(instruments_module, candidate)

        suffix = f"_{pair[:3]}_{pair[3:]}"
        for attr_name in dir(instruments_module):
            if attr_name.endswith(suffix):
                return getattr(instruments_module, attr_name)
        raise ValueError(f"Could not resolve Dukascopy instrument for pair {pair}.")

    @staticmethod
    def _normalize_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
        frame = dataframe.copy()
        frame.columns = [str(column).lower() for column in frame.columns]
        if "timestamp" not in frame.columns:
            frame = frame.reset_index()
            frame.columns = [str(column).lower() for column in frame.columns]
            if "timestamp" not in frame.columns and "index" in frame.columns:
                frame = frame.rename(columns={"index": "timestamp"})

        rename_map = {
            "bid_open": "open",
            "bid_high": "high",
            "bid_low": "low",
            "bid_close": "close",
            "tick_volume": "volume",
        }
        frame = frame.rename(columns=rename_map)

        if "volume" not in frame.columns:
            frame["volume"] = 0.0

        normalized = frame[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True)
        normalized = normalized.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        return normalized.reset_index(drop=True)
