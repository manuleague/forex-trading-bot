from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from forex_quant_bot.data.downloader import DukascopyDownloader
from forex_quant_bot.utils.time_utils import normalize_pair, split_pair


@dataclass(slots=True)
class DataCache:
    cache_dir: Path
    auto_download: bool = True
    sample_rows: int = 4_000
    downloader: DukascopyDownloader = field(init=False)

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.downloader = DukascopyDownloader(cache_dir=self.cache_dir)

    def get_history(
        self,
        pair: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        use_sample_data: bool = False,
    ) -> pd.DataFrame:
        pair = normalize_pair(pair)
        start = start.astimezone(timezone.utc)
        end = end.astimezone(timezone.utc)
        path = self.cache_path(pair, timeframe)

        if path.exists():
            dataframe = self._load_csv(path)
            coverage_ok = dataframe["timestamp"].min() <= start and dataframe["timestamp"].max() >= end
            if coverage_ok:
                return self._slice(dataframe, start, end)
            if not self.auto_download and not use_sample_data:
                raise ValueError(f"Cached data for {pair} {timeframe} does not fully cover {start} to {end}.")

        if not self.auto_download and not path.exists() and not use_sample_data:
            raise FileNotFoundError(f"No cached data found for {pair} {timeframe} at {path}.")

        if use_sample_data:
            dataframe = self.downloader.generate_sample_data(pair, start, end, timeframe, self.sample_rows, path)
        else:
            dataframe = self.downloader.download(pair, start, end, timeframe, path)
        return self._slice(dataframe, start, end)

    def get_quote_to_usd_series(
        self,
        pair: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        index: pd.Index,
        use_sample_data: bool = False,
    ) -> pd.Series:
        target_index = pd.DatetimeIndex(pd.to_datetime(index, utc=True))
        base, quote = split_pair(pair)
        if quote == "USD":
            return pd.Series(1.0, index=target_index, name="quote_to_usd")
        if base == "USD":
            pair_history = self.get_history(pair, timeframe, start, end, use_sample_data=use_sample_data)
            pair_series = self._align_rate_series(pair_history.set_index("timestamp")["close"], target_index, pair)
            converted = (1.0 / pair_series.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).ffill().bfill()
            if converted.isna().any():
                missing = int(converted.isna().sum())
                raise ValueError(f"Unable to derive quote_to_usd for {pair}; {missing} conversion values remain missing.")
            return converted.rename("quote_to_usd")

        direct_pair = f"{quote}USD"
        inverse_pair = f"USD{quote}"

        try:
            direct = self.get_history(direct_pair, timeframe, start, end, use_sample_data=use_sample_data)
            direct_series = self._align_rate_series(direct.set_index("timestamp")["close"], target_index, direct_pair)
            return direct_series.rename("quote_to_usd")
        except Exception:
            inverse = self.get_history(inverse_pair, timeframe, start, end, use_sample_data=use_sample_data)
            inverse_series = self._align_rate_series(inverse.set_index("timestamp")["close"], target_index, inverse_pair)
            converted = (1.0 / inverse_series.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).ffill().bfill()
            if converted.isna().any():
                missing = int(converted.isna().sum())
                raise ValueError(f"Unable to derive quote_to_usd for {pair}; {missing} conversion values remain missing.")
            return converted.rename("quote_to_usd")

    def cache_path(self, pair: str, timeframe: str) -> Path:
        return self.cache_dir / f"{pair}_{timeframe}.csv"

    @staticmethod
    def _load_csv(path: Path) -> pd.DataFrame:
        dataframe = pd.read_csv(path)
        dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], utc=True)
        dataframe = dataframe.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
        for column in ["open", "high", "low", "close", "volume"]:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")
        return dataframe.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

    @staticmethod
    def _slice(dataframe: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
        filtered = dataframe[(dataframe["timestamp"] >= start) & (dataframe["timestamp"] <= end)].copy()
        if filtered.empty:
            raise ValueError("Requested time range produced an empty historical dataset.")
        return filtered.reset_index(drop=True)

    @staticmethod
    def _align_rate_series(series: pd.Series, target_index: pd.DatetimeIndex, pair: str) -> pd.Series:
        aligned = pd.to_numeric(series, errors="coerce").sort_index().reindex(target_index).ffill().bfill()
        if aligned.isna().any():
            aligned = aligned.interpolate(method="time", limit_direction="both")
        if aligned.isna().any():
            fallback = pd.to_numeric(series, errors="coerce").dropna()
            if not fallback.empty:
                aligned = aligned.fillna(float(fallback.iloc[-1]))
        if aligned.isna().any():
            missing = int(aligned.isna().sum())
            raise ValueError(f"Unable to align conversion series for {pair}; {missing} quote_to_usd values are missing.")
        return aligned.astype(float)
