from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from forex_quant_bot.models import StrategyDecision
from forex_quant_bot.settings import StrategyConfig
from forex_quant_bot.utils.time_utils import TIMEFRAME_TO_MINUTES


class BaseStrategy(ABC):
    name: str
    min_bars: int = 50

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    def prepare_data(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        return dataframe

    def required_bars(self, history: pd.DataFrame) -> int:
        timeframe_minutes = self._infer_timeframe_minutes(history)
        scale = 1.0
        if timeframe_minutes >= TIMEFRAME_TO_MINUTES["W"]:
            scale = self.config.weekly_warmup_scale
        elif timeframe_minutes >= TIMEFRAME_TO_MINUTES["D"]:
            scale = self.config.daily_warmup_scale
        scaled = int(round(self.min_bars * scale))
        return max(self.config.adaptive_warmup_floor, scaled)

    @staticmethod
    def _infer_timeframe_minutes(history: pd.DataFrame) -> int:
        if "timestamp" not in history.columns or len(history) < 2:
            return 0
        timestamps = pd.to_datetime(history["timestamp"], utc=True)
        deltas = timestamps.diff().dropna()
        if deltas.empty:
            return 0
        median_minutes = deltas.dt.total_seconds().median() / 60.0
        return int(max(median_minutes, 0))

    @staticmethod
    def _current_regime(history: pd.DataFrame) -> str:
        if history.empty:
            return "range"
        return str(history.iloc[-1].get("regime", "range"))

    def _hold(self, reason: str, **metadata) -> StrategyDecision:
        payload = {"reason": reason}
        payload.update(metadata)
        return StrategyDecision(name=self.name, signal=0, confidence=0.0, metadata=payload)

    @abstractmethod
    def evaluate(self, history: pd.DataFrame) -> StrategyDecision:
        raise NotImplementedError
