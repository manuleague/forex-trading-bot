from __future__ import annotations

import numpy as np
import pandas as pd

from forex_quant_bot.models import RegimeSnapshot
from forex_quant_bot.utils.math_utils import adx, atr, ema, rolling_hurst, zscore


class RegimeDetector:
    def __init__(self, trend_ema_period: int = 50, atr_period: int = 14) -> None:
        self.trend_ema_period = trend_ema_period
        self.atr_period = atr_period

    def annotate(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        frame = dataframe.copy()
        frame["atr"] = atr(frame, self.atr_period)
        frame["adx"] = adx(frame, 14)
        frame["hurst"] = rolling_hurst(frame["close"], 100)
        frame["volatility_zscore"] = zscore(frame["atr"], 50).fillna(0.0)
        frame["trend_ema"] = ema(frame["close"], self.trend_ema_period)

        conditions = [
            frame["volatility_zscore"] >= 2.0,
            (frame["adx"] >= 25.0) & (frame["hurst"] >= 0.55) & (frame["close"] >= frame["trend_ema"]),
            (frame["adx"] >= 25.0) & (frame["hurst"] >= 0.55) & (frame["close"] < frame["trend_ema"]),
        ]
        choices = ["volatility_spike", "trend_up", "trend_down"]
        frame["regime"] = np.select(conditions, choices, default="range")
        return frame

    @staticmethod
    def snapshot(bar: pd.Series) -> RegimeSnapshot:
        return RegimeSnapshot(
            label=str(bar.get("regime", "range")),
            adx=float(bar.get("adx", 0.0) or 0.0),
            atr=float(bar.get("atr", 0.0) or 0.0),
            hurst=float(bar.get("hurst", 0.5) or 0.5),
            volatility_zscore=float(bar.get("volatility_zscore", 0.0) or 0.0),
        )
