from __future__ import annotations

import numpy as np
import pandas as pd

from forex_quant_bot.models import StrategyDecision
from forex_quant_bot.strategies.base_strategy import BaseStrategy
from forex_quant_bot.utils.math_utils import atr_zscore, donchian_channels, safe_div


class BreakoutStrategy(BaseStrategy):
    name = "breakout"
    min_bars = 60

    def prepare_data(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        frame = dataframe.copy()
        channel = donchian_channels(frame, self.config.breakout_lookback)
        volatility = atr_zscore(frame, atr_period=self.config.breakout_atr_period, lookback=100, min_periods=20)
        frame["breakout_channel_upper"] = channel["upper"]
        frame["breakout_channel_lower"] = channel["lower"]
        frame["breakout_prev_upper"] = channel["upper"].shift(1)
        frame["breakout_prev_lower"] = channel["lower"].shift(1)
        frame["breakout_atr"] = volatility["atr"]
        frame["breakout_atr_mean"] = volatility["atr"].rolling(20, min_periods=5).mean()
        frame["breakout_volatility_zscore"] = volatility["zscore"]
        return frame

    def evaluate(self, history: pd.DataFrame) -> StrategyDecision:
        required_bars = self.required_bars(history)
        if len(history) < required_bars:
            return self._hold("warmup", required_bars=required_bars, available_bars=len(history))
        if "breakout_prev_upper" not in history.columns:
            history = self.prepare_data(history)

        latest = history.iloc[-1]
        current_regime = self._current_regime(history)
        if current_regime not in {"trend_up", "trend_down", "volatility_spike"}:
            return self._hold("regime_filter", regime=current_regime)

        volatility_zscore = float(latest.get("breakout_volatility_zscore", np.nan))
        if not np.isfinite(volatility_zscore) or volatility_zscore <= 1.0:
            return self._hold("volatility_filter", regime=current_regime, volatility_zscore=volatility_zscore)

        latest_close = float(latest["close"])
        previous_upper = float(latest["breakout_prev_upper"])
        previous_lower = float(latest["breakout_prev_lower"])
        atr_value = float(latest["breakout_atr"])
        atr_mean = float(latest["breakout_atr_mean"])
        atr_ratio = safe_div(atr_value, atr_mean, default=1.0)

        signal = 0
        reason = "Breakout Neutral"
        breakout_distance = 0.0
        allow_long = current_regime in {"trend_up", "volatility_spike"}
        allow_short = current_regime in {"trend_down", "volatility_spike"}
        if allow_long and latest_close > previous_upper and atr_ratio >= self.config.breakout_atr_expansion:
            signal = 1
            reason = "Channel Breakout Long"
            breakout_distance = safe_div(latest_close - previous_upper, atr_value)
        elif allow_short and latest_close < previous_lower and atr_ratio >= self.config.breakout_atr_expansion:
            signal = -1
            reason = "Channel Breakout Short"
            breakout_distance = safe_div(previous_lower - latest_close, atr_value)

        confidence = 0.0
        if signal != 0:
            confidence = float(
                np.clip(
                    0.45 + min((atr_ratio - 1.0) * 0.35, 0.25) + min(breakout_distance * 0.10, 0.20) + min((volatility_zscore - 1.0) * 0.08, 0.10),
                    0.0,
                    1.0,
                )
            )

        return StrategyDecision(
            name=self.name,
            signal=signal,
            confidence=confidence,
            metadata={
                "reason": reason,
                "atr": atr_value,
                "atr_ratio": atr_ratio,
                "upper_channel": previous_upper,
                "lower_channel": previous_lower,
                "volatility_zscore": volatility_zscore,
                "regime": current_regime,
            },
        )
