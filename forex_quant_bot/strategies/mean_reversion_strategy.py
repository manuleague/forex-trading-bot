from __future__ import annotations

import numpy as np
import pandas as pd

from forex_quant_bot.models import StrategyDecision
from forex_quant_bot.strategies.base_strategy import BaseStrategy
from forex_quant_bot.utils.math_utils import adx, atr_zscore, bollinger_bands, rsi, safe_div


class MeanReversionStrategy(BaseStrategy):
    name = "mean_reversion"
    min_bars = 50

    def prepare_data(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        frame = dataframe.copy()
        bands = bollinger_bands(frame["close"], self.config.mean_bollinger_period, self.config.mean_bollinger_std)
        frame["mean_bb_middle"] = bands["middle"]
        frame["mean_bb_upper"] = bands["upper"]
        frame["mean_bb_lower"] = bands["lower"]
        frame["mean_rsi"] = rsi(frame["close"], self.config.mean_rsi_period)
        frame["mean_adx"] = adx(frame, period=14)
        volatility = atr_zscore(frame, atr_period=14, lookback=100, min_periods=20)
        frame["mean_atr"] = volatility["atr"]
        frame["mean_volatility_zscore"] = volatility["zscore"]
        return frame

    def evaluate(self, history: pd.DataFrame) -> StrategyDecision:
        required_bars = self.required_bars(history)
        if len(history) < max(required_bars, 2):
            return self._hold("warmup", required_bars=max(required_bars, 2), available_bars=len(history))
        if "mean_bb_upper" not in history.columns:
            history = self.prepare_data(history)

        latest = history.iloc[-1]
        previous = history.iloc[-2]
        current_regime = self._current_regime(history)
        if current_regime != "range":
            return self._hold("regime_filter", regime=current_regime)

        adx_value = float(latest.get("mean_adx", np.nan))
        if not np.isfinite(adx_value) or adx_value >= 25.0:
            return self._hold("adx_filter", regime=current_regime, adx=adx_value)

        volatility_zscore = float(latest.get("mean_volatility_zscore", np.nan))
        if not np.isfinite(volatility_zscore) or volatility_zscore > 1.5:
            return self._hold("volatility_filter", regime=current_regime, volatility_zscore=volatility_zscore)

        latest_open = float(latest["open"])
        latest_high = float(latest["high"])
        latest_low = float(latest["low"])
        latest_close = float(latest["close"])
        upper = float(latest["mean_bb_upper"])
        lower = float(latest["mean_bb_lower"])
        middle = float(latest["mean_bb_middle"])
        rsi_value = float(latest["mean_rsi"])
        band_width = max(upper - lower, 1e-9)

        signal = 0
        reason = "Range Neutral"
        distance = 0.0
        if latest_close <= lower and rsi_value <= 35:
            signal = 1
            reason = "Range Reversion Long"
            distance = safe_div(lower - latest_close, band_width)
        elif latest_close >= upper and rsi_value >= 65:
            signal = -1
            reason = "Range Reversion Short"
            distance = safe_div(latest_close - upper, band_width)
        elif latest_close < middle and rsi_value < 25:
            signal = 1
            reason = "RSI Oversold Long"
            distance = safe_div(middle - latest_close, band_width)
        elif latest_close > middle and rsi_value > 75:
            signal = -1
            reason = "RSI Overbought Short"
            distance = safe_div(latest_close - middle, band_width)

        inside_bar = float(latest_high) <= float(previous["high"]) and float(latest_low) >= float(previous["low"])
        body = abs(latest_close - latest_open)
        total_range = max(latest_high - latest_low, 1e-9)
        lower_wick = min(latest_open, latest_close) - latest_low
        upper_wick = latest_high - max(latest_open, latest_close)
        long_rejection = latest_low <= lower and latest_close >= lower and lower_wick >= max(body, total_range * 0.25)
        short_rejection = latest_high >= upper and latest_close <= upper and upper_wick >= max(body, total_range * 0.25)

        quality_pass = False
        if signal > 0:
            quality_pass = inside_bar or long_rejection
        elif signal < 0:
            quality_pass = inside_bar or short_rejection

        if signal != 0 and not quality_pass:
            return self._hold(
                "candle_quality_filter",
                regime=current_regime,
                inside_bar=inside_bar,
                long_rejection=long_rejection,
                short_rejection=short_rejection,
            )

        confidence = 0.0
        if signal != 0:
            rsi_pressure = abs(50 - rsi_value) / 50
            calm_bonus = 0.05 if volatility_zscore < 0 else 0.0
            quality_bonus = 0.05 if inside_bar else 0.10
            confidence = float(
                np.clip(
                    0.40 + min(distance * 1.50, 0.30) + min(rsi_pressure * 0.35, 0.30) + calm_bonus + quality_bonus,
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
                "rsi": rsi_value,
                "adx": adx_value,
                "upper_band": upper,
                "lower_band": lower,
                "middle_band": middle,
                "volatility_zscore": volatility_zscore,
                "inside_bar": inside_bar,
                "long_rejection": long_rejection,
                "short_rejection": short_rejection,
                "regime": current_regime,
            },
        )
