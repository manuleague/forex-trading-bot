from __future__ import annotations

import numpy as np
import pandas as pd

from forex_quant_bot.models import StrategyDecision
from forex_quant_bot.strategies.base_strategy import BaseStrategy
from forex_quant_bot.utils.math_utils import adx, atr_zscore, ema, macd, safe_div
from forex_quant_bot.utils.time_utils import TIMEFRAME_TO_MINUTES


class TrendStrategy(BaseStrategy):
    name = "trend"
    min_bars = 80

    def prepare_data(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        frame = dataframe.copy()
        frame["trend_ema_fast"] = ema(frame["close"], self.config.trend_fast_ema)
        frame["trend_ema_slow"] = ema(frame["close"], self.config.trend_slow_ema)
        frame["trend_adx"] = adx(frame, period=14)
        frame["trend_macd_hist"] = macd(frame["close"])["hist"]

        volatility = atr_zscore(frame, atr_period=14, lookback=100, min_periods=20)
        frame["trend_atr"] = volatility["atr"]
        frame["trend_volatility_zscore"] = volatility["zscore"]

        timeframe_minutes = self._infer_timeframe_minutes(frame)
        frame["mtf_4h_ema_50"] = np.nan
        frame["mtf_daily_ema_50"] = np.nan
        frame["mtf_weekly_ema_20"] = np.nan

        indexed = frame.set_index("timestamp").sort_index()
        if 0 < timeframe_minutes <= TIMEFRAME_TO_MINUTES["1h"]:
            h4_close = indexed["close"].resample("4h").last().dropna()
            h4_ema_50 = h4_close.ewm(span=50, adjust=False, min_periods=50).mean().shift(1)
            frame["mtf_4h_ema_50"] = h4_ema_50.reindex(indexed.index, method="ffill").to_numpy()
        elif timeframe_minutes == TIMEFRAME_TO_MINUTES["4h"]:
            daily_close = indexed["close"].resample("1D").last().dropna()
            daily_ema_50 = daily_close.ewm(span=50, adjust=False, min_periods=50).mean().shift(1)
            frame["mtf_daily_ema_50"] = daily_ema_50.reindex(indexed.index, method="ffill").to_numpy()
        elif timeframe_minutes == TIMEFRAME_TO_MINUTES["D"]:
            weekly_close = indexed["close"].resample("1W").last().dropna()
            weekly_ema_20 = weekly_close.ewm(span=20, adjust=False, min_periods=20).mean().shift(1)
            frame["mtf_weekly_ema_20"] = weekly_ema_20.reindex(indexed.index, method="ffill").to_numpy()
        return frame

    def evaluate(self, history: pd.DataFrame) -> StrategyDecision:
        required_bars = self.required_bars(history)
        if len(history) < required_bars:
            return self._hold("warmup", required_bars=required_bars, available_bars=len(history))
        if "trend_ema_fast" not in history.columns:
            history = self.prepare_data(history)

        latest = history.iloc[-1]
        current_regime = self._current_regime(history)
        if current_regime not in {"trend_up", "trend_down"}:
            return self._hold("regime_filter", regime=current_regime)

        latest_close = float(latest["close"])
        fast_value = float(latest["trend_ema_fast"])
        slow_value = float(latest["trend_ema_slow"])
        adx_value = float(latest["trend_adx"])
        macd_hist = float(latest["trend_macd_hist"])
        vol_zscore = float(latest.get("trend_volatility_zscore", np.nan))
        spread = safe_div(fast_value - slow_value, latest_close)

        signal = 0
        reason = "Trend Neutral"
        if current_regime == "trend_up" and fast_value > slow_value and adx_value >= self.config.trend_adx_threshold and macd_hist > 0:
            signal = 1
            reason = "Trend Bullish"
        elif current_regime == "trend_down" and fast_value < slow_value and adx_value >= self.config.trend_adx_threshold and macd_hist < 0:
            signal = -1
            reason = "Trend Bearish"

        timeframe_minutes = self._infer_timeframe_minutes(history)
        mtf_name = ""
        mtf_value = float("nan")
        if 0 < timeframe_minutes <= TIMEFRAME_TO_MINUTES["1h"]:
            mtf_name = "4h_ema_50"
            mtf_value = float(latest.get("mtf_4h_ema_50", np.nan))
        elif timeframe_minutes == TIMEFRAME_TO_MINUTES["4h"]:
            mtf_name = "daily_ema_50"
            mtf_value = float(latest.get("mtf_daily_ema_50", np.nan))
        elif timeframe_minutes == TIMEFRAME_TO_MINUTES["D"]:
            mtf_name = "weekly_ema_20"
            mtf_value = float(latest.get("mtf_weekly_ema_20", np.nan))

        if signal != 0 and mtf_name:
            if not np.isfinite(mtf_value):
                return self._hold("mtf_warmup", timeframe_minutes=timeframe_minutes, mtf_reference=mtf_name)
            if signal > 0 and latest_close <= mtf_value:
                return self._hold("mtf_filter", regime=current_regime, mtf_reference=mtf_name, mtf_value=mtf_value)
            if signal < 0 and latest_close >= mtf_value:
                return self._hold("mtf_filter", regime=current_regime, mtf_reference=mtf_name, mtf_value=mtf_value)

        confidence = 0.0
        if signal != 0:
            volatility_bonus = 0.05 if np.isfinite(vol_zscore) and vol_zscore > 0 else 0.0
            confidence = float(
                np.clip(
                    0.45
                    + min(abs(spread) * 250, 0.25)
                    + min(max(adx_value - self.config.trend_adx_threshold, 0.0) / 60, 0.20)
                    + min(abs(macd_hist) / latest_close * 150, 0.10)
                    + volatility_bonus,
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
                "adx": adx_value,
                "ema_fast": fast_value,
                "ema_slow": slow_value,
                "macd_hist": macd_hist,
                "volatility_zscore": vol_zscore,
                "mtf_reference": mtf_name,
                "mtf_value": mtf_value,
                "regime": current_regime,
            },
        )
