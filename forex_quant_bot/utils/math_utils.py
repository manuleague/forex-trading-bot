from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd



def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0 or np.isnan(denominator):
        return default
    return float(numerator / denominator)



def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()



def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()



def stddev(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).std(ddof=0)



def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    return result.fillna(50.0)



def true_range(df: pd.DataFrame) -> pd.Series:
    previous_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)



def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()



def atr_zscore(df: pd.DataFrame, atr_period: int = 14, lookback: int = 100, min_periods: int = 20) -> pd.DataFrame:
    atr_series = atr(df, atr_period)
    atr_mean = atr_series.rolling(lookback, min_periods=min_periods).mean()
    atr_std = atr_series.rolling(lookback, min_periods=min_periods).std(ddof=0)
    zscore = (atr_series - atr_mean) / atr_std.replace(0, np.nan)
    zscore = zscore.replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame({"atr": atr_series, "zscore": zscore})



def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(df)
    atr_series = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_series
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_series
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean().fillna(0.0)



def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist})



def bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    middle = sma(close, period)
    band_std = stddev(close, period)
    upper = middle + num_std * band_std
    lower = middle - num_std * band_std
    return pd.DataFrame({"middle": middle, "upper": upper, "lower": lower})



def donchian_channels(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    upper = df["high"].rolling(lookback, min_periods=lookback).max()
    lower = df["low"].rolling(lookback, min_periods=lookback).min()
    midpoint = (upper + lower) / 2.0
    return pd.DataFrame({"upper": upper, "lower": lower, "midpoint": midpoint})



def zscore(series: pd.Series, period: int = 50) -> pd.Series:
    mean = series.rolling(period, min_periods=period).mean()
    std = series.rolling(period, min_periods=period).std(ddof=0)
    return (series - mean) / std.replace(0, np.nan)



def hurst_exponent(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 32:
        return 0.5
    log_prices = np.log(np.clip(array, 1e-12, None))
    lags = range(2, min(20, log_prices.size // 2))
    tau = []
    valid_lags = []
    for lag in lags:
        diff = np.subtract(log_prices[lag:], log_prices[:-lag])
        if diff.size == 0:
            continue
        std = np.std(diff)
        if std <= 0 or not np.isfinite(std):
            continue
        tau.append(np.sqrt(std))
        valid_lags.append(lag)
    if len(valid_lags) < 2:
        return 0.5
    slope, _ = np.polyfit(np.log(valid_lags), np.log(tau), 1)
    return float(np.clip(2.0 * slope, 0.0, 1.0))



def rolling_hurst(close: pd.Series, window: int = 100, min_periods: int | None = None) -> pd.Series:
    effective_min_periods = min_periods if min_periods is not None else min(window, 32)
    return close.rolling(window, min_periods=effective_min_periods).apply(hurst_exponent, raw=False).fillna(0.5)



def safe_drawdown_pct(equity_curve: pd.Series, peaks: pd.Series | None = None) -> pd.Series:
    peaks = peaks if peaks is not None else equity_curve.cummax()
    return (equity_curve / peaks.replace(0, np.nan)) - 1.0



def max_drawdown(equity_curve: pd.Series) -> tuple[float, float]:
    if equity_curve.empty:
        return 0.0, 0.0
    peaks = equity_curve.cummax()
    drawdowns = equity_curve - peaks
    pct_drawdowns = safe_drawdown_pct(equity_curve, peaks)
    return float(drawdowns.min()), float(pct_drawdowns.min())



def profit_factor(pnl_values: Iterable[float]) -> float:
    pnl_array = np.asarray(list(pnl_values), dtype=float)
    gross_profit = pnl_array[pnl_array > 0].sum()
    gross_loss = -pnl_array[pnl_array < 0].sum()
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)



def sharpe_ratio(returns: pd.Series, periods_per_year: float) -> float:
    if returns.empty or returns.std(ddof=0) == 0:
        return 0.0
    return float((returns.mean() / returns.std(ddof=0)) * np.sqrt(periods_per_year))



def sortino_ratio(returns: pd.Series, periods_per_year: float) -> float:
    if returns.empty:
        return 0.0
    downside = returns[returns < 0]
    if downside.empty or downside.std(ddof=0) == 0:
        return 0.0
    return float((returns.mean() / downside.std(ddof=0)) * np.sqrt(periods_per_year))



def normalize_scores(values: dict[str, float]) -> dict[str, float]:
    clipped = {key: max(0.0, float(value)) for key, value in values.items()}
    total = sum(clipped.values())
    if total <= 0:
        default = 1.0 / max(len(clipped), 1)
        return {key: default for key in clipped}
    return {key: value / total for key, value in clipped.items()}



def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"
