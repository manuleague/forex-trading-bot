from __future__ import annotations

from datetime import datetime, timezone


TIMEFRAME_TO_PANDAS_FREQ: dict[str, str] = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "D": "1D",
    "W": "1W",
}

TIMEFRAME_TO_IB_BAR_SIZE: dict[str, str] = {
    "1m": "1 min",
    "5m": "5 mins",
    "15m": "15 mins",
    "30m": "30 mins",
    "1h": "1 hour",
    "4h": "4 hours",
    "D": "1 day",
    "W": "1 week",
}

TIMEFRAME_TO_DUKASCOPY_INTERVAL: dict[str, str] = {
    "1m": "INTERVAL_MIN_1",
    "5m": "INTERVAL_MIN_5",
    "15m": "INTERVAL_MIN_15",
    "30m": "INTERVAL_MIN_30",
    "1h": "INTERVAL_HOUR_1",
    "4h": "INTERVAL_HOUR_4",
    "D": "INTERVAL_DAY_1",
    "W": "INTERVAL_WEEK_1",
}

TIMEFRAME_TO_MINUTES: dict[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "D": 1_440,
    "W": 10_080,
}


def normalize_pair(pair: str) -> str:
    pair = pair.replace("/", "").replace(".", "").replace("_", "").upper().strip()
    if len(pair) != 6:
        raise ValueError(f"Forex pair must look like EURUSD, got {pair!r}.")
    return pair



def split_pair(pair: str) -> tuple[str, str]:
    normalized = normalize_pair(pair)
    return normalized[:3], normalized[3:]



def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)



def to_ib_duration(start: datetime, end: datetime) -> str:
    delta = end - start
    days = max(delta.days, 1)
    if days <= 30:
        return f"{days} D"
    if days <= 365:
        months = max(days // 30, 1)
        return f"{months} M"
    years = max(days // 365, 1)
    return f"{years} Y"



def pandas_freq_for_timeframe(timeframe: str) -> str:
    try:
        return TIMEFRAME_TO_PANDAS_FREQ[timeframe]
    except KeyError as exc:
        raise ValueError(f"Unsupported timeframe {timeframe!r}.") from exc



def ib_bar_size_for_timeframe(timeframe: str) -> str:
    try:
        return TIMEFRAME_TO_IB_BAR_SIZE[timeframe]
    except KeyError as exc:
        raise ValueError(f"Unsupported timeframe {timeframe!r}.") from exc



def dukascopy_interval_name(timeframe: str) -> str:
    try:
        return TIMEFRAME_TO_DUKASCOPY_INTERVAL[timeframe]
    except KeyError as exc:
        raise ValueError(f"Unsupported timeframe {timeframe!r}.") from exc



def annualization_factor(timeframe: str) -> float:
    minutes = TIMEFRAME_TO_MINUTES[timeframe]
    return (365 * 24 * 60) / minutes
