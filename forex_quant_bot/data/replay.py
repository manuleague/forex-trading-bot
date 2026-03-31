from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd


@dataclass(slots=True)
class ReplayEvent:
    index: int
    timestamp: pd.Timestamp
    history: pd.DataFrame
    bar: pd.Series


@dataclass(slots=True)
class BarReplay:
    dataframe: pd.DataFrame
    warmup_bars: int = 200
    history_lookback: int = 512

    def __iter__(self) -> Iterator[ReplayEvent]:
        for index in range(self.warmup_bars, len(self.dataframe)):
            start = max(0, index - self.history_lookback + 1)
            history = self.dataframe.iloc[start : index + 1]
            bar = self.dataframe.iloc[index]
            yield ReplayEvent(index=index, timestamp=bar["timestamp"], history=history, bar=bar)
