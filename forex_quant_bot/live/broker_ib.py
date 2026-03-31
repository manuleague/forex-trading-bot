from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from forex_quant_bot.settings import IBConfig
from forex_quant_bot.utils.time_utils import ib_bar_size_for_timeframe, normalize_pair, split_pair, to_ib_duration

try:  # pragma: no cover - runtime dependency
    from ib_insync import Forex, IB, MarketOrder, util
except ImportError:  # pragma: no cover - optional alias
    from ib_async import Forex, IB, MarketOrder, util  # type: ignore


@dataclass(slots=True)
class IBPaperBroker:
    config: IBConfig
    ib: Any = field(init=False)
    active_historical_streams: list[Any] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.ib = IB()

    async def connect(self) -> None:
        if self.config.paper_only and self.config.port not in self.config.allowed_paper_ports:
            raise ValueError(
                f"Refusing to connect to port {self.config.port}. Allowed paper ports are {self.config.allowed_paper_ports}."
            )
        if self.ib.isConnected():
            return
        await self.ib.connectAsync(
            host=self.config.host,
            port=self.config.port,
            clientId=self.config.client_id,
            account=self.config.account,
        )
        if not self.ib.isConnected():
            raise ConnectionError("Failed to connect to Interactive Brokers.")

    async def disconnect(self) -> None:
        await self._cancel_historical_streams()
        if self.ib.isConnected():
            self.ib.disconnect()

    async def _cancel_historical_streams(self) -> None:
        if not self.active_historical_streams:
            return
        streams = list(self.active_historical_streams)
        self.active_historical_streams.clear()
        if not self.ib.isConnected():
            return
        for stream in streams:
            try:
                self.ib.cancelHistoricalData(stream)
            except Exception:
                continue
        await asyncio.sleep(0)

    @staticmethod
    def _contract_symbol(pair: str) -> str:
        pair = normalize_pair(pair)
        base, quote = split_pair(pair)
        if base == "JPY" and quote == "USD":
            return "USDJPY"
        return f"{base}{quote}"

    async def qualify_forex(self, pair: str):
        contract = Forex(self._contract_symbol(pair))
        await self.ib.qualifyContractsAsync(contract)
        return contract

    async def request_bars(
        self,
        pair: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        keep_up_to_date: bool = False,
    ):
        contract = await self.qualify_forex(pair)
        duration = to_ib_duration(start, end)
        bars = await self.ib.reqHistoricalDataAsync(
            contract,
            endDateTime="" if keep_up_to_date else end.astimezone(timezone.utc),
            durationStr=duration,
            barSizeSetting=ib_bar_size_for_timeframe(timeframe),
            whatToShow="MIDPOINT",
            useRTH=False,
            formatDate=2,
            keepUpToDate=keep_up_to_date,
        )
        if keep_up_to_date and bars is not None:
            self.active_historical_streams.append(bars)
        return bars

    async def place_market_order(self, pair: str, action: str, size_units: int, timeout_seconds: float = 5.0):
        contract = await self.qualify_forex(pair)
        order = MarketOrder(action=action, totalQuantity=size_units)
        order.tif = "DAY"
        trade = self.ib.placeOrder(contract, order)

        pending_statuses = {"", "PendingSubmit", "ApiPending", "PreSubmitted", "Submitted"}
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        while self.ib.isConnected() and str(getattr(trade.orderStatus, "status", "")) in pending_statuses and loop.time() < deadline:
            await self.ib.sleep(0.25)
        return trade

    @staticmethod
    def bars_to_dataframe(bars: Any) -> pd.DataFrame:
        empty = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        if bars is None:
            return empty
        try:
            dataframe = util.df(bars)
        except Exception:
            return empty
        if dataframe is None or not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
            return empty
        dataframe = dataframe.rename(columns={"date": "timestamp"})
        dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], utc=True, errors="coerce")
        dataframe = dataframe.dropna(subset=["timestamp", "open", "high", "low", "close"])
        if dataframe.empty:
            return empty
        if "volume" not in dataframe.columns:
            dataframe["volume"] = 0.0
        return dataframe[["timestamp", "open", "high", "low", "close", "volume"]].copy()
