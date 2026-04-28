from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from forex_quant_bot.backtest.metrics import compile_backtest_report
from forex_quant_bot.core.allocator import StrategyAllocator
from forex_quant_bot.core.performance_tracker import PerformanceTracker
from forex_quant_bot.core.regime_detector import RegimeDetector
from forex_quant_bot.core.risk_overlay import RiskOverlay
from forex_quant_bot.live.broker_ib import IBPaperBroker
from forex_quant_bot.logs.csv_logger import CSVLogger
from forex_quant_bot.logs.summary_printer import SummaryPrinter
from forex_quant_bot.models import OrderEvent, Position, TradeEventRecord, TradeRecord
from forex_quant_bot.settings import BotConfig, RiskConfig, StrategyConfig
from forex_quant_bot.strategies import build_default_strategies
from forex_quant_bot.utils.time_utils import TIMEFRAME_TO_MINUTES, split_pair


@dataclass(slots=True)
class PairState:
    pair: str
    market_data: pd.DataFrame
    last_bar_timestamp: pd.Timestamp | None = None
    live_bar_timestamp: pd.Timestamp | None = None
    live_price: float = 0.0
    position: Position | None = None


@dataclass(slots=True)
class LiveRunner:
    ROMANIA_TZ = ZoneInfo("Europe/Bucharest")
    config: BotConfig
    broker: IBPaperBroker | None = None
    logger: CSVLogger | None = None
    regime_detector: RegimeDetector = field(default_factory=RegimeDetector)
    performance_tracker: PerformanceTracker | None = None
    allocator: StrategyAllocator | None = None
    risk_overlay: RiskOverlay | None = None
    strategies: list[Any] = field(default_factory=list)
    states: dict[str, PairState] = field(default_factory=dict)
    trades: list[TradeRecord] = field(default_factory=list)
    trade_events: list[TradeEventRecord] = field(default_factory=list)
    order_events: list[OrderEvent] = field(default_factory=list)
    risk_rows: list[dict[str, Any]] = field(default_factory=list)
    equity_rows: list[dict[str, Any]] = field(default_factory=list)
    realized_pnl: float = 0.0
    trade_id: int = 0
    last_conversion_rates: dict[str, float] = field(default_factory=dict)
    bar_updates_seen: dict[str, int] = field(default_factory=dict)
    primary_bar_streams: dict[str, Any] = field(default_factory=dict)
    conversion_bar_streams: dict[str, Any] = field(default_factory=dict)
    market_tickers: dict[str, Any] = field(default_factory=dict)
    stream_tasks: list[asyncio.Task[Any]] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    reconnect_event: asyncio.Event = field(default_factory=asyncio.Event)
    reconnect_cooldown_seconds: float = 10.0
    stream_poll_interval_seconds: float = 0.50
    market_price_poll_interval_seconds: float = 1.0
    heartbeat_interval_seconds: float = 5.0
    next_heartbeat_deadline: float = 0.0
    heartbeat_timeout_seconds: float = 4.0
    reconnect_reason: str = ""
    ib_error_handler_attached: bool = False
    ib_disconnect_handler_attached: bool = False
    ib_timeout_handler_attached: bool = False
    connection_loss_active: bool = False
    last_live_update_at: float = 0.0
    live_data_stale_after_seconds: float = 20.0
    terminal_status_line: str = ""
    terminal_status_by_pair: dict[str, dict[str, Any]] = field(default_factory=dict)

    async def run(self) -> dict[str, Any]:
        self._initialize_runtime()
        report: dict[str, Any] = {}
        should_shutdown = False

        while not should_shutdown:
            try:
                await self._connect_and_bootstrap()
                await self._connected_loop()
            except KeyboardInterrupt:
                self._emit_status("Live run interrupted. Writing any collected artifacts before disconnecting...")
                should_shutdown = True
            except asyncio.CancelledError:
                self._emit_status("Live run cancelled. Writing any collected artifacts before disconnecting...")
                should_shutdown = True
            except (ConnectionError, RuntimeError) as exc:
                self._emit_status(
                    f"[WARNING] {exc} Reconnecting in {int(self.reconnect_cooldown_seconds)} seconds..."
                )
            finally:
                await self._cancel_stream_tasks()
                if self.broker is not None:
                    await self.broker.disconnect()

            if not should_shutdown:
                await asyncio.sleep(self.reconnect_cooldown_seconds)

        built = self._build_report()
        if built:
            report = built
            if self.logger is not None:
                summary_text = SummaryPrinter.render(report)
                self.logger.persist_report(report, summary_text)
                report["summary_text"] = summary_text
                report["output_dir"] = self.logger.run_dir
                self._emit_status(f"Live artifacts saved to: {self.logger.run_dir}")
        return report

    def _initialize_runtime(self) -> None:
        if self.broker is None:
            self.broker = IBPaperBroker(self.config.ib)
        if self.logger is None:
            self.logger = CSVLogger(self.config.logging.output_dir, self.config.mode, self.config.primary_pair, self.config.timeframe)
        if not self.strategies:
            self.strategies = build_default_strategies(self.config.strategy)
        if self.performance_tracker is None:
            self.performance_tracker = PerformanceTracker(window=self.config.strategy.recent_trade_window)
        if self.allocator is None:
            self.allocator = StrategyAllocator(
                score_threshold=self.config.strategy.score_threshold,
                min_strategy_confidence=self.config.strategy.min_strategy_confidence,
            )
        if self.risk_overlay is None:
            self.risk_overlay = RiskOverlay(self.config.risk)

    async def _connect_and_bootstrap(self) -> None:
        if self.broker is None:
            raise RuntimeError("Broker is not initialized.")
        self.reconnect_event.clear()
        self.reconnect_reason = ""
        self._clear_terminal_status_line()
        self.terminal_status_by_pair.clear()
        self.terminal_status_line = ""
        self.primary_bar_streams.clear()
        self.conversion_bar_streams.clear()
        self.market_tickers.clear()
        self._emit_status(
            f"Connecting to IB paper at {self.config.ib.host}:{self.config.ib.port} "
            f"(client_id={self.config.ib.client_id})..."
        )
        await self.broker.connect()
        if not self.ib_error_handler_attached:
            self.broker.ib.errorEvent += self._on_ib_error
            self.ib_error_handler_attached = True
        if not self.ib_disconnect_handler_attached:
            self.broker.ib.disconnectedEvent += self._on_ib_disconnected
            self.ib_disconnect_handler_attached = True
        if not self.ib_timeout_handler_attached:
            self.broker.ib.timeoutEvent += self._on_ib_timeout
            self.ib_timeout_handler_attached = True
        if hasattr(self.broker.ib, "setTimeout"):
            self.broker.ib.setTimeout(self.live_data_stale_after_seconds)
        self._emit_status("Connected. Bootstrapping live forex streams...")
        await self._bootstrap_streams()
        loaded_pairs = [pair for pair in self.config.pairs if pair in self.states]
        latest_points = ", ".join(
            f"{pair} through {self._format_timestamp(self.states[pair].last_bar_timestamp)}" for pair in loaded_pairs
        )
        self._emit_status(
            f"Live paper mode active for {', '.join(loaded_pairs)} on {self.config.timeframe}. "
            f"Warm-up bars loaded: {latest_points}. Waiting for the next completed bar..."
        )
        self.last_live_update_at = asyncio.get_running_loop().time()
        self._set_next_heartbeat()
        if self.connection_loss_active:
            self._emit_status("[SUCCESS] Live streams resumed after reconnect.")
            self.connection_loss_active = False

    async def _connected_loop(self) -> None:
        while True:
            if self.reconnect_event.is_set():
                reason = self.reconnect_reason or "Reconnect requested."
                self.reconnect_event.clear()
                raise ConnectionError(reason)
            if self.broker is None or not self.broker.ib.isConnected():
                raise ConnectionError("IB connection is inactive.")
            await self._maybe_emit_heartbeat()
            await asyncio.sleep(1)

    def _request_reconnect(self, reason: str) -> None:
        if not self.reconnect_event.is_set():
            self.reconnect_reason = reason
            self.reconnect_event.set()
        elif not self.reconnect_reason:
            self.reconnect_reason = reason

    async def _cancel_stream_tasks(self) -> None:
        if not self.stream_tasks:
            return
        tasks = [task for task in self.stream_tasks if task is not None]
        self.stream_tasks = []
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _on_ib_error(self, req_id: int, error_code: int, error_string: str, contract: Any) -> None:
        if error_code in {1100, 10182}:
            if not self.connection_loss_active:
                self._emit_status(
                    f"[WARNING] IB connection issue ({error_code}): {error_string}. Scheduling reconnect."
                )
            self.connection_loss_active = True
            self._request_reconnect(f"IB connection issue {error_code}: {error_string}")
        elif error_code == 1102 and self.connection_loss_active:
            self._emit_status(f"[SUCCESS] IB connectivity restored (1102): {error_string}")

    def _on_ib_disconnected(self) -> None:
        if not self.connection_loss_active:
            self._emit_status("[WARNING] Interactive Brokers session disconnected. Scheduling reconnect.")
        self.connection_loss_active = True
        self._request_reconnect("Interactive Brokers disconnected.")

    def _on_ib_timeout(self, idle_period: float) -> None:
        if not self.connection_loss_active:
            self._emit_status(
                f"[WARNING] No incoming IB data for {idle_period:.1f}s. Scheduling reconnect."
            )
        self.connection_loss_active = True
        self._request_reconnect(f"IB data timeout after {idle_period:.1f}s.")

    def _set_next_heartbeat(self) -> None:
        self.next_heartbeat_deadline = asyncio.get_running_loop().time() + self.heartbeat_interval_seconds

    async def _maybe_emit_heartbeat(self) -> None:
        now = asyncio.get_running_loop().time()
        if self.next_heartbeat_deadline == 0.0:
            self._set_next_heartbeat()
            return
        if now < self.next_heartbeat_deadline:
            return
        self.next_heartbeat_deadline = now + self.heartbeat_interval_seconds
        if self.broker is None:
            raise ConnectionError("Broker is not initialized.")
        try:
            await self.broker.ping(timeout_seconds=self.heartbeat_timeout_seconds)
        except Exception as exc:
            self.connection_loss_active = True
            self._request_reconnect(f"IB heartbeat failed: {exc}")
            raise ConnectionError(f"IB heartbeat failed: {exc}") from exc
        now = asyncio.get_running_loop().time()
        if self.config.pairs and self.last_live_update_at and (now - self.last_live_update_at) > self.live_data_stale_after_seconds:
            self.connection_loss_active = True
            reason = (
                f"No live market updates received for {now - self.last_live_update_at:.1f}s."
            )
            self._request_reconnect(reason)
            raise ConnectionError(reason)

    @staticmethod
    def _looks_like_connection_issue(exc: Exception) -> bool:
        message = str(exc).lower()
        return isinstance(exc, ConnectionError) or any(
            token in message for token in ("not connected", "disconnected", "disconnect", "socket", "connection")
        )

    async def _bootstrap_streams(self) -> None:
        if self.broker is None:
            raise RuntimeError("Broker is not initialized.")

        end = datetime.now(timezone.utc)
        pairs_to_stream = set(self.config.pairs)
        for pair in self.config.pairs:
            _, quote = split_pair(pair)
            pairs_to_stream.update(self._conversion_pairs_for_quote(quote))

        previous_timestamps = {
            pair: pd.Timestamp(state.last_bar_timestamp)
            for pair, state in self.states.items()
            if state.last_bar_timestamp is not None
        }
        updated_states = dict(self.states)
        refreshed_conversion_rates = dict(self.last_conversion_rates)
        primary_streams: dict[str, Any] = {}
        conversion_streams: dict[str, Any] = {}
        market_tickers: dict[str, Any] = {}
        stream_errors: dict[str, str] = {}
        successful_primary: set[str] = set()

        for pair in sorted(pairs_to_stream):
            start = self._history_start(pair, end)
            try:
                bars = await self.broker.request_bars(pair, self.config.timeframe, start, end, keep_up_to_date=True)
            except Exception as exc:
                stream_errors[pair] = str(exc)
                continue

            incoming = self.broker.bars_to_dataframe(bars)
            if incoming.empty:
                stream_errors[pair] = "No historical bars returned from IB."
                continue

            apply_strategy_features = pair in self.config.pairs
            merged = self._merge_market_data(pair, incoming, apply_strategy_features=apply_strategy_features)
            if merged.empty:
                stream_errors[pair] = "Merged market data is empty."
                continue

            live_bar_timestamp = merged.iloc[-1]["timestamp"]
            live_price = float(merged.iloc[-1]["close"])
            last_closed_timestamp = live_bar_timestamp
            if apply_strategy_features and len(merged) >= 2:
                last_closed_timestamp = merged.iloc[-2]["timestamp"]

            existing_state = self.states.get(pair)
            updated_states[pair] = PairState(
                pair=pair,
                market_data=merged,
                last_bar_timestamp=last_closed_timestamp,
                live_bar_timestamp=live_bar_timestamp,
                live_price=live_price,
                position=existing_state.position if existing_state is not None else None,
            )
            if pair in self.config.pairs:
                successful_primary.add(pair)
                primary_streams[pair] = bars
                self._emit_status(self._stream_status_message(pair, merged, previous_timestamps.get(pair)))
                self._update_terminal_status_snapshot(
                    pair=pair,
                    price=live_price,
                    market_timestamp=datetime.now(timezone.utc),
                    bar_timestamp=live_bar_timestamp,
                )
            else:
                conversion_streams[pair] = bars
                refreshed_conversion_rates[pair] = live_price

            if hasattr(bars, "updateEvent"):
                bars.updateEvent += lambda bar_list, has_new_bar, p=pair: self._on_live_bar_tick_update(p, bar_list, has_new_bar)

        missing_primary = [pair for pair in self.config.pairs if pair not in successful_primary]
        if missing_primary:
            details = "; ".join(f"{pair}: {stream_errors.get(pair, 'No market data received from IB.')}" for pair in missing_primary)
            raise RuntimeError(
                "Unable to initialize live data for the requested pair(s). "
                f"Missing: {', '.join(missing_primary)}. Details: {details}. "
                "Check that TWS/Gateway paper is running, API access is enabled, and forex data permissions are available."
            )

        for pair in self.config.pairs:
            try:
                ticker = await self.broker.subscribe_tick_by_tick_midpoint(pair)
            except Exception as exc:
                self._emit_status(f"[WARNING] Tick-by-tick midpoint stream unavailable for {pair}: {exc}")
                try:
                    ticker = await self.broker.subscribe_market_data(pair)
                except Exception as fallback_exc:
                    self._emit_status(f"[WARNING] Real-time price stream unavailable for {pair}: {fallback_exc}")
                    continue
            market_tickers[pair] = ticker
            if hasattr(ticker, "updateEvent"):
                ticker.updateEvent += lambda *args, p=pair, current=ticker: self._on_market_data_update(p, args[0] if args else current)
            latest_price = self.broker.market_price(ticker)
            if latest_price > 0:
                self._update_terminal_status_snapshot(
                    pair=pair,
                    price=latest_price,
                    market_timestamp=datetime.now(timezone.utc),
                    bar_timestamp=updated_states[pair].live_bar_timestamp,
                )

        self.states = updated_states
        self.last_conversion_rates = refreshed_conversion_rates
        self.primary_bar_streams = primary_streams
        self.conversion_bar_streams = conversion_streams
        self.market_tickers = market_tickers
        await self._cancel_stream_tasks()
        self._start_stream_tasks()

    def _history_start(self, pair: str, end: datetime) -> datetime:
        timeframe_minutes = TIMEFRAME_TO_MINUTES[self.config.timeframe]
        overlap_bars = 600
        default_start = end - timedelta(minutes=timeframe_minutes * overlap_bars)
        state = self.states.get(pair)
        if state is None or state.last_bar_timestamp is None:
            return default_start
        last_seen = pd.Timestamp(state.last_bar_timestamp)
        if last_seen.tzinfo is None:
            last_seen = last_seen.tz_localize("UTC")
        else:
            last_seen = last_seen.tz_convert("UTC")
        return last_seen.to_pydatetime() - timedelta(minutes=timeframe_minutes * overlap_bars)

    def _merge_market_data(self, pair: str, incoming: pd.DataFrame, apply_strategy_features: bool) -> pd.DataFrame:
        if incoming.empty:
            return incoming
        base_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        existing_state = self.states.get(pair)
        if existing_state is None or existing_state.market_data.empty:
            merged = incoming[base_columns].copy()
        else:
            existing = existing_state.market_data[[column for column in base_columns if column in existing_state.market_data.columns]].copy()
            merged = pd.concat([existing, incoming[base_columns]], ignore_index=True)
            merged = merged.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)
        if len(merged) > 3000:
            merged = merged.tail(3000).reset_index(drop=True)
        return self._prepare_market_data(pair, merged, apply_strategy_features=apply_strategy_features)

    def _stream_status_message(self, pair: str, dataframe: pd.DataFrame, previous_timestamp) -> str:
        latest_label = self._format_timestamp(dataframe.iloc[-1]["timestamp"])
        if previous_timestamp is None:
            return f"Loaded {len(dataframe):,} warm-up bars for {pair} through {latest_label}."
        synced_bars = int((dataframe["timestamp"] > previous_timestamp).sum())
        if synced_bars > 0:
            return f"Re-synced {pair}: caught up {synced_bars} bar(s) through {latest_label}."
        return f"Re-synced {pair}: no completed-bar gap detected. Latest bar {latest_label}."

    def _start_stream_tasks(self) -> None:
        for pair, bar_list in self.conversion_bar_streams.items():
            self.stream_tasks.append(asyncio.create_task(self._poll_conversion_stream(pair, bar_list)))
        for pair, bar_list in self.primary_bar_streams.items():
            self.stream_tasks.append(asyncio.create_task(self._poll_primary_stream(pair, bar_list)))
        if self.market_tickers:
            self.stream_tasks.append(asyncio.create_task(self._poll_market_prices()))

    async def _poll_conversion_stream(self, pair: str, bar_list) -> None:
        while not self.reconnect_event.is_set():
            try:
                if self.broker is None or not self.broker.ib.isConnected():
                    return
                incoming = self.broker.bars_to_dataframe(bar_list)
                merged = self._merge_market_data(pair, incoming, apply_strategy_features=False)
                if not merged.empty:
                    existing_state = self.states.get(pair)
                    latest_ts = merged.iloc[-1]["timestamp"]
                    latest_price = float(merged.iloc[-1]["close"])
                    last_processed = existing_state.last_bar_timestamp if existing_state is not None else None
                    if existing_state is None or last_processed is None or pd.Timestamp(latest_ts) > pd.Timestamp(last_processed):
                        self.states[pair] = PairState(
                            pair=pair,
                            market_data=merged,
                            last_bar_timestamp=latest_ts,
                            live_bar_timestamp=latest_ts,
                            live_price=latest_price,
                            position=existing_state.position if existing_state is not None else None,
                        )
                        self.last_conversion_rates[pair] = latest_price
                        self.last_live_update_at = asyncio.get_running_loop().time()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                if self._looks_like_connection_issue(exc):
                    self.connection_loss_active = True
                    self._request_reconnect(f"Conversion stream error for {pair}: {exc}")
                    return
                self._emit_status(f"[WARNING] Conversion stream update failed for {pair}: {exc}")
            await asyncio.sleep(self.stream_poll_interval_seconds)

    async def _poll_primary_stream(self, pair: str, bar_list) -> None:
        while not self.reconnect_event.is_set():
            try:
                async with self.lock:
                    if self.broker is None or not self.broker.ib.isConnected():
                        return
                    incoming = self.broker.bars_to_dataframe(bar_list)
                    merged = self._merge_market_data(pair, incoming, apply_strategy_features=True)
                    if merged.empty or len(merged) < 2:
                        continue
                    state = self.states.get(pair)
                    live_bar_timestamp = merged.iloc[-1]["timestamp"]
                    latest_price = float(merged.iloc[-1]["close"])
                    latest_closed_ts = merged.iloc[-2]["timestamp"]
                    live_changed = (
                        state is None
                        or state.live_bar_timestamp is None
                        or pd.Timestamp(live_bar_timestamp) != pd.Timestamp(state.live_bar_timestamp)
                        or abs(float(state.live_price) - latest_price) > 1e-9
                    )
                    self.states[pair] = PairState(
                        pair=pair,
                        market_data=merged,
                        last_bar_timestamp=state.last_bar_timestamp if state is not None else None,
                        live_bar_timestamp=live_bar_timestamp,
                        live_price=latest_price,
                        position=state.position if state is not None else None,
                    )
                    self._update_terminal_status_snapshot(
                        pair=pair,
                        price=latest_price,
                        market_timestamp=datetime.now(timezone.utc),
                        bar_timestamp=live_bar_timestamp,
                        refresh_status_line=live_changed,
                    )
                    if live_changed:
                        self.last_live_update_at = asyncio.get_running_loop().time()
                    if state is not None and state.last_bar_timestamp is not None and pd.Timestamp(latest_closed_ts) <= pd.Timestamp(state.last_bar_timestamp):
                        continue
                    self.states[pair].last_bar_timestamp = latest_closed_ts
                    await self._process_pair(pair)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                if self._looks_like_connection_issue(exc):
                    self.connection_loss_active = True
                    self._request_reconnect(f"Primary stream error for {pair}: {exc}")
                    return
                self._emit_status(f"[WARNING] Live bar update failed for {pair}: {exc}")
            await asyncio.sleep(self.stream_poll_interval_seconds)

    def _on_market_data_update(self, pair: str, ticker: Any) -> None:
        if self.broker is None:
            return
        price = self.broker.market_price(ticker)
        if price <= 0:
            return
        self._update_terminal_status_snapshot(
            pair=pair,
            price=price,
            market_timestamp=datetime.now(timezone.utc),
            bar_timestamp=self.states.get(pair).live_bar_timestamp if self.states.get(pair) is not None else None,
        )
        self.last_live_update_at = asyncio.get_running_loop().time()

    def _on_live_bar_tick_update(self, pair: str, bar_list, has_new_bar: bool) -> None:
        if pair not in self.config.pairs:
            return
        try:
            if not bar_list:
                return
            live_bar = bar_list[-1]
            live_price = float(getattr(live_bar, "close", 0.0) or 0.0)
            if live_price <= 0:
                return
            live_bar_timestamp = getattr(live_bar, "date", None)
            state = self.states.get(pair)
            if state is not None:
                state.live_price = live_price
                if live_bar_timestamp is not None:
                    state.live_bar_timestamp = pd.Timestamp(live_bar_timestamp)
            self._update_terminal_status_snapshot(
                pair=pair,
                price=live_price,
                market_timestamp=datetime.now(timezone.utc),
                bar_timestamp=live_bar_timestamp,
                refresh_status_line=True,
            )
            self.last_live_update_at = asyncio.get_running_loop().time()
        except Exception:
            return

    async def _poll_market_prices(self) -> None:
        while not self.reconnect_event.is_set():
            try:
                if self.broker is None or not self.broker.ib.isConnected():
                    return
                updated_any = False
                now = datetime.now(timezone.utc)
                for pair in self.config.pairs:
                    ticker = self.market_tickers.get(pair)
                    price = self.broker.market_price(ticker) if ticker is not None else 0.0
                    if price <= 0:
                        state = self.states.get(pair)
                        if state is not None and state.live_price > 0:
                            price = state.live_price
                        elif state is not None and not state.market_data.empty:
                            price = float(state.market_data.iloc[-1]["close"])
                    if price <= 0:
                        continue
                    self._update_terminal_status_snapshot(
                        pair=pair,
                        price=price,
                        market_timestamp=now,
                        bar_timestamp=self.states.get(pair).live_bar_timestamp if self.states.get(pair) is not None else None,
                        refresh_status_line=False,
                    )
                    updated_any = True
                if updated_any:
                    self._refresh_terminal_status_line()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                if self._looks_like_connection_issue(exc):
                    self.connection_loss_active = True
                    self._request_reconnect(f"Real-time price polling failed: {exc}")
                    return
                self._emit_status(f"[WARNING] Real-time price refresh failed: {exc}")
            await asyncio.sleep(self.market_price_poll_interval_seconds)

    def _prepare_market_data(self, pair: str, dataframe: pd.DataFrame, apply_strategy_features: bool) -> pd.DataFrame:
        if dataframe.empty:
            return dataframe
        frame = self.regime_detector.annotate(dataframe)
        if apply_strategy_features:
            for strategy in build_default_strategies(self._strategy_config_for_pair(pair)):
                frame = strategy.prepare_data(frame)
        return frame

    async def _process_pair(self, pair: str) -> None:
        state = self.states[pair]
        market_data = state.market_data
        analysis_data = market_data.iloc[:-1].copy() if len(market_data) >= 2 else market_data.copy()
        strategy_config = self._strategy_config_for_pair(pair)
        risk_config = self._risk_config_for_pair(pair)
        strategies = build_default_strategies(strategy_config)
        allocator = StrategyAllocator(
            score_threshold=strategy_config.score_threshold,
            min_strategy_confidence=strategy_config.min_strategy_confidence,
        )
        if len(analysis_data) < max(strategy.required_bars(analysis_data) for strategy in strategies):
            return

        bar = analysis_data.iloc[-1]
        timestamp = pd.Timestamp(bar["timestamp"]).to_pydatetime()
        close_price = float(bar["close"])
        quote_to_usd = self._quote_to_usd(pair, close_price)
        if quote_to_usd <= 0:
            return

        if state.position is not None:
            state.position.bars_held += 1
            self._update_position_state(state.position, bar)

        open_risk_ratio = self._open_risk_ratio(state.position, quote_to_usd)
        unrealized_pnl = self._unrealized_pnl(state.position, close_price, quote_to_usd)
        equity = self.config.risk.initial_capital + self.realized_pnl + unrealized_pnl
        self.risk_overlay.update_equity(equity)

        if state.position is not None:
            stop_hit, stop_price = self._check_stop(state.position, bar)
            take_profit_hit, take_profit_price = self._check_take_profit(state.position, bar)
            if stop_hit:
                self.realized_pnl, state.position = await self._close_position(
                    pair=pair,
                    position=state.position,
                    exit_price=stop_price,
                    exit_time=timestamp,
                    exit_reason=self._stop_exit_reason(state.position),
                    exit_regime=str(bar["regime"]),
                    quote_to_usd=quote_to_usd,
                    action="SELL" if state.position.side == "Long" else "BUY",
                )
                if state.position is None:
                    unrealized_pnl = 0.0
                    equity = self.config.risk.initial_capital + self.realized_pnl
                    self.risk_overlay.update_equity(equity)
                    open_risk_ratio = 0.0
            elif take_profit_hit:
                self.realized_pnl, state.position = await self._close_position(
                    pair=pair,
                    position=state.position,
                    exit_price=take_profit_price,
                    exit_time=timestamp,
                    exit_reason="ATR Take Profit",
                    exit_regime=str(bar["regime"]),
                    quote_to_usd=quote_to_usd,
                    action="SELL" if state.position.side == "Long" else "BUY",
                )
                if state.position is None:
                    unrealized_pnl = 0.0
                    equity = self.config.risk.initial_capital + self.realized_pnl
                    self.risk_overlay.update_equity(equity)
                    open_risk_ratio = 0.0
            else:
                self._update_trailing_stop(state.position, bar, risk_config)

        decisions = {strategy.name: strategy.evaluate(analysis_data) for strategy in strategies}
        composite = allocator.allocate(decisions, str(bar["regime"]), self.performance_tracker)

        if state.position is not None:
            if self._should_signal_exit(state.position, composite.final_signal, close_price, strategy_config):
                self.realized_pnl, state.position = await self._close_position(
                    pair=pair,
                    position=state.position,
                    exit_price=close_price,
                    exit_time=timestamp,
                    exit_reason="Signal Exit",
                    exit_regime=str(bar["regime"]),
                    quote_to_usd=quote_to_usd,
                    action="SELL" if state.position.side == "Long" else "BUY",
                )
                if state.position is None:
                    unrealized_pnl = 0.0
                    equity = self.config.risk.initial_capital + self.realized_pnl
                    self.risk_overlay.update_equity(equity)
                    open_risk_ratio = 0.0

        if state.position is None:
            risk_decision = self.risk_overlay.evaluate_entry(
                bias=composite.bias,
                equity=equity,
                current_open_risk_ratio=open_risk_ratio,
                close_price=close_price,
                atr_value=float(bar.get("atr", 0.0) or 0.0),
                quote_to_usd_rate=quote_to_usd,
                pair=pair,
                open_positions=self._open_positions(),
                current_bar_range=float(max(float(bar["high"]) - float(bar["low"]), 0.0)),
                average_bar_range=float(analysis_data["high"].sub(analysis_data["low"]).clip(lower=0.0).tail(20).mean() or 0.0),
                average_atr_value=float(analysis_data["atr"].tail(100).mean() or 0.0),
            )
            self.risk_rows.append(
                {
                    "timestamp": timestamp,
                    "pair": pair,
                    "equity": equity,
                    "equity_peak": self.risk_overlay.equity_peak,
                    "drawdown_pct": 1.0 - (equity / self.risk_overlay.equity_peak if self.risk_overlay.equity_peak else 1.0),
                    "signal_bias": composite.bias,
                    "final_signal": composite.final_signal,
                    "risk_approved": risk_decision.approved,
                    "risk_reason": risk_decision.reason,
                    "risk_amount_usd": risk_decision.risk_amount_usd,
                    "regime": str(bar["regime"]),
                    "trading_enabled": self.risk_overlay.trading_enabled,
                    "circuit_breaker_active": self.risk_overlay.circuit_breaker_active,
                    "cooldown_bars_remaining": self.risk_overlay.cooldown_bars_remaining,
                }
            )
            if risk_decision.approved:
                candidate_trade_id = self.trade_id + 1
                modeled_entry_price = self._apply_slippage(close_price, composite.bias)
                side = "Long" if composite.bias > 0 else "Short"
                action = "BUY" if side == "Long" else "SELL"
                trade = await self.broker.place_market_order(pair, action, risk_decision.size_units)
                execution_time = datetime.now(timezone.utc)
                order_status = self._trade_status(trade)
                order_details = self._trade_message(trade) or "paper entry"
                filled_entry_price = self._trade_fill_price(trade, modeled_entry_price)
                self.order_events.append(
                    OrderEvent(
                        timestamp=execution_time,
                        pair=pair,
                        action=action,
                        size_units=risk_decision.size_units,
                        order_type="MKT",
                        status=order_status,
                        price=filled_entry_price,
                        order_id=str(getattr(trade.order, "orderId", "")),
                        details=order_details,
                    )
                )
                if self._trade_succeeded(trade):
                    self.trade_id = candidate_trade_id
                    entry_reason = self._dominant_reason(composite, composite.bias)
                    take_profit_price = self._take_profit_price(filled_entry_price, composite.bias, float(bar.get("atr", 0.0) or 0.0), risk_config)
                    contributing_strategies = self._contributing_strategies(composite, composite.bias)
                    state.position = Position(
                        trade_id=self.trade_id,
                        pair=pair,
                        side=side,
                        size_units=risk_decision.size_units,
                        entry_time=execution_time,
                        entry_price=filled_entry_price,
                        stop_price=risk_decision.stop_price,
                        take_profit_price=take_profit_price,
                        entry_reason=entry_reason,
                        entry_regime=str(bar["regime"]),
                        atr_at_entry=float(bar.get("atr", 0.0) or 0.0),
                        equity_at_entry=equity,
                        risk_amount_usd=risk_decision.risk_amount_usd,
                        strategy_scores=composite.strategy_scores,
                        contributing_strategies=contributing_strategies,
                        signal_strength=composite.final_signal,
                        highest_price_seen=filled_entry_price,
                        lowest_price_seen=filled_entry_price,
                    )
                    self.trade_events.append(
                        TradeEventRecord(
                            trade_id=self.trade_id,
                            position_type=side,
                            step_type="Entry",
                            timestamp=execution_time,
                            signal_reason=entry_reason,
                            price=filled_entry_price,
                            size=float(risk_decision.size_units),
                            net_pnl_usd=0.0,
                            net_pnl_pct=0.0,
                            cumulative_pnl_usd=self.realized_pnl,
                            cumulative_pnl_pct=self.realized_pnl / self.config.risk.initial_capital,
                        )
                    )
                    self._emit_status(
                        f"TRADE OPEN | {pair} | {side} | {execution_time:%Y-%m-%d %H:%M:%S UTC} | "
                        f"exec={filled_entry_price:.5f} | size={risk_decision.size_units:,} | "
                        f"stop={risk_decision.stop_price:.5f} | tp={take_profit_price:.5f} | "
                        f"signal={composite.final_signal:+.3f} | reason={entry_reason} | "
                        f"strategies={', '.join(contributing_strategies)} | signal_bar={timestamp:%Y-%m-%d %H:%M UTC}"
                    )
                else:
                    self._emit_status(
                        f"ORDER REJECTED | {pair} | {action} | {execution_time:%Y-%m-%d %H:%M:%S UTC} | details={order_details}"
                    )

        self._maybe_emit_bar_status(
            pair=pair,
            timestamp=timestamp,
            close_price=close_price,
            regime=str(bar["regime"]),
            composite=composite,
            position=state.position,
        )

        current_unrealized = self._unrealized_pnl(state.position, close_price, quote_to_usd)
        self.equity_rows.append(
            {
                "timestamp": timestamp,
                "pair": pair,
                "equity": self.config.risk.initial_capital + self.realized_pnl + current_unrealized,
                "realized_pnl_usd": self.realized_pnl,
                "unrealized_pnl_usd": current_unrealized,
                "position_side": state.position.side if state.position else "Flat",
                "regime": str(bar["regime"]),
                "close": close_price,
                "quote_to_usd": quote_to_usd,
            }
        )

    def _strategy_config_for_pair(self, pair: str) -> StrategyConfig:
        overrides = self.config.pair_specific_config.get(pair.upper(), {}).get("strategy", {})
        return replace(self.config.strategy, **overrides) if overrides else self.config.strategy

    def _risk_config_for_pair(self, pair: str) -> RiskConfig:
        overrides = self.config.pair_specific_config.get(pair.upper(), {}).get("risk", {})
        return replace(self.config.risk, **overrides) if overrides else self.config.risk

    @staticmethod
    def _conversion_pairs_for_quote(quote: str) -> set[str]:
        if quote == "USD":
            return set()
        if quote == "JPY":
            return {"USDJPY"}
        return {f"{quote}USD", f"USD{quote}"}

    def _quote_to_usd(self, pair: str, close_price: float | None = None) -> float:
        base, quote = split_pair(pair)
        if quote == "USD":
            return 1.0
        if quote == "JPY":
            usd_jpy = self.last_conversion_rates.get("USDJPY", 0.0)
            if usd_jpy > 0:
                return 1.0 / usd_jpy
            fallback_close = None
            if pair == "USDJPY" and close_price is not None and close_price > 0:
                fallback_close = close_price
            elif "USDJPY" in self.states and not self.states["USDJPY"].market_data.empty:
                fallback_close = float(self.states["USDJPY"].market_data.iloc[-1]["close"])
            if fallback_close is not None and fallback_close > 0:
                return 1.0 / fallback_close
            return 0.0
        direct = f"{quote}USD"
        inverse = f"USD{quote}"
        if direct in self.last_conversion_rates and self.last_conversion_rates[direct] > 0:
            return self.last_conversion_rates[direct]
        if inverse in self.last_conversion_rates and self.last_conversion_rates[inverse] > 0:
            return 1.0 / self.last_conversion_rates[inverse]
        fallback_close = close_price
        if (fallback_close is None or fallback_close <= 0) and pair in self.states and not self.states[pair].market_data.empty:
            fallback_close = float(self.states[pair].market_data.iloc[-1]["close"])
        if base == "USD" and fallback_close is not None and fallback_close > 0:
            return 1.0 / fallback_close
        return 0.0

    def _open_risk_ratio(self, position: Position | None, quote_to_usd: float) -> float:
        if position is None:
            return 0.0
        equity = self.config.risk.initial_capital + self.realized_pnl
        if equity <= 0:
            return 1.0
        stop_distance = abs(position.entry_price - position.stop_price)
        return (stop_distance * position.size_units * quote_to_usd) / equity

    def _open_positions(self) -> list[Position]:
        return [state.position for state in self.states.values() if state.position is not None]

    @staticmethod
    def _update_position_state(position: Position, bar: pd.Series) -> None:
        current_high = float(bar["high"])
        current_low = float(bar["low"])
        position.highest_price_seen = max(position.highest_price_seen, current_high)
        position.lowest_price_seen = min(position.lowest_price_seen, current_low)
        if position.side == "Long":
            favorable = (current_high - position.entry_price) / position.entry_price
            adverse = (current_low - position.entry_price) / position.entry_price
        else:
            favorable = (position.entry_price - current_low) / position.entry_price
            adverse = (position.entry_price - current_high) / position.entry_price
        position.max_favorable_excursion_pct = max(position.max_favorable_excursion_pct, favorable)
        position.max_adverse_excursion_pct = min(position.max_adverse_excursion_pct, adverse)

    def _update_trailing_stop(self, position: Position, bar: pd.Series, risk_config: RiskConfig) -> None:
        current_atr = float(bar.get("atr", 0.0) or 0.0)
        trigger_atr = position.atr_at_entry if position.atr_at_entry > 0 else current_atr
        reference_atr = max(trigger_atr, current_atr)
        if trigger_atr <= 0 or reference_atr <= 0:
            return

        break_even_distance = trigger_atr * risk_config.break_even_atr_multiplier
        break_even_buffer = trigger_atr * risk_config.break_even_buffer_atr_multiplier
        trailing_activation_multiplier = risk_config.trailing_activation_atr_multiplier
        if TIMEFRAME_TO_MINUTES.get(self.config.timeframe, 0) >= TIMEFRAME_TO_MINUTES["4h"]:
            trailing_activation_multiplier = max(trailing_activation_multiplier, 2.0)
        trailing_distance = reference_atr * risk_config.trailing_stop_atr_multiplier
        trailing_step_multiplier = max(risk_config.trailing_stop_step_atr_multiplier, 0.1)

        if position.side == "Long":
            favorable_distance = position.highest_price_seen - position.entry_price
            if not position.break_even_armed and favorable_distance >= break_even_distance:
                position.stop_price = max(position.stop_price, position.entry_price + break_even_buffer)
                position.break_even_armed = True
            if favorable_distance >= trigger_atr * trailing_activation_multiplier:
                step_count = 1 + int((favorable_distance - (trigger_atr * trailing_activation_multiplier)) / (trigger_atr * trailing_step_multiplier))
                if not position.trailing_stop_active or step_count > position.trailing_step_count:
                    position.trailing_stop_active = True
                    position.trailing_step_count = step_count
                    position.stop_price = max(position.stop_price, position.highest_price_seen - trailing_distance)
        else:
            favorable_distance = position.entry_price - position.lowest_price_seen
            if not position.break_even_armed and favorable_distance >= break_even_distance:
                position.stop_price = min(position.stop_price, position.entry_price - break_even_buffer)
                position.break_even_armed = True
            if favorable_distance >= trigger_atr * trailing_activation_multiplier:
                step_count = 1 + int((favorable_distance - (trigger_atr * trailing_activation_multiplier)) / (trigger_atr * trailing_step_multiplier))
                if not position.trailing_stop_active or step_count > position.trailing_step_count:
                    position.trailing_stop_active = True
                    position.trailing_step_count = step_count
                    position.stop_price = min(position.stop_price, position.lowest_price_seen + trailing_distance)

    @staticmethod
    def _check_stop(position: Position, bar: pd.Series) -> tuple[bool, float]:
        if position.side == "Long" and float(bar["low"]) <= position.stop_price:
            return True, position.stop_price
        if position.side == "Short" and float(bar["high"]) >= position.stop_price:
            return True, position.stop_price
        return False, position.stop_price

    @staticmethod
    def _check_take_profit(position: Position, bar: pd.Series) -> tuple[bool, float]:
        if position.side == "Long" and float(bar["high"]) >= position.take_profit_price:
            return True, position.take_profit_price
        if position.side == "Short" and float(bar["low"]) <= position.take_profit_price:
            return True, position.take_profit_price
        return False, position.take_profit_price

    @staticmethod
    def _stop_exit_reason(position: Position) -> str:
        if position.trailing_stop_active:
            return "Trailing Stop Exit"
        if position.break_even_armed:
            return "Break-Even Stop"
        return "ATR Exit"

    def _should_signal_exit(self, position: Position, final_signal: float, close_price: float, strategy_config: StrategyConfig) -> bool:
        reversal_threshold = strategy_config.score_threshold / 2
        opposite_signal = (position.side == "Long" and final_signal < -reversal_threshold) or (
            position.side == "Short" and final_signal > reversal_threshold
        )
        if not opposite_signal:
            return False
        profit_in_atr = self._profit_in_atr(position, close_price)
        if profit_in_atr > 1.0 and abs(final_signal) <= 0.5:
            return False
        return True

    @staticmethod
    def _profit_in_atr(position: Position, close_price: float) -> float:
        if position.atr_at_entry <= 0:
            return 0.0
        if position.side == "Long":
            return (close_price - position.entry_price) / position.atr_at_entry
        return (position.entry_price - close_price) / position.atr_at_entry

    def _take_profit_price(self, entry_price: float, bias: int, atr_value: float, risk_config: RiskConfig) -> float:
        stop_distance = atr_value * risk_config.atr_stop_multiplier
        distance = max(
            atr_value * risk_config.take_profit_atr_multiplier,
            stop_distance * 2.5,
        )
        return entry_price + bias * distance

    def _apply_slippage(self, price: float, bias: int) -> float:
        if self.config.slippage_bps == 0:
            return price
        return price * (1 + (self.config.slippage_bps / 10_000) * bias)

    @staticmethod
    def _pnl_quote(side: str, size_units: int, entry_price: float, exit_price: float) -> float:
        if side == "Long":
            return size_units * (exit_price - entry_price)
        return size_units * (entry_price - exit_price)

    def _unrealized_pnl(self, position: Position | None, close_price: float, quote_to_usd: float) -> float:
        if position is None:
            return 0.0
        return self._pnl_quote(position.side, position.size_units, position.entry_price, close_price) * quote_to_usd

    async def _close_position(
        self,
        pair: str,
        position: Position,
        exit_price: float,
        exit_time,
        exit_reason: str,
        exit_regime: str,
        quote_to_usd: float,
        action: str,
    ) -> tuple[float, Position | None]:
        modeled_exit_price = self._apply_slippage(exit_price, -1 if position.side == "Long" else 1)
        trade = await self.broker.place_market_order(pair, action, position.size_units)
        execution_time = datetime.now(timezone.utc)
        order_status = self._trade_status(trade)
        order_details = self._trade_message(trade) or "paper exit"
        filled_exit_price = self._trade_fill_price(trade, modeled_exit_price)
        self.order_events.append(
            OrderEvent(
                timestamp=execution_time,
                pair=pair,
                action=action,
                size_units=position.size_units,
                order_type="MKT",
                status=order_status,
                price=filled_exit_price,
                order_id=str(getattr(trade.order, "orderId", "")),
                details=order_details,
            )
        )
        if not self._trade_succeeded(trade):
            self._emit_status(
                f"ORDER REJECTED | {pair} | {action} | {execution_time:%Y-%m-%d %H:%M:%S UTC} | "
                f"details={order_details} | position remains open"
            )
            return self.realized_pnl, position

        pnl_quote = self._pnl_quote(position.side, position.size_units, position.entry_price, filled_exit_price)
        pnl_usd = pnl_quote * quote_to_usd - self.config.risk.commission_per_trade_usd
        realized_pnl = self.realized_pnl + pnl_usd
        pnl_pct = pnl_usd / position.equity_at_entry if position.equity_at_entry else 0.0
        cumulative_pct = realized_pnl / self.config.risk.initial_capital if self.config.risk.initial_capital else 0.0
        trade_record = TradeRecord(
            trade_id=position.trade_id,
            pair=position.pair,
            side=position.side,
            entry_time=position.entry_time,
            exit_time=execution_time,
            entry_price=position.entry_price,
            exit_price=filled_exit_price,
            size_units=position.size_units,
            pnl_quote=pnl_quote,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            cumulative_pnl_usd=realized_pnl,
            cumulative_pnl_pct=cumulative_pct,
            entry_reason=position.entry_reason,
            exit_reason=exit_reason,
            entry_regime=position.entry_regime,
            exit_regime=exit_regime,
            bars_held=position.bars_held,
            max_favorable_excursion_pct=position.max_favorable_excursion_pct,
            max_adverse_excursion_pct=position.max_adverse_excursion_pct,
            commission_paid_usd=self.config.risk.commission_per_trade_usd,
            signal_strength=position.signal_strength,
            equity_at_entry=position.equity_at_entry,
            strategy_scores=position.strategy_scores,
        )
        self.trades.append(trade_record)
        self.trade_events.append(
            TradeEventRecord(
                trade_id=position.trade_id,
                position_type=position.side,
                step_type="Exit",
                timestamp=execution_time,
                signal_reason=exit_reason,
                price=filled_exit_price,
                size=float(position.size_units),
                net_pnl_usd=pnl_usd,
                net_pnl_pct=pnl_pct,
                cumulative_pnl_usd=realized_pnl,
                cumulative_pnl_pct=cumulative_pct,
                favorite_excursion_pct=position.max_favorable_excursion_pct,
                adverse_excursion_pct=position.max_adverse_excursion_pct,
            )
        )
        self.performance_tracker.record_trade(
            strategy_names=position.contributing_strategies,
            pair=position.pair,
            side=position.side,
            entry_time=position.entry_time,
            exit_time=execution_time,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            regime=position.entry_regime,
        )
        self.realized_pnl = realized_pnl
        self._emit_status(
            f"TRADE CLOSE | {pair} | {position.side} | {execution_time:%Y-%m-%d %H:%M:%S UTC} | "
            f"exec={filled_exit_price:.5f} | exit={exit_reason} | entry_reason={position.entry_reason} | "
            f"strategies={', '.join(position.contributing_strategies)} | "
            f"pnl={pnl_usd:+.2f} USD ({pnl_pct:+.2%}) | signal_bar={exit_time:%Y-%m-%d %H:%M UTC}"
        )
        return realized_pnl, None

    @staticmethod
    def _dominant_reason(composite, bias: int) -> str:
        ranked = sorted(composite.strategy_scores.items(), key=lambda item: item[1], reverse=True)
        for name, _ in ranked:
            decision = composite.strategy_decisions[name]
            if decision.signal == bias:
                return str(decision.metadata.get("reason", f"{name} signal"))
        return "Allocator Signal"

    @staticmethod
    def _contributing_strategies(composite, bias: int) -> list[str]:
        names = [name for name, decision in composite.strategy_decisions.items() if decision.signal == bias and composite.strategy_scores.get(name, 0.0) > 0]
        return names or [max(composite.strategy_scores, key=composite.strategy_scores.get)]

    def _maybe_emit_bar_status(self, pair: str, timestamp, close_price: float, regime: str, composite, position: Position | None) -> None:
        bias_label = {1: "Long", -1: "Short", 0: "Flat"}[composite.bias]
        position_label = position.side if position is not None else "Flat"
        updates = self.bar_updates_seen.get(pair, 0) + 1
        self.bar_updates_seen[pair] = updates
        current_price = self._current_display_price(pair, close_price)
        self._update_terminal_status_snapshot(
            pair=pair,
            price=current_price,
            market_timestamp=self._status_market_timestamp(pair, timestamp),
            signal_timestamp=timestamp,
            regime=regime,
            signal=composite.final_signal,
            bias=bias_label,
            position=position_label,
        )

    def _update_terminal_status_snapshot(
        self,
        pair: str,
        price: float | None = None,
        market_timestamp=None,
        bar_timestamp=None,
        signal_timestamp=None,
        regime: str | None = None,
        signal: float | None = None,
        bias: str | None = None,
        position: str | None = None,
        refresh_status_line: bool = True,
    ) -> None:
        snapshot = dict(self.terminal_status_by_pair.get(pair, {}))
        if price is not None and price > 0:
            snapshot["price"] = float(price)
        if market_timestamp is not None:
            snapshot["market_timestamp"] = pd.Timestamp(market_timestamp)
        if bar_timestamp is not None:
            snapshot["bar_timestamp"] = pd.Timestamp(bar_timestamp)
        if signal_timestamp is not None:
            snapshot["signal_timestamp"] = pd.Timestamp(signal_timestamp)
        if regime is not None:
            snapshot["regime"] = regime
        if signal is not None:
            snapshot["signal"] = float(signal)
        if bias is not None:
            snapshot["bias"] = bias
        if position is not None:
            snapshot["position"] = position
        self.terminal_status_by_pair[pair] = snapshot
        if refresh_status_line:
            self._refresh_terminal_status_line()

    def _current_display_price(self, pair: str, fallback_price: float | None = None) -> float:
        state = self.states.get(pair)
        if state is not None and state.live_price > 0:
            return state.live_price
        ticker = self.market_tickers.get(pair)
        if ticker is not None and self.broker is not None:
            live_price = self.broker.market_price(ticker)
            if live_price > 0:
                return live_price
        if fallback_price is not None and fallback_price > 0:
            return fallback_price
        if state is not None and not state.market_data.empty:
            return float(state.market_data.iloc[-1]["close"])
        return 0.0

    def _status_market_timestamp(self, pair: str, fallback_timestamp) -> pd.Timestamp:
        snapshot = self.terminal_status_by_pair.get(pair, {})
        value = snapshot.get("market_timestamp", fallback_timestamp)
        return pd.Timestamp(value)

    @staticmethod
    def _trade_status(trade) -> str:
        status = str(getattr(trade.orderStatus, "status", "") or "Unknown")
        return status

    @staticmethod
    def _trade_message(trade) -> str:
        for entry in reversed(getattr(trade, "log", []) or []):
            message = str(getattr(entry, "message", "") or "").strip()
            if message:
                return message
        advanced_error = str(getattr(trade, "advancedError", "") or "").strip()
        return advanced_error

    @classmethod
    def _trade_succeeded(cls, trade) -> bool:
        status = cls._trade_status(trade)
        filled = float(getattr(trade.orderStatus, "filled", 0.0) or 0.0)
        return status == "Filled" or filled > 0

    @staticmethod
    def _trade_fill_price(trade, fallback_price: float) -> float:
        avg_fill = float(getattr(trade.orderStatus, "avgFillPrice", 0.0) or 0.0)
        return avg_fill if avg_fill > 0 else fallback_price

    def _emit_status(self, message: str) -> None:
        if not self.config.summary_to_stdout:
            return
        self._clear_terminal_status_line()
        print(message, flush=True)
        self._refresh_terminal_status_line()

    def _clear_terminal_status_line(self) -> None:
        if not self.config.summary_to_stdout or not self.terminal_status_line:
            return
        clear_width = max(len(self.terminal_status_line), 200)
        print("\r" + (" " * clear_width) + "\r", end="", flush=True)

    def _refresh_terminal_status_line(self) -> None:
        if not self.config.summary_to_stdout:
            return
        ordered_statuses = [self._render_terminal_status(pair) for pair in self.config.pairs if pair in self.terminal_status_by_pair]
        if not ordered_statuses:
            self.terminal_status_line = ""
            return
        line = "LIVE | " + " || ".join(ordered_statuses)
        display_width = max(len(line), len(self.terminal_status_line), 200)
        print(f"\r{line:<{display_width}}", end="", flush=True)
        self.terminal_status_line = line

    def _render_terminal_status(self, pair: str) -> str:
        snapshot = self.terminal_status_by_pair.get(pair, {})
        price = float(snapshot.get("price", 0.0) or 0.0)
        market_timestamp = snapshot.get("market_timestamp")
        if market_timestamp is not None:
            market_label = self._format_romania_display_timestamp(market_timestamp, include_seconds=True)
        else:
            market_label = "--"
        bar_timestamp = snapshot.get("bar_timestamp")
        if bar_timestamp is not None:
            bar_label = self._format_romania_display_timestamp(bar_timestamp, include_seconds=False)
        else:
            bar_label = "--"
        regime = str(snapshot.get("regime", "waiting"))
        signal = float(snapshot.get("signal", 0.0) or 0.0)
        bias = str(snapshot.get("bias", "Flat"))
        position = str(snapshot.get("position", "Flat"))
        return (
            f"{pair} {price:.5f} | live={market_label} RO | bar={bar_label} RO | "
            f"reg={regime} | sig={signal:+.3f} | bias={bias} | pos={position}"
        )

    @classmethod
    def _format_romania_display_timestamp(cls, value, include_seconds: bool) -> str:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        localized = timestamp.tz_convert(cls.ROMANIA_TZ)
        return localized.strftime("%Y-%m-%d %H:%M:%S" if include_seconds else "%Y-%m-%d %H:%M")

    @staticmethod
    def _format_timestamp(value) -> str:
        if value is None:
            return "unknown"
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        return timestamp.strftime("%Y-%m-%d %H:%M UTC")

    def _build_report(self) -> dict[str, Any] | None:
        if self.logger is None or self.performance_tracker is None:
            return None
        equity_curve = pd.DataFrame(self.equity_rows)
        if equity_curve.empty or self.config.primary_pair not in self.states:
            return None
        primary_market_data = self.states[self.config.primary_pair].market_data.copy()
        primary_market_data["quote_to_usd"] = primary_market_data["close"].map(
            lambda close: self._quote_to_usd(self.config.primary_pair, float(close))
        )
        strategy_metrics = self.performance_tracker.metrics_frame(
            current_regime=str(self.states[self.config.primary_pair].market_data.iloc[-1]["regime"])
        )
        report = compile_backtest_report(
            pair=self.config.primary_pair,
            timeframe=self.config.timeframe,
            initial_capital=self.config.risk.initial_capital,
            trades=self.trades,
            trade_events=self.trade_events,
            equity_curve=equity_curve,
            market_data=primary_market_data,
            strategy_metrics=strategy_metrics,
            risk_metrics=pd.DataFrame(self.risk_rows),
        )
        report["order_events"] = pd.DataFrame([event.to_dict() for event in self.order_events])
        report["output_dir"] = self.logger.run_dir
        return report









