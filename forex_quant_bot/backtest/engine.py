from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import pandas as pd

from forex_quant_bot.backtest.metrics import compile_backtest_report
from forex_quant_bot.core.allocator import StrategyAllocator
from forex_quant_bot.core.performance_tracker import PerformanceTracker
from forex_quant_bot.core.regime_detector import RegimeDetector
from forex_quant_bot.core.risk_overlay import RiskOverlay
from forex_quant_bot.data.cache import DataCache
from forex_quant_bot.data.replay import BarReplay
from forex_quant_bot.logs.csv_logger import CSVLogger
from forex_quant_bot.logs.summary_printer import SummaryPrinter
from forex_quant_bot.models import Position, RiskDecision, TradeEventRecord, TradeRecord
from forex_quant_bot.settings import BotConfig, RiskConfig, StrategyConfig
from forex_quant_bot.strategies import build_default_strategies
from forex_quant_bot.utils.time_utils import TIMEFRAME_TO_MINUTES, split_pair


@dataclass(slots=True)
class BacktestEngine:
    config: BotConfig

    def run(self) -> dict[str, Any]:
        if self.config.start is None or self.config.end is None:
            raise ValueError("Backtest mode requires --start and --end.")

        pair = self.config.primary_pair
        strategy_config = self._strategy_config_for_pair(pair)
        risk_config = self._risk_config_for_pair(pair)

        self._emit_status(
            f"Loading {pair} {self.config.timeframe} data from {self.config.start.date()} to {self.config.end.date()}..."
        )
        data_cache = DataCache(
            self.config.data.cache_dir,
            auto_download=self.config.data.auto_download,
            sample_rows=self.config.data.sample_rows,
        )
        raw_data = data_cache.get_history(
            pair=pair,
            timeframe=self.config.timeframe,
            start=self.config.start,
            end=self.config.end,
            use_sample_data=self.config.use_sample_data,
        )
        self._emit_status(f"Loaded {len(raw_data):,} bars. Preparing regime and strategy features...")

        strategies = build_default_strategies(strategy_config)
        regime_detector = RegimeDetector()
        market_data = regime_detector.annotate(raw_data)
        for strategy in strategies:
            market_data = strategy.prepare_data(market_data)
        market_data["quote_to_usd"] = data_cache.get_quote_to_usd_series(
            pair=pair,
            timeframe=self.config.timeframe,
            start=self.config.start,
            end=self.config.end,
            index=market_data["timestamp"],
            use_sample_data=self.config.use_sample_data,
        ).values
        market_data["bar_range"] = (market_data["high"] - market_data["low"]).clip(lower=0.0)
        market_data["bar_range_avg_20"] = market_data["bar_range"].rolling(20, min_periods=5).mean().shift(1)
        market_data["atr_avg_100"] = market_data["atr"].rolling(100, min_periods=20).mean()

        allocator = StrategyAllocator(
            score_threshold=strategy_config.score_threshold,
            min_strategy_confidence=strategy_config.min_strategy_confidence,
        )
        tracker = PerformanceTracker(window=strategy_config.recent_trade_window)
        risk_overlay = RiskOverlay(risk_config)
        run_label = f"{self.config.start:%Y%m%d}_to_{self.config.end:%Y%m%d}"
        logger = CSVLogger(self.config.logging.output_dir, self.config.mode, pair, self.config.timeframe, run_label=run_label)

        trades: list[TradeRecord] = []
        trade_events: list[TradeEventRecord] = []
        risk_rows: list[dict[str, Any]] = []
        equity_rows: list[dict[str, Any]] = []
        position: Position | None = None
        realized_pnl = 0.0
        trade_id = 0

        warmup_bars = max(strategy.required_bars(market_data) for strategy in strategies)
        total_replay_bars = max(len(market_data) - warmup_bars, 0)
        progress_checkpoints = self._progress_checkpoints(total_replay_bars)
        next_checkpoint_index = 0
        self._emit_status(f"Running backtest replay over {total_replay_bars:,} bars...")

        for processed_bars, event in enumerate(BarReplay(market_data, warmup_bars=warmup_bars), start=1):
            bar = event.bar
            timestamp = pd.Timestamp(bar["timestamp"]).to_pydatetime()
            close_price = float(bar["close"])
            quote_to_usd = self._sanitize_quote_to_usd(bar.get("quote_to_usd"), pair, close_price)
            if quote_to_usd <= 0:
                continue

            if position is not None:
                position.bars_held += 1
                self._update_position_state(position, bar)

            open_risk_ratio = self._open_risk_ratio(position, quote_to_usd, realized_pnl, risk_config)
            unrealized_pnl = self._unrealized_pnl(position, close_price, quote_to_usd) if position else 0.0
            equity = risk_config.initial_capital + realized_pnl + unrealized_pnl
            risk_overlay.update_equity(equity)

            if position is not None:
                stop_hit, stop_price = self._check_stop(position, bar)
                take_profit_hit, take_profit_price = self._check_take_profit(position, bar)
                if stop_hit:
                    realized_pnl, position = self._close_position(
                        position=position,
                        exit_price=stop_price,
                        exit_time=timestamp,
                        exit_reason=self._stop_exit_reason(position),
                        exit_regime=str(bar["regime"]),
                        quote_to_usd=quote_to_usd,
                        realized_pnl=realized_pnl,
                        trade_events=trade_events,
                        trades=trades,
                        tracker=tracker,
                        risk_config=risk_config,
                    )
                    unrealized_pnl = 0.0
                    equity = risk_config.initial_capital + realized_pnl
                    risk_overlay.update_equity(equity)
                    open_risk_ratio = 0.0
                elif take_profit_hit:
                    realized_pnl, position = self._close_position(
                        position=position,
                        exit_price=take_profit_price,
                        exit_time=timestamp,
                        exit_reason="ATR Take Profit",
                        exit_regime=str(bar["regime"]),
                        quote_to_usd=quote_to_usd,
                        realized_pnl=realized_pnl,
                        trade_events=trade_events,
                        trades=trades,
                        tracker=tracker,
                        risk_config=risk_config,
                    )
                    unrealized_pnl = 0.0
                    equity = risk_config.initial_capital + realized_pnl
                    risk_overlay.update_equity(equity)
                    open_risk_ratio = 0.0
                else:
                    self._update_trailing_stop(position, bar, risk_config)

            decisions = {strategy.name: strategy.evaluate(event.history) for strategy in strategies}
            composite = allocator.allocate(decisions, current_regime=str(bar["regime"]), performance_tracker=tracker)

            if position is not None:
                if self._should_signal_exit(position, composite.final_signal, close_price, strategy_config):
                    realized_pnl, position = self._close_position(
                        position=position,
                        exit_price=close_price,
                        exit_time=timestamp,
                        exit_reason="Signal Exit",
                        exit_regime=str(bar["regime"]),
                        quote_to_usd=quote_to_usd,
                        realized_pnl=realized_pnl,
                        trade_events=trade_events,
                        trades=trades,
                        tracker=tracker,
                        risk_config=risk_config,
                    )
                    unrealized_pnl = 0.0
                    equity = risk_config.initial_capital + realized_pnl
                    risk_overlay.update_equity(equity)
                    open_risk_ratio = 0.0

            current_bar_range = float(bar.get("bar_range", 0.0) or 0.0)
            average_bar_range = float(bar.get("bar_range_avg_20", 0.0) or 0.0)
            session_allowed = self._within_trading_session(timestamp, strategy_config)
            if position is None:
                if composite.bias != 0 and not session_allowed:
                    risk_decision = RiskDecision(False, 0, 0.0, close_price, 0.0, open_risk_ratio, "session_filter")
                else:
                    risk_decision = risk_overlay.evaluate_entry(
                        pair=pair,
                        open_positions=[],
                        bias=composite.bias,
                        equity=equity,
                        current_open_risk_ratio=open_risk_ratio,
                        close_price=close_price,
                        atr_value=float(bar.get("atr", 0.0) or 0.0),
                        quote_to_usd_rate=quote_to_usd,
                        current_bar_range=current_bar_range,
                        average_bar_range=average_bar_range,
                        average_atr_value=float(bar.get("atr_avg_100", 0.0) or 0.0),
                    )
                risk_rows.append(
                    {
                        "timestamp": timestamp,
                        "equity": equity,
                        "equity_peak": risk_overlay.equity_peak,
                        "drawdown_pct": 1.0 - (equity / risk_overlay.equity_peak if risk_overlay.equity_peak else 1.0),
                        "trading_enabled": risk_overlay.trading_enabled,
                        "circuit_breaker_active": risk_overlay.circuit_breaker_active,
                        "cooldown_bars_remaining": risk_overlay.cooldown_bars_remaining,
                        "current_open_risk_ratio": open_risk_ratio,
                        "signal_bias": composite.bias,
                        "final_signal": composite.final_signal,
                        "session_allowed": session_allowed,
                        "risk_approved": risk_decision.approved,
                        "risk_reason": risk_decision.reason,
                        "risk_amount_usd": risk_decision.risk_amount_usd,
                        "atr": float(bar.get("atr", 0.0) or 0.0),
                        "atr_avg_100": float(bar.get("atr_avg_100", 0.0) or 0.0),
                        "bar_range": current_bar_range,
                        "bar_range_avg_20": average_bar_range,
                        "regime": str(bar["regime"]),
                    }
                )
                if risk_decision.approved:
                    trade_id += 1
                    side = "Long" if composite.bias > 0 else "Short"
                    entry_price = self._apply_slippage(close_price, composite.bias)
                    entry_reason = self._dominant_reason(composite, composite.bias)
                    contributing = self._contributing_strategies(composite, composite.bias)
                    take_profit_price = self._take_profit_price(
                        entry_price,
                        composite.bias,
                        float(bar.get("atr", 0.0) or 0.0),
                        risk_config,
                    )
                    position = Position(
                        trade_id=trade_id,
                        pair=pair,
                        side=side,
                        size_units=risk_decision.size_units,
                        entry_time=timestamp,
                        entry_price=entry_price,
                        stop_price=risk_decision.stop_price,
                        take_profit_price=take_profit_price,
                        entry_reason=entry_reason,
                        entry_regime=str(bar["regime"]),
                        atr_at_entry=float(bar.get("atr", 0.0) or 0.0),
                        equity_at_entry=equity,
                        risk_amount_usd=risk_decision.risk_amount_usd,
                        strategy_scores=composite.strategy_scores,
                        contributing_strategies=contributing,
                        signal_strength=composite.final_signal,
                        highest_price_seen=entry_price,
                        lowest_price_seen=entry_price,
                    )
                    trade_events.append(
                        TradeEventRecord(
                            trade_id=trade_id,
                            position_type=side,
                            step_type="Entry",
                            timestamp=timestamp,
                            signal_reason=entry_reason,
                            price=entry_price,
                            size=float(risk_decision.size_units),
                            net_pnl_usd=0.0,
                            net_pnl_pct=0.0,
                            cumulative_pnl_usd=realized_pnl,
                            cumulative_pnl_pct=realized_pnl / risk_config.initial_capital,
                            favorite_excursion_pct=0.0,
                            adverse_excursion_pct=0.0,
                        )
                    )

            unrealized_pnl = self._unrealized_pnl(position, close_price, quote_to_usd) if position else 0.0
            equity_rows.append(
                {
                    "timestamp": timestamp,
                    "equity": risk_config.initial_capital + realized_pnl + unrealized_pnl,
                    "realized_pnl_usd": realized_pnl,
                    "unrealized_pnl_usd": unrealized_pnl,
                    "position_side": position.side if position else "Flat",
                    "regime": str(bar["regime"]),
                    "close": close_price,
                    "quote_to_usd": quote_to_usd,
                }
            )

            if next_checkpoint_index < len(progress_checkpoints) and processed_bars >= progress_checkpoints[next_checkpoint_index]:
                self._emit_progress(
                    processed_bars=processed_bars,
                    total_bars=total_replay_bars,
                    trade_count=len(trades),
                    equity=risk_config.initial_capital + realized_pnl + unrealized_pnl,
                    timestamp=timestamp,
                )
                next_checkpoint_index += 1

        if position is not None:
            final_bar = market_data.iloc[-1]
            final_quote_to_usd = self._sanitize_quote_to_usd(final_bar.get("quote_to_usd"), pair, float(final_bar["close"]))
            realized_pnl, position = self._close_position(
                position=position,
                exit_price=float(final_bar["close"]),
                exit_time=pd.Timestamp(final_bar["timestamp"]).to_pydatetime(),
                exit_reason="End Of Data",
                exit_regime=str(final_bar["regime"]),
                quote_to_usd=final_quote_to_usd,
                realized_pnl=realized_pnl,
                trade_events=trade_events,
                trades=trades,
                tracker=tracker,
                risk_config=risk_config,
            )

        self._emit_status("Backtest replay complete. Computing metrics and writing CSV artifacts...")
        equity_curve = pd.DataFrame(equity_rows)
        risk_metrics = pd.DataFrame(risk_rows)
        strategy_metrics = tracker.metrics_frame(current_regime=str(market_data.iloc[-1]["regime"]))

        report = compile_backtest_report(
            pair=pair,
            timeframe=self.config.timeframe,
            initial_capital=risk_config.initial_capital,
            trades=trades,
            trade_events=trade_events,
            equity_curve=equity_curve,
            market_data=market_data,
            strategy_metrics=strategy_metrics,
            risk_metrics=risk_metrics,
        )
        summary_text = SummaryPrinter.render(report)
        logger.persist_report(report, summary_text)
        report["summary_text"] = summary_text
        report["output_dir"] = logger.run_dir
        return report

    def _strategy_config_for_pair(self, pair: str) -> StrategyConfig:
        overrides = self.config.pair_specific_config.get(pair.upper(), {}).get("strategy", {})
        return replace(self.config.strategy, **overrides) if overrides else self.config.strategy

    def _risk_config_for_pair(self, pair: str) -> RiskConfig:
        overrides = self.config.pair_specific_config.get(pair.upper(), {}).get("risk", {})
        return replace(self.config.risk, **overrides) if overrides else self.config.risk

    def _emit_status(self, message: str) -> None:
        if self.config.summary_to_stdout:
            print(message, flush=True)

    def _emit_progress(self, processed_bars: int, total_bars: int, trade_count: int, equity: float, timestamp) -> None:
        if not self.config.summary_to_stdout or total_bars <= 0:
            return
        progress_pct = processed_bars / total_bars
        print(
            "Backtest progress: "
            f"{progress_pct:.0%} ({processed_bars:,}/{total_bars:,} bars, trades={trade_count}, "
            f"equity={equity:,.2f} USD, last_bar={timestamp:%Y-%m-%d %H:%M UTC})",
            flush=True,
        )

    @staticmethod
    def _progress_checkpoints(total_bars: int) -> list[int]:
        if total_bars < 10_000:
            return []
        return sorted({max(1, int(total_bars * pct / 100)) for pct in range(10, 100, 10)})

    def _within_trading_session(self, timestamp, strategy_config: StrategyConfig) -> bool:
        timeframe_minutes = TIMEFRAME_TO_MINUTES.get(self.config.timeframe, 0)
        if timeframe_minutes >= TIMEFRAME_TO_MINUTES["4h"]:
            return True
        session_hour = pd.Timestamp(timestamp).hour
        start_hour = strategy_config.start_trading_hour
        end_hour = strategy_config.end_trading_hour
        if start_hour == end_hour:
            return True
        if start_hour < end_hour:
            return start_hour <= session_hour < end_hour
        return session_hour >= start_hour or session_hour < end_hour

    @staticmethod
    def _sanitize_quote_to_usd(value, pair: str, close_price: float) -> float:
        base, quote = split_pair(pair)
        try:
            rate = float(value)
        except (TypeError, ValueError):
            rate = float("nan")
        if not pd.isna(rate) and rate > 0:
            return rate
        if quote == "USD":
            return 1.0
        if base == "USD" and close_price > 0:
            return 1.0 / close_price
        return 0.0

    def _open_risk_ratio(self, position: Position | None, quote_to_usd: float, realized_pnl: float, risk_config: RiskConfig) -> float:
        if position is None:
            return 0.0
        equity = risk_config.initial_capital + realized_pnl
        if equity <= 0:
            return 1.0
        stop_distance = abs(position.entry_price - position.stop_price)
        risk_usd = stop_distance * position.size_units * quote_to_usd
        return risk_usd / equity

    def _update_position_state(self, position: Position, bar: pd.Series) -> None:
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

    def _check_stop(self, position: Position, bar: pd.Series) -> tuple[bool, float]:
        if position.side == "Long" and float(bar["low"]) <= position.stop_price:
            return True, position.stop_price
        if position.side == "Short" and float(bar["high"]) >= position.stop_price:
            return True, position.stop_price
        return False, position.stop_price

    def _check_take_profit(self, position: Position, bar: pd.Series) -> tuple[bool, float]:
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
        slippage = self.config.slippage_bps / 10_000
        return price * (1 + slippage * bias)

    def _unrealized_pnl(self, position: Position | None, close_price: float, quote_to_usd: float) -> float:
        if position is None:
            return 0.0
        pnl_quote = self._pnl_quote(position.side, position.size_units, position.entry_price, close_price)
        return pnl_quote * quote_to_usd

    @staticmethod
    def _pnl_quote(side: str, size_units: int, entry_price: float, exit_price: float) -> float:
        if side == "Long":
            return size_units * (exit_price - entry_price)
        return size_units * (entry_price - exit_price)

    def _close_position(
        self,
        position: Position,
        exit_price: float,
        exit_time,
        exit_reason: str,
        exit_regime: str,
        quote_to_usd: float,
        realized_pnl: float,
        trade_events: list[TradeEventRecord],
        trades: list[TradeRecord],
        tracker: PerformanceTracker,
        risk_config: RiskConfig,
    ) -> tuple[float, None]:
        exit_price = self._apply_slippage(exit_price, -1 if position.side == "Long" else 1)
        pnl_quote = self._pnl_quote(position.side, position.size_units, position.entry_price, exit_price)
        pnl_usd = pnl_quote * quote_to_usd - risk_config.commission_per_trade_usd
        realized_pnl += pnl_usd
        pnl_pct = pnl_usd / position.equity_at_entry if position.equity_at_entry else 0.0
        cumulative_pct = realized_pnl / risk_config.initial_capital if risk_config.initial_capital else 0.0
        trade = TradeRecord(
            trade_id=position.trade_id,
            pair=position.pair,
            side=position.side,
            entry_time=position.entry_time,
            exit_time=exit_time,
            entry_price=position.entry_price,
            exit_price=exit_price,
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
            commission_paid_usd=risk_config.commission_per_trade_usd,
            signal_strength=position.signal_strength,
            equity_at_entry=position.equity_at_entry,
            strategy_scores=position.strategy_scores,
        )
        trades.append(trade)
        trade_events.append(
            TradeEventRecord(
                trade_id=position.trade_id,
                position_type=position.side,
                step_type="Exit",
                timestamp=exit_time,
                signal_reason=exit_reason,
                price=exit_price,
                size=float(position.size_units),
                net_pnl_usd=pnl_usd,
                net_pnl_pct=pnl_pct,
                cumulative_pnl_usd=realized_pnl,
                cumulative_pnl_pct=cumulative_pct,
                favorite_excursion_pct=position.max_favorable_excursion_pct,
                adverse_excursion_pct=position.max_adverse_excursion_pct,
            )
        )
        tracker.record_trade(
            strategy_names=position.contributing_strategies,
            pair=position.pair,
            side=position.side,
            entry_time=position.entry_time,
            exit_time=exit_time,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            regime=position.entry_regime,
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
        names = [
            name
            for name, decision in composite.strategy_decisions.items()
            if decision.signal == bias and composite.strategy_scores.get(name, 0.0) > 0
        ]
        return names or [max(composite.strategy_scores, key=composite.strategy_scores.get)]


