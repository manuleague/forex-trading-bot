from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from forex_quant_bot.models import Position, RiskDecision
from forex_quant_bot.settings import RiskConfig
from forex_quant_bot.utils.time_utils import split_pair


@dataclass(slots=True)
class RiskOverlay:
    config: RiskConfig
    equity_peak: float = 0.0
    trading_enabled: bool = True
    circuit_breaker_active: bool = False
    cooldown_bars_remaining: int = 0
    last_drawdown_pct: float = 0.0

    def __post_init__(self) -> None:
        self.equity_peak = self.config.initial_capital

    def update_equity(self, equity: float) -> None:
        self.equity_peak = max(self.equity_peak, equity)
        if self.equity_peak <= 0:
            return

        drawdown_pct = max(0.0, 1.0 - (equity / self.equity_peak))
        recovery_ratio = equity / self.equity_peak if self.equity_peak else 0.0
        breached_now = self.last_drawdown_pct < self.config.drawdown_circuit_breaker <= drawdown_pct

        if not self.circuit_breaker_active and breached_now:
            self.trading_enabled = False
            self.circuit_breaker_active = True
            self.cooldown_bars_remaining = max(self.config.circuit_breaker_cooldown_bars, 0)
        elif self.circuit_breaker_active:
            if self.cooldown_bars_remaining > 0:
                self.cooldown_bars_remaining -= 1
            recovery_ready = recovery_ratio >= self.config.circuit_breaker_recovery_threshold
            cooldown_complete = self.cooldown_bars_remaining <= 0
            if cooldown_complete and recovery_ready:
                self.trading_enabled = True
                self.circuit_breaker_active = False
                self.cooldown_bars_remaining = 0

        self.last_drawdown_pct = drawdown_pct

    def evaluate_entry(
        self,
        bias: int,
        equity: float,
        current_open_risk_ratio: float,
        close_price: float,
        atr_value: float,
        quote_to_usd_rate: float,
        pair: str = "",
        open_positions: list[Position] | None = None,
        current_bar_range: float = 0.0,
        average_bar_range: float = 0.0,
        average_atr_value: float = 0.0,
    ) -> RiskDecision:
        open_positions = open_positions or []

        if not self.trading_enabled:
            reason = "circuit_breaker_recovery" if self.circuit_breaker_active else "circuit_breaker"
            return RiskDecision(False, 0, 0.0, close_price, 0.0, current_open_risk_ratio, reason)
        if bias == 0:
            return RiskDecision(False, 0, 0.0, close_price, 0.0, current_open_risk_ratio, "flat_signal")
        if atr_value <= 0 or quote_to_usd_rate <= 0 or close_price <= 0 or equity <= 0:
            return RiskDecision(False, 0, 0.0, close_price, 0.0, current_open_risk_ratio, "invalid_atr_or_price")
        if average_atr_value > 0 and atr_value < average_atr_value * self.config.minimum_atr_ratio:
            return RiskDecision(False, 0, 0.0, close_price, 0.0, current_open_risk_ratio, "quiet_market_atr")
        if average_bar_range > 0 and current_bar_range > average_bar_range * self.config.liquidity_spike_range_multiplier:
            return RiskDecision(False, 0, 0.0, close_price, 0.0, current_open_risk_ratio, "liquidity_range_spike")

        if pair:
            blocked_currency = self._blocked_currency(pair, open_positions)
            if blocked_currency is not None:
                return RiskDecision(False, 0, 0.0, close_price, 0.0, current_open_risk_ratio, f"currency_exposure_{blocked_currency}")

        target_risk_amount = equity * self.config.risk_per_trade
        stop_distance = atr_value * self.config.atr_stop_multiplier
        usd_risk_per_unit = stop_distance * quote_to_usd_rate
        if usd_risk_per_unit <= 0:
            return RiskDecision(False, 0, stop_distance, close_price, target_risk_amount, current_open_risk_ratio, "invalid_usd_risk")

        usd_notional_per_unit = close_price * quote_to_usd_rate
        if usd_notional_per_unit <= 0:
            return RiskDecision(False, 0, stop_distance, close_price, target_risk_amount, current_open_risk_ratio, "invalid_notional")

        size_by_risk = int(max(target_risk_amount / usd_risk_per_unit, 1))
        max_notional_usd = equity * self.config.max_notional_leverage
        size_by_notional = int(max(max_notional_usd / usd_notional_per_unit, 0))
        if size_by_notional < 1:
            return RiskDecision(False, 0, stop_distance, close_price, target_risk_amount, current_open_risk_ratio, "max_notional_leverage")

        size_units = min(size_by_risk, size_by_notional)
        actual_risk_amount = size_units * usd_risk_per_unit
        open_risk_after_trade = current_open_risk_ratio + (actual_risk_amount / equity)
        if open_risk_after_trade > self.config.max_total_exposure:
            return RiskDecision(False, 0, stop_distance, close_price, actual_risk_amount, open_risk_after_trade, "max_exposure")

        stop_price = close_price - bias * stop_distance
        return RiskDecision(True, size_units, stop_distance, stop_price, actual_risk_amount, open_risk_after_trade, "approved")

    def _blocked_currency(self, pair: str, open_positions: list[Position]) -> str | None:
        base, quote = split_pair(pair)
        counts = self._currency_counts(open_positions)
        if counts.get(base, 0) >= self.config.max_positions_per_currency:
            return base
        if counts.get(quote, 0) >= self.config.max_positions_per_currency:
            return quote
        return None

    @staticmethod
    def _currency_counts(open_positions: list[Position]) -> Counter[str]:
        counts: Counter[str] = Counter()
        for position in open_positions:
            base, quote = split_pair(position.pair)
            counts[base] += 1
            counts[quote] += 1
        return counts
