from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class StrategyDecision:
    name: str
    signal: int
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RegimeSnapshot:
    label: str
    adx: float
    atr: float
    hurst: float
    volatility_zscore: float


@dataclass(slots=True)
class StrategyTradeRecord:
    strategy_name: str
    pair: str
    side: str
    entry_time: datetime
    exit_time: datetime
    pnl_usd: float
    pnl_pct: float
    regime: str


@dataclass(slots=True)
class StrategyPerformanceSnapshot:
    trade_count: int
    win_rate: float
    avg_pnl_usd: float
    avg_pnl_pct: float
    profit_factor: float
    max_drawdown_pct: float
    regime_fit: float


@dataclass(slots=True)
class CompositeDecision:
    final_signal: float
    bias: int
    strategy_scores: dict[str, float]
    strategy_decisions: dict[str, StrategyDecision]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RiskDecision:
    approved: bool
    size_units: int
    stop_distance: float
    stop_price: float
    risk_amount_usd: float
    exposure_after_trade: float
    reason: str


@dataclass(slots=True)
class Position:
    trade_id: int
    pair: str
    side: str
    size_units: int
    entry_time: datetime
    entry_price: float
    stop_price: float
    take_profit_price: float
    entry_reason: str
    entry_regime: str
    atr_at_entry: float
    equity_at_entry: float
    risk_amount_usd: float
    strategy_scores: dict[str, float]
    contributing_strategies: list[str]
    signal_strength: float
    highest_price_seen: float = 0.0
    lowest_price_seen: float = 0.0
    break_even_armed: bool = False
    trailing_stop_active: bool = False
    trailing_step_count: int = 0
    bars_held: int = 0
    max_favorable_excursion_pct: float = 0.0
    max_adverse_excursion_pct: float = 0.0


@dataclass(slots=True)
class TradeRecord:
    trade_id: int
    pair: str
    side: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    size_units: int
    pnl_quote: float
    pnl_usd: float
    pnl_pct: float
    cumulative_pnl_usd: float
    cumulative_pnl_pct: float
    entry_reason: str
    exit_reason: str
    entry_regime: str
    exit_regime: str
    bars_held: int
    max_favorable_excursion_pct: float
    max_adverse_excursion_pct: float
    commission_paid_usd: float
    signal_strength: float
    equity_at_entry: float
    strategy_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["entry_time"] = self.entry_time.isoformat()
        payload["exit_time"] = self.exit_time.isoformat()
        payload["position_type"] = self.side
        return payload


@dataclass(slots=True)
class TradeEventRecord:
    trade_id: int
    position_type: str
    step_type: str
    timestamp: datetime
    signal_reason: str
    price: float
    size: float
    net_pnl_usd: float
    net_pnl_pct: float
    cumulative_pnl_usd: float
    cumulative_pnl_pct: float
    favorite_excursion_pct: float = 0.0
    adverse_excursion_pct: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload


@dataclass(slots=True)
class OrderEvent:
    timestamp: datetime
    pair: str
    action: str
    size_units: int
    order_type: str
    status: str
    price: float | None = None
    order_id: str | None = None
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload
