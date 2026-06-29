from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from forex_quant_bot.models import CompositeDecision, StrategyDecision
from forex_quant_bot.utils.math_utils import normalize_scores


@dataclass(slots=True)
class StrategyAllocator:
    score_threshold: float
    min_strategy_confidence: float
    history_window: int = 200
    signal_history: dict[str, deque[int]] = field(default_factory=dict)
    w_performance: float = 0.4
    w_confidence: float = 0.3
    w_regime_fit: float = 0.2
    w_diversification: float = 0.1
    w_win_rate: float = 0.45
    w_profit_factor: float = 0.25
    w_avg_pnl: float = 0.20
    w_drawdown: float = 0.10

    def __post_init__(self) -> None:
        self._validate_weights()
        self.signal_history = defaultdict(lambda: deque(maxlen=self.history_window))

    def allocate(self, decisions: dict[str, StrategyDecision], current_regime: str, performance_tracker) -> CompositeDecision:
        raw_scores: dict[str, float] = {}
        components: dict[str, dict[str, float]] = {}

        for name, decision in decisions.items():
            snapshot = performance_tracker.snapshot(name, current_regime)
            performance_score = self._performance_score(snapshot)
            raw_confidence = decision.confidence if decision.confidence >= self.min_strategy_confidence else 0.0
            regime_fit = snapshot.regime_fit
            diversification = self._diversification_bonus(name)
            composite = (
                self.w_performance * performance_score
                + self.w_confidence * raw_confidence
                + self.w_regime_fit * regime_fit
                + self.w_diversification * diversification
            )
            if decision.signal == 0:
                composite *= 0.25
            raw_scores[name] = composite
            components[name] = {
                "performance": performance_score,
                "confidence": raw_confidence,
                "regime_fit": regime_fit,
                "diversification": diversification,
                "composite": composite,
            }

        scores = normalize_scores(raw_scores)
        final_signal = float(sum(decisions[name].signal * scores[name] for name in decisions))
        if final_signal > self.score_threshold:
            bias = 1
        elif final_signal < -self.score_threshold:
            bias = -1
        else:
            bias = 0

        for name, decision in decisions.items():
            self.signal_history[name].append(int(decision.signal))

        return CompositeDecision(
            final_signal=final_signal,
            bias=bias,
            strategy_scores=scores,
            strategy_decisions=decisions,
            metadata={"regime": current_regime, "components": components},
        )

    def _performance_score(self, snapshot) -> float:
        pf_component = 1.0 if snapshot.profit_factor == float("inf") else np.tanh(max(snapshot.profit_factor - 1.0, -1.0)) * 0.5 + 0.5
        avg_pnl_component = np.tanh(snapshot.avg_pnl_pct * 50.0) * 0.5 + 0.5
        drawdown_penalty = max(0.0, 1.0 - snapshot.max_drawdown_pct)
        return float(
            np.clip(
                self.w_win_rate * snapshot.win_rate
                + self.w_profit_factor * pf_component
                + self.w_avg_pnl * avg_pnl_component
                + self.w_drawdown * drawdown_penalty,
                0.0,
                1.0,
            )
        )

    def _diversification_bonus(self, strategy_name: str) -> float:
        history = self.signal_history[strategy_name]
        if len(history) < 20:
            return 0.5

        correlations = []
        base = np.asarray(history, dtype=float)
        for other_name, other_history in self.signal_history.items():
            if other_name == strategy_name or len(other_history) < 20:
                continue
            other = np.asarray(other_history, dtype=float)
            size = min(len(base), len(other))
            if size < 20:
                continue
            lhs = base[-size:]
            rhs = other[-size:]
            if np.std(lhs) == 0 or np.std(rhs) == 0:
                corr = 0.0
            else:
                corr = float(np.corrcoef(lhs, rhs)[0, 1])
                if not np.isfinite(corr):
                    corr = 0.0
            correlations.append(corr)
        if not correlations:
            return 0.5
        avg_corr = float(np.mean(correlations))
        return float(np.clip((1.0 - avg_corr) / 2.0, 0.0, 1.0))

    def _validate_weights(self) -> None:
        composite_total = self.w_performance + self.w_confidence + self.w_regime_fit + self.w_diversification
        performance_total = self.w_win_rate + self.w_profit_factor + self.w_avg_pnl + self.w_drawdown
        if not np.isclose(composite_total, 1.0):
            raise ValueError(f"Allocator composite weights must sum to 1.0, got {composite_total:.4f}.")
        if not np.isclose(performance_total, 1.0):
            raise ValueError(f"Allocator performance weights must sum to 1.0, got {performance_total:.4f}.")
