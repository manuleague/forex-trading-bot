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

    def __post_init__(self) -> None:
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
            composite = 0.4 * performance_score + 0.3 * raw_confidence + 0.2 * regime_fit + 0.1 * diversification
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

    @staticmethod
    def _performance_score(snapshot) -> float:
        pf_component = 1.0 if snapshot.profit_factor == float("inf") else np.tanh(max(snapshot.profit_factor - 1.0, -1.0)) * 0.5 + 0.5
        avg_pnl_component = np.tanh(snapshot.avg_pnl_pct * 50.0) * 0.5 + 0.5
        drawdown_penalty = max(0.0, 1.0 - snapshot.max_drawdown_pct)
        return float(np.clip(0.45 * snapshot.win_rate + 0.25 * pf_component + 0.20 * avg_pnl_component + 0.10 * drawdown_penalty, 0.0, 1.0))

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
