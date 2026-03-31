from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from forex_quant_bot.models import StrategyPerformanceSnapshot, StrategyTradeRecord
from forex_quant_bot.utils.math_utils import profit_factor, safe_div


@dataclass(slots=True)
class PerformanceTracker:
    window: int = 100
    trades_by_strategy: dict[str, deque[StrategyTradeRecord]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.trades_by_strategy = defaultdict(lambda: deque(maxlen=self.window))

    def record_trade(
        self,
        strategy_names: list[str],
        pair: str,
        side: str,
        entry_time,
        exit_time,
        pnl_usd: float,
        pnl_pct: float,
        regime: str,
    ) -> None:
        record = StrategyTradeRecord(
            strategy_name="",
            pair=pair,
            side=side,
            entry_time=entry_time,
            exit_time=exit_time,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            regime=regime,
        )
        for name in strategy_names:
            self.trades_by_strategy[name].append(
                StrategyTradeRecord(
                    strategy_name=name,
                    pair=record.pair,
                    side=record.side,
                    entry_time=record.entry_time,
                    exit_time=record.exit_time,
                    pnl_usd=record.pnl_usd,
                    pnl_pct=record.pnl_pct,
                    regime=record.regime,
                )
            )

    def snapshot(self, strategy_name: str, current_regime: str) -> StrategyPerformanceSnapshot:
        trades = list(self.trades_by_strategy[strategy_name])
        if not trades:
            return StrategyPerformanceSnapshot(
                trade_count=0,
                win_rate=0.5,
                avg_pnl_usd=0.0,
                avg_pnl_pct=0.0,
                profit_factor=1.0,
                max_drawdown_pct=0.0,
                regime_fit=0.5,
            )

        pnl_usd = np.array([trade.pnl_usd for trade in trades], dtype=float)
        pnl_pct = np.array([trade.pnl_pct for trade in trades], dtype=float)
        wins = pnl_usd > 0
        equity = np.cumsum(pnl_usd)
        peaks = np.maximum.accumulate(equity)
        drawdowns = safe_div_array(equity - peaks, np.where(peaks == 0, np.nan, peaks))
        regime_trades = [trade for trade in trades if trade.regime == current_regime]
        if regime_trades:
            regime_fit = np.mean([1.0 if trade.pnl_usd > 0 else 0.0 for trade in regime_trades])
        else:
            regime_fit = float(np.mean(wins)) if wins.size else 0.5

        return StrategyPerformanceSnapshot(
            trade_count=len(trades),
            win_rate=float(np.mean(wins)) if wins.size else 0.5,
            avg_pnl_usd=float(np.mean(pnl_usd)) if pnl_usd.size else 0.0,
            avg_pnl_pct=float(np.mean(pnl_pct)) if pnl_pct.size else 0.0,
            profit_factor=profit_factor(pnl_usd),
            max_drawdown_pct=float(abs(np.nanmin(drawdowns))) if drawdowns.size else 0.0,
            regime_fit=float(np.clip(regime_fit, 0.0, 1.0)),
        )

    def metrics_frame(self, current_regime: str) -> pd.DataFrame:
        rows = []
        for name in sorted(self.trades_by_strategy):
            snapshot = self.snapshot(name, current_regime)
            rows.append(
                {
                    "strategy": name,
                    "trade_count": snapshot.trade_count,
                    "win_rate": snapshot.win_rate,
                    "avg_pnl_usd": snapshot.avg_pnl_usd,
                    "avg_pnl_pct": snapshot.avg_pnl_pct,
                    "profit_factor": snapshot.profit_factor,
                    "max_drawdown_pct": snapshot.max_drawdown_pct,
                    "regime_fit": snapshot.regime_fit,
                }
            )
        return pd.DataFrame(rows)



def safe_div_array(numerator, denominator):
    result = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=np.isfinite(denominator) & (denominator != 0))
    return result
