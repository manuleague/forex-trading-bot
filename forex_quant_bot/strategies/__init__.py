from .base_strategy import BaseStrategy
from .breakout_strategy import BreakoutStrategy
from .mean_reversion_strategy import MeanReversionStrategy
from .trend_strategy import TrendStrategy


def build_default_strategies(strategy_config):
    return [
        TrendStrategy(strategy_config),
        MeanReversionStrategy(strategy_config),
        BreakoutStrategy(strategy_config),
    ]
