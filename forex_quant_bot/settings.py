from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PAIR_SPECIFIC_OVERRIDES: dict[str, dict[str, dict[str, Any]]] = {
    "USDJPY": {
        "risk": {
            "take_profit_atr_multiplier": 5.0,
            "trailing_stop_activation": 3.5,
        }
    },
    "EURUSD": {
        "strategy": {
            "score_threshold": 0.50,
        }
    },
    "GBPUSD": {
        "strategy": {
            "score_threshold": 0.60,
            "adx_threshold": 30.0,
        },
        "risk": {
            "take_profit_atr_multiplier": 3.5,
            "atr_stop_multiplier": 2.2,
        },
    },
}


@dataclass(slots=True)
class StrategyConfig:
    trend_fast_ema: int = 20
    trend_slow_ema: int = 50
    trend_adx_threshold: float = 20.0
    mean_rsi_period: int = 14
    mean_bollinger_period: int = 20
    mean_bollinger_std: float = 2.0
    breakout_lookback: int = 20
    breakout_atr_period: int = 14
    breakout_atr_expansion: float = 1.1
    score_threshold: float = 0.50
    min_strategy_confidence: float = 0.20
    recent_trade_window: int = 100
    adaptive_warmup_floor: int = 15
    daily_warmup_scale: float = 0.35
    weekly_warmup_scale: float = 0.25
    start_trading_hour: int = 8
    end_trading_hour: int = 20


@dataclass(slots=True)
class RiskConfig:
    initial_capital: float = 10_000.0
    risk_per_trade: float = 0.01
    max_total_exposure: float = 0.05
    drawdown_circuit_breaker: float = 0.05
    circuit_breaker_cooldown_bars: int = 50
    circuit_breaker_recovery_threshold: float = 0.97
    atr_stop_multiplier: float = 1.5
    take_profit_atr_multiplier: float = 5.0
    break_even_atr_multiplier: float = 1.5
    break_even_buffer_atr_multiplier: float = 0.2
    trailing_activation_atr_multiplier: float = 3.0
    trailing_stop_atr_multiplier: float = 2.0
    trailing_stop_step_atr_multiplier: float = 0.5
    minimum_atr_ratio: float = 0.80
    liquidity_spike_range_multiplier: float = 2.5
    max_positions_per_currency: int = 2
    max_concurrent_positions: int = 1
    commission_per_trade_usd: float = 0.0
    max_notional_leverage: float = 1.0


@dataclass(slots=True)
class LoggingConfig:
    output_dir: Path = Path("output")
    write_summary_txt: bool = True


@dataclass(slots=True)
class DataConfig:
    cache_dir: Path = Path("data_cache")
    provider: str = "dukascopy"
    auto_download: bool = True
    sample_rows: int = 4_000


@dataclass(slots=True)
class IBConfig:
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 7
    account: str | None = None
    paper_only: bool = True
    allowed_paper_ports: tuple[int, ...] = (7497, 4002)


@dataclass(slots=True)
class BotConfig:
    mode: str
    pairs: list[str]
    timeframe: str
    start: datetime | None = None
    end: datetime | None = None
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    ib: IBConfig = field(default_factory=IBConfig)
    pair_specific_config: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    use_sample_data: bool = False
    slippage_bps: float = 0.0
    summary_to_stdout: bool = True

    @property
    def primary_pair(self) -> str:
        return self.pairs[0]


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _normalize_pair_specific_config(raw: dict[str, Any] | None) -> dict[str, dict[str, dict[str, Any]]]:
    if not raw:
        return {}

    strategy_fields = set(StrategyConfig.__dataclass_fields__.keys())
    risk_fields = set(RiskConfig.__dataclass_fields__.keys())
    strategy_aliases = {
        "adx_threshold": "trend_adx_threshold",
    }
    risk_aliases = {
        "trailing_stop_activation": "trailing_activation_atr_multiplier",
    }
    normalized: dict[str, dict[str, dict[str, Any]]] = {}

    for pair, overrides in raw.items():
        if not isinstance(overrides, dict):
            continue
        pair_key = str(pair).upper()
        pair_payload: dict[str, dict[str, Any]] = {"strategy": {}, "risk": {}}

        strategy_overrides = overrides.get("strategy", {})
        if isinstance(strategy_overrides, dict):
            for key, value in strategy_overrides.items():
                mapped = strategy_aliases.get(key, key)
                pair_payload["strategy"][mapped] = value

        risk_overrides = overrides.get("risk", {})
        if isinstance(risk_overrides, dict):
            for key, value in risk_overrides.items():
                mapped = risk_aliases.get(key, key)
                pair_payload["risk"][mapped] = value

        for key, value in overrides.items():
            if key in {"strategy", "risk"}:
                continue
            mapped_strategy = strategy_aliases.get(key, key)
            mapped_risk = risk_aliases.get(key, key)
            if mapped_strategy in strategy_fields:
                pair_payload["strategy"][mapped_strategy] = value
            elif mapped_risk in risk_fields:
                pair_payload["risk"][mapped_risk] = value

        normalized[pair_key] = {section: values for section, values in pair_payload.items() if values}

    return normalized


def _merge_pair_specific_config(
    base: dict[str, dict[str, dict[str, Any]]],
    overrides: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    merged = deepcopy(base)
    for pair, sections in overrides.items():
        target = merged.setdefault(pair, {})
        for section, values in sections.items():
            target.setdefault(section, {})
            target[section].update(values)
    return merged


def build_config_from_args(args: Any) -> BotConfig:
    strategy_yaml = load_yaml_config(Path(args.strategy_config)) if getattr(args, "strategy_config", None) else {}
    pairs_yaml = load_yaml_config(Path(args.pairs_config)) if getattr(args, "pairs_config", None) else {}
    risk_yaml = strategy_yaml.get("risk", {})
    pair_specific_yaml = _normalize_pair_specific_config(strategy_yaml.get("pair_specific", {}))

    strategy = StrategyConfig(
        **strategy_yaml.get("strategy", {}),
    )
    risk = RiskConfig(
        initial_capital=getattr(args, "initial_capital", None) or risk_yaml.get("initial_capital", 10_000.0),
        risk_per_trade=getattr(args, "risk_per_trade", None) or risk_yaml.get("risk_per_trade", 0.01),
        max_total_exposure=getattr(args, "max_total_exposure", None) or risk_yaml.get("max_total_exposure", 0.05),
        drawdown_circuit_breaker=risk_yaml.get("drawdown_circuit_breaker", 0.05),
        circuit_breaker_cooldown_bars=risk_yaml.get("circuit_breaker_cooldown_bars", 50),
        circuit_breaker_recovery_threshold=risk_yaml.get("circuit_breaker_recovery_threshold", 0.97),
        atr_stop_multiplier=getattr(args, "atr_stop_multiplier", None) or risk_yaml.get("atr_stop_multiplier", 1.5),
        take_profit_atr_multiplier=risk_yaml.get("take_profit_atr_multiplier", 5.0),
        break_even_atr_multiplier=risk_yaml.get("break_even_atr_multiplier", 1.5),
        break_even_buffer_atr_multiplier=risk_yaml.get("break_even_buffer_atr_multiplier", 0.2),
        trailing_activation_atr_multiplier=risk_yaml.get("trailing_activation_atr_multiplier", 3.0),
        trailing_stop_atr_multiplier=risk_yaml.get("trailing_stop_atr_multiplier", 2.0),
        trailing_stop_step_atr_multiplier=risk_yaml.get("trailing_stop_step_atr_multiplier", 0.5),
        minimum_atr_ratio=risk_yaml.get("minimum_atr_ratio", 0.80),
        liquidity_spike_range_multiplier=risk_yaml.get("liquidity_spike_range_multiplier", 2.5),
        max_positions_per_currency=risk_yaml.get("max_positions_per_currency", 2),
        max_concurrent_positions=risk_yaml.get("max_concurrent_positions", 1),
        commission_per_trade_usd=risk_yaml.get("commission_per_trade_usd", 0.0),
        max_notional_leverage=getattr(args, "max_notional_leverage", None) or risk_yaml.get("max_notional_leverage", 1.0),
    )

    pairs: list[str] = []
    cli_pairs = getattr(args, "pair", None) or getattr(args, "pairs", None)
    if cli_pairs:
        for item in str(cli_pairs).split(","):
            symbol = item.strip().upper()
            if symbol and symbol not in pairs:
                pairs.append(symbol)
    else:
        for item in pairs_yaml.get("pairs", []):
            symbol = str(item.get("symbol", "")).strip().upper()
            if symbol and symbol not in pairs:
                pairs.append(symbol)
    if not pairs:
        raise ValueError("At least one forex pair must be provided via --pair/--pairs or config/pairs.yaml.")

    logging_cfg = LoggingConfig(output_dir=Path(getattr(args, "output_dir", "output")))
    data_cfg = DataConfig(
        cache_dir=Path(getattr(args, "cache_dir", "data_cache")),
        provider=getattr(args, "provider", "dukascopy"),
        auto_download=not getattr(args, "no_auto_download", False),
        sample_rows=getattr(args, "sample_rows", 4_000),
    )
    ib_cfg = IBConfig(
        host=getattr(args, "ib_host", "127.0.0.1"),
        port=getattr(args, "ib_port", 7497),
        client_id=getattr(args, "client_id", 7),
        account=getattr(args, "account", None),
        paper_only=not getattr(args, "allow_live_ib", False),
    )

    return BotConfig(
        mode=args.mode,
        pairs=pairs,
        timeframe=args.timeframe,
        start=_parse_datetime(getattr(args, "start", None)),
        end=_parse_datetime(getattr(args, "end", None)),
        strategy=strategy,
        risk=risk,
        logging=logging_cfg,
        data=data_cfg,
        ib=ib_cfg,
        pair_specific_config=_merge_pair_specific_config(_normalize_pair_specific_config(PAIR_SPECIFIC_OVERRIDES), pair_specific_yaml),
        use_sample_data=getattr(args, "use_sample_data", False),
        slippage_bps=getattr(args, "slippage_bps", 0.0),
        summary_to_stdout=not getattr(args, "quiet", False),
    )
