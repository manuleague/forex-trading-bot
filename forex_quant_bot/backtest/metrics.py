from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from forex_quant_bot.models import TradeEventRecord, TradeRecord
from forex_quant_bot.utils.math_utils import max_drawdown, profit_factor, sharpe_ratio, sortino_ratio
from forex_quant_bot.utils.time_utils import annualization_factor



def compile_backtest_report(
    pair: str,
    timeframe: str,
    initial_capital: float,
    trades: list[TradeRecord],
    trade_events: list[TradeEventRecord],
    equity_curve: pd.DataFrame,
    market_data: pd.DataFrame,
    strategy_metrics: pd.DataFrame,
    risk_metrics: pd.DataFrame,
) -> dict[str, Any]:
    trades_df = pd.DataFrame([trade.to_dict() for trade in trades])
    events_df = pd.DataFrame([event.to_dict() for event in trade_events])
    if trades_df.empty:
        trades_df = pd.DataFrame(
            columns=[
                "trade_id",
                "pair",
                "side",
                "entry_time",
                "exit_time",
                "entry_price",
                "exit_price",
                "size_units",
                "pnl_quote",
                "pnl_usd",
                "pnl_pct",
                "cumulative_pnl_usd",
                "cumulative_pnl_pct",
                "entry_reason",
                "exit_reason",
                "entry_regime",
                "exit_regime",
                "bars_held",
                "max_favorable_excursion_pct",
                "max_adverse_excursion_pct",
                "commission_paid_usd",
                "signal_strength",
                "equity_at_entry",
                "strategy_scores",
            ]
        )
    if events_df.empty:
        events_df = pd.DataFrame(
            columns=[
                "trade_id",
                "position_type",
                "step_type",
                "timestamp",
                "signal_reason",
                "price",
                "size",
                "net_pnl_usd",
                "net_pnl_pct",
                "cumulative_pnl_usd",
                "cumulative_pnl_pct",
                "favorite_excursion_pct",
                "adverse_excursion_pct",
            ]
        )

    if not trades_df.empty:
        trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"], utc=True)
        trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"], utc=True)
    if not events_df.empty:
        events_df["timestamp"] = pd.to_datetime(events_df["timestamp"], utc=True)

    summary = _build_summary(trades_df, equity_curve, initial_capital)
    returns_table = _build_returns_table(trades_df, initial_capital)
    benchmark = _build_benchmark(trades_df, market_data, equity_curve, timeframe, initial_capital)
    trade_analysis = _build_trade_analysis(trades_df)
    details_table = _build_details_table(trades_df)
    bars_summary = _build_bars_summary(trades_df)
    regime_metrics = _build_regime_metrics(trades_df)
    transactions = _build_transactions(events_df)

    return {
        "pair": pair,
        "timeframe": timeframe,
        "initial_capital": initial_capital,
        "summary": summary,
        "returns_table": returns_table,
        "benchmark": benchmark,
        "trade_analysis": trade_analysis,
        "win_loss": {
            "losses": trade_analysis["losses"],
            "wins": trade_analysis["wins"],
            "total_trades": trade_analysis["total_trades"],
        },
        "details_table": details_table,
        "bars_summary": bars_summary,
        "transactions": transactions,
        "trade_records": trades_df,
        "trade_events": events_df,
        "equity_curve": equity_curve,
        "strategy_metrics": strategy_metrics,
        "regime_metrics": regime_metrics,
        "risk_metrics": risk_metrics,
    }



def _build_summary(trades_df: pd.DataFrame, equity_curve: pd.DataFrame, initial_capital: float) -> dict[str, Any]:
    total_pnl = float(trades_df["pnl_usd"].sum()) if not trades_df.empty else 0.0
    total_trades = int(len(trades_df))
    profitable = int((trades_df["pnl_usd"] > 0).sum()) if not trades_df.empty else 0
    equity_series = equity_curve["equity"] if not equity_curve.empty else pd.Series(dtype=float)
    max_dd_usd, max_dd_pct = max_drawdown(equity_series)
    return {
        "total_pnl_usd": total_pnl,
        "total_pnl_pct": total_pnl / initial_capital if initial_capital else 0.0,
        "max_drawdown_usd": abs(max_dd_usd),
        "max_drawdown_pct": abs(max_dd_pct),
        "total_trades": total_trades,
        "profitable_trades_count": profitable,
        "profitable_trades_pct": profitable / total_trades if total_trades else 0.0,
        "profit_factor": profit_factor(trades_df["pnl_usd"]) if not trades_df.empty else 0.0,
    }



def _slice_side(trades_df: pd.DataFrame, side: str | None) -> pd.DataFrame:
    if trades_df.empty:
        return trades_df.copy()
    if side is None:
        return trades_df.copy()
    return trades_df[trades_df["side"] == side].copy()



def _money_pct(value: float, initial_capital: float) -> str:
    pct = value / initial_capital if initial_capital else 0.0
    return f"{value:,.2f} USD ({pct:.2%})"



def _trade_stat_frame(trades_df: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    rows = []
    for label, side in [("All", None), ("Long", "Long"), ("Short", "Short")]:
        subset = _slice_side(trades_df, side)
        net = float(subset["pnl_usd"].sum()) if not subset.empty else 0.0
        gross_profit = float(subset.loc[subset["pnl_usd"] > 0, "pnl_usd"].sum()) if not subset.empty else 0.0
        gross_loss = float(-subset.loc[subset["pnl_usd"] < 0, "pnl_usd"].sum()) if not subset.empty else 0.0
        commissions = float(subset["commission_paid_usd"].sum()) if not subset.empty else 0.0
        expected = float(subset["pnl_usd"].mean()) if not subset.empty else 0.0
        rows.append(
            {
                "column": label,
                "Initial capital": f"{initial_capital:,.2f} USD" if label == "All" else "",
                "Open P&L": "0.00 USD (0.00%)" if label == "All" else "",
                "Net P&L": _money_pct(net, initial_capital),
                "Gross profit": _money_pct(gross_profit, initial_capital),
                "Gross loss": _money_pct(gross_loss, initial_capital),
                "Profit factor": f"{profit_factor(subset['pnl_usd']) if not subset.empty else 0.0:.3f}",
                "Commission paid": f"{commissions:,.2f} USD",
                "Expected payoff": f"{expected:,.2f} USD",
            }
        )
    return pd.DataFrame(rows).set_index("column").T



def _build_returns_table(trades_df: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    return _trade_stat_frame(trades_df, initial_capital)



def _build_benchmark(
    trades_df: pd.DataFrame,
    market_data: pd.DataFrame,
    equity_curve: pd.DataFrame,
    timeframe: str,
    initial_capital: float,
) -> dict[str, Any]:
    buy_hold_return = 0.0
    if not market_data.empty:
        first_row = market_data.iloc[0]
        last_row = market_data.iloc[-1]
        units = initial_capital / max(float(first_row["close"] * first_row["quote_to_usd"]), 1e-9)
        buy_hold_quote = units * (float(last_row["close"]) - float(first_row["close"]))
        buy_hold_return = float(buy_hold_quote * float(last_row["quote_to_usd"]))

    total_pnl = float(trades_df["pnl_usd"].sum()) if not trades_df.empty else 0.0
    returns = equity_curve["equity"].pct_change().fillna(0.0) if not equity_curve.empty else pd.Series(dtype=float)
    factor = annualization_factor(timeframe)
    return {
        "buy_and_hold_return_usd": buy_hold_return,
        "buy_and_hold_return_pct": buy_hold_return / initial_capital if initial_capital else 0.0,
        "strategy_outperformance_usd": total_pnl - buy_hold_return,
        "sharpe_ratio": sharpe_ratio(returns, factor),
        "sortino_ratio": sortino_ratio(returns, factor),
    }



def _build_trade_analysis(trades_df: pd.DataFrame) -> dict[str, Any]:
    total = int(len(trades_df))
    wins_df = trades_df[trades_df["pnl_usd"] > 0] if not trades_df.empty else trades_df
    losses_df = trades_df[trades_df["pnl_usd"] < 0] if not trades_df.empty else trades_df
    breakeven_df = trades_df[trades_df["pnl_usd"] == 0] if not trades_df.empty else trades_df
    wins = int(len(wins_df))
    losses = int(len(losses_df))
    breakeven = int(len(breakeven_df))
    return {
        "average_loss_pct": float(losses_df["pnl_pct"].mean()) if losses else 0.0,
        "average_profit_pct": float(wins_df["pnl_pct"].mean()) if wins else 0.0,
        "total_trades": total,
        "wins": {"count": wins, "pct": wins / total if total else 0.0},
        "losses": {"count": losses, "pct": losses / total if total else 0.0},
        "break_even": {"count": breakeven, "pct": breakeven / total if total else 0.0},
    }



def _build_details_table(trades_df: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for label, side in [("All", None), ("Long", "Long"), ("Short", "Short")]:
        subset = _slice_side(trades_df, side)
        wins = subset[subset["pnl_usd"] > 0]
        losses = subset[subset["pnl_usd"] < 0]
        avg_loss_usd = float(losses["pnl_usd"].mean()) if not losses.empty else 0.0
        avg_loss_pct = float(losses["pnl_pct"].mean()) if not losses.empty else 0.0
        avg_win_usd = float(wins["pnl_usd"].mean()) if not wins.empty else 0.0
        avg_win_pct = float(wins["pnl_pct"].mean()) if not wins.empty else 0.0
        rows[label] = {
            "Total trades": int(len(subset)),
            "Total open trades": 0,
            "Winning trades": int(len(wins)),
            "Losing trades": int(len(losses)),
            "Percent profitable": f"{(len(wins) / len(subset)):.2%}" if len(subset) else "0.00%",
            "Avg P&L": f"{subset['pnl_usd'].mean() if len(subset) else 0.0:,.2f} USD ({subset['pnl_pct'].mean() if len(subset) else 0.0:.2%})",
            "Avg winning trade": f"{avg_win_usd:,.2f} USD ({avg_win_pct:.2%})",
            "Avg losing trade": f"{avg_loss_usd:,.2f} USD ({avg_loss_pct:.2%})",
            "Ratio avg win / avg loss": f"{abs(avg_win_usd / avg_loss_usd):.3f}" if avg_loss_usd != 0 else "0.000",
            "Largest winning trade": f"{wins['pnl_usd'].max() if not wins.empty else 0.0:,.2f} USD ({wins['pnl_pct'].max() if not wins.empty else 0.0:.2%})",
            "Largest losing trade": f"{losses['pnl_usd'].min() if not losses.empty else 0.0:,.2f} USD ({losses['pnl_pct'].min() if not losses.empty else 0.0:.2%})",
            "Average number of bars in trades": f"{subset['bars_held'].mean() if len(subset) else 0.0:.2f}",
        }
    return pd.DataFrame(rows)



def _build_bars_summary(trades_df: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, dict[str, float]] = {"winning": {}, "losing": {}}
    for label, side in [("All", None), ("Long", "Long"), ("Short", "Short")]:
        subset = _slice_side(trades_df, side)
        winners = subset[subset["pnl_usd"] > 0]
        losers = subset[subset["pnl_usd"] < 0]
        output["winning"][label] = float(winners["bars_held"].mean()) if not winners.empty else 0.0
        output["losing"][label] = float(losers["bars_held"].mean()) if not losers.empty else 0.0
    return output



def _build_regime_metrics(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame(columns=["regime", "trade_count", "win_rate", "net_pnl_usd", "avg_pnl_pct", "avg_bars_held"])
    grouped = trades_df.groupby("entry_regime")
    rows = []
    for regime, subset in grouped:
        rows.append(
            {
                "regime": regime,
                "trade_count": int(len(subset)),
                "win_rate": float((subset["pnl_usd"] > 0).mean()),
                "net_pnl_usd": float(subset["pnl_usd"].sum()),
                "avg_pnl_pct": float(subset["pnl_pct"].mean()),
                "avg_bars_held": float(subset["bars_held"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("regime")



def _build_transactions(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return events_df
    output = events_df.copy()
    output["_step_order"] = output["step_type"].map({"Exit": 0, "Entry": 1}).fillna(2)
    output = output.sort_values(["trade_id", "_step_order", "timestamp"], ascending=[False, True, False]).drop(columns=["_step_order"])
    return output[
        [
            "trade_id",
            "position_type",
            "step_type",
            "timestamp",
            "signal_reason",
            "price",
            "size",
            "net_pnl_usd",
            "net_pnl_pct",
            "cumulative_pnl_usd",
            "cumulative_pnl_pct",
            "favorite_excursion_pct",
            "adverse_excursion_pct",
        ]
    ]
