from __future__ import annotations

import pandas as pd


class SummaryPrinter:
    MAX_TRANSACTION_ROWS = 60

    @staticmethod
    def render(report: dict) -> str:
        summary = report["summary"]
        benchmark = report["benchmark"]
        trade_analysis = report["trade_analysis"]
        win_loss = report["win_loss"]
        details_table = report["details_table"]
        returns_table = report["returns_table"]
        transactions = report["transactions"]
        bars_summary = report["bars_summary"]

        sections = [
            SummaryPrinter._render_summary(summary),
            SummaryPrinter._render_transactions(transactions),
            SummaryPrinter._render_table("Returns", returns_table),
            SummaryPrinter._render_benchmark(benchmark),
            SummaryPrinter._render_trade_analysis(trade_analysis),
            SummaryPrinter._render_win_loss(win_loss),
            SummaryPrinter._render_table("Details", details_table),
            SummaryPrinter._render_bars(bars_summary),
        ]
        return "\n\n".join(section for section in sections if section)

    @staticmethod
    def _render_summary(summary: dict) -> str:
        lines = [
            "A. Metrics Results",
            f"Total P&L: {summary['total_pnl_usd']:+,.2f} USD ({summary['total_pnl_pct']:+.2%})",
            f"Max equity drawdown: {summary['max_drawdown_usd']:,.2f} USD ({summary['max_drawdown_pct']:.2%})",
            f"Total trades: {summary['total_trades']}",
            f"Profitable trades: {summary['profitable_trades_pct']:.2%} ({summary['profitable_trades_count']}/{summary['total_trades']})",
            f"Profit factor: {summary['profit_factor']:.3f}",
        ]
        return "\n".join(lines)

    @staticmethod
    def _render_transactions(transactions: pd.DataFrame) -> str:
        if transactions.empty:
            return "B. List of Transactions\nNo trades were generated."
        formatted = transactions.copy()
        total_rows = len(formatted)
        if total_rows > SummaryPrinter.MAX_TRANSACTION_ROWS:
            formatted = formatted.head(SummaryPrinter.MAX_TRANSACTION_ROWS).copy()
        formatted["timestamp"] = pd.to_datetime(formatted["timestamp"], utc=True).dt.strftime("%b %d, %Y %H:%M")
        numeric_cols = [
            "price",
            "size",
            "net_pnl_usd",
            "net_pnl_pct",
            "cumulative_pnl_usd",
            "cumulative_pnl_pct",
            "favorite_excursion_pct",
            "adverse_excursion_pct",
        ]
        for column in numeric_cols:
            if column in {"net_pnl_pct", "cumulative_pnl_pct", "favorite_excursion_pct", "adverse_excursion_pct"}:
                formatted[column] = formatted[column].map(lambda value: f"{float(value):+.2%}")
            else:
                formatted[column] = formatted[column].map(lambda value: f"{float(value):,.5f}" if column == "price" else f"{float(value):,.2f}")
        header = "B. List of Transactions"
        if total_rows > SummaryPrinter.MAX_TRANSACTION_ROWS:
            header += f"\nShowing latest {SummaryPrinter.MAX_TRANSACTION_ROWS} of {total_rows} transaction rows. Full history is stored in transactions.csv."
        return header + "\n" + formatted.to_string(index=False, line_width=2000)

    @staticmethod
    def _render_table(title: str, dataframe: pd.DataFrame) -> str:
        if dataframe.empty:
            return f"{title}\nNo data."
        return f"{title}\n{dataframe.to_string(line_width=2000)}"

    @staticmethod
    def _render_benchmark(benchmark: dict) -> str:
        lines = [
            "Benchmark & Risk-adjusted Performance",
            f"Buy & hold return: {benchmark['buy_and_hold_return_usd']:+,.2f} USD ({benchmark['buy_and_hold_return_pct']:+.2%})",
            f"Buy & hold % gain: {benchmark['buy_and_hold_return_pct']:+.2%}",
            f"Strategy outperformance: {benchmark['strategy_outperformance_usd']:+,.2f} USD",
            f"Sharpe ratio: {benchmark['sharpe_ratio']:.3f}",
            f"Sortino ratio: {benchmark['sortino_ratio']:.3f}",
        ]
        return "\n".join(lines)

    @staticmethod
    def _render_trade_analysis(trade_analysis: dict) -> str:
        lines = [
            "Trades Analysis",
            f"Average loss: {trade_analysis['average_loss_pct']:.2%}",
            f"Average profit: {trade_analysis['average_profit_pct']:.2%}",
            f"Total trades: {trade_analysis['total_trades']}",
            f"Wins: {trade_analysis['wins']['count']} trades ({trade_analysis['wins']['pct']:.2%})",
            f"Losses: {trade_analysis['losses']['count']} trades ({trade_analysis['losses']['pct']:.2%})",
            f"Break even: {trade_analysis['break_even']['count']} trades ({trade_analysis['break_even']['pct']:.2%})",
        ]
        return "\n".join(lines)

    @staticmethod
    def _render_win_loss(win_loss: dict) -> str:
        return "\n".join(
            [
                "Win/Loss Ratio",
                f"Losses: {win_loss['losses']['count']} trades ({win_loss['losses']['pct']:.2%})",
                f"Wins: {win_loss['wins']['count']} trades ({win_loss['wins']['pct']:.2%})",
                f"Total trades: {win_loss['total_trades']}",
            ]
        )

    @staticmethod
    def _render_bars(bars_summary: dict) -> str:
        winning = bars_summary["winning"]
        losing = bars_summary["losing"]
        lines = [
            "Average Bars in Trade",
            f"Avg # bars in winning trades: {winning['All']:.2f} (All), {winning['Long']:.2f} (Long), {winning['Short']:.2f} (Short)",
            f"Avg # bars in losing trades: {losing['All']:.2f} (All), {losing['Long']:.2f} (Long), {losing['Short']:.2f} (Short)",
        ]
        return "\n".join(lines)
