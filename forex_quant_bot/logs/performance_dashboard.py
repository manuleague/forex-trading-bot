from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import pandas as pd


class PerformanceDashboardWriter:
    WIDTH = 1660
    HEIGHT = 1320
    BACKGROUND = "#f6f8fb"
    PANEL = "#ffffff"
    PANEL_BORDER = "#d7dde8"
    TEXT = "#172033"
    MUTED = "#5b667a"
    EQUITY = "#2563eb"
    REALIZED = "#16a34a"
    UNREALIZED = "#f59e0b"
    DRAWDOWN = "#dc2626"
    GRID = "#e7ecf3"

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir

    def persist(self, report: dict[str, Any]) -> dict[str, Path]:
        svg_content = self._render_svg(report)
        svg_path = self.run_dir / "performance_dashboard.svg"
        svg_path.write_text(svg_content, encoding="utf-8")

        html_path = self.run_dir / "dashboard.html"
        html_path.write_text(self._render_html(svg_path.name), encoding="utf-8")
        return {"dashboard_svg": svg_path, "dashboard_html": html_path}

    def _render_html(self, svg_name: str) -> str:
        return (
            "<!doctype html>\n"
            "<html lang=\"en\">\n"
            "<head>\n"
            "  <meta charset=\"utf-8\">\n"
            "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            "  <title>Strategy Performance Dashboard</title>\n"
            "  <style>body{margin:0;background:#eef2f7;font-family:Segoe UI,Arial,sans-serif;}img{display:block;width:100%;height:auto;max-width:1660px;margin:24px auto;box-shadow:0 12px 40px rgba(15,23,42,.12);}</style>\n"
            "</head>\n"
            "<body>\n"
            f"  <img src=\"{escape(svg_name)}\" alt=\"Performance dashboard\">\n"
            "</body>\n"
            "</html>\n"
        )

    def _render_svg(self, report: dict[str, Any]) -> str:
        summary = report.get("summary", {})
        benchmark = report.get("benchmark", {})
        trade_analysis = report.get("trade_analysis", {})
        equity_curve = self._prepare_frame(report.get("equity_curve"))
        strategy_metrics = self._prepare_frame(report.get("strategy_metrics"))
        regime_metrics = self._prepare_frame(report.get("regime_metrics"))

        cards = [
            ("Total P&L", f"{float(summary.get('total_pnl_usd', 0.0)):+,.2f} USD"),
            ("Return", f"{float(summary.get('total_pnl_pct', 0.0)):+.2%}"),
            ("Max Drawdown", f"{float(summary.get('max_drawdown_pct', 0.0)):.2%}"),
            ("Profit Factor", self._format_number(float(summary.get('profit_factor', 0.0)), 3)),
            ("Sharpe", f"{float(benchmark.get('sharpe_ratio', 0.0)):.3f}"),
            ("Sortino", f"{float(benchmark.get('sortino_ratio', 0.0)):.3f}"),
            ("Trades", f"{int(summary.get('total_trades', 0)):,}"),
            ("Win Rate", f"{float(trade_analysis.get('wins', {}).get('pct', 0.0)):.2%}"),
        ]

        page_margin = 40
        card_gap_x = 18
        card_gap_y = 18
        card_width = int((self.WIDTH - (2 * page_margin) - (3 * card_gap_x)) / 4)
        card_height = 74
        cards_top = 104
        chart_gap = 24
        left_x = 40
        chart_width = int((self.WIDTH - (2 * left_x) - chart_gap) / 2)
        right_x = left_x + chart_width + chart_gap
        chart_area_top = cards_top + (2 * card_height) + card_gap_y + 28
        chart_height = 300
        lower_top = chart_area_top + chart_height + 28
        bottom_top = lower_top + chart_height + 28
        bottom_height = 300

        svg_parts = [
            f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{self.WIDTH}\" height=\"{self.HEIGHT}\" viewBox=\"0 0 {self.WIDTH} {self.HEIGHT}\">",
            f"<rect width=\"100%\" height=\"100%\" fill=\"{self.BACKGROUND}\"/>",
            f"<text x=\"40\" y=\"48\" font-family=\"Segoe UI, Arial, sans-serif\" font-size=\"30\" font-weight=\"700\" fill=\"{self.TEXT}\">Forex Quant Bot Performance Dashboard</text>",
            f"<text x=\"40\" y=\"78\" font-family=\"Segoe UI, Arial, sans-serif\" font-size=\"15\" fill=\"{self.MUTED}\">{escape(str(report.get('pair', 'N/A')))} | {escape(str(report.get('timeframe', 'N/A')))} </text>",
        ]

        for index, (label, value) in enumerate(cards):
            row = index // 4
            col = index % 4
            x = page_margin + col * (card_width + card_gap_x)
            y = cards_top + row * (card_height + card_gap_y)
            svg_parts.extend(
                [
                    f"<rect x=\"{x}\" y=\"{y}\" width=\"{card_width}\" height=\"{card_height}\" rx=\"14\" fill=\"{self.PANEL}\" stroke=\"{self.PANEL_BORDER}\"/>",
                    f"<text x=\"{x + 16}\" y=\"{y + 28}\" font-family=\"Segoe UI, Arial, sans-serif\" font-size=\"14\" fill=\"{self.MUTED}\">{escape(label)}</text>",
                    f"<text x=\"{x + 16}\" y=\"{y + 54}\" font-family=\"Segoe UI, Arial, sans-serif\" font-size=\"22\" font-weight=\"700\" fill=\"{self.TEXT}\">{escape(value)}</text>",
                ]
            )

        svg_parts.append(
            self._render_line_chart(
                x=left_x,
                y=chart_area_top,
                width=chart_width,
                height=chart_height,
                title="Equity Curve",
                dataframe=equity_curve,
                columns=[("equity", self.EQUITY)],
            )
        )
        svg_parts.append(
            self._render_line_chart(
                x=right_x,
                y=chart_area_top,
                width=chart_width,
                height=chart_height,
                title="Realized vs Unrealized P&L",
                dataframe=equity_curve,
                columns=[("realized_pnl_usd", self.REALIZED), ("unrealized_pnl_usd", self.UNREALIZED)],
            )
        )

        drawdown_frame = equity_curve.copy()
        if not drawdown_frame.empty and "equity" in drawdown_frame.columns:
            peaks = drawdown_frame["equity"].cummax()
            drawdown_frame["drawdown_pct"] = (drawdown_frame["equity"] / peaks) - 1.0
        svg_parts.append(
            self._render_line_chart(
                x=left_x,
                y=lower_top,
                width=chart_width,
                height=chart_height,
                title="Drawdown",
                dataframe=drawdown_frame,
                columns=[("drawdown_pct", self.DRAWDOWN)],
                value_formatter=lambda value: f"{value:.2%}",
            )
        )
        svg_parts.append(
            self._render_table_panel(
                x=right_x,
                y=lower_top,
                width=chart_width,
                height=chart_height,
                title="Strategy Metrics",
                dataframe=self._format_strategy_metrics(strategy_metrics),
            )
        )
        svg_parts.append(
            self._render_table_panel(
                x=left_x,
                y=bottom_top,
                width=chart_width,
                height=bottom_height,
                title="Regime Metrics",
                dataframe=self._format_regime_metrics(regime_metrics),
            )
        )
        svg_parts.append(
            self._render_notes_panel(
                x=right_x,
                y=bottom_top,
                width=chart_width,
                height=bottom_height,
                summary=summary,
                benchmark=benchmark,
                trade_analysis=trade_analysis,
            )
        )

        svg_parts.append("</svg>")
        return "\n".join(svg_parts)

    @staticmethod
    def _prepare_frame(value: Any) -> pd.DataFrame:
        if isinstance(value, pd.DataFrame):
            frame = value.copy()
            if "timestamp" in frame.columns:
                frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
                frame = frame.sort_values("timestamp").reset_index(drop=True)
            return frame
        return pd.DataFrame()

    def _render_line_chart(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        title: str,
        dataframe: pd.DataFrame,
        columns: list[tuple[str, str]],
        value_formatter=None,
    ) -> str:
        panel = [
            f"<g>",
            f"<rect x=\"{x}\" y=\"{y}\" width=\"{width}\" height=\"{height}\" rx=\"16\" fill=\"{self.PANEL}\" stroke=\"{self.PANEL_BORDER}\"/>",
            f"<text x=\"{x + 18}\" y=\"{y + 28}\" font-family=\"Segoe UI, Arial, sans-serif\" font-size=\"18\" font-weight=\"700\" fill=\"{self.TEXT}\">{escape(title)}</text>",
        ]

        plot_x = x + 18
        plot_y = y + 42
        plot_width = width - 36
        plot_height = height - 70
        panel.append(f"<rect x=\"{plot_x}\" y=\"{plot_y}\" width=\"{plot_width}\" height=\"{plot_height}\" rx=\"10\" fill=\"#fbfcfe\" stroke=\"{self.GRID}\"/>")

        if dataframe.empty or not any(column in dataframe.columns for column, _ in columns):
            panel.append(f"<text x=\"{x + 18}\" y=\"{y + 64}\" font-family=\"Segoe UI, Arial, sans-serif\" font-size=\"14\" fill=\"{self.MUTED}\">No data available.</text>")
            panel.append("</g>")
            return "\n".join(panel)

        series_values: list[float] = []
        for column, _ in columns:
            if column in dataframe.columns:
                series_values.extend(pd.to_numeric(dataframe[column], errors="coerce").dropna().astype(float).tolist())
        if not series_values:
            panel.append(f"<text x=\"{x + 18}\" y=\"{y + 64}\" font-family=\"Segoe UI, Arial, sans-serif\" font-size=\"14\" fill=\"{self.MUTED}\">No numeric data available.</text>")
            panel.append("</g>")
            return "\n".join(panel)

        min_value = min(series_values)
        max_value = max(series_values)
        if min_value == max_value:
            padding = max(abs(min_value) * 0.05, 1.0)
            min_value -= padding
            max_value += padding
        range_value = max_value - min_value

        for index in range(5):
            grid_y = plot_y + (plot_height * index / 4)
            panel.append(f"<line x1=\"{plot_x}\" y1=\"{grid_y:.2f}\" x2=\"{plot_x + plot_width}\" y2=\"{grid_y:.2f}\" stroke=\"{self.GRID}\" stroke-width=\"1\"/>")
            label_value = max_value - (range_value * index / 4)
            label = value_formatter(label_value) if value_formatter else f"{label_value:,.2f}"
            panel.append(f"<text x=\"{plot_x + 8}\" y=\"{grid_y - 6:.2f}\" font-family=\"Segoe UI, Arial, sans-serif\" font-size=\"11\" fill=\"{self.MUTED}\">{escape(label)}</text>")

        for column, color in columns:
            if column not in dataframe.columns:
                continue
            series = pd.to_numeric(dataframe[column], errors="coerce").astype(float).tolist()
            if not series:
                continue
            points = self._polyline_points(series, plot_x, plot_y, plot_width, plot_height, min_value, max_value)
            panel.append(f"<polyline fill=\"none\" stroke=\"{color}\" stroke-width=\"2.4\" points=\"{points}\"/>")

        legend_y = y + height - 18
        legend_x = x + 20
        for column, color in columns:
            if column not in dataframe.columns:
                continue
            panel.append(f"<line x1=\"{legend_x}\" y1=\"{legend_y}\" x2=\"{legend_x + 18}\" y2=\"{legend_y}\" stroke=\"{color}\" stroke-width=\"3\"/>")
            panel.append(f"<text x=\"{legend_x + 24}\" y=\"{legend_y + 4}\" font-family=\"Segoe UI, Arial, sans-serif\" font-size=\"12\" fill=\"{self.MUTED}\">{escape(column)}</text>")
            legend_x += 160

        panel.append("</g>")
        return "\n".join(panel)

    @staticmethod
    def _polyline_points(values: list[float], x: int, y: int, width: int, height: int, min_value: float, max_value: float) -> str:
        clean = [float(value) for value in values]
        if len(clean) == 1:
            clean = [clean[0], clean[0]]
        span = max(max_value - min_value, 1e-9)
        points = []
        for index, value in enumerate(clean):
            px = x + (width * index / max(len(clean) - 1, 1))
            py = y + height - ((value - min_value) / span) * height
            points.append(f"{px:.2f},{py:.2f}")
        return " ".join(points)

    def _render_table_panel(self, x: int, y: int, width: int, height: int, title: str, dataframe: pd.DataFrame) -> str:
        panel = [
            "<g>",
            f"<rect x=\"{x}\" y=\"{y}\" width=\"{width}\" height=\"{height}\" rx=\"16\" fill=\"{self.PANEL}\" stroke=\"{self.PANEL_BORDER}\"/>",
            f"<text x=\"{x + 18}\" y=\"{y + 28}\" font-family=\"Segoe UI, Arial, sans-serif\" font-size=\"18\" font-weight=\"700\" fill=\"{self.TEXT}\">{escape(title)}</text>",
        ]
        if dataframe.empty:
            panel.append(f"<text x=\"{x + 18}\" y=\"{y + 64}\" font-family=\"Segoe UI, Arial, sans-serif\" font-size=\"14\" fill=\"{self.MUTED}\">No data available.</text>")
            panel.append("</g>")
            return "\n".join(panel)

        rows = dataframe.head(8)
        header_y = y + 56
        panel.append(f"<rect x=\"{x + 14}\" y=\"{header_y - 18}\" width=\"{width - 28}\" height=\"26\" rx=\"8\" fill=\"#f2f5fa\"/>")
        available_width = width - 48
        column_step = available_width / max(len(rows.columns), 1)
        column_x = [x + 24 + (index * column_step) for index in range(len(rows.columns))]
        for cx, column in zip(column_x, rows.columns):
            panel.append(f"<text x=\"{cx}\" y=\"{header_y}\" font-family=\"Segoe UI, Arial, sans-serif\" font-size=\"12\" font-weight=\"700\" fill=\"{self.TEXT}\">{escape(str(column))}</text>")

        row_y = header_y + 28
        for _, row in rows.iterrows():
            for cx, column in zip(column_x, rows.columns):
                panel.append(f"<text x=\"{cx}\" y=\"{row_y}\" font-family=\"Consolas, monospace\" font-size=\"12\" fill=\"{self.TEXT}\">{escape(str(row[column]))}</text>")
            row_y += 24

        panel.append("</g>")
        return "\n".join(panel)

    def _render_notes_panel(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        summary: dict[str, Any],
        benchmark: dict[str, Any],
        trade_analysis: dict[str, Any],
    ) -> str:
        lines = [
            f"Buy & hold: {float(benchmark.get('buy_and_hold_return_usd', 0.0)):+,.2f} USD ({float(benchmark.get('buy_and_hold_return_pct', 0.0)):+.2%})",
            f"Strategy outperformance: {float(benchmark.get('strategy_outperformance_usd', 0.0)):+,.2f} USD",
            f"Profitable trades: {int(summary.get('profitable_trades_count', 0))}/{int(summary.get('total_trades', 0))}",
            f"Average win: {float(trade_analysis.get('average_profit_pct', 0.0)):+.2%}",
            f"Average loss: {float(trade_analysis.get('average_loss_pct', 0.0)):+.2%}",
            f"Break-even trades: {int(trade_analysis.get('break_even', {}).get('count', 0))}",
        ]
        panel = [
            "<g>",
            f"<rect x=\"{x}\" y=\"{y}\" width=\"{width}\" height=\"{height}\" rx=\"16\" fill=\"{self.PANEL}\" stroke=\"{self.PANEL_BORDER}\"/>",
            f"<text x=\"{x + 18}\" y=\"{y + 28}\" font-family=\"Segoe UI, Arial, sans-serif\" font-size=\"18\" font-weight=\"700\" fill=\"{self.TEXT}\">Run Notes</text>",
        ]
        line_y = y + 64
        for line in lines:
            panel.append(f"<text x=\"{x + 18}\" y=\"{line_y}\" font-family=\"Segoe UI, Arial, sans-serif\" font-size=\"14\" fill=\"{self.TEXT}\">{escape(line)}</text>")
            line_y += 28
        panel.append("</g>")
        return "\n".join(panel)

    @staticmethod
    def _format_strategy_metrics(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=["strategy", "trades", "win rate", "PF"])
        formatted = frame.copy()
        for column in ("trade_count", "win_rate", "profit_factor"):
            if column not in formatted.columns:
                formatted[column] = 0.0
        formatted["trade_count"] = formatted["trade_count"].fillna(0).astype(int)
        formatted["win_rate"] = pd.to_numeric(formatted["win_rate"], errors="coerce").fillna(0.0).map(lambda value: f"{value:.2%}")
        formatted["profit_factor"] = pd.to_numeric(formatted["profit_factor"], errors="coerce").fillna(0.0).map(lambda value: PerformanceDashboardWriter._format_number(value, 3))
        return formatted[["strategy", "trade_count", "win_rate", "profit_factor"]].rename(
            columns={"trade_count": "trades", "profit_factor": "PF"}
        )

    @staticmethod
    def _format_regime_metrics(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=["regime", "trades", "win rate", "net pnl"])
        formatted = frame.copy()
        for column in ("trade_count", "win_rate", "net_pnl_usd"):
            if column not in formatted.columns:
                formatted[column] = 0.0
        formatted["trade_count"] = formatted["trade_count"].fillna(0).astype(int)
        formatted["win_rate"] = pd.to_numeric(formatted["win_rate"], errors="coerce").fillna(0.0).map(lambda value: f"{value:.2%}")
        formatted["net_pnl_usd"] = pd.to_numeric(formatted["net_pnl_usd"], errors="coerce").fillna(0.0).map(lambda value: f"{value:+,.2f}")
        return formatted[["regime", "trade_count", "win_rate", "net_pnl_usd"]].rename(
            columns={"trade_count": "trades", "net_pnl_usd": "net pnl"}
        )

    @staticmethod
    def _format_number(value: float, digits: int) -> str:
        if pd.isna(value):
            return "0.000"
        if value == float("inf"):
            return "INF"
        if value == float("-inf"):
            return "-INF"
        return f"{value:.{digits}f}"
