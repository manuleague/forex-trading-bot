from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


class CSVLogger:
    def __init__(self, base_dir: Path, mode: str, pair: str, timeframe: str, run_label: str | None = None) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_parts = [mode, pair, timeframe]
        if run_label:
            run_parts.append(run_label.strip().replace(" ", "_"))
        run_parts.append(timestamp)
        run_name = "_".join(part for part in run_parts if part)
        self.run_dir = base_dir / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def write_dataframe(self, name: str, dataframe: pd.DataFrame, include_index: bool = False) -> Path:
        path = self.run_dir / f"{name}.csv"
        dataframe.to_csv(path, index=include_index)
        return path

    def write_text(self, name: str, content: str) -> Path:
        path = self.run_dir / name
        path.write_text(content, encoding="utf-8")
        return path

    def persist_report(self, report: dict[str, Any], summary_text: str) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        for key, filename, include_index in [
            ("trade_records", "trades.csv", False),
            ("transactions", "transactions.csv", False),
            ("strategy_metrics", "strategy_metrics.csv", False),
            ("regime_metrics", "regime_metrics.csv", False),
            ("risk_metrics", "risk_overlay_metrics.csv", False),
            ("equity_curve", "equity_curve.csv", False),
            ("returns_table", "returns.csv", True),
            ("details_table", "details.csv", True),
            ("order_events", "order_events.csv", False),
        ]:
            value = report.get(key)
            if isinstance(value, pd.DataFrame):
                path = self.run_dir / filename
                value.to_csv(path, index=include_index)
                paths[key] = path
        paths["summary"] = self.write_text("summary.txt", summary_text)
        return paths
