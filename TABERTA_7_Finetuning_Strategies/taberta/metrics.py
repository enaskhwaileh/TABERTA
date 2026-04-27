"""
Metrics tracking and CSV export for training runs.

Each training script creates a MetricsLogger that records per-epoch metrics
and writes them to a CSV file for easy graphing.

CSV columns (per model):
    epoch, train_loss, val_loss, learning_rate, best_val_loss, pearson,
    early_stopped, elapsed_seconds, timestamp
"""

import csv
import os
import time
from typing import Any, Dict, Optional


class MetricsLogger:
    """
    Records per-epoch training metrics and writes to CSV.

    Usage:
        ml = MetricsLogger("metrics/PC_metrics.csv", strategy="PC")
        for epoch in range(1, epochs+1):
            ...training...
            ml.log(epoch=epoch, train_loss=avg_loss, val_loss=avg_val, lr=current_lr)
        ml.save()  # also called automatically if using context manager
    """

    COLUMNS = [
        "epoch", "train_loss", "val_loss", "learning_rate",
        "best_val_loss", "pearson", "spearman",
        "early_stopped", "elapsed_seconds", "timestamp",
    ]

    def __init__(self, csv_path: str, strategy: str = ""):
        self.csv_path = csv_path
        self.strategy = strategy
        self.rows = []
        self._start_time = time.time()

        os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)

    def log(self, epoch: int, train_loss: Optional[float] = None, **kwargs):
        """Record metrics for one epoch."""
        row = {col: "" for col in self.COLUMNS}
        row["epoch"] = epoch
        row["train_loss"] = f"{train_loss:.6f}" if train_loss is not None else ""
        row["elapsed_seconds"] = f"{time.time() - self._start_time:.1f}"
        row["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

        for key, val in kwargs.items():
            if key in row:
                if isinstance(val, float):
                    row[key] = f"{val:.6f}"
                else:
                    row[key] = str(val)

        self.rows.append(row)
        # Auto-save after each epoch so data is never lost
        self._write()

    def _write(self):
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writeheader()
            writer.writerows(self.rows)

    def save(self):
        """Explicitly save (also called after each log)."""
        self._write()
