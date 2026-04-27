#!/usr/bin/env python3
"""
Export all TABERTA experiment metrics from JSON files to a single CSV.

Reads metrics/*.json files and aggregates them into:
  - results/all_results.csv           (full results table)
  - results/summary_pivot.csv         (pivot: rows=dataset, cols=model, values=throughput)
  - results/summary_pivot_time.csv    (pivot: encoding_time_seconds)
"""

import json
import csv
import sys
from pathlib import Path

METRICS_DIR = Path(__file__).parent / "metrics"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Fields to export
CSV_FIELDS = [
    "dataset",
    "model",
    "embedding_dimension",
    "processed_tables",
    "skipped_tables",
    "embeddings_count",
    "encoding_inference_time_seconds",
    "indexing_time_seconds",
    "total_time_seconds",
    "throughput_embeddings_per_sec",
    "avg_time_per_embedding_ms",
    "embeddings_size_mb",
    "index_size_mb",
    "storage_per_1k_embeddings_mb",
    "gpu_available",
    "device",
    "model_path",
    "timestamp_start",
    "timestamp_end",
]


def load_all_metrics():
    """Load all *_metrics.json files from the metrics directory."""
    rows = []
    for f in sorted(METRICS_DIR.glob("*_metrics.json")):
        if f.name in ("summary.json", "pipeline_summary.json"):
            continue
        try:
            data = json.loads(f.read_text())
            rows.append(data)
        except Exception as e:
            print(f"⚠️  Skipping {f.name}: {e}")
    return rows


def export_full_csv(rows):
    """Write all results to a single CSV."""
    out = RESULTS_DIR / "all_results.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"✅ {out}  ({len(rows)} rows)")
    return out


def export_pivot(rows, value_key, filename):
    """Create a pivot CSV: rows=dataset, columns=model, values=value_key."""
    datasets = sorted(set(r["dataset"] for r in rows))
    models = sorted(set(r["model"] for r in rows))

    lookup = {(r["dataset"], r["model"]): r.get(value_key, "") for r in rows}

    out = RESULTS_DIR / filename
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset"] + models)
        for ds in datasets:
            writer.writerow([ds] + [lookup.get((ds, m), "") for m in models])
    print(f"✅ {out}  ({len(datasets)} datasets × {len(models)} models)")
    return out


def main():
    rows = load_all_metrics()
    if not rows:
        print("❌ No metrics found in", METRICS_DIR)
        sys.exit(1)

    print(f"\n📊 Found {len(rows)} experiment results\n")

    export_full_csv(rows)
    export_pivot(rows, "throughput_embeddings_per_sec", "pivot_throughput.csv")
    export_pivot(rows, "encoding_inference_time_seconds", "pivot_encoding_time.csv")
    export_pivot(rows, "avg_time_per_embedding_ms", "pivot_avg_time_per_embedding.csv")
    export_pivot(rows, "index_size_mb", "pivot_index_size_mb.csv")
    export_pivot(rows, "embedding_dimension", "pivot_embedding_dimension.csv")

    print(f"\n📁 All CSVs saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
