#!/usr/bin/env python3
"""Aggregate all metrics for paper presentation"""

import json
import glob
from pathlib import Path
from collections import defaultdict

metrics_dir = Path("./metrics")
output_file = metrics_dir / "summary.json"

# Load all metrics
all_metrics = []
for filepath in sorted(metrics_dir.glob("*_metrics.json")):
    with open(filepath) as f:
        all_metrics.append(json.load(f))

# Organize by dataset and model
by_dataset = defaultdict(dict)
by_model = defaultdict(dict)

for m in all_metrics:
    dataset = m['dataset']
    model = m['model']
    
    by_dataset[dataset][model] = {
        'throughput': m['throughput_embeddings_per_sec'],
        'storage_efficiency': m['storage_per_1k_embeddings_mb'],
        'encoding_time': m['encoding_inference_time_seconds'],
        'total_time': m['total_time_seconds'],
        'embeddings': m['embeddings_count']
    }
    
    by_model[model][dataset] = by_dataset[dataset][model]

# Calculate averages per model
model_averages = {}
for model, datasets in by_model.items():
    model_averages[model] = {
        'avg_throughput': sum(d['throughput'] for d in datasets.values()) / len(datasets),
        'avg_storage': sum(d['storage_efficiency'] for d in datasets.values()) / len(datasets),
        'total_embeddings': sum(d['embeddings'] for d in datasets.values())
    }

# Summary
summary = {
    'total_experiments': len(all_metrics),
    'datasets': list(by_dataset.keys()),
    'models': list(by_model.keys()),
    'by_dataset': by_dataset,
    'by_model': by_model,
    'model_averages': model_averages,
    'total_embeddings': sum(m['embeddings_count'] for m in all_metrics)
}

# Save
with open(output_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"✓ Summary saved: {output_file}")
print(f"  Experiments: {len(all_metrics)}")
print(f"  Datasets: {len(by_dataset)}")
print(f"  Models: {len(by_model)}")
print(f"  Total embeddings: {summary['total_embeddings']:,}")
