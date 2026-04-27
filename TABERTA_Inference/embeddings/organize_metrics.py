#!/usr/bin/env python3
"""
Organize and consolidate TABERTA metrics for analysis and visualization
Creates a unified CSV and enhanced JSON summary
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

METRICS_DIR = Path("./metrics")

def load_all_metrics():
    """Load all individual metrics files"""
    metrics_files = list(METRICS_DIR.glob("*_metrics.json"))
    
    all_metrics = []
    for file in sorted(metrics_files):
        if file.name == "pipeline_summary.json":
            continue
            
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                all_metrics.append(data)
        except Exception as e:
            print(f"⚠️  Error loading {file.name}: {e}")
    
    return all_metrics

def create_consolidated_dataframe(metrics_list):
    """Create a pandas DataFrame from metrics"""
    
    # Flatten metrics into rows
    rows = []
    for m in metrics_list:
        row = {
            'dataset': m.get('dataset', 'unknown'),
            'model': m.get('model', 'unknown'),
            'collection': m.get('collection', 'unknown'),
            'embeddings_count': m.get('embeddings_count', 0),
            'processed_tables': m.get('processed_tables', 0),
            'skipped_tables': m.get('skipped_tables', 0),
            'embedding_dimension': m.get('embedding_dimension', 0),
            'encoding_time_sec': m.get('encoding_inference_time_seconds', 0),
            'indexing_time_sec': m.get('indexing_time_seconds', 0),
            'total_time_sec': m.get('total_time_seconds', 0),
            'throughput_emb_per_sec': m.get('throughput_embeddings_per_sec', 0),
            'avg_time_per_emb_ms': m.get('avg_time_per_embedding_ms', 0),
            'storage_mb': m.get('embeddings_size_mb', 0),
            'storage_per_1k_mb': m.get('storage_per_1k_embeddings_mb', 0),
            'gpu_available': m.get('gpu_available', False),
            'device': m.get('device', 'unknown'),
            'timestamp_start': m.get('timestamp_start', ''),
            'timestamp_end': m.get('timestamp_end', '')
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Add computed columns
    if len(df) > 0:
        # Model category
        df['model_type'] = df['model'].apply(lambda x: 
            'baseline' if 'baseline' in x else
            'supervised' if 'supervised' in x else
            'unsupervised' if 'unsupervised' in x else
            'hybrid' if 'hybrid' in x else 'other'
        )
        
        # Dataset size category
        df['dataset_size'] = df['processed_tables'].apply(lambda x:
            'small' if x < 1000 else
            'medium' if x < 10000 else
            'large' if x < 50000 else
            'xlarge'
        )
        
        # Performance category based on throughput
        df['performance'] = df['throughput_emb_per_sec'].apply(lambda x:
            'slow' if x < 5 else
            'medium' if x < 15 else
            'fast'
        )
    
    return df

def create_summary_statistics(df):
    """Create summary statistics"""
    summary = {
        'generated_at': datetime.now().isoformat(),
        'total_combinations': len(df),
        'total_embeddings': int(df['embeddings_count'].sum()),
        'total_tables_processed': int(df['processed_tables'].sum()),
        'avg_throughput_emb_per_sec': float(df['throughput_emb_per_sec'].mean()),
        'avg_encoding_time_sec': float(df['encoding_time_sec'].mean()),
        'avg_indexing_time_sec': float(df['indexing_time_sec'].mean()),
        'total_storage_mb': float(df['storage_mb'].sum()),
        
        'by_dataset': {},
        'by_model': {},
        'by_model_type': {}
    }
    
    # Aggregate by dataset
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        summary['by_dataset'][dataset] = {
            'combinations': len(dataset_df),
            'total_embeddings': int(dataset_df['embeddings_count'].sum()),
            'total_tables': int(dataset_df['processed_tables'].max()),  # Same for all models
            'avg_throughput': float(dataset_df['throughput_emb_per_sec'].mean()),
            'total_storage_mb': float(dataset_df['storage_mb'].sum())
        }
    
    # Aggregate by model
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        summary['by_model'][model] = {
            'combinations': len(model_df),
            'total_embeddings': int(model_df['embeddings_count'].sum()),
            'avg_throughput': float(model_df['throughput_emb_per_sec'].mean()),
            'avg_encoding_time': float(model_df['encoding_time_sec'].mean()),
            'total_storage_mb': float(model_df['storage_mb'].sum())
        }
    
    # Aggregate by model type
    for model_type in df['model_type'].unique():
        type_df = df[df['model_type'] == model_type]
        summary['by_model_type'][model_type] = {
            'combinations': len(type_df),
            'total_embeddings': int(type_df['embeddings_count'].sum()),
            'avg_throughput': float(type_df['throughput_emb_per_sec'].mean())
        }
    
    return summary

def main():
    print("=" * 80)
    print("📊 TABERTA Metrics Organizer")
    print("=" * 80)
    print()
    
    # Load all metrics
    print("📂 Loading metrics files...")
    metrics_list = load_all_metrics()
    print(f"✅ Loaded {len(metrics_list)} metrics files")
    print()
    
    if not metrics_list:
        print("❌ No metrics files found!")
        return
    
    # Create DataFrame
    print("📊 Creating consolidated DataFrame...")
    df = create_consolidated_dataframe(metrics_list)
    print(f"✅ Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
    print()
    
    # Save to CSV
    csv_path = METRICS_DIR / "consolidated_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"💾 Saved consolidated CSV: {csv_path}")
    print()
    
    # Create summary statistics
    print("📈 Computing summary statistics...")
    summary = create_summary_statistics(df)
    
    # Save summary JSON
    summary_path = METRICS_DIR / "metrics_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"💾 Saved summary JSON: {summary_path}")
    print()
    
    # Display summary
    print("=" * 80)
    print("📊 SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total Combinations: {summary['total_combinations']}")
    print(f"Total Embeddings: {summary['total_embeddings']:,}")
    print(f"Total Tables Processed: {summary['total_tables_processed']:,}")
    print(f"Average Throughput: {summary['avg_throughput_emb_per_sec']:.2f} emb/sec")
    print(f"Total Storage: {summary['total_storage_mb']:.2f} MB")
    print()
    
    print("By Dataset:")
    for dataset, stats in summary['by_dataset'].items():
        print(f"  {dataset:15s}: {stats['total_tables']:6,} tables, {stats['combinations']} models, "
              f"{stats['avg_throughput']:5.2f} emb/sec")
    print()
    
    print("By Model:")
    for model, stats in summary['by_model'].items():
        print(f"  {model:20s}: {stats['total_embeddings']:8,} embeddings, "
              f"{stats['avg_throughput']:5.2f} emb/sec")
    print()
    
    print("=" * 80)
    print("✅ Metrics organization complete!")
    print()
    print(f"📁 Output files:")
    print(f"  • CSV: {csv_path}")
    print(f"  • JSON: {summary_path}")
    print()
    print("📊 Ready for visualization in the Jupyter notebook!")
    print("=" * 80)

if __name__ == "__main__":
    main()
