#!/usr/bin/env python3
"""
TABERTA Embedding Generation Pipeline with Efficiency Metrics

Generates embeddings for all dataset × model combinations and tracks:
- Encoding time (GPU inference)
- Indexing time (Qdrant storage)
- Throughput (embeddings/second)
- Storage efficiency (MB per 1K embeddings)
- Memory usage
"""

import qdrant_client
from qdrant_client.models import VectorParams, Distance, PointStruct, OptimizersConfigDiff
from sentence_transformers import SentenceTransformer
import numpy as np
import hashlib
import json
import time
import os
import sys
import subprocess
from pathlib import Path
from pymongo import MongoClient
from tqdm import tqdm
import torch

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MONGODB_URI = os.getenv("TABERTA_MONGODB_URI", "mongodb://localhost:27017/")
QDRANT_HOST = os.getenv("TABERTA_QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("TABERTA_QDRANT_PORT", "6333"))
QDRANT_CONTAINER = os.getenv("TABERTA_QDRANT_CONTAINER", "qdrant_taberta")
QDRANT_STORAGE_ROOT = os.getenv("TABERTA_QDRANT_STORAGE_ROOT", "/qdrant/storage/collections")
MODELS_DIR = Path(os.getenv("TABERTA_MODELS_DIR", PROJECT_ROOT / "models"))

# Model paths
MODEL_PATHS = {
    "supervised_v1": str(MODELS_DIR / "2_Triplet Contrastive (TC) "),
    "supervised_v2": str(MODELS_DIR / "4_Triplet Contrastive (Optimized) (TC-opt) "),
    "supervised_v3": str(MODELS_DIR / "3_Triplet Contrastive (SmartBatch) (TC-SB)"),
    "supervised_v4": str(MODELS_DIR / "5_Self-Supervised Contrastive (SimCSE) (SS-C) "),
    "supervised_v5": str(MODELS_DIR / "1_Pairwise Contrastive (PC)"),
    "unsupervised_v6": str(MODELS_DIR / "6_Masked Language Modeling (MLM) "),
    "hybrid_v7": str(MODELS_DIR / "7_hybrid_model_reg"),
    "baseline_sbert": "sentence-transformers/all-MiniLM-L6-v2",
    "baseline_mpnet": "sentence-transformers/all-mpnet-base-v2",
    "qwen3_8b": "Qwen/Qwen3-Embedding-8B"
}

# Dataset configurations
DATASETS = {
    "wikidbs-10k": {"db_name": "wikidbs_10k", "type": "collections"},
    "spider": {"db_name": "spider", "type": "collections"},
    "fetaqa": {"db_name": "fetaqa", "type": "collections"},
    "ottqa": {"db_name": "ottqa", "type": "collections"},
    "tabfact": {"db_name": "tabfact", "type": "collections"}
}

# Output directory for metrics
METRICS_DIR = Path(os.getenv("TABERTA_METRICS_DIR", PROJECT_ROOT / "embeddings" / "metrics"))
METRICS_DIR.mkdir(exist_ok=True)


def generate_table_id(db_name, table_name):
    """Generate consistent MD5 hash ID for a table"""
    combined = f"{db_name}_{table_name}"
    return hashlib.md5(combined.encode('utf-8')).hexdigest()


def create_empty_metrics(dataset_name, model_name, collection_name):
    """Initialize empty metrics dictionary"""
    return {
        "dataset": dataset_name,
        "model": model_name,
        "collection": collection_name,
        "model_path": MODEL_PATHS.get(model_name, "unknown"),
        "encoding_inference_time_seconds": 0.0,
        "indexing_time_seconds": 0.0,
        "index_size_bytes": 0,
        "index_size_mb": 0.0,
        "embeddings_count": 0,
        "processed_tables": 0,
        "skipped_tables": 0,
        "embedding_dimension": None,
        "embeddings_size_bytes": 0,
        "embeddings_size_mb": 0.0,
        "timestamp_start": time.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp_end": None,
        "total_time_seconds": 0.0,
        "throughput_embeddings_per_sec": 0.0,
        "avg_time_per_embedding_ms": 0.0,
        "storage_per_1k_embeddings_mb": 0.0,
        "gpu_available": torch.cuda.is_available(),
        "device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "CPU"
    }


def add_embedding_size(metrics, embedding):
    """Track size metrics for each embedding"""
    arr = np.asarray(embedding, dtype=np.float32)
    metrics["embeddings_count"] += 1
    metrics["embedding_dimension"] = int(arr.shape[-1])
    metrics["embeddings_size_bytes"] += int(arr.nbytes)
    metrics["embeddings_size_mb"] = round(metrics["embeddings_size_bytes"] / (1024 * 1024), 4)


def finalize_metrics(metrics, start_time):
    """Calculate final efficiency metrics"""
    elapsed = time.time() - start_time
    metrics["timestamp_end"] = time.strftime("%Y-%m-%d %H:%M:%S")
    metrics["total_time_seconds"] = round(elapsed, 2)
    
    if metrics["embeddings_count"] > 0:
        # Throughput
        if metrics["total_time_seconds"] > 0:
            metrics["throughput_embeddings_per_sec"] = round(
                metrics["embeddings_count"] / metrics["total_time_seconds"], 2
            )
        
        # Average time per embedding
        metrics["avg_time_per_embedding_ms"] = round(
            (metrics["total_time_seconds"] * 1000) / metrics["embeddings_count"], 2
        )
        
        # Storage per 1K embeddings
        metrics["storage_per_1k_embeddings_mb"] = round(
            (metrics["embeddings_size_mb"] / metrics["embeddings_count"]) * 1000, 4
        )
    
    return metrics


def save_metrics(metrics, dataset_name, model_name):
    """Save metrics to JSON file"""
    filename = METRICS_DIR / f"{dataset_name}_{model_name}_metrics.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"📊 Metrics saved to {filename}")


def measure_qdrant_collection_size_bytes(collection_name):
    """Return actual on-disk bytes allocated for a Qdrant collection, or None if unreachable.

    Uses `du -s --block-size=1` (real allocated blocks) rather than `du -sb`
    (apparent size). Qdrant pre-allocates 32 MB sparse mmap files per segment
    for vectors/payload/WAL; apparent size includes the unused holes and stays
    constant regardless of how many vectors were written, so it is useless as
    a scaling metric.
    """
    path = f"{QDRANT_STORAGE_ROOT}/{collection_name}"
    try:
        result = subprocess.run(
            ["docker", "exec", QDRANT_CONTAINER, "du", "-s", "--block-size=1", path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            print(f"⚠️  Could not measure collection size: {result.stderr.strip()}")
            return None
        return int(result.stdout.split()[0])
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
        print(f"⚠️  Could not measure collection size: {e}")
        return None


def record_collection_size(metrics, collection_name):
    """Populate index_size_bytes/index_size_mb in metrics from on-disk measurement."""
    size_bytes = measure_qdrant_collection_size_bytes(collection_name)
    if size_bytes is None:
        return
    metrics["index_size_bytes"] = size_bytes
    metrics["index_size_mb"] = round(size_bytes / (1024 * 1024), 4)
    print(f"💾 Collection on-disk size: {size_bytes} bytes ({metrics['index_size_mb']} MB)")


def enable_hnsw_indexing(qdrant, collection_name, threshold):
    """Enable HNSW indexing and wait until the optimizer finishes building the index."""
    qdrant.update_collection(
        collection_name=collection_name,
        optimizer_config=OptimizersConfigDiff(indexing_threshold=threshold),
    )
    print(f"🗂️  Enabled HNSW indexing on '{collection_name}' (indexing_threshold={threshold})")

    deadline = time.time() + 300
    while time.time() < deadline:
        info = qdrant.get_collection(collection_name)
        points = info.points_count or 0
        indexed = getattr(info, "indexed_vectors_count", 0) or 0
        optimizer_ok = str(getattr(info, "optimizer_status", "")) in (
            "ok", "OptimizersStatus.OK", "OptimizersStatusOneOf.OK",
        )
        if points > 0 and indexed >= points and optimizer_ok:
            print(f"✅ HNSW build complete: {indexed}/{points} vectors indexed")
            return
        time.sleep(1.0)
    print("⚠️  Timed out waiting for HNSW indexing to complete")





def generate_embeddings_for_dataset(dataset_name, model_name, limit=None, index_threshold=None):
    """
    Generate embeddings for one dataset × model combination

    Args:
        dataset_name: Name of dataset (e.g., 'spider')
        model_name: Name of model (e.g., 'supervised_v1')
        limit: Optional limit on number of tables to process
        index_threshold: If set (int), enable HNSW indexing with this threshold after upload
    """
    print(f"\n{'='*80}")
    print(f"🚀 Starting: {dataset_name} × {model_name}")
    print(f"{'='*80}\n")
    
    # Validate inputs
    if dataset_name not in DATASETS:
        print(f"❌ Unknown dataset: {dataset_name}")
        return False
    
    if model_name not in MODEL_PATHS:
        print(f"❌ Unknown model: {model_name}")
        return False
    
    # Initialize connections
    start_time = time.time()
    qdrant = qdrant_client.QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=600)
    mongo_client = MongoClient(MONGODB_URI)
    
    dataset_config = DATASETS[dataset_name]
    db = mongo_client[dataset_config["db_name"]]

    # Collection name
    collection_name = f"{dataset_name}_{model_name}"
    print(f"📁 Qdrant collection: {collection_name}")
    
    # Initialize metrics
    metrics = create_empty_metrics(dataset_name, model_name, collection_name)
    
    # Load model
    print(f"📦 Loading model: {MODEL_PATHS[model_name]}")
    try:
        model_kwargs = {}
        if "qwen3" in model_name.lower() or "Qwen3" in MODEL_PATHS[model_name]:
            model_kwargs = {"dtype": torch.float16}
            print("  Using float16 for Qwen3 (8B params)")
        model = SentenceTransformer(MODEL_PATHS[model_name], model_kwargs=model_kwargs)
        if torch.cuda.is_available():
            model = model.to("cuda")
            print(f"✅ Model loaded on GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  GPU not available, using CPU")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Get embedding dimension (use base_dim for consistent sizing across all models)
    base_dim = model.get_sentence_embedding_dimension()
    

    # Force-delete any existing Qdrant collection, then wait until it's really gone
    if qdrant.collection_exists(collection_name):
        print(f"⚠️  Collection {collection_name} exists, force-deleting...")
        qdrant.delete_collection(collection_name, timeout=120)
        deadline = time.time() + 60
        while qdrant.collection_exists(collection_name):
            if time.time() > deadline:
                raise RuntimeError(f"Timed out waiting for '{collection_name}' to be deleted")
            time.sleep(0.5)
        print(f"🗑️  Deleted '{collection_name}'")

    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=base_dim,
            distance=Distance.COSINE
        ),
        optimizers_config=OptimizersConfigDiff(indexing_threshold=0)
    )
    print(f"✅ Created Qdrant collection: {collection_name}")
    
    # Check if this is a single-collection dataset (spider, fetaqa, etc.) or multi-collection (wikidbs-10k)
    collections = db.list_collection_names()

    # Detect per-document-table format: each document in a collection IS a table
    # (e.g. tabfact corpus_* — doc has a `table` field shaped [[header], [row], ...]).
    table_doc_collections = []
    for coll_name in collections:
        sample = db[coll_name].find_one()
        if (
            sample
            and isinstance(sample.get("table"), list)
            and sample["table"]
            and isinstance(sample["table"][0], list)
        ):
            table_doc_collections.append(coll_name)

    if table_doc_collections:
        print(f"📊 Detected per-document-table format in: {table_doc_collections}")

        # Materialize ALL docs into memory to avoid MongoDB cursor timeout
        # (cursors expire after 10 min inactivity; encoding is slow per-doc)
        all_docs = []
        for coll_name in table_doc_collections:
            cursor = db[coll_name].find({}, {"_id": 0})
            for doc in cursor:
                all_docs.append((coll_name, doc))
            cursor.close()
        total_tables = len(all_docs)
        if limit:
            all_docs = all_docs[:limit]
            total_tables = len(all_docs)
        print(f"📊 Loaded {total_tables} docs into memory, processing...\n")

        batch_points = []
        BATCH_SIZE = 100
        processed = 0

        with tqdm(total=total_tables, desc=f"{dataset_name} × {model_name}") as pbar:
            for coll_name, doc in all_docs:
                try:
                    table_id = doc.get("table_id") or f"{coll_name}_{processed}"
                    table = doc.get("table") or []
                    if not table or not isinstance(table[0], list):
                        metrics["skipped_tables"] += 1
                        processed += 1
                        pbar.update(1)
                        continue

                    header = table[0]
                    rows = table[1:4]  # first 3 data rows
                    schema_text = f"[TABLE]{table_id} [SCHEMA]{','.join(str(s) for s in header)}"
                    row_text = f"[ROWS]{'|'.join(','.join(str(v) for v in row) for row in rows)}"
                    combined_text = f"{schema_text} {row_text}"

                    encode_start = time.time()
                    embedding = model.encode(combined_text, batch_size=128, show_progress_bar=False)
                    metrics["encoding_inference_time_seconds"] += time.time() - encode_start

                    add_embedding_size(metrics, embedding)
                    metrics["processed_tables"] += 1

                    point_id = generate_table_id(dataset_config["db_name"], str(table_id))
                    point = PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload={
                            "dataset": dataset_name,
                            "model": model_name,
                            "table_name": str(table_id),
                            "db_name": dataset_config["db_name"],
                            "schema": header,
                            "num_rows": len(table) - 1,
                            "source_collection": coll_name,
                        },
                    )
                    batch_points.append(point)

                    if len(batch_points) >= BATCH_SIZE:
                        index_start = time.time()
                        qdrant.upsert(collection_name=collection_name, points=batch_points)
                        metrics["indexing_time_seconds"] += time.time() - index_start
                        batch_points = []

                    processed += 1
                    pbar.update(1)
                except Exception as e:
                    print(f"\n⚠️  Error processing doc in {coll_name}: {e}")
                    metrics["skipped_tables"] += 1
                    processed += 1
                    pbar.update(1)

    elif "tables" in collections and len(collections) <= 2:
        # Single collection format: all tables are documents in "tables" collection
        print(f"📊 Detected single-collection format (tables collection)")
        tables_collection = db["tables"]
        table_docs = list(tables_collection.find({}, {"_id": 0}))
        
        if limit:
            table_docs = table_docs[:limit]
        
        total_tables = len(table_docs)
        print(f"📊 Processing {total_tables} tables from 'tables' collection...\n")
        
        # Process tables
        batch_points = []
        BATCH_SIZE = 100
        
        with tqdm(total=total_tables, desc=f"{dataset_name} × {model_name}") as pbar:
            for idx, table_doc in enumerate(table_docs):
                try:
                    # Extract table info from document
                    table_id = table_doc.get('table_id', table_doc.get('db_id', f'table_{idx}'))
                    
                    # Extract schema - try different formats
                    if 'column_names' in table_doc:
                        schema = table_doc['column_names']
                        if isinstance(schema, list) and len(schema) > 0:
                            if isinstance(schema[0], list):  # [[table_idx, col_name], ...]
                                schema = [col[1] if isinstance(col, list) else str(col) for col in schema]
                            schema_text = f"[TABLE]{table_id} [SCHEMA]{','.join(str(s) for s in schema)}"
                        else:
                            schema_text = f"[TABLE]{table_id} [SCHEMA]unknown"
                    elif 'header' in table_doc:
                        schema = table_doc['header']
                        schema_text = f"[TABLE]{table_id} [SCHEMA]{','.join(str(s) for s in schema)}"
                    else:
                        # Fall back to document keys
                        schema = list(table_doc.keys())
                        schema_text = f"[TABLE]{table_id} [SCHEMA]{','.join(schema)}"
                    
                    # Extract rows - try different formats
                    rows = []
                    if 'rows' in table_doc:
                        rows = table_doc['rows'][:3]  # First 3 rows
                    elif 'row_data' in table_doc:
                        rows = table_doc['row_data'][:3]
                    
                    if rows:
                        row_text = f"[ROWS]{'|'.join(','.join(str(v) for v in row) for row in rows)}"
                    else:
                        # Use document values as row
                        row_text = f"[ROWS]{','.join(str(v) for k, v in list(table_doc.items())[:10] if k not in ['_id', 'table_id', 'db_id'])}"
                    
                    # Generate embedding from combined text (TRACK TIME)
                    combined_text = f"{schema_text} {row_text}"
                    encode_start = time.time()
                    embedding = model.encode(combined_text, batch_size=128, show_progress_bar=False)
                    metrics["encoding_inference_time_seconds"] += time.time() - encode_start
                    
                    # Track size
                    add_embedding_size(metrics, embedding)
                    metrics["processed_tables"] += 1
                    
                    # Generate ID
                    point_id = generate_table_id(dataset_config["db_name"], str(table_id))
                    
                    # Create point
                    point = PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload={
                            "dataset": dataset_name,
                            "model": model_name,
                            "table_name": str(table_id),
                            "db_name": dataset_config["db_name"],
                            "schema": schema,
                            "num_rows": len(rows) if rows else 0
                        }
                    )
                    batch_points.append(point)
                    
                    # Index batch
                    if len(batch_points) >= BATCH_SIZE:
                        index_start = time.time()
                        qdrant.upsert(collection_name=collection_name, points=batch_points)
                        metrics["indexing_time_seconds"] += time.time() - index_start
                        batch_points = []
                    
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"\n⚠️  Error processing table {idx}: {e}")
                    metrics["skipped_tables"] += 1
                    pbar.update(1)
    
    else:
        # Multi-collection format: each collection is a table (wikidbs-10k)
        print(f"📊 Detected multi-collection format ({len(collections)} collections)")
        table_names = collections
        if limit:
            table_names = table_names[:limit]
        
        total_tables = len(table_names)
        print(f"📊 Processing {total_tables} tables...\n")
        
        # Process tables
        batch_points = []
        BATCH_SIZE = 100
        
        with tqdm(total=total_tables, desc=f"{dataset_name} × {model_name}") as pbar:
            for table_name in table_names:
                try:
                    # Get table data
                    table_cursor = db[table_name].find({}, {"_id": 0}).sort("_id", 1)
                    table_data = list(table_cursor)
                    
                    if not table_data:
                        metrics["skipped_tables"] += 1
                        pbar.update(1)
                        continue
                    
                    # Extract schema
                    schema = list(table_data[0].keys())
                    schema_text = f"[TABLE]{table_name} [SCHEMA]{','.join(schema)}"
                    
                    # Extract rows (first 3)
                    row_text = f"[ROWS]{'|'.join(','.join(str(v) for v in doc.values()) for doc in table_data[:3])}"
                    
                    # Generate embedding from combined text (TRACK TIME)
                    combined_text = f"{schema_text} {row_text}"
                    encode_start = time.time()
                    embedding = model.encode(combined_text, batch_size=128, show_progress_bar=False)
                    metrics["encoding_inference_time_seconds"] += time.time() - encode_start
                    
                    # Track size
                    add_embedding_size(metrics, embedding)
                    metrics["processed_tables"] += 1
                    
                    # Generate ID
                    point_id = generate_table_id(dataset_config["db_name"], table_name)
                    
                    # Create point
                    point = PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload={
                            "dataset": dataset_name,
                            "model": model_name,
                            "table_name": table_name,
                            "db_name": dataset_config["db_name"],
                            "schema": schema,
                            "num_rows": len(table_data)
                        }
                    )
                    batch_points.append(point)
                    
                    # Index batch
                    if len(batch_points) >= BATCH_SIZE:
                        index_start = time.time()
                        qdrant.upsert(collection_name=collection_name, points=batch_points)
                        metrics["indexing_time_seconds"] += time.time() - index_start
                        batch_points = []
                    
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"\n⚠️  Error processing {table_name}: {e}")
                    metrics["skipped_tables"] += 1
                    pbar.update(1)
    
    # Index remaining points
    if batch_points:
        index_start = time.time()
        qdrant.upsert(collection_name=collection_name, points=batch_points)
        metrics["indexing_time_seconds"] += time.time() - index_start
    
    # Optional: enable HNSW indexing and wait for the build to complete
    if index_threshold is not None:
        enable_hnsw_indexing(qdrant, collection_name, threshold=index_threshold)

    # Record on-disk collection size (includes HNSW graph if indexing was enabled)
    record_collection_size(metrics, collection_name)

    # Finalize metrics
    metrics = finalize_metrics(metrics, start_time)

    # Save metrics
    save_metrics(metrics, dataset_name, model_name)

    # Print summary
    print(f"\n{'='*80}")
    print(f"✅ Completed: {dataset_name} × {model_name}")
    print(f"{'='*80}")
    print(f"📊 Processed: {metrics['processed_tables']} tables")
    print(f"📊 Generated: {metrics['embeddings_count']} embeddings")
    print(f"⏱️  Encoding time: {metrics['encoding_inference_time_seconds']:.2f}s")
    print(f"⏱️  Indexing time: {metrics['indexing_time_seconds']:.2f}s")
    print(f"💾 Index size: {metrics['index_size_mb']:.4f} MB ({metrics['index_size_bytes']} bytes)")
    print(f"⏱️  Total time: {metrics['total_time_seconds']:.2f}s")
    print(f"⚡ Throughput: {metrics['throughput_embeddings_per_sec']:.2f} embeddings/sec")
    print(f"💾 Embeddings size: {metrics['embeddings_size_mb']:.2f} MB")
    print(f"{'='*80}\n")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return True


def run_all_combinations(limit=None, index_threshold=None, skip_completed=False, skip_models=None):
    """Run all dataset × model combinations"""
    skip_models = skip_models or []
    
    # Build list of combos, optionally filtering out already-completed ones
    combos = []
    skipped = []
    for dataset_name in DATASETS.keys():
        for model_name in MODEL_PATHS.keys():
            if model_name in skip_models:
                skipped.append((dataset_name, model_name, "skipped model"))
                continue
            if skip_completed:
                metrics_file = METRICS_DIR / f"{dataset_name}_{model_name}_metrics.json"
                if metrics_file.exists():
                    skipped.append((dataset_name, model_name, "already completed"))
                    continue
            combos.append((dataset_name, model_name))
    
    print("\n" + "="*80)
    print("🚀 TABERTA FULL PIPELINE - ALL COMBINATIONS")
    print("="*80)
    print(f"📊 Datasets: {len(DATASETS)}")
    print(f"🤖 Models: {len(MODEL_PATHS)}")
    if skipped:
        print(f"⏭️  Skipping {len(skipped)} combinations ({len([s for s in skipped if s[2]=='already completed'])} completed, {len([s for s in skipped if s[2]=='skipped model'])} excluded models)")
    print(f"🔢 Combinations to run: {len(combos)}")
    if limit:
        print(f"⚠️  Limited to {limit} tables per dataset")
    print("="*80 + "\n")
    
    results = []
    total = len(combos)
    completed = 0
    
    for dataset_name, model_name in combos:
        try:
            success = generate_embeddings_for_dataset(
                dataset_name, model_name, limit, index_threshold=index_threshold,
            )
            results.append({
                "dataset": dataset_name,
                "model": model_name,
                "success": success
            })
            if success:
                completed += 1
        except Exception as e:
            print(f"❌ Failed: {dataset_name} × {model_name}: {e}")
            results.append({
                "dataset": dataset_name,
                "model": model_name,
                "success": False,
                "error": str(e)
            })
    
    # Final summary
    print("\n" + "="*80)
    print("📊 FINAL SUMMARY")
    print("="*80)
    print(f"✅ Completed: {completed}/{total}")
    print(f"❌ Failed: {total - completed}/{total}")
    print("="*80 + "\n")
    
    # Save summary
    summary_file = METRICS_DIR / "pipeline_summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "total_combinations": total,
            "completed": completed,
            "failed": total - completed,
            "results": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    print(f"📄 Summary saved to {summary_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate TABERTA embeddings and upsert to Qdrant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s spider supervised_v1 100 --index 20\n"
            "  %(prog)s tabfact baseline_sbert --index 100\n"
            "  %(prog)s all 500 --index 20\n"
        ),
    )
    parser.add_argument("dataset", help="Dataset name or 'all'")
    parser.add_argument("arg2", nargs="?", default=None,
                        help="Model name (single mode) OR limit (when dataset='all')")
    parser.add_argument("arg3", nargs="?", default=None,
                        help="Limit (single mode only)")
    parser.add_argument("--index", type=int, metavar="THRESHOLD", default=None,
                        help="Enable HNSW indexing after upload with the given indexing_threshold "
                             "(required value, e.g. --index 20). Omit to skip indexing.")
    parser.add_argument("--skip-completed", action="store_true",
                        help="Skip dataset×model combos that already have a metrics JSON file")
    parser.add_argument("--skip-models", nargs="+", default=[],
                        help="Model names to exclude (e.g. --skip-models qwen3_8b)")
    args = parser.parse_args()

    if args.index is not None and args.index < 0:
        parser.error("--index THRESHOLD must be a non-negative integer")

    if args.dataset == "all":
        limit = int(args.arg2) if args.arg2 is not None else None
        run_all_combinations(limit, index_threshold=args.index,
                             skip_completed=args.skip_completed,
                             skip_models=args.skip_models)
    else:
        model = args.arg2 or "supervised_v1"
        limit = int(args.arg3) if args.arg3 is not None else None
        generate_embeddings_for_dataset(
            args.dataset, model, limit, index_threshold=args.index,
        )
