#!/usr/bin/env python3
"""
Load TABERTA datasets from tar.gz files into MongoDB
Supports: wikidbs-10k, spider, bird, fetaqa, ottqa, tabfact
"""

import tarfile
import json
import os
import sys
from pathlib import Path
from pymongo import MongoClient
from tqdm import tqdm
import time
import pandas as pd
import numpy as np

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = Path(os.getenv("TABERTA_DATASETS_DIR", PROJECT_ROOT / "datasets"))
MONGODB_URI = os.getenv("TABERTA_MONGODB_URI", "mongodb://localhost:27017/")
TEMP_EXTRACT_DIR = Path(
    os.getenv("TABERTA_TEMP_EXTRACT_DIR", PROJECT_ROOT / "embeddings" / "temp_extract")
)

# DATASETS = {
#     "wikidbs-10k": {
#         "file": "wikidbs-10k.tar.gz",
#         "db_name": "wikidbs_10k",
#         "format": "mongodb_collections"  # Each table is a collection
#     },
#     "spider": {
#         "file": "spider.tar.gz", 
#         "db_name": "spider",
#         "format": "tables_json"  # tables.json format
#     },
#     "bird": {
#         "file": "bird.tar.gz",
#         "db_name": "bird", 
#         "format": "tables_json"
#     },
#     "fetaqa": {
#         "file": "fetaqa.tar.gz",
#         "db_name": "fetaqa",
#         "format": "tables_json"
#     },
#     "ottqa": {
#         "file": "ottqa.tar.gz",
#         "db_name": "ottqa",
#         "format": "tables_json"
#     },
#     "tabfact": {
#         "file": "tabfact.tar.gz",
#         "db_name": "tabfact",
#         "format": "tables_json"
#     }
# }

DATASETS = {
    "bird": {
        "file": "bird.tar.gz",
        "db_name": "bird",
        "format": "parquet_validation"
    },
    "spider": {
        "file": "spider.tar.gz",
        "db_name": "spider",
        "format": "spider_schema"
    },
    "tabfact": {
        "file": "tabfact.tar.gz",
        "db_name": "tabfact",
        "format": "parquet_validation"
    }
}


def check_mongodb_connection():
    """Verify MongoDB is running"""
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=2000)
        client.server_info()
        print("✅ MongoDB connection successful")
        return True
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        print("Start MongoDB with: docker start mongo_container")
        return False


def load_wikidbs_from_tar(tar_path, db_name, client):
    """Load wikidbs-10k format (MongoDB dump in tar.gz)"""
    print(f"\n📂 Extracting {tar_path.name}...")
    
    # Extract tar
    TEMP_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    extract_path = TEMP_EXTRACT_DIR / "wikidbs"
    
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(extract_path)
    
    print(f"✅ Extracted to {extract_path}")
    
    # Find bson files and load
    db = client[db_name]
    bson_files = list(extract_path.rglob("*.bson"))
    
    if not bson_files:
        print("⚠️  No .bson files found, checking for JSON format...")
        return load_tables_json_format(extract_path, db_name, client)
    
    print(f"📊 Found {len(bson_files)} BSON files")
    
    # Use mongorestore if available
    import subprocess
    for bson_file in tqdm(bson_files, desc="Restoring collections"):
        collection_name = bson_file.stem
        result = subprocess.run([
            "mongorestore",
            "--host", "localhost",
            "--port", "27017",
            "--db", db_name,
            "--collection", collection_name,
            str(bson_file)
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"⚠️  Failed to restore {collection_name}: {result.stderr}")
    
    # Cleanup
    import shutil
    shutil.rmtree(extract_path, ignore_errors=True)
    
    return db.list_collection_names()


def load_tables_json_format(extract_path, db_name, client):
    """Load spider/bird/fetaqa/ottqa/tabfact format (tables.json)"""
    db = client[db_name]
    
    # Find tables.json or similar
    json_files = list(extract_path.rglob("tables.json")) + list(extract_path.rglob("*.jsonl"))
    
    if not json_files:
        print(f"⚠️  No tables.json or .jsonl files found in {extract_path}")
        return []
    
    print(f"📄 Found {len(json_files)} JSON files")
    
    collections_created = []
    
    for json_file in json_files:
        print(f"\n📂 Loading {json_file.name}...")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            if json_file.suffix == '.jsonl':
                # JSONL format - one JSON object per line
                tables = [json.loads(line) for line in f if line.strip()]
            else:
                # Regular JSON format
                data = json.load(f)
                if isinstance(data, list):
                    tables = data
                elif isinstance(data, dict):
                    # Sometimes wrapped in a key
                    tables = data.get('tables', [data])
                else:
                    tables = [data]
        
        print(f"📊 Found {len(tables)} tables")
        
        # Create collection for each table
        for i, table in enumerate(tqdm(tables, desc="Loading tables")):
            # Generate collection name from table_id or name
            table_id = table.get('table_id') or table.get('id') or table.get('name') or f"table_{i}"
            collection_name = f"{db_name}_{table_id}".replace('-', '_').replace('.', '_')
            
            # Extract header/columns and rows
            header = table.get('header') or table.get('columns') or table.get('column_names', [])
            rows = table.get('rows') or table.get('data', [])
            
            if not header or not rows:
                continue
            
            # Convert to document format: [{col1: val1, col2: val2, ...}, ...]
            documents = []
            for row in rows:
                doc = {}
                for j, col_name in enumerate(header):
                    value = row[j] if j < len(row) else None
                    doc[str(col_name)] = value
                documents.append(doc)
            
            # Insert into MongoDB
            if documents:
                db[collection_name].drop()  # Clear if exists
                db[collection_name].insert_many(documents)
                collections_created.append(collection_name)
    
    return collections_created


def load_spider_schema_format(extract_path, db_name, client):
    """Load spider/tables.json as per-table schema docs (one doc per table).

    Each doc uses the same shape as tabfact (`table: [[header]]`) so the
    existing per-doc-table embedding pipeline handles it unchanged. Spider
    ships no row data, so the `table` field contains only the header row.
    """
    db = client[db_name]

    json_files = list(extract_path.rglob("tables.json"))
    json_files = [p for p in json_files if ".ipynb_checkpoints" not in p.parts]
    if not json_files:
        print(f"⚠️  No tables.json found in {extract_path}")
        return []

    tables_json = json_files[0]
    print(f"\n📂 Loading {tables_json.relative_to(extract_path)}")

    with open(tables_json, "r", encoding="utf-8") as f:
        databases = json.load(f)

    print(f"📊 Found {len(databases)} databases in tables.json")

    documents = []
    for entry in databases:
        db_id = entry.get("db_id")
        table_names_original = entry.get("table_names_original") or entry.get("table_names") or []
        table_names = entry.get("table_names") or table_names_original
        column_names_original = entry.get("column_names_original") or entry.get("column_names") or []
        column_types = entry.get("column_types") or []
        primary_keys = entry.get("primary_keys") or []
        foreign_keys = entry.get("foreign_keys") or []

        # Group columns by their owning table index (skip the special -1 "*" entry)
        cols_by_table = {i: [] for i in range(len(table_names_original))}
        types_by_table = {i: [] for i in range(len(table_names_original))}
        for col_idx, (tbl_idx, col_name) in enumerate(column_names_original):
            if tbl_idx < 0:
                continue
            cols_by_table.setdefault(tbl_idx, []).append(col_name)
            col_type = column_types[col_idx] if col_idx < len(column_types) else None
            types_by_table.setdefault(tbl_idx, []).append(col_type)

        for tbl_idx, tbl_name in enumerate(table_names_original):
            header = cols_by_table.get(tbl_idx, [])
            if not header:
                continue
            display_name = table_names[tbl_idx] if tbl_idx < len(table_names) else tbl_name
            documents.append({
                "database_id": db_id,
                "table_id": f"{db_id}::{tbl_name}",
                "table_name": display_name,
                "table_name_original": tbl_name,
                "column_types": types_by_table.get(tbl_idx, []),
                "primary_keys": primary_keys,
                "foreign_keys": foreign_keys,
                "table": [list(header)],
            })

    if not documents:
        print("⚠️  No tables extracted from tables.json")
        return []

    collection_name = "tables_schema"
    db[collection_name].drop()
    batch_size = 1000
    for i in tqdm(range(0, len(documents), batch_size), desc="Inserting"):
        db[collection_name].insert_many(documents[i:i + batch_size])

    print(f"📊 Inserted {len(documents)} table-schema docs into {db_name}.{collection_name}")
    return [collection_name]


def load_validation_parquet_format(extract_path, db_name, client):
    """Load validation.parquet files into MongoDB, one collection per containing folder."""
    db = client[db_name]

    parquet_files = list(extract_path.rglob("validation.parquet"))

    if not parquet_files:
        print(f"⚠️  No validation.parquet files found in {extract_path}")
        return []

    print(f"📄 Found {len(parquet_files)} validation.parquet files")

    collections_created = []

    for parquet_file in parquet_files:
        # Name collection after the parent folder (e.g. "corpus", "queries")
        parent_name = parquet_file.parent.name
        collection_name = f"{parent_name}_validation".replace('-', '_').replace('.', '_')

        print(f"\n📂 Loading {parquet_file.relative_to(extract_path)} → {collection_name}")

        df = pd.read_parquet(parquet_file)
        print(f"📊 Rows: {len(df)}, columns: {list(df.columns)}")

        def to_bson_safe(value):
            if isinstance(value, np.ndarray):
                return [to_bson_safe(v) for v in value.tolist()]
            if isinstance(value, list):
                return [to_bson_safe(v) for v in value]
            if isinstance(value, dict):
                return {k: to_bson_safe(v) for k, v in value.items()}
            if isinstance(value, (np.integer,)):
                return int(value)
            if isinstance(value, (np.floating,)):
                return float(value)
            if isinstance(value, (bytes, bytearray)):
                return value.decode('utf-8', errors='replace')
            return value

        documents = [
            {k: to_bson_safe(v) for k, v in row.items()}
            for row in df.to_dict(orient='records')
        ]

        if documents:
            db[collection_name].drop()
            # Insert in batches to avoid huge single inserts
            batch_size = 1000
            for i in tqdm(range(0, len(documents), batch_size), desc="Inserting"):
                db[collection_name].insert_many(documents[i:i + batch_size])
            collections_created.append(collection_name)

    return collections_created


def load_dataset_from_tar(dataset_name):
    """Load a specific dataset from tar.gz into MongoDB"""
    if dataset_name not in DATASETS:
        print(f"❌ Unknown dataset: {dataset_name}")
        print(f"Available: {list(DATASETS.keys())}")
        return False
    
    config = DATASETS[dataset_name]
    tar_path = DATASETS_DIR / config["file"]
    
    if not tar_path.exists():
        print(f"❌ File not found: {tar_path}")
        return False
    
    print(f"\n{'='*80}")
    print(f"📦 Loading dataset: {dataset_name}")
    print(f"📁 File: {tar_path} ({tar_path.stat().st_size / (1024**3):.2f} GB)")
    print(f"🗄️  Database: {config['db_name']}")
    print(f"{'='*80}")
    
    client = MongoClient(MONGODB_URI)
    start_time = time.time()
    
    try:
        if config["format"] == "mongodb_collections":
            collections = load_wikidbs_from_tar(tar_path, config["db_name"], client)
        else:
            # Extract first
            print(f"📂 Extracting {tar_path.name}...")
            TEMP_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
            extract_path = TEMP_EXTRACT_DIR / dataset_name

            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(extract_path)

            if config["format"] == "parquet_validation":
                collections = load_validation_parquet_format(extract_path, config["db_name"], client)
            elif config["format"] == "spider_schema":
                collections = load_spider_schema_format(extract_path, config["db_name"], client)
            else:
                collections = load_tables_json_format(extract_path, config["db_name"], client)

            # Cleanup
            import shutil
            shutil.rmtree(extract_path, ignore_errors=True)
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*80}")
        print(f"✅ Successfully loaded {dataset_name}")
        print(f"📊 Collections created: {len(collections)}")
        print(f"⏱️  Time taken: {elapsed:.2f} seconds")
        print(f"{'='*80}\n")
        
        # Show sample
        db = client[config["db_name"]]
        print(f"📋 Sample collections: {collections[:5]}")
        if collections:
            sample_coll = db[collections[0]]
            print(f"📄 Sample document count: {sample_coll.count_documents({})}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_existing_data():
    """Check what's already loaded in MongoDB"""
    client = MongoClient(MONGODB_URI)
    print(f"\n{'='*80}")
    print("📊 Current MongoDB Status")
    print(f"{'='*80}\n")
    
    for dataset_name, config in DATASETS.items():
        db_name = config["db_name"]
        db = client[db_name]
        collections = db.list_collection_names()
        
        if collections:
            total_docs = sum(db[c].count_documents({}) for c in collections)
            print(f"✅ {dataset_name:15} → {len(collections):6} collections, {total_docs:10,} documents")
        else:
            print(f"⚠️  {dataset_name:15} → NOT LOADED")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    print("🚀 TABERTA Dataset Loader")
    print("="*80)

    # Check MongoDB
    if not check_mongodb_connection():
        sys.exit(1)

    # Check current status
    check_existing_data()

    # If a specific dataset is given, load only that one
    if len(sys.argv) > 1:
        target = sys.argv[1]
        print(f"\n📦 Loading dataset: {target}")
        load_dataset_from_tar(target)
    else:
        print("\n📦 Loading ALL datasets...")
        for ds in DATASETS.keys():
            load_dataset_from_tar(ds)
