#!/usr/bin/env python3
"""
STREAMING Load TABERTA datasets from tar.gz files into MongoDB
Reads directly from tar.gz without extracting to disk - solves space issues!
Supports: wikidbs-10k, spider, bird, fetaqa, ottqa, tabfact
"""

import tarfile
import json
import io
import os
import sys
from pathlib import Path
from pymongo import MongoClient
from tqdm import tqdm
import time

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = Path(os.getenv("TABERTA_DATASETS_DIR", PROJECT_ROOT / "datasets"))
MONGODB_URI = os.getenv("TABERTA_MONGODB_URI", "mongodb://localhost:27017/")

DATASETS = {
    "tabfact": {
        "file": "tabfact.tar.gz",
        "db_name": "tabfact",
        "format": "tables_json"
    }
}

# DATASETS = {
#     "wikidbs-10k": {
#         "file": "wikidbs-10k.tar.gz",
#         "db_name": "wikidbs_10k",
#         "format": "mongodb_dump"
#     },
#     "spider": {
#         "file": "spider.tar.gz", 
#         "db_name": "spider",
#         "format": "tables_json"
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


def stream_load_tables_json(tar_path, db_name, client):
    """
    Stream load tables.json format directly from tar.gz
    NO extraction to disk - reads tar.gz in memory!
    """
    print(f"\n🌊 Streaming from {tar_path.name} (no disk extraction)...")
    
    db = client[db_name]
    collection = db["tables"]
    
    # Clear existing collection
    collection.drop()
    
    tables_found = []
    json_files_found = 0
    
    # Open tar.gz and stream through files
    with tarfile.open(tar_path, 'r:gz') as tar:
        for member in tar.getmembers():
            # Look for tables.json or .jsonl files
            if member.name.endswith('tables.json') or member.name.endswith('.jsonl'):
                json_files_found += 1
                print(f"📄 Found: {member.name}")
                
                # Extract file content to memory (not disk!)
                file_obj = tar.extractfile(member)
                if file_obj:
                    content = file_obj.read().decode('utf-8')
                    
                    # Parse JSON
                    try:
                        if member.name.endswith('.jsonl'):
                            # JSONL format - one table per line
                            for line in content.strip().split('\n'):
                                if line.strip():
                                    table = json.loads(line)
                                    tables_found.append(table)
                        else:
                            # Regular JSON - array of tables
                            data = json.loads(content)
                            if isinstance(data, list):
                                tables_found.extend(data)
                            elif isinstance(data, dict):
                                # Single table or dict of tables
                                if 'tables' in data:
                                    tables_found.extend(data['tables'])
                                else:
                                    tables_found.append(data)
                    except json.JSONDecodeError as e:
                        print(f"⚠️  JSON decode error in {member.name}: {e}")
                        continue
    
    if not tables_found:
        print(f"⚠️  No tables found in {tar_path.name}")
        return 0
    
    # Insert all tables into MongoDB
    print(f"📊 Found {len(tables_found)} tables, inserting into MongoDB...")
    
    if tables_found:
        # Add table_id if missing
        for i, table in enumerate(tables_found):
            if 'table_id' not in table and '_id' not in table:
                table['table_id'] = f"{db_name}_table_{i}"
        
        # Batch insert
        collection.insert_many(tables_found)
    
    return len(tables_found)


def stream_load_wikidbs(tar_path, db_name, client):
    """
    Stream load wikidbs-10k MongoDB dump format
    For BSON files, we'll use a hybrid approach - extract one collection at a time
    """
    print(f"\n🌊 Processing {tar_path.name}...")
    print("⚠️  Note: MongoDB dumps require mongorestore, falling back to JSON extraction")
    
    db = client[db_name]
    
    # Try to find JSON files in the tar
    json_tables = []
    with tarfile.open(tar_path, 'r:gz') as tar:
        for member in tar.getmembers():
            if member.name.endswith('.json') and not member.name.endswith('metadata.json'):
                print(f"📄 Found JSON: {member.name}")
                file_obj = tar.extractfile(member)
                if file_obj:
                    try:
                        content = file_obj.read().decode('utf-8')
                        data = json.loads(content)
                        if isinstance(data, list):
                            json_tables.extend(data)
                        else:
                            json_tables.append(data)
                    except:
                        pass
    
    if json_tables:
        collection = db["tables"]
        collection.drop()
        collection.insert_many(json_tables)
        return len(json_tables)
    
    print("⚠️  No JSON files found. For BSON dumps, use mongorestore manually.")
    print("    Or use the existing wikidbs_10k database (42,474 tables already loaded)")
    return 0


def load_dataset_streaming(dataset_name, client):
    """Load a single dataset using streaming (no disk extraction)"""
    if dataset_name not in DATASETS:
        print(f"❌ Unknown dataset: {dataset_name}")
        return False
    
    config = DATASETS[dataset_name]
    tar_path = DATASETS_DIR / config["file"]
    
    if not tar_path.exists():
        print(f"❌ File not found: {tar_path}")
        return False
    
    print("=" * 80)
    print(f"📦 Loading dataset: {dataset_name}")
    print(f"📁 File: {tar_path} ({tar_path.stat().st_size / 1e9:.2f} GB)")
    print(f"🗄️  Database: {config['db_name']}")
    print(f"🌊 Method: STREAMING (no disk extraction)")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        if config["format"] == "tables_json":
            tables_count = stream_load_tables_json(tar_path, config["db_name"], client)
        elif config["format"] == "mongodb_dump":
            tables_count = stream_load_wikidbs(tar_path, config["db_name"], client)
        else:
            print(f"❌ Unknown format: {config['format']}")
            return False
        
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 80)
        print(f"✅ Successfully loaded {dataset_name}")
        print(f"📊 Tables loaded: {tables_count:,}")
        print(f"⏱️  Time taken: {elapsed:.2f} seconds")
        print("=" * 80)
        
        # Show sample collections
        db = client[config["db_name"]]
        collections = db.list_collection_names()
        print(f"\n📋 Collections created: {collections}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error loading {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point"""
    print("🚀 TABERTA Dataset Loader (STREAMING MODE)")
    print("=" * 80)
    
    # Check MongoDB connection
    if not check_mongodb_connection():
        sys.exit(1)
    
    client = MongoClient(MONGODB_URI)
    
    # Parse arguments
    if len(sys.argv) < 2:
        print("\n📖 Usage:")
        print(f"  {sys.argv[0]} <dataset_name>    # Load single dataset")
        print(f"  {sys.argv[0]} all               # Load all datasets")
        print(f"\n📦 Available datasets: {', '.join(DATASETS.keys())}")
        sys.exit(1)
    
    dataset_arg = sys.argv[1].lower()
    
    if dataset_arg == "all":
        print("\n📦 Loading all datasets...")
        print("⏱️  Estimated time: ~10-20 minutes")
        print("")
        
        success_count = 0
        for dataset_name in DATASETS.keys():
            if load_dataset_streaming(dataset_name, client):
                success_count += 1
            print("")
        
        print("=" * 80)
        print(f"✅ Completed: {success_count}/{len(DATASETS)} datasets loaded successfully")
        print("=" * 80)
    else:
        # Load single dataset
        load_dataset_streaming(dataset_arg, client)
    
    client.close()


if __name__ == "__main__":
    main()
