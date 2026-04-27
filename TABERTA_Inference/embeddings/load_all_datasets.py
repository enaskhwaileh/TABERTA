#!/usr/bin/env python3
"""
Load ALL TABERTA datasets from tar.gz into MongoDB.
Handles: parquet corpus files (fetaqa, ottqa, tabfact), tables.json (spider),
         CSV-based wikidbs-10k.

Each dataset's tables go into db[dataset_name].corpus_train, corpus_validation,
corpus_test — or a single "tables" collection for spider/wikidbs-10k.
"""

import tarfile
import json
import io
import csv
import os
import sys
import time
from pathlib import Path
from pymongo import MongoClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = Path(os.getenv("TABERTA_DATASETS_DIR", PROJECT_ROOT / "datasets"))
MONGODB_URI = os.getenv("TABERTA_MONGODB_URI", "mongodb://localhost:27017/")


def check_mongodb():
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=3000)
        client.server_info()
        print("✅ MongoDB connected")
        return client
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Parquet-based datasets: fetaqa, ottqa, tabfact
# corpus/{train,validation,test}.parquet => each row is a table document
# Columns: database_id, table_id, table (list-of-lists), context
# ---------------------------------------------------------------------------

def load_parquet_dataset(tar_path, db_name, client):
    """Load corpus parquet files into MongoDB collections."""
    import pyarrow.parquet as pq

    db = client[db_name]
    total = 0

    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            # Only load corpus splits (the actual tables)
            if "/corpus/" not in member.name or not member.name.endswith(".parquet"):
                continue

            split = member.name.rsplit("/", 1)[-1].replace(".parquet", "")
            coll_name = f"corpus_{split}"

            print(f"  📄 {member.name} -> {db_name}.{coll_name}")

            f = tar.extractfile(member)
            if not f:
                continue

            df = pq.read_table(io.BytesIO(f.read())).to_pandas()

            # Deep-convert numpy types to native Python for MongoDB
            import numpy as np

            def to_native(obj):
                if isinstance(obj, np.ndarray):
                    return [to_native(x) for x in obj]
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, np.bool_):
                    return bool(obj)
                if isinstance(obj, dict):
                    return {k: to_native(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [to_native(x) for x in obj]
                return obj

            docs = []
            for _, row in df.iterrows():
                doc = {col: to_native(row[col]) for col in df.columns}
                docs.append(doc)

            if not docs:
                print(f"    ⚠️  Empty: {member.name}")
                continue

            db[coll_name].drop()
            db[coll_name].insert_many(docs)
            print(f"    ✅ {len(docs):,} docs inserted")
            total += len(docs)

    return total


# ---------------------------------------------------------------------------
# Spider: tables.json (array of {column_names, db_id, ...})
# ---------------------------------------------------------------------------

def load_spider(tar_path, db_name, client):
    db = client[db_name]
    db["tables"].drop()

    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith("tables.json") and "checkpoint" not in member.name:
                print(f"  📄 {member.name}")
                f = tar.extractfile(member)
                if not f:
                    continue
                data = json.loads(f.read().decode("utf-8"))
                if isinstance(data, list):
                    db["tables"].insert_many(data)
                    print(f"    ✅ {len(data):,} tables inserted")
                    return len(data)
    return 0


# ---------------------------------------------------------------------------
# WikiDBs-10k: databases/*/tables/*.csv — each CSV is a table
# ---------------------------------------------------------------------------

def load_wikidbs(tar_path, db_name, client):
    db = client[db_name]

    # Drop all existing non-system collections
    for coll in db.list_collection_names():
        if not coll.startswith("system."):
            db[coll].drop()

    table_count = 0
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            # Only load tables/ CSVs (not tables_with_ids/)
            if "/tables/" not in member.name or not member.name.endswith(".csv"):
                continue
            if "/tables_with_ids/" in member.name:
                continue

            f = tar.extractfile(member)
            if not f:
                continue

            # Collection name: dbname__tablename
            parts = member.name.split("/")
            # wikidbs-10k/databases/<db-name>/tables/<table>.csv
            if len(parts) >= 5:
                wiki_db = parts[2]
                table_file = parts[4].replace(".csv", "")
                coll_name = f"{wiki_db}__{table_file}"
            else:
                coll_name = member.name.replace("/", "__").replace(".csv", "")

            try:
                content = f.read().decode("utf-8")
                reader = csv.DictReader(io.StringIO(content))
                rows = list(reader)
                if rows:
                    db[coll_name].drop()
                    db[coll_name].insert_many(rows)
                    table_count += 1
            except Exception as e:
                print(f"    ⚠️  Error loading {member.name}: {e}")

            if table_count % 500 == 0 and table_count > 0:
                print(f"    ... {table_count:,} tables loaded so far")

    print(f"    ✅ {table_count:,} tables loaded")
    return table_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DATASETS = {
    "spider":      {"file": "spider.tar.gz",      "loader": load_spider},
    "fetaqa":      {"file": "fetaqa.tar.gz",       "loader": load_parquet_dataset},
    "ottqa":       {"file": "ottqa.tar.gz",        "loader": load_parquet_dataset},
    "tabfact":     {"file": "tabfact.tar.gz",      "loader": load_parquet_dataset},
    "wikidbs-10k": {"file": "wikidbs-10k.tar.gz",  "loader": load_wikidbs},
    # bird excluded: 18GB tar with SQLite databases, needs special handling
}

DB_NAMES = {
    "spider": "spider",
    "fetaqa": "fetaqa",
    "ottqa": "ottqa",
    "tabfact": "tabfact",
    "wikidbs-10k": "wikidbs_10k",
}


def load_dataset(name, client):
    config = DATASETS[name]
    tar_path = DATASETS_DIR / config["file"]
    db_name = DB_NAMES[name]

    if not tar_path.exists():
        print(f"❌ File not found: {tar_path}")
        return False

    print(f"\n{'='*60}")
    print(f"📦 {name} ({tar_path.stat().st_size / 1e6:.1f} MB) -> {db_name}")
    print(f"{'='*60}")

    start = time.time()
    count = config["loader"](tar_path, db_name, client)
    elapsed = time.time() - start

    print(f"✅ {name}: {count:,} tables in {elapsed:.1f}s")
    return True


def main():
    client = check_mongodb()

    targets = sys.argv[1:] if len(sys.argv) > 1 else []

    if not targets or targets == ["all"]:
        targets = list(DATASETS.keys())

    for name in targets:
        if name not in DATASETS:
            print(f"❌ Unknown dataset: {name}. Available: {list(DATASETS.keys())}")
            continue
        load_dataset(name, client)

    # Summary
    print(f"\n{'='*60}")
    print("📊 MongoDB Summary")
    print(f"{'='*60}")
    for db_name in sorted(set(DB_NAMES.values())):
        if db_name in client.list_database_names():
            db = client[db_name]
            colls = db.list_collection_names()
            total = sum(db[c].count_documents({}) for c in colls)
            print(f"  {db_name}: {len(colls)} collections, {total:,} docs")
        else:
            print(f"  {db_name}: NOT LOADED")


if __name__ == "__main__":
    main()
