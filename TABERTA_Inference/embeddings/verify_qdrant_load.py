#!/usr/bin/env python3
"""Verify embeddings landed correctly in Qdrant and match MongoDB table counts."""

import subprocess

from qdrant_client import QdrantClient
from pymongo import MongoClient

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
MONGODB_URI = "mongodb://localhost:27017/"
QDRANT_CONTAINER = "qdrant_container"
QDRANT_STORAGE_ROOT = "/qdrant/storage/collections"


def measure_qdrant_collection_size_bytes(collection_name):
    """Return actual on-disk bytes for a Qdrant collection via `du -s --block-size=1`.

    Uses real allocated blocks (not apparent size) because Qdrant pre-allocates
    sparse 32 MB mmap files per segment — apparent size stays constant regardless
    of how many vectors were written.
    """
    path = f"{QDRANT_STORAGE_ROOT}/{collection_name}"
    try:
        result = subprocess.run(
            ["docker", "exec", QDRANT_CONTAINER, "du", "-s", "--block-size=1", path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return None
        return int(result.stdout.split()[0])
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        return None


def mongo_table_count(db_name):
    """Count tabfact-style table documents across collections whose docs have a `table` list."""
    db = MongoClient(MONGODB_URI)[db_name]
    total = 0
    for coll_name in db.list_collection_names():
        sample = db[coll_name].find_one()
        if (
            sample
            and isinstance(sample.get("table"), list)
            and sample["table"]
            and isinstance(sample["table"][0], list)
        ):
            total += db[coll_name].count_documents({})
    return total


def infer_mongo_db_name(collection_name):
    """Qdrant collections are named '<dataset>_<model>'. The dataset prefix is the Mongo DB name."""
    mongo_client = MongoClient(MONGODB_URI)
    existing_dbs = set(mongo_client.list_database_names())
    parts = collection_name.split("_")
    for i in range(len(parts) - 1, 0, -1):
        candidate = "_".join(parts[:i])
        if candidate in existing_dbs:
            return candidate
    return None


def verify(q, collection_name, mongo_db_name=None):
    info = q.get_collection(collection_name)
    count = q.count(collection_name, exact=True).count
    vec_size = info.config.params.vectors.size
    distance = info.config.params.vectors.distance
    indexing_threshold = info.config.optimizer_config.indexing_threshold
    indexed_cnt = getattr(info, "indexed_vectors_count", None)
    is_indexed = indexing_threshold and indexing_threshold > 0 and (indexed_cnt or 0) > 0

    size_bytes = measure_qdrant_collection_size_bytes(collection_name)
    if size_bytes is not None:
        size_mb = round(size_bytes / (1024 * 1024), 4)
        size_str = f"{size_mb} MB ({size_bytes} bytes)"
    else:
        size_str = "unavailable (docker exec failed)"

    print(f"📦 Qdrant collection: {collection_name}")
    print(f"   vectors_count       : {count}")
    print(f"   indexed_vectors_cnt : {indexed_cnt}")
    print(f"   indexing_threshold  : {indexing_threshold}  ({'disabled' if not indexing_threshold else 'enabled'})")
    print(f"   vector_size         : {vec_size}")
    print(f"   distance            : {distance}")
    print(f"   collection_status   : {info.status}")
    print(f"   optimizer_status    : {info.optimizer_status}")
    print(f"   HNSW indexed        : {'✅ yes' if is_indexed else '❌ no (brute-force only)'}")
    print(f"   on_disk_size        : {size_str}")

    if mongo_db_name:
        mongo_total = mongo_table_count(mongo_db_name)
        match = "✅" if mongo_total == count else "❌"
        print(f"   mongo tables  : {mongo_total}  {match}")

    # Sample point
    points, _ = q.scroll(
        collection_name=collection_name,
        limit=1,
        with_payload=True,
        with_vectors=True,
    )
    if not points:
        print("   ⚠️  no points found")
        return
    #p = points[0]
    #vec = p.vector if isinstance(p.vector, list) else list(p.vector)
    #print(f"\n🔎 Sample point id: {p.id}")
    #print(f"   payload keys : {list(p.payload.keys())}")
    #print(f"   table_name   : {p.payload.get('table_name')}")
    #print(f"   schema[:5]   : {list(p.payload.get('schema', []))[:5]}")
    #print(f"   num_rows     : {p.payload.get('num_rows')}")
    #print(f"   vector dim   : {len(vec)}")
    #print(f"   vector[:4]   : {[round(float(x), 4) for x in vec[:4]]}")


def verify_all():
    q = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    colls = q.get_collections().collections
    if not colls:
        print("⚠️  No Qdrant collections found")
        return

    print(f"📚 Found {len(colls)} Qdrant collection(s)\n")
    for i, c in enumerate(colls):
        if i > 0:
            print("─" * 40)
        mongo_db = infer_mongo_db_name(c.name)
        verify(q, c.name, mongo_db)
        print()


if __name__ == "__main__":
    verify_all()
