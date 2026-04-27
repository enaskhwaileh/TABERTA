#!/usr/bin/env python3
"""Verify dataset content loaded into MongoDB — mirrors verify_qdrant_load.py."""

from pymongo import MongoClient

MONGODB_URI = "mongodb://localhost:27017/"
SYSTEM_DBS = {"admin", "config", "local"}


def is_per_doc_table(sample):
    """True if a doc represents a full table (has `table` field shaped [[header], [row], ...])."""
    return (
        bool(sample)
        and isinstance(sample.get("table"), list)
        and len(sample["table"]) > 0
        and isinstance(sample["table"][0], list)
    )


def verify_collection(db, coll_name):
    coll = db[coll_name]
    count = coll.count_documents({})
    sample = coll.find_one()

    print(f"📦 Collection: {db.name}.{coll_name}")
    print(f"   document_count : {count}")

    if not sample:
        print("   ⚠️  empty collection")
        return

    keys = [k for k in sample.keys() if k != "_id"]
    print(f"   field_keys     : {keys}")

    if is_per_doc_table(sample):
        table = sample["table"]
        header = table[0]
        data_rows = table[1:]
        print(f"   format         : per-document-table (each doc = 1 table)")
        print(f"   total_tables   : {count}")
        #print(f"\n🔎 Sample table")
        #print(f"   table_id       : {sample.get('table_id')}")
        #print(f"   database_id    : {sample.get('database_id')}")
        #print(f"   header[:5]     : {list(header)[:5]}")
        #print(f"   num_data_rows  : {len(data_rows)}")
        #if data_rows:
        #    print(f"   first_row[:5]  : {list(data_rows[0])[:5]}")
        #ctx = sample.get("context")
        #if ctx is not None:
        #    ctx_preview = str(ctx)[:120]
        #    print(f"   context        : {ctx_preview}{'...' if len(str(ctx)) > 120 else ''}")
    else:
        print(f"   format         : flat documents")
        #print(f"\n🔎 Sample document")
        #for k in keys[:6]:
        #    v = sample[k]
        #    preview = str(v)
        #    if len(preview) > 120:
        #        preview = preview[:120] + "..."
        #    print(f"   {k:14} : {preview}")


def verify_database(client, db_name):
    db = client[db_name]
    collections = db.list_collection_names()
    total_docs = sum(db[c].estimated_document_count() for c in collections)
    print(f"🗄️  Database: {db_name}")
    print(f"   collections    : {len(collections)}")
    print(f"   total_docs     : {total_docs}")
    if not collections:
        print("   ⚠️  no collections\n")
        return
    print()
    for i, coll_name in enumerate(collections):
        if i > 0:
            print("─" * 40)
        verify_collection(db, coll_name)
        print()


def verify_all():
    client = MongoClient(MONGODB_URI)
    try:
        client.admin.command("ping")
    except Exception as e:
        print(f"❌ Cannot reach MongoDB at {MONGODB_URI}: {e}")
        return

    dbs = [d for d in client.list_database_names() if d not in SYSTEM_DBS]
    if not dbs:
        print("⚠️  No user databases found")
        return

    print(f"📚 Found {len(dbs)} user database(s): {dbs}\n")
    for db_name in dbs:
        print("=" * 80)
        verify_database(client, db_name)


if __name__ == "__main__":
    verify_all()
