"""
Training sample preparation for all 7 fine-tuning strategies.

Converts WikiDBs data + serialization views into the training samples
expected by each strategy's loss function:

  PC      → pairs  (anchor, other, label)
  SS-C    → pairs  (text, text)  — SimCSE self-pairing
  TC      → triplets (anchor, positive, negative) — FK-based positives
  TC-Opt  → triplets (anchor, positive, negative) — same-DB positives
  TC-SB   → triplets (anchor, positive, negative) — same-DB, smart-batched
  MLM     → flat texts (one per table)
  Hybrid  → (stage 1: flat texts for MLM, stage 2: triplets)
"""

import logging
import random
from typing import Dict, List, Optional, Tuple

from taberta.data_loading import WikiDBsCorpus, DatabaseRecord
from taberta.serialization import serialize

logger = logging.getLogger(__name__)


# ============================================================================
# Pairs  (used by PC, SS-C)
# ============================================================================

def prepare_pc_pairs(
    corpus: WikiDBsCorpus,
    view: str = "row",
    max_databases: Optional[int] = None,
    seed: int = 42,
) -> List[Tuple[str, str, float]]:
    """
    Pairwise Contrastive (PC) — generate (text_a, text_b, label) pairs.

    Positive pairs (label=0.5): two tables from the *same* database.
    Negative pairs (label=0.0): two tables from *different* databases.

    Returns list of (text_a, text_b, label) tuples.
    """
    rng = random.Random(seed)
    db_names = corpus.database_names
    if max_databases:
        db_names = db_names[:max_databases]

    # Pre-serialize all tables grouped by database
    db_tables: Dict[str, List[str]] = {}
    for db_name in db_names:
        try:
            db = corpus.load_database(db_name)
        except Exception as e:
            logger.warning(f"Skipping {db_name}: {e}")
            continue
        texts = [serialize(t, view) for t in db.tables.values() if t.rows]
        if texts:
            db_tables[db_name] = texts

    all_db_names = list(db_tables.keys())
    pairs = []

    for db_name in all_db_names:
        texts = db_tables[db_name]
        # Positive pairs: tables within the same database
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                pairs.append((texts[i], texts[j], 0.5))

        # Negative pairs: table from this DB vs table from another DB
        other_dbs = [d for d in all_db_names if d != db_name]
        if other_dbs:
            neg_db = rng.choice(other_dbs)
            neg_text = rng.choice(db_tables[neg_db])
            anchor_text = rng.choice(texts)
            pairs.append((anchor_text, neg_text, 0.0))

    rng.shuffle(pairs)
    logger.info(f"PC: generated {len(pairs)} pairs from {len(all_db_names)} databases")
    return pairs


def prepare_ssc_pairs(
    corpus: WikiDBsCorpus,
    view: str = "full",
    max_databases: Optional[int] = None,
) -> List[Tuple[str, str]]:
    """
    Self-Supervised Contrastive / SimCSE (SS-C) — generate self-pairs.

    Each table is paired with itself; dropout during encoding provides
    the two stochastic views.

    Returns list of (text, text) tuples.
    """
    db_names = corpus.database_names
    if max_databases:
        db_names = db_names[:max_databases]

    pairs = []
    for db_name in db_names:
        try:
            db = corpus.load_database(db_name)
        except Exception:
            continue
        for table in db.tables.values():
            if not table.rows:
                continue
            text = serialize(table, view)
            pairs.append((text, text))

    logger.info(f"SS-C: generated {len(pairs)} self-pairs")
    return pairs


# ============================================================================
# Triplets  (used by TC, TC-Opt, TC-SB, Hybrid stage 2)
# ============================================================================

def prepare_tc_triplets(
    corpus: WikiDBsCorpus,
    view: str = "full",
    target_triplets: int = 10000,
    use_fk_positives: bool = True,
    seed: int = 42,
    max_databases: Optional[int] = None,
) -> List[Tuple[str, str, str]]:
    """
    Triplet Contrastive (TC) — generate (anchor, positive, negative) triplets.

    Positive selection:
      - If ``use_fk_positives`` and the anchor table has FK targets, pick one.
      - Otherwise, pick another table from the same database.
    Negative: a table from a different database.

    Returns list of (anchor_text, positive_text, negative_text) tuples.
    """
    rng = random.Random(seed)
    db_names = corpus.database_names
    if max_databases:
        db_names = db_names[:max_databases]

    # Pre-load and serialize
    db_serialized: Dict[str, Dict[str, str]] = {}
    db_records: Dict[str, DatabaseRecord] = {}
    for db_name in db_names:
        try:
            db_rec = corpus.load_database(db_name)
        except Exception:
            continue
        texts = {}
        for t in db_rec.tables.values():
            if t.rows:
                texts[t.table_name] = serialize(t, view)
        if texts:
            db_serialized[db_name] = texts
            db_records[db_name] = db_rec

    all_db_names = list(db_serialized.keys())
    if len(all_db_names) < 2:
        logger.warning("Need at least 2 databases for triplet generation")
        return []

    triplets = []
    attempts = 0
    max_attempts = target_triplets * 3

    while len(triplets) < target_triplets and attempts < max_attempts:
        attempts += 1

        # Pick anchor database and table
        anchor_db_name = rng.choice(all_db_names)
        anchor_tables = db_serialized[anchor_db_name]
        anchor_table_name = rng.choice(list(anchor_tables.keys()))
        anchor_text = anchor_tables[anchor_table_name]

        # Pick positive: FK target or same-DB table
        positive_text = None
        if use_fk_positives:
            db_rec = db_records[anchor_db_name]
            fk_targets = db_rec.get_fk_targets(anchor_table_name)
            fk_targets = [t for t in fk_targets if t in anchor_tables]
            if fk_targets:
                positive_text = anchor_tables[rng.choice(fk_targets)]

        if positive_text is None:
            # Fallback: another table from the same database
            other_tables = [t for t in anchor_tables if t != anchor_table_name]
            if not other_tables:
                continue
            positive_text = anchor_tables[rng.choice(other_tables)]

        # Pick negative: table from a different database
        neg_db_name = rng.choice([d for d in all_db_names if d != anchor_db_name])
        neg_tables = db_serialized[neg_db_name]
        negative_text = rng.choice(list(neg_tables.values()))

        triplets.append((anchor_text, positive_text, negative_text))

    logger.info(
        f"TC: generated {len(triplets)} triplets from {len(all_db_names)} databases "
        f"(FK positives={'on' if use_fk_positives else 'off'})"
    )
    return triplets


def prepare_tc_opt_triplets(
    corpus: WikiDBsCorpus,
    views: List[str] = None,
    target_triplets: int = 10000,
    seed: int = 42,
    max_databases: Optional[int] = None,
) -> List[Tuple[str, str, str]]:
    """
    TC-Opt — triplets using both SchemaView and FullView.

    Anchor and positive use FullView; occasionally SchemaView is mixed
    in to expose the encoder to both during training. Negative always
    from a different database.
    """
    if views is None:
        views = ["full", "schema"]

    rng = random.Random(seed)
    db_names = corpus.database_names
    if max_databases:
        db_names = db_names[:max_databases]

    # Pre-serialize under both views
    db_multi: Dict[str, Dict[str, Dict[str, str]]] = {}
    for db_name in db_names:
        try:
            db_rec = corpus.load_database(db_name)
        except Exception:
            continue
        table_views: Dict[str, Dict[str, str]] = {}
        for t in db_rec.tables.values():
            if t.rows:
                table_views[t.table_name] = {v: serialize(t, v) for v in views}
        if table_views:
            db_multi[db_name] = table_views

    all_db_names = list(db_multi.keys())
    if len(all_db_names) < 2:
        return []

    triplets = []
    attempts = 0

    while len(triplets) < target_triplets and attempts < target_triplets * 3:
        attempts += 1

        anchor_db = rng.choice(all_db_names)
        tables_a = db_multi[anchor_db]
        tnames = list(tables_a.keys())

        anchor_name = rng.choice(tnames)
        # Alternate between views for anchor
        anchor_view = rng.choice(views)
        anchor_text = tables_a[anchor_name][anchor_view]

        # Positive: different table, same DB, always full view
        others = [t for t in tnames if t != anchor_name]
        if not others:
            continue
        pos_name = rng.choice(others)
        positive_text = tables_a[pos_name]["full"]

        # Negative: different DB, full view
        neg_db = rng.choice([d for d in all_db_names if d != anchor_db])
        neg_tables = db_multi[neg_db]
        neg_name = rng.choice(list(neg_tables.keys()))
        negative_text = neg_tables[neg_name]["full"]

        triplets.append((anchor_text, positive_text, negative_text))

    logger.info(f"TC-Opt: generated {len(triplets)} multi-view triplets")
    return triplets


def prepare_tc_sb_triplets(
    corpus: WikiDBsCorpus,
    view: str = "full",
    target_triplets: int = 10000,
    seed: int = 42,
    max_databases: Optional[int] = None,
) -> List[Tuple[str, str, str]]:
    """
    TC-SB — triplets with smart batching (same-DB grouping).

    Same-DB positives; the returned triplets are grouped so that
    consecutive batches contain tables from related databases,
    ensuring more competitive negatives within each batch.
    """
    # Generate standard same-DB triplets
    triplets = prepare_tc_triplets(
        corpus,
        view=view,
        target_triplets=target_triplets,
        use_fk_positives=False,
        seed=seed,
        max_databases=max_databases,
    )
    # Smart batching: sort by anchor prefix to cluster same-DB tables
    triplets.sort(key=lambda t: t[0][:60])
    logger.info(f"TC-SB: {len(triplets)} triplets (smart-batch sorted)")
    return triplets


# ============================================================================
# Flat texts  (used by MLM, Hybrid stage 1)
# ============================================================================

def prepare_mlm_texts(
    corpus: WikiDBsCorpus,
    view: str = "full",
    max_databases: Optional[int] = None,
) -> List[str]:
    """
    MLM — flat serialized texts, one per table.

    Used to train masked language modeling on table structure.
    """
    db_names = corpus.database_names
    if max_databases:
        db_names = db_names[:max_databases]

    texts = []
    for db_name in db_names:
        try:
            db = corpus.load_database(db_name)
        except Exception:
            continue
        for table in db.tables.values():
            if not table.rows:
                continue
            texts.append(serialize(table, view))

    logger.info(f"MLM: collected {len(texts)} table texts")
    return texts
