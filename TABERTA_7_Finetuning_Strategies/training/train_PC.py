"""
Train PC — Pairwise Contrastive (Paper Table 1, Row 1)
======================================================
Supervision : Supervised
Objective   : InfoNCE (CosineSimilarityLoss — pairwise labels)
View        : RowView
Granularity : Row
Base model  : all-mpnet-base-v2

Positive pairs: two tables from the same database (label=0.5)
Negative pairs: two tables from different databases (label=0.0)
"""

import argparse
import logging
import os
import sys
import random
import torch

from contextlib import nullcontext
from sentence_transformers import SentenceTransformer, losses, InputExample, evaluation
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from taberta.data_loading import WikiDBsCorpus
from taberta.data_preparation import prepare_pc_pairs
from taberta.config import PCConfig
from taberta.metrics import MetricsLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("train_PC.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def _to_device(batch, device):
    if isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [_to_device(item, device) for item in batch]
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)
    return batch


def main():
    parser = argparse.ArgumentParser(description="Train PC — Pairwise Contrastive")
    parser.add_argument("--data-dir", type=str, default="data/wikidbs-10k/databases")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-databases", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    cfg = PCConfig()
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"PC Training | device={device} | epochs={cfg.epochs} | batch={cfg.batch_size}")

    # ---- Data ----
    corpus = WikiDBsCorpus(args.data_dir, row_limit=cfg.row_limit)
    logger.info(f"Corpus: {corpus.num_databases} databases")

    pairs = prepare_pc_pairs(corpus, view=cfg.views[0], max_databases=args.max_databases, seed=cfg.seed)

    random.seed(cfg.seed)
    random.shuffle(pairs)
    split = int(0.9 * len(pairs))
    train_pairs, val_pairs = pairs[:split], pairs[split:]

    train_examples = [InputExample(texts=[a, b], label=label) for a, b, label in train_pairs]
    val_examples = [InputExample(texts=[a, b], label=label) for a, b, label in val_pairs]
    logger.info(f"Train: {len(train_examples)} pairs | Val: {len(val_examples)} pairs")

    # ---- Model ----
    model = SentenceTransformer(cfg.base_model).to(device)
    loss_func = losses.CosineSimilarityLoss(model)

    collate_fn = lambda batch: batch
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=cfg.batch_size, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_examples, shuffle=False, batch_size=cfg.batch_size, collate_fn=collate_fn)

    # Evaluator for Pearson/Spearman
    evaluator = evaluation.EmbeddingSimilarityEvaluator(
        [ex.texts[0] for ex in val_examples],
        [ex.texts[1] for ex in val_examples],
        [ex.label for ex in val_examples],
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    # ---- Metrics ----
    os.makedirs(cfg.output_dir, exist_ok=True)
    metrics_csv = os.path.join(cfg.output_dir, "metrics.csv")
    ml = MetricsLogger(metrics_csv, strategy="PC")

    # ---- Train with early stopping ----
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_train_loss = 0.0
        for batch in train_dataloader:
            optimizer.zero_grad()
            features, labels = model.smart_batching_collate(batch)
            features = [_to_device(f, device) for f in features]

            ctx = torch.amp.autocast("cuda", dtype=torch.float16) if scaler else nullcontext()
            with ctx:
                loss = loss_func(features, labels.to(device))

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_train_loss += loss.item()

        avg_train = total_train_loss / max(len(train_dataloader), 1)

        # -- Validation loss --
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                features, labels = model.smart_batching_collate(batch)
                features = [_to_device(f, device) for f in features]
                ctx = torch.amp.autocast("cuda", dtype=torch.float16) if scaler else nullcontext()
                with ctx:
                    val_loss = loss_func(features, labels.to(device))
                total_val_loss += val_loss.item()
        avg_val = total_val_loss / max(len(val_dataloader), 1)

        # -- Evaluator (Pearson / Spearman) --
        eval_scores = evaluator(model, output_path=cfg.output_dir, epoch=epoch)
        # Extract primary scores from the returned dict
        pearson = None
        spearman = None
        for k, v in eval_scores.items():
            if "pearson" in k.lower():
                pearson = v
            if "spearman" in k.lower():
                spearman = v
        # Fallback: use first value if keys don't match expected names
        if pearson is None and eval_scores:
            pearson = next(iter(eval_scores.values()))

        pearson_str = f"{pearson:.4f}" if pearson is not None else "N/A"
        logger.info(f"Epoch {epoch}/{cfg.epochs}: train_loss={avg_train:.4f} | val_loss={avg_val:.4f} | pearson={pearson_str}")

        current_lr = optimizer.param_groups[0]["lr"]
        ml.log(epoch=epoch, train_loss=avg_train, val_loss=avg_val,
               learning_rate=current_lr, pearson=pearson, spearman=spearman,
               best_val_loss=min(best_val_loss, avg_val))

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            model.save(os.path.join(cfg.output_dir, "best"))
            logger.info(f"  -> New best model (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            logger.info(f"  -> No improvement ({patience_counter}/{cfg.patience})")
            if patience_counter >= cfg.patience:
                logger.info("Early stopping triggered.")
                break

    model.save(cfg.output_dir)
    logger.info(f"PC model saved to {cfg.output_dir}")
    logger.info(f"Metrics CSV: {metrics_csv}")


if __name__ == "__main__":
    main()
