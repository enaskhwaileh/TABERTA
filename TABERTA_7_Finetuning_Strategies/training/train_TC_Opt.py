"""
Train TC-Opt — Triplet Contrastive Optimized (Paper Table 1, Row 4)
===================================================================
Supervision : Supervised
Objective   : Margin-based triplet loss + AMP + gradient checkpointing
Views       : FullView + SchemaView (multi-view exposure)
Granularity : Table
Base model  : all-mpnet-base-v2

Same triplet objective as TC, but with:
  - Automatic Mixed Precision (AMP) for memory efficiency
  - Multi-view serialization: anchor alternates between FullView/SchemaView
  - Enables larger batches and longer inputs
"""

import argparse
import logging
import os
import sys
import random
import torch

from contextlib import nullcontext
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from taberta.data_loading import WikiDBsCorpus
from taberta.data_preparation import prepare_tc_opt_triplets
from taberta.config import TCOptConfig
from taberta.metrics import MetricsLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("train_TC_Opt.log"), logging.StreamHandler()],
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
    parser = argparse.ArgumentParser(description="Train TC-Opt — Triplet Contrastive Optimized")
    parser.add_argument("--data-dir", type=str, default="data/wikidbs-10k/databases")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-databases", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--target-triplets", type=int, default=None)
    args = parser.parse_args()

    cfg = TCOptConfig()
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.target_triplets:
        cfg.target_triplets = args.target_triplets

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"TC-Opt Training | device={device} | epochs={cfg.epochs} | AMP={cfg.use_amp}")

    # ---- Data ----
    corpus = WikiDBsCorpus(args.data_dir, row_limit=cfg.row_limit)
    logger.info(f"Corpus: {corpus.num_databases} databases")

    triplets = prepare_tc_opt_triplets(
        corpus, views=cfg.views, target_triplets=cfg.target_triplets,
        seed=cfg.seed, max_databases=args.max_databases,
    )

    all_examples = [InputExample(texts=[a, p, n]) for a, p, n in triplets]
    random.seed(cfg.seed)
    random.shuffle(all_examples)
    split_idx = int(0.9 * len(all_examples))
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]
    logger.info(f"Train: {len(train_examples)} | Val: {len(val_examples)}")

    # ---- Model ----
    model = SentenceTransformer(cfg.base_model).to(device)
    loss_func = losses.TripletLoss(model=model)

    collate_fn = lambda batch: batch
    train_dataloader = DataLoader(
        train_examples, shuffle=True, batch_size=cfg.batch_size,
        pin_memory=True, collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(val_examples, shuffle=False, batch_size=cfg.batch_size, collate_fn=collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scaler = torch.amp.GradScaler("cuda") if (device == "cuda" and cfg.use_amp) else None

    # ---- Metrics ----
    os.makedirs(cfg.output_dir, exist_ok=True)
    metrics_csv = os.path.join(cfg.output_dir, "metrics.csv")
    ml = MetricsLogger(metrics_csv, strategy="TC-Opt")

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

        logger.info(f"Epoch {epoch}/{cfg.epochs}: train_loss={avg_train:.4f} | val_loss={avg_val:.4f}")

        current_lr = optimizer.param_groups[0]["lr"]
        ml.log(epoch=epoch, train_loss=avg_train, val_loss=avg_val,
               learning_rate=current_lr, best_val_loss=min(best_val_loss, avg_val))

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
    logger.info(f"TC-Opt model saved to {cfg.output_dir}")
    logger.info(f"Metrics CSV: {metrics_csv}")


if __name__ == "__main__":
    main()
