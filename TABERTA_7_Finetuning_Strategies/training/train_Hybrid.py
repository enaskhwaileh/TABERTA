"""
Train Hybrid — Hybrid MLM + TC (Paper Table 1, Row 7)
=====================================================
Supervision : Hybrid (staged)
Objective   : Stage 1: MLM loss → Stage 2: Triplet loss
Views       : Stage 1: FullView → Stage 2: SchemaView + FullView
Granularity : Table
Base model  : Stage 1 output (MLM fine-tuned MPNet)

Two-stage training:
  1. MLM pre-training adapts the encoder to table structure (run train_MLM.py first)
  2. Supervised triplet fine-tuning aligns embeddings for retrieval

This script handles stage 2. It loads the MLM model, wraps it as a
SentenceTransformer, and fine-tunes with triplet loss + early stopping.
"""

import argparse
import logging
import os
import random
import sys
import torch

from contextlib import nullcontext
from sentence_transformers import SentenceTransformer, losses, InputExample, models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from taberta.data_loading import WikiDBsCorpus
from taberta.data_preparation import prepare_tc_opt_triplets
from taberta.config import HybridConfig
from taberta.metrics import MetricsLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("train_Hybrid.log"), logging.StreamHandler()],
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


def _load_mlm_as_sentence_transformer(mlm_model_path: str) -> SentenceTransformer:
    """
    Convert an MLM-pretrained HuggingFace model into a SentenceTransformer.

    The original code had a bug here: script 6 saves as AutoModelForMaskedLM
    but script 7 tried to load it as SentenceTransformer directly — incompatible
    formats. This function properly bridges the two by extracting the base
    transformer and adding a pooling layer.
    """
    word_embedding_model = models.Transformer(mlm_model_path)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
    )
    normalize_model = models.Normalize()
    return SentenceTransformer(modules=[word_embedding_model, pooling_model, normalize_model])


def main():
    parser = argparse.ArgumentParser(description="Train Hybrid — MLM + TC (Stage 2)")
    parser.add_argument("--data-dir", type=str, default="data/wikidbs-10k/databases")
    parser.add_argument("--mlm-model-path", type=str, default=None,
                        help="Path to the MLM stage 1 model (output of train_MLM.py)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-databases", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--target-triplets", type=int, default=None)
    args = parser.parse_args()

    cfg = HybridConfig()
    if args.mlm_model_path:
        cfg.mlm_model_path = args.mlm_model_path
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.target_triplets:
        cfg.target_triplets = args.target_triplets

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Hybrid Training (Stage 2) | device={device} | epochs={cfg.epochs}")
    logger.info(f"Loading MLM model from: {cfg.mlm_model_path}")

    if not os.path.exists(cfg.mlm_model_path):
        logger.error(
            f"MLM model not found at '{cfg.mlm_model_path}'. "
            "Run train_MLM.py first to produce the stage 1 model."
        )
        sys.exit(1)

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

    # ---- Model (load MLM stage 1 → wrap as SentenceTransformer) ----
    model = _load_mlm_as_sentence_transformer(cfg.mlm_model_path)
    model = model.to(device)
    loss_func = losses.TripletLoss(model)

    collate_fn = lambda batch: batch
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=cfg.batch_size, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_examples, shuffle=False, batch_size=cfg.batch_size, collate_fn=collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    # ---- Metrics ----
    os.makedirs(cfg.output_dir, exist_ok=True)
    metrics_csv = os.path.join(cfg.output_dir, "metrics.csv")
    ml = MetricsLogger(metrics_csv, strategy="Hybrid")

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

        # -- Validation --
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
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Epoch {epoch}/{cfg.epochs}: train_loss={avg_train:.4f} | val_loss={avg_val:.4f} | lr={current_lr:.2e}")

        # -- Early stopping --
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            model.save(os.path.join(cfg.output_dir, "best"))
            logger.info(f"  -> New best model (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            logger.info(f"  -> No improvement ({patience_counter}/{cfg.patience})")

        ml.log(epoch=epoch, train_loss=avg_train, val_loss=avg_val,
               learning_rate=current_lr, best_val_loss=best_val_loss)

        if patience_counter >= cfg.patience:
            logger.info("Early stopping triggered.")
            ml.log(epoch=epoch, train_loss=avg_train, val_loss=avg_val,
                   learning_rate=current_lr, best_val_loss=best_val_loss,
                   early_stopped=True)
            break

    model.save(cfg.output_dir)
    logger.info(f"Hybrid model saved to {cfg.output_dir}")
    logger.info(f"Metrics CSV: {metrics_csv}")


if __name__ == "__main__":
    main()
