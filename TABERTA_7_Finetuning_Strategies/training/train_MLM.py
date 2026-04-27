"""
Train MLM — Masked Language Modeling (Paper Table 1, Row 6)
===========================================================
Supervision : Self-supervised
Objective   : MLM loss (reconstruct masked tokens)
View        : FullView (masked)
Granularity : Token
Base model  : microsoft/mpnet-base  (raw MPNet with MLM head)

Trains the encoder to recover masked tokens from serialized table context.
The resulting model is also used as stage 1 of the Hybrid strategy.
"""

import argparse
import logging
import math
import os
import sys
import torch

from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from taberta.data_loading import WikiDBsCorpus
from taberta.data_preparation import prepare_mlm_texts
from taberta.config import MLMConfig
from taberta.metrics import MetricsLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("train_MLM.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class TokenizedTableDataset(Dataset):
    """Tokenizes serialized table texts for MLM training."""

    def __init__(self, texts, tokenizer, max_length=512):
        self.encodings = []
        for text in texts:
            enc = tokenizer(
                text, truncation=True, padding="max_length",
                max_length=max_length, return_tensors="pt",
            )
            enc = {k: v.squeeze(0) for k, v in enc.items()}
            enc["labels"] = enc["input_ids"].clone()
            self.encodings.append(enc)

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]


def main():
    parser = argparse.ArgumentParser(description="Train MLM — Masked Language Modeling")
    parser.add_argument("--data-dir", type=str, default="data/wikidbs-10k/databases")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-databases", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    cfg = MLMConfig()
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"MLM Training | device={device} | epochs={cfg.epochs} | mlm_prob={cfg.mlm_probability}")

    # ---- Data ----
    corpus = WikiDBsCorpus(args.data_dir, row_limit=cfg.row_limit)
    logger.info(f"Corpus: {corpus.num_databases} databases")

    texts = prepare_mlm_texts(corpus, view=cfg.views[0], max_databases=args.max_databases)
    logger.info(f"Table texts for MLM: {len(texts)}")

    # ---- Tokenizer & Model ----
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    model = AutoModelForMaskedLM.from_pretrained(cfg.base_model)
    model.to(device)

    full_dataset = TokenizedTableDataset(texts, tokenizer, max_length=cfg.max_seq_length)

    # Train/val split
    val_size = max(1, int(0.1 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    logger.info(f"Train: {train_size} | Val: {val_size}")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=cfg.mlm_probability,
    )

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=cfg.batch_size, collate_fn=data_collator)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=cfg.batch_size, collate_fn=data_collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    # ---- Metrics ----
    os.makedirs(cfg.output_dir, exist_ok=True)
    metrics_csv = os.path.join(cfg.output_dir, "metrics.csv")
    ml = MetricsLogger(metrics_csv, strategy="MLM")

    # ---- Train with early stopping ----
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_train_loss = 0.0
        num_batches = 0
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = model(**batch)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

            total_train_loss += loss.item()
            num_batches += 1

        avg_train = total_train_loss / max(num_batches, 1)
        perplexity_train = math.exp(min(avg_train, 30))  # cap to avoid overflow

        # -- Validation --
        model.eval()
        total_val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                if scaler:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        outputs = model(**batch)
                else:
                    outputs = model(**batch)
                total_val_loss += outputs.loss.item()
                val_batches += 1

        avg_val = total_val_loss / max(val_batches, 1)
        perplexity_val = math.exp(min(avg_val, 30))

        logger.info(
            f"Epoch {epoch}/{cfg.epochs}: train_loss={avg_train:.4f} (ppl={perplexity_train:.2f}) | "
            f"val_loss={avg_val:.4f} (ppl={perplexity_val:.2f})"
        )

        current_lr = optimizer.param_groups[0]["lr"]
        ml.log(epoch=epoch, train_loss=avg_train, val_loss=avg_val,
               learning_rate=current_lr, best_val_loss=min(best_val_loss, avg_val))

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            model.save_pretrained(os.path.join(cfg.output_dir, "best"))
            tokenizer.save_pretrained(os.path.join(cfg.output_dir, "best"))
            logger.info(f"  -> New best model (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            logger.info(f"  -> No improvement ({patience_counter}/{cfg.patience})")
            if patience_counter >= cfg.patience:
                logger.info("Early stopping triggered.")
                break

    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    logger.info(f"MLM model saved to {cfg.output_dir}")
    logger.info(f"Metrics CSV: {metrics_csv}")


if __name__ == "__main__":
    main()
