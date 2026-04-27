# TABERTA 7 Fine-Tuning Strategies

![TABERTA owl](taberta_owl.png)

This folder contains the public fine-tuning package for TABERTA. It includes the seven training strategies used to adapt table encoders for dataset discovery and table retrieval. Model checkpoints and datasets are intentionally not stored in Git; publish trained weights through Hugging Face or another model registry and keep local datasets outside version control.

## Strategies

| Strategy | Script | Supervision | Objective | Serialization view |
| --- | --- | --- | --- | --- |
| PC | `training/train_PC.py` | Supervised | Cosine similarity over pairs | RowView |
| SS-C | `training/train_SSC.py` | Self-supervised | Multiple negatives ranking | FullView |
| TC | `training/train_TC.py` | Supervised | Triplet loss | FullView |
| TC-Opt | `training/train_TC_Opt.py` | Supervised | Triplet loss with AMP/checkpointing | FullView + SchemaView |
| TC-SB | `training/train_TC_SB.py` | Supervised | Triplet loss with smart batching | FullView |
| MLM | `training/train_MLM.py` | Self-supervised | Masked language modeling | FullView |
| Hybrid | `training/train_Hybrid.py` | Hybrid | MLM stage followed by triplet loss | FullView, then FullView + SchemaView |

## Repository Layout

```text
TABERTA_7_Finetuning_Strategies/
|-- README.md
|-- requirements.txt
|-- model_manifest.md
|-- taberta/
|   |-- config.py
|   |-- data_loading.py
|   |-- data_preparation.py
|   |-- metrics.py
|   `-- serialization.py
`-- training/
    |-- train_PC.py
    |-- train_SSC.py
    |-- train_TC.py
    |-- train_TC_Opt.py
    |-- train_TC_SB.py
    |-- train_MLM.py
    `-- train_Hybrid.py
```

## Installation

Use Python 3.10 or newer.

```bash
cd TABERTA_7_Finetuning_Strategies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Format

Training scripts expect a WikiDBs-style directory:

```text
data/wikidbs-10k/databases/
`-- <database-name>/
    |-- info_full.json
    `-- tables/
        `-- <table>.csv
```

You can also pass any compatible path with `--data-dir`.

## Quick Start

Run a small smoke training job before launching a full run:

```bash
python training/train_TC.py \
  --data-dir data/wikidbs-10k/databases \
  --max-databases 50 \
  --epochs 1 \
  --output-dir models/TC_triplet_contrastive_smoke
```

Run the seven strategies independently:

```bash
python training/train_PC.py --data-dir data/wikidbs-10k/databases
python training/train_SSC.py --data-dir data/wikidbs-10k/databases
python training/train_TC.py --data-dir data/wikidbs-10k/databases
python training/train_TC_Opt.py --data-dir data/wikidbs-10k/databases
python training/train_TC_SB.py --data-dir data/wikidbs-10k/databases
python training/train_MLM.py --data-dir data/wikidbs-10k/databases
python training/train_Hybrid.py --data-dir data/wikidbs-10k/databases --mlm-model-path models/MLM_masked_lm
```

## Outputs

Each script writes checkpoints and metrics under `models/` by default. These outputs are ignored by Git because they can be large. For public release, upload model weights to Hugging Face and update `model_manifest.md` with the final model IDs.

## Notes

- `MLM` is a pre-training stage and is not intended as a standalone retriever.
- `Hybrid` expects the output of `train_MLM.py` through `--mlm-model-path`.
- The code uses CUDA when available and falls back to CPU.
