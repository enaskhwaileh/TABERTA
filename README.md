# TABERTA

<p align="center">
  <img src="taberta_owl.png" alt="TABERTA owl" width="380">
</p>

<p align="center">
  <strong>Structure-aware bi-encoders for table retrieval, dataset discovery, and table-grounded reasoning.</strong>
</p>

<p align="center">
  <a href="https://huggingface.co/EnasKhwaileh/TABERTA">Hugging Face Models</a> |
  <a href="TABERTA_Inference/README.md">Inference Guide</a> |
  <a href="TABERTA_7_Finetuning_Strategies/README.md">Fine-Tuning Guide</a>
</p>

TABERTA is a structure-aware fine-tuning framework for learning dense
representations of relational tables. Instead of treating a table as one flat
block of text, TABERTA exposes schema, row, and hybrid table views to
sentence-transformer bi-encoders, then fine-tunes those encoders with
retrieval-oriented objectives.

The goal is simple: given a natural-language query and a large corpus of
tables, TABERTA ranks the most relevant tables using efficient vector search.
This makes it useful for dataset discovery, evidence selection for table
question answering, fact verification, and schema grounding for text-to-SQL.

## What Is In This Repository

```text
TABERTA/
|-- README.md
|-- taberta_owl.png
|-- TABERTA_Inference/
|   |-- README.md
|   |-- download_models.sh
|   |-- download_datasets.sh
|   |-- embeddings/
|   |-- evaluation/
|   `-- datasets/sample_data/
`-- TABERTA_7_Finetuning_Strategies/
    |-- README.md
    |-- taberta/
    `-- training/
```

- `TABERTA_Inference/` contains the public workflow for downloading released
  checkpoints, preparing datasets, generating embeddings, indexing with Qdrant,
  and running retrieval evaluation.
- `TABERTA_7_Finetuning_Strategies/` contains training code for the seven
  TABERTA fine-tuning strategies.
- Large checkpoints, dataset archives, generated embeddings, vector indexes,
  and local database files are intentionally excluded from Git.

## Released Models

The released checkpoints are hosted on Hugging Face:

`https://huggingface.co/EnasKhwaileh/TABERTA`

| ID | Model | Main idea |
| --- | --- | --- |
| `supervised_v5` | Pairwise Contrastive (PC) | Learns relevance boundaries from positive and negative table pairs. |
| `supervised_v1` | Triplet Contrastive (TC) | Learns relative ranking with anchor, positive, and negative examples. |
| `supervised_v2` | Optimized Triplet Contrastive (TC-opt) | Adds harder negatives and optimization improvements for retrieval. |
| `supervised_v3` | SmartBatch Triplet Contrastive (TC-SB) | Uses smarter batching for triplet training. |
| `supervised_v4` | Self-Supervised Contrastive (SS-C / SimCSE) | Learns table representations from stochastic views without labels. |
| `unsupervised_v6` | Masked Language Modeling (MLM) | Uses token reconstruction as a table-aware pretraining signal. |
| `hybrid_v7` | Hybrid MLM + Contrastive | Combines pretraining and retrieval alignment. Recommended default. |

## Table Serialization Views

TABERTA studies how a table should be shown to an encoder:

- `SchemaView`: table names, column names, and type-like schema signals. This
  is compact and strong for dataset discovery.
- `RowView`: rows paired with schema context. This helps when relevance depends
  on specific values.
- `Hybrid` or `FullView`: schema plus sampled or aggregated content. This
  balances structural intent with value grounding and is the default style for
  broad evaluation.

## Quick Start: Inference

```bash
git clone https://github.com/enaskhwaileh/TABERTA.git
cd TABERTA/TABERTA_Inference

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

bash download_models.sh
```

Load a released encoder:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("models/7_hybrid_model_reg")

table_texts = [
    "[TABLE] countries [SCHEMA] country, capital, population "
    "[ROWS] France, Paris, 67M | Germany, Berlin, 83M"
]
query = "European countries and their populations"

table_embeddings = model.encode(table_texts, normalize_embeddings=True)
query_embedding = model.encode(query, normalize_embeddings=True)
scores = table_embeddings @ query_embedding
```

For the full MongoDB and Qdrant pipeline, see
[TABERTA_Inference/README.md](TABERTA_Inference/README.md).

## Quick Start: Fine-Tuning

```bash
cd TABERTA_7_Finetuning_Strategies

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python training/train_TC.py \
  --data-dir data/wikidbs-10k/databases \
  --max-databases 50 \
  --epochs 1 \
  --output-dir models/TC_triplet_contrastive_smoke
```

For the full training workflow, see
[TABERTA_7_Finetuning_Strategies/README.md](TABERTA_7_Finetuning_Strategies/README.md).

## Data And Evaluation

TABERTA is trained on WikiDBs-style relational databases and evaluated across
heterogeneous table retrieval settings, including schema-driven search,
value-grounded evidence retrieval, and schema/table grounding. The public
workflow supports datasets such as WikiDBs, WikiTables, TabFact, FeTaQA, OTTQA,
Spider, and BIRD when users download them locally.

Only small sample files are committed to this repository. Use the scripts in
`TABERTA_Inference/` to download or prepare larger datasets.

## Repository Hygiene

This repository is designed to stay lightweight and reproducible:

- Model checkpoints live on Hugging Face.
- Large datasets and generated embeddings stay outside Git.
- MongoDB data, Qdrant indexes, experiment outputs, logs, and local
  environments are ignored.
- Source code, documentation, sample data, and reproducible scripts are tracked.

## Citation

If TABERTA is useful in your work, please cite the TABERTA paper when the final
citation is available. A placeholder citation is included in the Hugging Face
model card.
