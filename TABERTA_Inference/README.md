# TABERTA Inference

TABERTA Inference contains the public workflow for using TABERTA models to
encode tables, prepare benchmark datasets, build vector indexes, and run
retrieval evaluation.

## Repository Layout

```text
TABERTA_Inference/
├── datasets/
│   └── sample_data/          # Small public sample tables and queries
├── embeddings/               # Dataset loading, embedding, indexing, metrics
├── evaluation/               # Retrieval and baseline evaluation scripts
├── models/                   # Created locally by download_models.sh
├── utils/                    # Table conversion helpers
├── download_datasets.sh      # Optional public dataset downloader
├── download_models.sh        # TABERTA checkpoint downloader
└── requirements.txt
```

## Models

The released TABERTA checkpoints are hosted on Hugging Face under
`EnasKhwaileh/TABERTA`. To download them locally:

```bash
bash download_models.sh
```

The script places checkpoints under `models/`. The main TABERTA variants are:

| ID | Model |
| --- | --- |
| `supervised_v5` | Pairwise Contrastive (PC) |
| `supervised_v4` | Self-Supervised Contrastive / SimCSE (SS-C) |
| `supervised_v1` | Triplet Contrastive (TC) |
| `supervised_v2` | Optimized Triplet Contrastive (TC-opt) |
| `supervised_v3` | SmartBatch Triplet Contrastive (TC-SB) |
| `unsupervised_v6` | Masked Language Modeling (MLM) |
| `hybrid_v7` | Hybrid MLM + Triplet Contrastive |

Load a model directly with Sentence Transformers:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("models/7_hybrid_model_reg")
embeddings = model.encode(["table text to encode"], normalize_embeddings=True)
```

## Setup

Create an environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the full pipeline, run MongoDB and Qdrant locally:

```bash
docker run -d --name taberta-mongo -p 27017:27017 mongo:7
docker run -d --name taberta-qdrant -p 6333:6333 qdrant/qdrant
```

The scripts default to:

- MongoDB: `mongodb://localhost:27017/`
- Qdrant: `localhost:6333`
- datasets directory: `TABERTA_Inference/datasets`
- models directory: `TABERTA_Inference/models`

You can override paths with environment variables where supported, for example:

```bash
export TABERTA_DATASETS_DIR=/path/to/datasets
export TABERTA_MODELS_DIR=/path/to/models
export TABERTA_MONGODB_URI=mongodb://localhost:27017/
```

## Dataset Preparation

Only small sample files are committed. Full datasets are intentionally excluded
from Git.

To download public datasets:

```bash
bash download_datasets.sh
```

To load prepared dataset archives into MongoDB:

```bash
python embeddings/load_all_datasets.py
```

For targeted loaders:

```bash
python embeddings/load_datasets_to_mongodb.py spider
python embeddings/load_datasets_streaming.py tabfact
```

## Generate Embeddings

After downloading models and loading datasets, generate embeddings and create
Qdrant collections:

```bash
python embeddings/generate_all_embeddings_with_metrics.py spider hybrid_v7 100
```

Arguments:

- dataset name, such as `spider`, `tabfact`, `fetaqa`, `ottqa`, or
  `wikidbs-10k`
- model id, such as `hybrid_v7` or `supervised_v1`
- optional row limit for quick tests

Run all configured combinations:

```bash
python embeddings/generate_all_embeddings_with_metrics.py all
```

Generated metrics are written to `embeddings/metrics/` and vector collections
are written to Qdrant. These outputs are ignored by Git.

## Evaluate Retrieval

Run the baseline retrieval evaluation:

```bash
python evaluation/vanilla_sbert_benchmark.py
```

Run the Qwen3 embedding comparison when Qwen3 collections exist locally:

```bash
python evaluation/qwen3_effectiveness_evaluation.py --batch-size 8
```

Evaluation outputs are written to `evaluation/results/` and ignored by Git.

## Notes For Public Use

- Keep large datasets, checkpoints, generated embeddings, MongoDB data, and
  Qdrant storage outside Git.
- Commit only source code, sample data, and documentation.
- The fine-tuning workflow belongs in a separate top-level folder and is not
  included in this inference package.
