#!/bin/bash
# Download all TABERTA models from HuggingFace
# Usage: bash download_models.sh

set -e

echo "🤗 Downloading TABERTA Models from HuggingFace"
echo "=============================================="

# Check if hf CLI is installed
if ! command -v hf &> /dev/null; then
    echo "❌ hf CLI not found. Installing..."
    pip install --upgrade huggingface-hub[cli]
fi

# Create models directory
mkdir -p models

# All models are folders inside the single EnasKhwaileh/TABERTA repo
REPO="EnasKhwaileh/TABERTA"

models=(
    "1_Pairwise Contrastive (PC)"
    "2_Triplet Contrastive (TC) "
    "3_Triplet Contrastive (SmartBatch) (TC-SB)"
    "4_Triplet Contrastive (Optimized) (TC-opt) "
    "5_Self-Supervised Contrastive (SimCSE) (SS-C) "
    "6_Masked Language Modeling (MLM) "
    "7_hybrid_model_reg"
)

# Download each model folder from the repo
for model in "${models[@]}"; do
    echo ""
    echo "📥 Downloading $model..."
    hf download "$REPO" \
        --include "${model}/*" \
        --local-dir "./models"

    if [ $? -eq 0 ]; then
        echo "✅ $model downloaded successfully"
    else
        echo "⚠️  Failed to download $model"
    fi
done

echo ""
echo "✨ Download complete!"
echo "Models saved in ./models/"
echo ""
echo "To use a model:"
echo "  from sentence_transformers import SentenceTransformer"
echo "  model = SentenceTransformer('./models/7_hybrid_model_reg')"
