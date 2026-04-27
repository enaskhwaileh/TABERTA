#!/bin/bash
# Download all TABERTA datasets from HuggingFace
# Usage: bash download_datasets.sh

set -e

echo "🤗 Downloading TABERTA Datasets from HuggingFace"
echo "================================================"

# Check if hf CLI is installed
if ! command -v hf &> /dev/null; then
    echo "❌ hf CLI not found. Installing..."
    pip install --upgrade huggingface-hub[cli]
fi

# Create datasets directory
mkdir -p datasets

# Array of datasets to download
declare -A datasets
datasets=(
    # Fine tuning dataset
    #["structured-wikipedia"]="wikimedia/structured-wikipedia"
    
    # Inference datasets
    ["spider"]="xlangai/spider"
    ["bird-sql"]="birdsql/bird_sql_dev_20251106"
    ["tab-fact"]="wenhu/tab_fact"
    ["wikitables"]="hfhgj/wikitables"
    ["fetaqa"]="DongfuJiang/FeTaQA"
    ["ottqa-corpus"]="target-benchmark/ottqa-corpus"
)

# Download each dataset
for dataset_name in "${!datasets[@]}"; do
    dataset_repo="${datasets[$dataset_name]}"
    echo ""
    echo "📥 Downloading $dataset_name..."
    
    hf download "$dataset_repo" \
        --repo-type dataset \
        --local-dir "./datasets/$dataset_name"
    
    if [ $? -eq 0 ]; then
        echo "✅ $dataset_name downloaded successfully"
    else
        echo "⚠️  Failed to download $dataset_name (it may not exist yet on HuggingFace)"
    fi
done

echo ""
echo "✨ Download complete!"
echo "Datasets saved in ./datasets/"
echo ""
echo "Note: Sample data (100 tables, 20 queries) is already included in datasets/sample_data/"
