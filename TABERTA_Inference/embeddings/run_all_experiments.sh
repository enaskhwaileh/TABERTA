#!/bin/bash

# Run all dataset × model combinations for paper

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS=("spider" "wikidbs-10k" "fetaqa" "ottqa")
MODELS=("supervised_v1" "supervised_v2" "baseline_sbert" "baseline_mpnet")

echo "=========================================="
echo "TABERTA EXPERIMENTS"
echo "=========================================="
echo "Datasets: ${#DATASETS[@]}"
echo "Models: ${#MODELS[@]}"
echo "Total: $((${#DATASETS[@]} * ${#MODELS[@]})) combinations"
echo "=========================================="
echo ""

TOTAL=$((${#DATASETS[@]} * ${#MODELS[@]}))
CURRENT=0

for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        CURRENT=$((CURRENT + 1))
        echo ""
        echo "[$CURRENT/$TOTAL] Running: $dataset × $model"
        python3 "$SCRIPT_DIR/generate_all_embeddings_with_metrics.py" "$dataset" "$model"
        
        if [ $? -eq 0 ]; then
            echo "✓ Success: $dataset × $model"
        else
            echo "✗ Failed: $dataset × $model"
        fi
    done
done

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "=========================================="
echo ""
echo "Results in: $SCRIPT_DIR/metrics/"
find "$SCRIPT_DIR/metrics" -name "*.json" | wc -l
echo "JSON files created"
