#!/bin/bash

# TABERTA Full Pipeline Execution Script
# Expected runtime: ~10-15 hours
# Output: Qdrant collections and metrics files for configured datasets/models

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "════════════════════════════════════════════════════════════════"
echo "🚀 TABERTA Full Pipeline - Starting at $(date)"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Step 1: Load all datasets to MongoDB using STREAMING loader (no disk extraction!)
echo "📥 STEP 1/2: Loading all datasets to MongoDB..."
echo "────────────────────────────────────────────────────────────────"
echo "🌊 Using STREAMING mode - no disk extraction needed!"
echo ""
python3 "$SCRIPT_DIR/load_datasets_streaming.py" all
LOAD_EXIT=$?

if [ $LOAD_EXIT -ne 0 ]; then
    echo ""
    echo "❌ ERROR: Dataset loading failed with exit code $LOAD_EXIT"
    echo "Pipeline stopped at $(date)"
    exit $LOAD_EXIT
fi

echo ""
echo "✅ All datasets loaded successfully"
echo ""

# Step 2: Generate embeddings for all 54 combinations
echo "🔄 STEP 2/2: Generating embeddings for configured combinations..."
echo "────────────────────────────────────────────────────────────────"
echo "Expected time: ~10-15 hours"
echo ""

python3 "$SCRIPT_DIR/generate_all_embeddings_with_metrics.py" all
GEN_EXIT=$?

echo ""
echo "════════════════════════════════════════════════════════════════"
if [ $GEN_EXIT -eq 0 ]; then
    echo "✅ PIPELINE COMPLETED SUCCESSFULLY at $(date)"
else
    echo "❌ Pipeline failed with exit code $GEN_EXIT at $(date)"
fi
echo "════════════════════════════════════════════════════════════════"

exit $GEN_EXIT
