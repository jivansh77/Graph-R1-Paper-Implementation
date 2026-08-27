#!/bin/bash
# Push and run Graph-R1 experiments on Kaggle
# Usage: bash scripts/run_kaggle.sh

set -e

echo "=== Graph-R1 Kaggle Experiment Runner ==="

# Check kaggle CLI
if ! command -v kaggle &> /dev/null; then
    echo "Error: kaggle CLI not installed. Run: pip install kaggle"
    exit 1
fi

# Push the notebook
echo "Pushing notebook to Kaggle..."
cd notebooks/
kaggle kernels push
cd ..

echo ""
echo "Notebook submitted! Monitor at:"
echo "  kaggle kernels status jivanshc/graph-r1-reproduction"
echo ""
echo "To check output:"
echo "  kaggle kernels output jivanshc/graph-r1-reproduction -p experiments/"
echo ""
echo "To view logs:"
echo "  kaggle kernels status jivanshc/graph-r1-reproduction"
