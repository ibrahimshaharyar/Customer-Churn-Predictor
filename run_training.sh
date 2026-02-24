#!/bin/bash
# Run the complete training pipeline

echo "=================================="
echo "Starting Training Pipeline"
echo "=================================="
echo ""

cd "$(dirname "$0")"
python -m src.main_train

echo ""
echo "=================================="
echo "Training Pipeline Complete!"
echo "=================================="
