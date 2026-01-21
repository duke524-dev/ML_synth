#!/bin/bash
# Setup script for artifacts directory structure

mkdir -p artifacts/current
mkdir -p artifacts/staging

echo "Created artifacts directory structure"
echo "Place model checkpoints in artifacts/current/:"
echo "  - hf_model.pt"
echo "  - lf_crypto_model.pt"
echo "  - lf_equity_model.pt"
echo ""
echo "Place normalization_stats.json and calibration.json in artifacts/current/"
