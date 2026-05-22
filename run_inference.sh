#!/bin/bash

# PromptMR+ Inference Script
# This script sets up the Docker container and runs inference

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "🚀 PromptMR+ Inference"
echo "====================="

# Stop and remove old container if exists
echo "🔧 Cleaning up old container..."
sudo docker rm -f pmr_reconstruction 2>/dev/null || true

# Start new container with proper settings
echo "📦 Starting Docker container with 2GB shared memory..."
sudo docker run -dit \
  --name pmr_reconstruction \
  --shm-size=2gb \
  -v "$PROJECT_DIR:/app" \
  -v "/media/sbme26-600k/bitlocker_mount/CardiacCreed/cmrxrecon2025:/data" \
  --gpus all \
  pmr-plus-env > /dev/null

echo "✅ Container started"

# Run inference
echo "🎯 Running inference..."
sudo docker exec -it pmr_reconstruction python main.py predict \
  --config configs/base.yaml \
  --config configs/model/pmr-plus.yaml \
  --config configs/inference/pmr-plus/cmr24-cardiac.yaml

echo "✨ Done! Reconstructions saved to _predict/cmr24-cardiac/test_plus/reconstructions"
