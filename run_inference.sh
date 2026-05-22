#!/bin/bash

# PromptMR+ Inference Script for CMRxRecon 2025
# This script sets up the Docker container and runs inference for all acceleration factors (8, 16, 24)

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "🚀 PromptMR+ Inference for CMRxRecon 2025"
echo "=========================================="

# Stop and remove old container if exists
echo "🔧 Cleaning up old container..."
sudo docker rm -f pmr_reconstruction 2>/dev/null || true

# Start new container with proper settings
echo "📦 Starting Docker container with 2GB shared memory..."
sudo docker run -dit \
  --name pmr_reconstruction \
  --shm-size=2gb \
  -v "$PROJECT_DIR:/app" \
  -v "/media/sbme26-600k/bitlocker_mount/CardiacCreed:/data" \
  --gpus all \
  pmr-plus-env > /dev/null

echo "✅ Container started"

# Define paths
DATA_PATH="/data/cmrxrecon2025/ValidationTestDataTaskR1/TaskR1_Data/TaskR1/MultiCoil/Cine/ValidationSet/FullSample_TaskR1"
CKPT_PATH="/data/Models/PromptMR-plus/weights/cmr24-cardiac/promptmr-plus-epoch=11-step=337764.ckpt"
OUTPUT_BASE="/app/_predict/cmr25-cardiac"

# Run inference for each acceleration factor
for acc in 8 16 24; do
    echo ""
    echo "🎯 Running inference with acceleration factor: ${acc}x"
    echo "----------------------------------------------"
    
    # Create output directory for this acceleration factor
    mkdir -p "${OUTPUT_BASE}/acc${acc}/reconstructions"
    
    # Run inference
    sudo docker exec -it pmr_reconstruction python main.py predict \
      --config configs/base.yaml \
      --config configs/model/pmr-plus.yaml \
      --config configs/inference/pmr-plus/cmr25-cardiac.yaml \
      --data.init_args.mask_acc ${acc} \
      --data.init_args.test_transform.init_args.mask_func.init_args.mask_acc ${acc} \
      --trainer.callbacks.0.init_args.output_dir "${OUTPUT_BASE}/acc${acc}"
    
    echo "✅ Inference complete for ${acc}x acceleration"
done

echo ""
echo "✨ All reconstructions complete!"
echo "📁 Results saved to: ${OUTPUT_BASE}"
echo ""
echo "📊 To evaluate performance metrics, run:"
echo "   python Performance-Metric_anas.py \\"
echo "     --recon_dir ${OUTPUT_BASE} \\"
echo "     --gt_dir ${DATA_PATH} \\"
echo "     --output_dir ${OUTPUT_BASE}/metrics \\"
echo "     --acc_factors 8 16 24"
echo ""

