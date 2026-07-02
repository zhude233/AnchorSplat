#!/bin/bash
# Inference script for LGM and Trellis generated PLY files
#
# Usage:
#   ./scripts/inference_external.sh <input_ply> <output_ply> <model_type>
#
# model_type: lgm or trellis
#
# Optional:
#   WEIGHTS=checkpoints/anchorsplat_20x.pth ./scripts/inference_external.sh input.ply output.ply lgm
#   NORMALIZATION=bbox MAX_INPUT_GAUSSIANS=200000 ./scripts/inference_external.sh input.ply output.ply lgm

INPUT_PLY=${1:-""}
OUTPUT_PLY=${2:-""}
MODEL_TYPE=${3:-"lgm"}
WEIGHTS=${WEIGHTS:-"checkpoints/anchorsplat_20x.pth"}
NORMALIZATION=${NORMALIZATION:-"auto"}
MAX_INPUT_GAUSSIANS=${MAX_INPUT_GAUSSIANS:-0}

if [ -z "$INPUT_PLY" ] || [ -z "$OUTPUT_PLY" ]; then
    echo "Usage: $0 <input_ply> <output_ply> <model_type>"
    echo "  model_type: lgm or trellis (default: lgm)"
    exit 1
fi

echo "================================================"
echo "Running inference on external PLY file"
echo "  Input: $INPUT_PLY"
echo "  Output: $OUTPUT_PLY"
echo "  Model Type: $MODEL_TYPE"
echo "  Weights: $WEIGHTS"
echo "  Normalization: $NORMALIZATION"
echo "  Max Input Gaussians: $MAX_INPUT_GAUSSIANS"
echo "================================================"

python inference_external.py \
    --weights "$WEIGHTS" \
    --input_ply "$INPUT_PLY" \
    --output_ply "$OUTPUT_PLY" \
    --model_type "$MODEL_TYPE" \
    --normalization "$NORMALIZATION" \
    --max_input_gaussians "$MAX_INPUT_GAUSSIANS"
