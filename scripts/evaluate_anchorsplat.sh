#!/usr/bin/env bash
set -euo pipefail

GPUS=${GPUS:-"0"}
NPROC=${NPROC:-1}
PORT=${PORT:-29519}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/eval_anchorsplat_20x"}
CHECKPOINT=${CHECKPOINT:-"checkpoints/anchorsplat_20x.pth"}
POINT_MULTIPLY_FACTOR=${POINT_MULTIPLY_FACTOR:-20}
BATCH_SIZE=${BATCH_SIZE:-${NPROC}}
TEST_DATA_DIR=${TEST_DATA_DIR:-""}
DATA_CONFIG=${DATA_CONFIG:-"configs/dataset/objaverse.gin"}
MODEL_CONFIG=${MODEL_CONFIG:-"configs/model/ptv3.gin"}
TRAIN_CONFIG=${TRAIN_CONFIG:-"configs/train/default.gin"}

GIN_PARAMS=(
    --gin_param="build_trainloader.batch_size=${BATCH_SIZE}"
    --gin_param="FeaturePredictor.resume_ckpt='${CHECKPOINT}'"
    --gin_param="FeaturePredictor.point_multiply_factor=${POINT_MULTIPLY_FACTOR}"
)

if [[ -n "${TEST_DATA_DIR}" ]]; then
    GIN_PARAMS+=(
        --gin_param="test_dataset/GaussianSceneDataset.dataset_folder='${TEST_DATA_DIR}'"
    )
fi

CUDA_VISIBLE_DEVICES="${GPUS}" torchrun \
    --nnodes=1 \
    --nproc_per_node="${NPROC}" \
    --rdzv-endpoint="localhost:${PORT}" \
    train.py \
    --output_dir="${OUTPUT_DIR}" \
    --gin_file="${DATA_CONFIG}" \
    --gin_file="${MODEL_CONFIG}" \
    --gin_file="${TRAIN_CONFIG}" \
    "${GIN_PARAMS[@]}" \
    --only_eval \
    --eval_subdir="test" \
    --compare_with_input
