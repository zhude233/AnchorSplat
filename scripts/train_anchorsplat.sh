#!/usr/bin/env bash
set -euo pipefail

GPUS=${GPUS:-"0,1,2,3,4,5,6,7"}
NPROC=${NPROC:-8}
ACCUMULATE_STEP=${ACCUMULATE_STEP:-1}
BATCH_SIZE=${BATCH_SIZE:-$((NPROC * ACCUMULATE_STEP))}
PORT=${PORT:-29518}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/anchorsplat_20x"}
DATA_CONFIG=${DATA_CONFIG:-"configs/dataset/objaverse.gin"}
MODEL_CONFIG=${MODEL_CONFIG:-"configs/model/ptv3.gin"}
TRAIN_CONFIG=${TRAIN_CONFIG:-"configs/train/default.gin"}

CUDA_VISIBLE_DEVICES="${GPUS}" torchrun \
    --nnodes=1 \
    --nproc_per_node="${NPROC}" \
    --rdzv-endpoint="localhost:${PORT}" \
    train.py \
    --output_dir="${OUTPUT_DIR}" \
    --gin_file="${DATA_CONFIG}" \
    --gin_file="${MODEL_CONFIG}" \
    --gin_file="${TRAIN_CONFIG}" \
    --gin_param="build_trainloader.batch_size=${BATCH_SIZE}" \
    --gin_param="build_trainloader.accumulate_step=${ACCUMULATE_STEP}"
