#!/bin/bash
#SBATCH --job-name=mv_4dgs
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#SBATCH --partition=default_partition
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-high"
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --time=4-8:00:00
#SBATCH --mem=32G

set -euo pipefail

source /home/wl757/.bashrc
export TMPDIR=/tmp

WORKDIR="/home/wl757/multiplexed-pixels/4DGaussians"
cd "${WORKDIR}"

# ---------------------------------------------------------------------------- #
# User-set configuration
# ---------------------------------------------------------------------------- #
DATASET_NAME="vision_pro_spinner"
RAW_VIDEO_ROOT="${WORKDIR}/data/${DATASET_NAME}"
CONVERT_EXTRA_ARGS=(
    # "--recursive"
    # "--frame_stride" "2"
)

TRAIN_EXTRA_ARGS=(
    # "--skip_post_eval"
)

# ---------------------------------------------------------------------------- #
# Inferred paths
# ---------------------------------------------------------------------------- #
SLURM_LOG_DIR="slurm_logs"
DATASET_OUTPUT="${WORKDIR}/data/multipleview/${DATASET_NAME}"
EXPERIMENT_NAME="multipleview/${DATASET_NAME}"

mkdir -p "${SLURM_LOG_DIR}" "${DATASET_OUTPUT}"

# ---------------------------------------------------------------------------- #
# Stage 1: Convert raw multi-view videos into 4DGS dataset structure
# ---------------------------------------------------------------------------- #
set +u
conda activate 4dgaussians
set -u

python video_to_4dgs.py \
    --videos "${RAW_VIDEO_ROOT}" \
    --output "${DATASET_OUTPUT}" \
    --dataset_name "${DATASET_NAME}" \
    "${CONVERT_EXTRA_ARGS[@]}"

# ---------------------------------------------------------------------------- #
# Stage 2: Train 4D Gaussians (renders + metrics run automatically afterward)
# ---------------------------------------------------------------------------- #
python train.py \
    -s "${DATASET_OUTPUT}" \
    --expname "${EXPERIMENT_NAME}" \
    "${TRAIN_EXTRA_ARGS[@]}"

conda deactivate || true
