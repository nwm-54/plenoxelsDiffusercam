#!/bin/bash
#SBATCH --job-name=preprocess-shared-gsplat
#SBATCH --partition=default_partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --constraint="gpu-high"
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/preprocess_shared_%j.out
#SBATCH --error=slurm_logs/preprocess_shared_%j.err

# SLURM wrapper for scripts/preprocess_shared_gsplat.py.
#
# This script now materializes images/ automatically from raw inputs:
#   - If <train|eval>-dir contains videos, frames are extracted with ffmpeg.
#   - If it contains a single .lfr, it is decoded via lf_to_colmap.py.
#   - If it already contains images/, they are used as-is.
#
# Example:
#   sbatch scripts/preprocess_shared_gsplat.slurm \
#       --train-dir ../gs7/input_data/dog/iphone4 \
#       --eval-dir test=../gs7/input_data/dog/eval/test \
#       --fps 10 --covisible

set -euo pipefail
source ~/.bashrc

CONDA_ENV="${CONDA_ENV:-transformers}"
TRAIN_DIR=""
EVAL_COUNT=0
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --train-dir)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --train-dir expects a path." >&2
                exit 1
            fi
            TRAIN_DIR="$2"
            FORWARD_ARGS+=("$1" "$2")
            shift 2
            ;;
        --eval-dir)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --eval-dir expects name=path." >&2
                exit 1
            fi
            ((EVAL_COUNT+=1))
            FORWARD_ARGS+=("$1" "$2")
            shift 2
            ;;
        --conda-env)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --conda-env expects an environment name." >&2
                exit 1
            fi
            CONDA_ENV="$2"
            FORWARD_ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            FORWARD_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "${TRAIN_DIR}" ]]; then
    echo "ERROR: --train-dir is required." >&2
    exit 1
fi

if [[ ${EVAL_COUNT} -eq 0 ]]; then
    echo "ERROR: At least one --eval-dir is required." >&2
    exit 1
fi

CMD=(python "scripts/preprocess_shared_gsplat.py" "${FORWARD_ARGS[@]}")
echo ">>> Running in conda env '${CONDA_ENV}': ${CMD[*]}"
conda run -n "${CONDA_ENV}" "${CMD[@]}"
