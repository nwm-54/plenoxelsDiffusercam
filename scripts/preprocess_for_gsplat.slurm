#!/bin/bash
#SBATCH --job-name=preprocess-gsplat
#SBATCH --partition=default_partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --constraint="gpu-high"
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/preprocess_%j.out
#SBATCH --error=slurm_logs/preprocess_%j.err

# Simple SLURM wrapper for scripts/preprocess_for_gsplat.py.
# Examples:
#   # Single-subset preprocessing from a video
#   sbatch scripts/preprocess_for_gsplat.slurm --input input_data/dog/iphone1 --fps 15 --overwrite
#
#   # Shared reconstruction (train + eval) — videos or .lfr also supported
#   sbatch scripts/preprocess_for_gsplat.slurm \
#       --input input_data/dog/iphone4 \
#       --eval-dir test=input_data/dog/eval/test \
#       --fps 10 --overwrite

set -euo pipefail
source ~/.bashrc

CONDA_ENV="${CONDA_ENV:-transformers}"

INPUT_PATH=""
FPS_VALUE=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)
            INPUT_PATH="$2"
            shift 2
            ;;
        --fps)
            FPS_VALUE="$2"
            shift 2
            ;;
        --dynamic|--overwrite)
            EXTRA_ARGS+=("$1")
            shift
            ;;
        --stage|--stage-cache|--scene-name|--lf-inner|--lf-downscale|--conda-env|--output-root)
            EXTRA_ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "${INPUT_PATH}" ]]; then
    echo "ERROR: --input path is required." >&2
    exit 1
fi

CMD=(python "scripts/preprocess_for_gsplat.py" --input "${INPUT_PATH}")
if [[ -n "${FPS_VALUE}" ]]; then
    CMD+=(--fps "${FPS_VALUE}")
fi
CMD+=("${EXTRA_ARGS[@]}")

echo ">>> Running in conda env '${CONDA_ENV}': ${CMD[*]}"
conda run -n "${CONDA_ENV}" "${CMD[@]}"
