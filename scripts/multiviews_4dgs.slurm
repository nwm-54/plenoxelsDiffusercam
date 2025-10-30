#!/bin/bash
#SBATCH --job-name=4dgs_multiview
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --error=slurm_logs/%A_%a.err
#SBATCH --partition=default_partition
#SBATCH --gres=gpu:1
#SBATCH --exclude=lil-compute-05,unicorn-compute-01,bhattacharjee-compute-02
#SBATCH --constraint="gpu-high"
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --array=0-8

source /home/wl757/.bashrc

WORKDIR="/home/wl757/multiplexed-pixels/gs7"
cd "${WORKDIR}"

set -eo pipefail

if [[ -z "${ADDR2LINE:-}" ]]; then
    if command -v addr2line >/dev/null 2>&1; then
        export ADDR2LINE="$(command -v addr2line)"
    else
        export ADDR2LINE=""
    fi
fi

CAMERA_TYPES=(monocular stereo iphone)
VIEW_COUNTS=(1 3 5)
TOTAL_TYPES=${#CAMERA_TYPES[@]}
TOTAL_VIEWS=${#VIEW_COUNTS[@]}
TOTAL_JOBS=$((TOTAL_TYPES * TOTAL_VIEWS))

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= TOTAL_JOBS )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID ${TASK_ID} out of range (expected 0-$((TOTAL_JOBS - 1)))." >&2
    exit 1
fi

CAMERA_INDEX=$((TASK_ID / TOTAL_VIEWS))
VIEW_INDEX=$((TASK_ID % TOTAL_VIEWS))

CAMERA_MODE="${CAMERA_TYPES[CAMERA_INDEX]}"
N_TRAIN_VIEWS="${VIEW_COUNTS[VIEW_INDEX]}"
ANIMATION_VIEWS=60

# ---------------------------------------------------------------------------- #
# User-set configuration shared by all jobs
# ---------------------------------------------------------------------------- #
DATASET_BASE_NAME="demo"
DATASET_SOURCE="/home/wl757/multiplexed-pixels/plenoxels/blender_data/lego_gen12"
FOURD_REPO="../4DGaussians"

# ---------------------------------------------------------------------------- #
# Derived configuration for the selected array task
# ---------------------------------------------------------------------------- #
printf -v DATASET_NAME "%s_%s_%dv" "${DATASET_BASE_NAME}" "${CAMERA_MODE}" "${N_TRAIN_VIEWS}"
MODEL_DIR="./output5/${DATASET_NAME}"
TRANSFORMS_JSON="${MODEL_DIR}/transforms.json"
BLENDER_BLEND="blender/lego.blend"
BLENDER_SCRIPT="blender/render_blender.py"
RENDER_DIR="./renders/${DATASET_NAME}"
FOURD_DATASET="${FOURD_REPO}/data/multipleview/${DATASET_NAME}"
FOURD_EXPERIMENT="multipleview/${DATASET_NAME}"
FOURD_MODEL_PATH_REL="output/${FOURD_EXPERIMENT}"
FOURD_MODEL_DIR="${FOURD_REPO}/${FOURD_MODEL_PATH_REL}"

TRAIN_ARGS=()
case "${CAMERA_MODE}" in
    monocular)
        CAMERA_DESC="Monocular"
        ;;
stereo)
    CAMERA_DESC="Stereo"
    TRAIN_ARGS+=(--use_stereo --angle_deg 10)
    ;;
iphone)
    CAMERA_DESC="iPhone"
    TRAIN_ARGS+=(--use_iphone --angle_deg 10)
    ;;
    *)
        echo "ERROR: Unsupported camera mode '${CAMERA_MODE}'." >&2
        exit 1
        ;;
esac

BLENDER_ARGS=(
    --trajectory random_static
    --num-cams "${N_TRAIN_VIEWS}"
    --loc-upper
    --views "${ANIMATION_VIEWS}"
    --arm-mode range
    --arm-z-start 11.0
    --arm-z-end 3.0
)

FOURD_CONVERT_ARGS=(
    --test_blender "/share/monakhova/shamus_data/multiplexed_pixels/dnerf/lego"
    --test_transforms "transforms_val.json"
    --video_blender "/share/monakhova/shamus_data/multiplexed_pixels/dnerf/lego"
    --video_transforms "transforms_val.json"
    --vggt_skip_tracks
)

FOURD_TRAIN_ARGS=()

# Prefer dataset-specific config if available, otherwise fall back to default.
FOURD_CONFIG="arguments/multipleview/${DATASET_NAME}.py"
if [[ ! -f "${FOURD_REPO}/${FOURD_CONFIG}" ]]; then
    FOURD_CONFIG="arguments/multipleview/default.py"
fi

if [[ -d "${FOURD_MODEL_DIR}" ]]; then
    echo ">>> Removing existing 4DGaussians output at ${FOURD_MODEL_DIR}..."
    rm -rf "${FOURD_MODEL_DIR}"
fi

if [[ -d "${FOURD_DATASET}" ]]; then
    echo ">>> Removing existing converted dataset at ${FOURD_DATASET}..."
    rm -rf "${FOURD_DATASET}"
fi

mkdir -p slurm_logs "${MODEL_DIR}" "${RENDER_DIR}" "${FOURD_REPO}/data/multipleview"

echo "=========================================="
echo "4DGAUSSIANS MULTI-VIEW PIPELINE"
echo "=========================================="
echo "Job ID: ${SLURM_ARRAY_JOB_ID:-N/A}"
echo "Array Task ID: ${TASK_ID}"
echo "Camera mode: ${CAMERA_DESC} (${CAMERA_MODE})"
echo "Train views: ${N_TRAIN_VIEWS}"
echo "Dataset base: ${DATASET_SOURCE}"
echo "Dataset name: ${DATASET_NAME}"
echo "Model dir: ${MODEL_DIR}"
echo "Render dir: ${RENDER_DIR}"
echo "4D dataset: ${FOURD_DATASET}"
echo "Experiment: ${FOURD_EXPERIMENT}"
echo "Config file: ${FOURD_CONFIG}"
echo "=========================================="
echo ""

TRAIN_BASE_ARGS=(
    -s "${DATASET_SOURCE}"
    -m "${MODEL_DIR}"
    --skip_train
    --n_train_images "${N_TRAIN_VIEWS}"
)

BLENDER_BASE_ARGS=(
    --transforms-json "${TRANSFORMS_JSON}"
    --results "${RENDER_DIR}"
)

FOURD_CONVERT_BASE_ARGS=(
    --blender "${RENDER_DIR}"
    --output "${FOURD_DATASET}"
    --dataset_name "${DATASET_NAME}"
)

FOURD_TRAIN_BASE_ARGS=(
    -s "${FOURD_DATASET}"
    --expname "${FOURD_EXPERIMENT}"
    --configs "${FOURD_CONFIG}"
)

# ---------------------------------------------------------------------------- #
# Pipeline
# ---------------------------------------------------------------------------- #
echo ">>> Generating camera trajectories (${CAMERA_DESC}, ${N_TRAIN_VIEWS} view[s])..."
conda run -n gaussian_splatting python train_sim_multiviews.py \
    "${TRAIN_BASE_ARGS[@]}" \
    "${TRAIN_ARGS[@]}"
echo ""

echo ">>> Rendering Blender frames..."
conda run -n 4dgaussians blender -b "${BLENDER_BLEND}" -P "${BLENDER_SCRIPT}" -- \
    "${BLENDER_BASE_ARGS[@]}" \
    "${BLENDER_ARGS[@]}"
echo ""

echo ">>> Converting Blender outputs to 4DGaussians dataset..."
conda run -n 4dgaussians python "${FOURD_REPO}/blender_to_4dgs.py" \
    "${FOURD_CONVERT_BASE_ARGS[@]}" \
    "${FOURD_CONVERT_ARGS[@]}"
echo ""

echo ">>> Clearing Python caches in ${FOURD_REPO}..."
find "${FOURD_REPO}" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find "${FOURD_REPO}" -name "*.pyc" -delete 2>/dev/null || true
echo ""

FOURD_MODEL_DIR="${FOURD_REPO}/${FOURD_MODEL_PATH_REL}"
mkdir -p "${FOURD_MODEL_DIR}"
if [[ -f "${FOURD_MODEL_DIR}/cfg_args" ]]; then
    echo ">>> Removing stale cfg_args at ${FOURD_MODEL_DIR}/cfg_args..."
    rm -f "${FOURD_MODEL_DIR}/cfg_args"
fi

pushd "${FOURD_REPO}" >/dev/null
echo ">>> Training 4DGaussians model (includes evaluation & metrics)..."
conda run -n 4dgaussians python train.py \
    "${FOURD_TRAIN_BASE_ARGS[@]}" \
    "${FOURD_TRAIN_ARGS[@]}"
popd >/dev/null

echo ""
echo "=========================================="
echo "ARRAY TASK ${TASK_ID} COMPLETE (${CAMERA_DESC}, ${N_TRAIN_VIEWS} view[s])"
echo "Results saved to: ${FOURD_REPO}/${FOURD_MODEL_PATH_REL}"
echo "=========================================="
