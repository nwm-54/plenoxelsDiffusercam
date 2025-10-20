#!/bin/bash
#SBATCH --job-name=lego_4dgs
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#SBATCH --partition=default_partition
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-high"
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --time=4-8:00:00
#SBATCH --mem=32G

source /home/wl757/.bashrc

export TMPDIR=/tmp

WORKDIR="/home/wl757/multiplexed-pixels/gs7"
cd "${WORKDIR}"

# ---------------------------------------------------------------------------- #
# User-set configuration
# ---------------------------------------------------------------------------- #
DATASET_NAME="demo_stereo"
DATASET_SOURCE="/home/wl757/multiplexed-pixels/plenoxels/blender_data/lego_gen12"
N_TRAIN_VIEWS=3
ANIMATION_VIEWS=60

TRAIN_ARGS=(
    # --use_iphone
    --use_stereo
    # "--dls"
    # "--use_multiplexing"
)

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
    # "--skip_vggt"
    --test_blender "/share/monakhova/shamus_data/multiplexed_pixels/dnerf/lego"
    --test_transforms "transforms_val.json"
    --video_blender "/share/monakhova/shamus_data/multiplexed_pixels/dnerf/lego"
    --video_transforms "transforms_val.json"
)

FOURD_TRAIN_ARGS=(
    # "--test_iterations" "1000"
)

FOURD_RENDER_ARGS=(
    # "--skip_train"
    # "--skip_test"
    # "--skip_video"
    # "--iteration" "15000"
)

FOURD_METRICS_ARGS=(
    # "/another/output/path"
)

# ---------------------------------------------------------------------------- #
# Inferred paths and arguments (derive from the user settings above)
# ---------------------------------------------------------------------------- #
MODEL_DIR="./output5/${DATASET_NAME}"
TRANSFORMS_JSON="${MODEL_DIR}/transforms.json"
BLENDER_BLEND="blender/lego.blend"
BLENDER_SCRIPT="blender/render_blender.py"
RENDER_DIR="./renders/${DATASET_NAME}"
FOURD_REPO="../4DGaussians"
FOURD_DATASET="${FOURD_REPO}/data/multipleview/${DATASET_NAME}"
FOURD_CONFIG="arguments/multipleview/${DATASET_NAME}.py"
FOURD_EXPERIMENT="multipleview/${DATASET_NAME}"
FOURD_MODEL_PATH_REL="output/${FOURD_EXPERIMENT}"

mkdir -p slurm_logs "${MODEL_DIR}" "${RENDER_DIR}"

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

FOURD_RENDER_BASE_ARGS=(
    --model_path "${FOURD_MODEL_PATH_REL}"
    --configs "${FOURD_CONFIG}"
)

FOURD_METRICS_BASE_ARGS=(
    "${FOURD_MODEL_PATH_REL}"
)

# ---------------------------------------------------------------------------- #
# Pipeline
# ---------------------------------------------------------------------------- #
set +u
conda activate gaussian_splatting
set -u
echo "Generating views with Gaussian Splatting..."
python train_sim_multiviews.py \
    "${TRAIN_BASE_ARGS[@]}" \
    "${TRAIN_ARGS[@]}"
conda deactivate || true

set +u
conda activate 4dgaussians
set -u
echo "Rendering with Blender..."
blender -b "${BLENDER_BLEND}" -P "${BLENDER_SCRIPT}" -- \
    "${BLENDER_BASE_ARGS[@]}" \
    "${BLENDER_ARGS[@]}"

python "${FOURD_REPO}/blender_to_4dgs.py" \
    "${FOURD_CONVERT_BASE_ARGS[@]}" \
    "${FOURD_CONVERT_ARGS[@]}"

pushd "${FOURD_REPO}" >/dev/null
python train.py \
    "${FOURD_TRAIN_BASE_ARGS[@]}" \
    "${FOURD_TRAIN_ARGS[@]}"

python render.py \
    "${FOURD_RENDER_BASE_ARGS[@]}" \
    "${FOURD_RENDER_ARGS[@]}"

python metrics.py \
    --model_paths \
    "${FOURD_METRICS_BASE_ARGS[@]}" \
    "${FOURD_METRICS_ARGS[@]}"
popd >/dev/null
