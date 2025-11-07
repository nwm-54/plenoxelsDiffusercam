#!/usr/bin/env bash
# Lightweight AWS runner for gs7 preprocessing (no SLURM).
# - Iterates all datasets under `dataset/` except `dynamic`.
# - Runs `scripts/preprocess_for_gsplat.py` directly.
# - Uses fps 20 for stereo, fps 10 for iPhone, when the corresponding
#   train/eval folders exist.
# - Writes per-scene logs into `logs/`.
#
# Usage:
#   bash scripts/run_preprocess_aws.sh                  # sequential
#   PYTHON_BIN=python3 bash scripts/run_preprocess_aws.sh
#   OVERWRITE=1 bash scripts/run_preprocess_aws.sh      # force rebuilds
#
# Env vars:
#   PYTHON_BIN   Python executable to use (default: python)
#   LOG_DIR      Logs directory (default: <repo>/logs)
#   DATASET_DIR  Dataset root (default: <repo>/dataset)
#   FPS_STEREO   FPS for stereo videos (default: 20)
#   FPS_IPHONE   FPS for iPhone videos (default: 10)
#   EXAMPLES_ROOT Path to gsplat/examples (default auto-detected)
#   OVERWRITE    If set, passes --overwrite to the Python script

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)

PYTHON_BIN=${PYTHON_BIN:-python}
LOG_DIR=${LOG_DIR:-"${REPO_ROOT}/logs"}
DATASET_DIR=${DATASET_DIR:-"${REPO_ROOT}/dataset"}
FPS_STEREO=${FPS_STEREO:-20}
FPS_IPHONE=${FPS_IPHONE:-10}
EXAMPLES_ROOT=${EXAMPLES_ROOT:-"${REPO_ROOT}/../gsplat/examples"}

mkdir -p "${LOG_DIR}"

timestamp() { date +%Y%m%d_%H%M%S; }

run_one() {
  local scene="$1"       # e.g., action-figure
  local mode="$2"        # stereo|iphone
  local fps="$3"         # e.g., 20 or 10

  local train_dir="${DATASET_DIR}/${scene}/${mode}-train"
  local eval_dir="${DATASET_DIR}/${scene}/${mode}-eval"
  local out_dir="${DATASET_DIR}/${scene}/${mode}"

  if [[ ! -d "${train_dir}" || ! -d "${eval_dir}" ]]; then
    echo "[skip] ${scene}/${mode}: missing train or eval dir" >&2
    return 0
  fi

  mkdir -p "${LOG_DIR}/${scene}"
  local ts
  ts=$(timestamp)
  local log_file="${LOG_DIR}/${scene}/${mode}_${ts}.log"

  echo "[run] ${scene}/${mode} -> ${out_dir} (fps=${fps})"
  echo "      log: ${log_file}"

  # Build command
  cmd=(
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/preprocess_for_gsplat.py"
    --input "${train_dir}"
    --eval-dir "${eval_dir}"
    --output-dir "${out_dir}"
    --fps "${fps}"
    --examples-root "${EXAMPLES_ROOT}"
  )

  if [[ "${OVERWRITE:-}" != "" ]]; then
    cmd+=(--overwrite)
  fi

  # Stream to log (stdout+stderr)
  (
    echo "===== $(date -Is) START ${scene}/${mode} =====";
    printf 'CMD: '; printf '%q ' "${cmd[@]}"; echo;
    "${cmd[@]}"
    rc=$?
    echo "===== $(date -Is) END ${scene}/${mode} (rc=${rc}) ====="
    exit ${rc}
  ) |& tee -a "${log_file}"
}

main() {
  if [[ ! -d "${DATASET_DIR}" ]]; then
    echo "ERROR: DATASET_DIR not found: ${DATASET_DIR}" >&2
    exit 1
  fi

  # Iterate top-level scenes, excluding 'dynamic'
  shopt -s nullglob
  for scene_path in "${DATASET_DIR}"/*/; do
    scene=$(basename "${scene_path}")
    [[ "${scene}" == "dynamic" ]] && continue

    run_one "${scene}" stereo  "${FPS_STEREO}"
    run_one "${scene}" iphone  "${FPS_IPHONE}"
  done
}

main "$@"

