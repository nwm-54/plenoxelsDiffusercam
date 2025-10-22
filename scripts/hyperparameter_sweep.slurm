#!/bin/bash
#SBATCH --job-name=gs_hyper_sweep
#SBATCH --output=slurm_logs/hyper_sweep_%A_%a.out
#SBATCH --error=slurm_logs/hyper_sweep_%A_%a.err
#SBATCH --partition=default_partition
#SBATCH --exclude=lil-compute-05,unicorn-compute-01,bhattacharjee-compute-02
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-high"
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -t 12:00:00
#SBATCH --mem 32gb
#
# Default array covers 6 configs × 5 camera models = 30 runs (0-29).
# Use `python hyperparameter_sweep.py --list_tasks ...` to confirm and adjust if needed.
#SBATCH --array=0-29

set -euo pipefail

source /home/wl757/.bashrc
conda activate gaussian_splatting

mkdir -p slurm_logs

echo "=========================================="
echo "GSPLAT HYPERPARAMETER SWEEP - ARRAY JOB"
echo "=========================================="
echo "Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "=========================================="
echo ""

SWEEP_ARGS=("$@")

if [[ ${#SWEEP_ARGS[@]} -eq 0 ]]; then
    echo "No sweep arguments provided. Defaulting to all camera models."
    SWEEP_ARGS=(--camera_models base stereo iphone multiplexing lightfield)
fi

CMD=(python hyperparameter_sweep.py "${SWEEP_ARGS[@]}")

echo "Executing: ${CMD[*]}"
echo ""

"${CMD[@]}"

echo ""
echo "=========================================="
echo "ARRAY TASK ${SLURM_ARRAY_TASK_ID} COMPLETE"
echo "=========================================="
