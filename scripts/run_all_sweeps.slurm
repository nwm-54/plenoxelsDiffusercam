#!/bin/bash
#SBATCH --job-name=wandb_sweep_agents
#SBATCH --output=slurm_logs/sweep_agent_%A_%a.out
#SBATCH --error=slurm_logs/sweep_agent_%A_%a.err
#SBATCH --partition=default_partition
#SBATCH --exclude=lil-compute-05,unicorn-compute-01,bhattacharjee-compute-02
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-high"
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -t 12:00:00
#SBATCH --mem 32gb
#SBATCH --array=0-39

set -euo pipefail

source /home/wl757/.bashrc
conda activate gaussian_splatting

mkdir -p slurm_logs

echo "=========================================="
echo "WANDB SWEEP AGENT - ARRAY JOB"
echo "=========================================="
echo "Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "=========================================="
echo ""

# Define all sweep IDs (8 sweeps, each will get 5 agents)
SWEEP_IDS=(
    "shamus-team/hyperparameters/q0obf4ky"  # sweep_stereo_1v
    "shamus-team/hyperparameters/q74vg6fo"  # sweep_stereo_3v
    "shamus-team/hyperparameters/4usvwnzn"  # sweep_iphone_1v
    "shamus-team/hyperparameters/vy0q2anm"  # sweep_iphone_3v
    "shamus-team/hyperparameters/d53voo7q"  # sweep_lightfield_dls12_1v
    "shamus-team/hyperparameters/vxfnyelf"  # sweep_lightfield_dls12_3v
    "shamus-team/hyperparameters/mi5rnyyz"  # sweep_multiplexing_dls20_1v
    "shamus-team/hyperparameters/f95l04pd"  # sweep_multiplexing_dls20_3v
)

# Map array task ID to sweep ID (each sweep gets 5 agents)
SWEEP_INDEX=$((SLURM_ARRAY_TASK_ID / 5))
AGENT_NUMBER=$((SLURM_ARRAY_TASK_ID % 5 + 1))

SWEEP_ID="${SWEEP_IDS[$SWEEP_INDEX]}"

echo "Running agent #${AGENT_NUMBER} for sweep: ${SWEEP_ID}"
echo "Command: wandb agent ${SWEEP_ID}"
echo ""

# Run the wandb agent
wandb agent "${SWEEP_ID}"

echo ""
echo "=========================================="
echo "ARRAY TASK ${SLURM_ARRAY_TASK_ID} COMPLETE"
echo "Agent #${AGENT_NUMBER} for ${SWEEP_ID} finished"
echo "=========================================="
