#!/bin/bash
#SBATCH --job-name=3dgs_multiviews
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=kim-compute-03
#SBATCH --constraint="gpu-high"
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -t 12:00:00                          
#SBATCH --mem 32gb

source /home/wl757/.bashrc
conda activate gaussian_splatting

wandb agent shamus-team/multiplexed-pixels/2769qn27
wandb agent shamus-team/multiplexed-pixels/2ont5h8e
wandb agent shamus-team/multiplexed-pixels/lfqmfp02
wandb agent shamus-team/multiplexed-pixels/nvtka2rv