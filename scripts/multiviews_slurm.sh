#!/bin/bash
#SBATCH --job-name=3dgs_multiviews
#SBATCH --output=slurm_logs/%j.out
#SBATCH --partition=gpu --gres=gpu:1
#SBATCH --constraint="gpu-high"
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 24:00:00                          
#SBATCH --mem 32gb

source /home/wl757/.bashrc
conda activate gaussian_splatting

wandb agent shamus-team/multiplexed-pixels/2a91lniq