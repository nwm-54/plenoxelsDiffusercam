#!/bin/bash
#SBATCH --job-name=3dgs_multiviews
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#SBATCH --partition=default_partition
#SBATCH --exclude=lil-compute-05,unicorn-compute-01,bhattacharjee-compute-02
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-high"
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -t 12:00:00                          
#SBATCH --mem 32gb

source /home/wl757/.bashrc
conda activate gaussian_splatting

wandb agent shamus-team/multiplexed-pixels/rogikpm8
wandb agent shamus-team/multiplexed-pixels/c939y7p2
wandb agent shamus-team/multiplexed-pixels/6ytlyq27
