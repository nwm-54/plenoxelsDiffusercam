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
#SBATCH -t 2-00:00:00
#SBATCH --mem 32gb

source /home/wl757/.bashrc
conda activate gaussian_splatting

# # n_train_images
# # stereo
# wandb agent shamus-team/multiplexed-pixels/ob61noxo
# # multiplexing
# wandb agent shamus-team/multiplexed-pixels/um9e3psf
# # iphone
# wandb agent shamus-team/multiplexed-pixels/ween6fgc

# # angles
# # stereo
# wandb agent shamus-team/multiplexed-pixels/a9ptglqq
# # multiplexing
# wandb agent shamus-team/multiplexed-pixels/pju94m22
# # iphone
# wandb agent shamus-team/multiplexed-pixels/6b16mfme

# sweep_iphone_30v
wandb agent shamus-team/multiplexed-pixels/00rpfh6z