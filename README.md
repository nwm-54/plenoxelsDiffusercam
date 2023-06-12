# Plenoxels and Diffusercam


## Setup

We recommend setup with a conda environment, using the packages provided in `requirements.txt`.
'''
conda create --name plenoxels python=3.9
conda activate plenoxels
pip install https://storage.googleapis.com/jax-releases/cuda111/jaxlib-0.1.72+cuda111-cp39-none-manylinux2010_x86_64.whl
cd plenoxels
pip install --upgrade -r requirements.txt
'''

## Downloading data

Currently, this implementation only supports NeRF-Blender, which is available at:

<https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1>

TODO

## Voxel Optimization (aka Training)

Sample training command

'''
python plenoptimize_multiplexedN.py --data_dir /home/vitran/plenoxels/blender_data/ --expname dls_psnr_multiplex100_dls_15_blocked_lenslet2 --scene lego_gen10 --log_dir jax_logs10/ --physical_batch_size 500 --lr_rmsprop 0. --logical_batch_size 1000 --resolution 128 --num_epochs 1 --val_interval 1 --render_interval 100 --dim_lens 80 --optimizer sgd --num_lens 100 --max_per_pixel 4 --d_lenses 1.6 --d_lens_sensor 15 --train_json multilens_100_dls_15_blocked_lenslet2 --tv_rgb 1e-3 --tv_sigma 1e-3 --blocked_lenslet "18, 95, 65, 5, 15, 64, 33, 70, 49, 47, 80, 96, 8, 35, 54, 74, 97, 2, 83, 84, 88, 25, 19, 23, 63, 37, 28, 87, 75, 72, 20, 81, 53, 9, 22, 29, 0, 85, 16, 27, 99, 82, 71, 3, 44, 57, 98, 50, 58, 4"
'''
