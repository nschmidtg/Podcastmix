#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python test_real.py --target_model ConvTasNet --test_dir podcastmix/real_podcasts/metadata --out_dir=/home/marius/data/Podcastmix/real/ConvTasNet --exp_dir=/mnt/sda1/nicolas/tesis/thesis/ConvTasNet_model/exp/tmp --use_gpu=1 --n_save_ex=-1
CUDA_VISIBLE_DEVICES=1 python test_real.py --target_model UNet --test_dir podcastmix/real_podcasts/metadata --out_dir=/home/marius/data/Podcastmix/real/UNet --exp_dir=/mnt/sda1/nicolas/tesis/thesis/UNet_model/exp/tmp --use_gpu=1 --n_save_ex=-1
CUDA_VISIBLE_DEVICES=1 python test.py --target_model ConvTasNet --test_dir podcastmix/metadata/test --out_dir=/home/marius/data/Podcastmix/synth/ConvTasNet --exp_dir=/mnt/sda1/nicolas/tesis/thesis/ConvTasNet_model/exp/tmp --use_gpu=1 --n_save_ex=-1
CUDA_VISIBLE_DEVICES=1 python test.py --target_model UNet --test_dir podcastmix/metadata/test --out_dir=/home/marius/data/Podcastmix/synth/UNet --exp_dir=/mnt/sda1/nicolas/tesis/thesis/UNet_model/exp/tmp --use_gpu=1 --n_save_ex=-1
