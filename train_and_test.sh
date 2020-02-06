#!/bin/bash

# ffhq-base [0.0, 15.1]
# wider-face [0.000000 , 14.69332]

CUDA_VISIBLE_DEVICES=0 python3 main.py --phase "train" \
                --datasetA="cliped_ball" --datasetB="ffhq-npy-tfrecords" --sub_dir="cliped-ball2ffhq" \
                --gpu_num 1 --global_batch_size 1 --epoch 300 --lr_decay_epoch 150 --B_range 15.1

CUDA_VISIBLE_DEVICES=0 python3 main.py --phase "train" \
                --datasetA="cliped_aeroplane" --datasetB="ffhq-npy-tfrecords" --sub_dir="cliped-aeroplane2ffhq" \
                --gpu_num 1 --global_batch_size 1 --epoch 300 --lr_decay_epoch 150 --B_range 15.1

CUDA_VISIBLE_DEVICES=0 python3 main.py --phase "train" \
                --datasetA="cliped_ball" --datasetB="ffhq-npy-tfrecords" --sub_dir="cliped-ball2ffhq-epoch2000" \
                --gpu_num 1 --global_batch_size 1 --epoch 2000 --lr_decay_epoch 1000 --B_range 15.1
