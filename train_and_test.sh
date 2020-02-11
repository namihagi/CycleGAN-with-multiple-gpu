#!/bin/bash

# ffhq-base [0.0, 15.1]
# wider-face [0.000000, 17.067213]

## train
#CUDA_VISIBLE_DEVICES=3 python3 main.py \
#  --phase train \
#  --class_name aeroplane \
#  --train_A_path ./datasets/voc2007-with-annotation/aeroplane/train \
#  --test_A_path ./datasets/voc2007-with-annotation/aeroplane/test \
#  --train_B_path ./datasets/wider-face-with-annotation \
#  --test_B_path ./datasets/FDDB-with-annotation \
#  --sub_dir aeroplane2wider-only-detector \
#  --batch_size 1 \
#  --epoch 300 \
#  --lr_decay_epoch 150 \
#  --B_range 17.1

# test
CUDA_VISIBLE_DEVICES=3 python3 main.py \
  --phase test \
  --class_name aeroplane \
  --train_A_path ./datasets/voc2007-with-annotation/aeroplane/train \
  --test_A_path ./datasets/voc2007-with-annotation/aeroplane/test \
  --train_B_path ./datasets/wider-face-with-annotation \
  --test_B_path ./datasets/FDDB-with-annotation \
  --sub_dir aeroplane2wider-only-detector \
  --batch_size 1 \
  --B_range 17.1
