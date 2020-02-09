#!/bin/bash

# ffhq-base [0.0, 15.1]
# wider-face [0.000000, 17.067213]

CUDA_VISIBLE_DEVICES=0 python3 main.py \
  --phase train \
  --class_name aeroplane \
  --datasetA_path ./datasets/voc2007-with-annotation/aeroplane/train \
  --datasetB_path ./datasets/wider-face-with-annotation \
  --sub_dir aeroplane2wider-with-annotation \
  --batch_size 1 \
  --epoch 300 \
  --lr_decay_epoch 150 \
  --B_range 17.1
