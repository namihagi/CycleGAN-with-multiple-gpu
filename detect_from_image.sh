#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 detector_by_pretrained_model_for_images.py \
  --input_dir ../datasets/wider_face/jpg_and_json/test/images \
  --output_prediction_dir ./result/wider_face/test/predictions
