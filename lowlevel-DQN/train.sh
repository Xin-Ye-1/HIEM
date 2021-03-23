#!/bin/bash

set -e

CURRENT_DIR=$(pwd)

CUDA_VISIBLE_DEVICES=-1 python "${CURRENT_DIR}"/train.py \
  --is_approaching_policy=True \
  --max_episodes=100000 \
  --load_model=False \
  --continuing_training=False \
  --model_path="${CURRENT_DIR}/result1_for_pretrain/model" \
  --num_threads=1 \
  --num_scenes=1 \
  --num_targets=78 \
  --use_default_scenes=True \
  --use_default_targets=False \
  --default_scenes='5cf0e1e9493994e483e985c436b9d3bc' \
  --default_targets='music' \
  --default_targets='television' \
  --default_targets='table' \
  --default_targets='stand' \
  --default_targets='dressing_table' \
  --default_targets='heater' \
#  --default_scenes='0c9a666391cc08db7d6ca1a926183a76' \
#  --default_targets='sofa' \
#  --default_targets='television' \
#  --default_targets='tv_stand' \
#  --default_targets='bed' \
#  --default_targets='toilet' \
#  --default_targets='bathtub' \
#  --default_scenes='00d9be7210856e638fa3b1addf2237d6' \
#  --default_targets='sofa' \
#  --default_targets='television' \
#  --default_targets='tv_stand' \
#  --default_targets='stand' \
#  --default_targets='dressing_table' \
#  --default_targets='music' \
#  --default_scenes='0880799c157b4dff08f90db221d7f884' \
#  --default_targets='sofa' \
#  --default_targets='television' \
#  --default_targets='tv_stand' \
#  --default_targets='bed' \
#  --default_targets='bathtub' \
#  --default_targets='toilet' \