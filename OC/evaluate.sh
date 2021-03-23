#!/bin/bash

set -e

CURRENT_DIR=$(pwd)

CUDA_VISIBLE_DEVICES=-1 python "${CURRENT_DIR}"/evaluate.py \
  --max_episodes=1 \
  --load_model=True \
  --model_path="result1_mt/model" \
  --evaluate_file='../random_method/1s6t_1.txt' \
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


