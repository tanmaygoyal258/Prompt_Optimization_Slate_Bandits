#!/bin/bash

python3 main.py \
--dataset glue/sst2 \
--seed 2 \
--num_shots 8 \
--example_pool_size 16 \
--embedding_dim 64 \
--failure_level 0.05 \
--only_test \
--comments "precision linearly scheduled and param_norm_ub 1" \
