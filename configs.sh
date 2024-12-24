#!/bin/bash

python3 main.py \
--dataset glue/sst2 \
--seed 2 \
--num_shots 4 \
--example_pool_size 8 \
--embedding_dim 64 \
--failure_level 0.05 \
--only_test \
--comments "creating random pools within the example pool" \
--warmup_length 1128
