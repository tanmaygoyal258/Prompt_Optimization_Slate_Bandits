#!/bin/bash



python3 main.py \
--dataset glue/sst2 \
--seed 2 \
--num_shots 4 \
--example_pool_size 16 \
--embedding_dim 64 \
--failure_level 0.05 \
--repeat_examples \
--warmup_length 4128 \
--only_test

python3 main.py \
--dataset glue/sst2 \
--seed 2 \
--num_shots 4 \
--example_pool_size 32 \
--embedding_dim 64 \
--failure_level 0.05 \
--repeat_examples \
--warmup_length 4128 \
--only_test


