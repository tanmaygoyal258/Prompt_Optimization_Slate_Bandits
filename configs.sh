#!/bin/bash


python3 main.py \
--dataset glue/mrpc \
--seed 2 \
--num_shots 4 \
--example_pool_size 8 \
--embedding_dim 64 \
--failure_level 0.05 \
--random_baseline

python3 main.py \
--dataset glue/mrpc \
--seed 2 \
--num_shots 4 \
--example_pool_size 16 \
--embedding_dim 64 \
--failure_level 0.05 \
--random_baseline


python3 main.py \
--dataset glue/mrpc \
--seed 2 \
--num_shots 4 \
--example_pool_size 32 \
--embedding_dim 64 \
--failure_level 0.05 \
--random_baseline
