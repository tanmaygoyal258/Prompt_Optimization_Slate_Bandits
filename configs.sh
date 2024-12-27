#!/bin/bash

python3 main.py \
--dataset yelp_polarity \
--seed 2 \
--num_shots 4 \
--example_pool_size 8 \
--embedding_dim 64 \
--failure_level 0.05 \
--only_test \
--test_length 1000 \
--random_baseline