#!/bin/bash

python3 main.py \
--dataset glue/sst2 \
--seed 2 \
--num_shots 4 \
--example_pool_size 1
