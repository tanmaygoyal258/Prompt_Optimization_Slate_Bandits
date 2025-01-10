#!/bin/bash



python3 main.py \
--dataset glue/mrpc \
--seed 2 \
--num_shots 4 \
--example_pool_size 16 \
--embedding_dim 64 \
--failure_level 0.05 \
--repeat_examples \
--load_chkpt_idx 1794 \
--load_data_path glue_mrpc_result/09-01_06-54

python3 main.py \
--dataset glue/mrpc \
--seed 2 \
--num_shots 4 \
--example_pool_size 32 \
--embedding_dim 64 \
--failure_level 0.05 \
--repeat_examples


