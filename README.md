To run the experiments, you can use the following command:

```
python3 main.py --dataset [DATASET] --seed [SEED] --num_shots [NUM_SHOTS] --example_pool_size [EXAMPLE_POOL_SIZE] --embedding_dim [EMBEDDING_DIM] --failure_level [FAILURE_LEVEL] --only_test --load_chkpt_idx [LOAD_CHKPT_IDX] --load_data_path [LOAD_DATA_PATH] --comments [COMMENTS] --random_baseline --seperate_pools --warmup_length [WARMUP_LENGTH] --test_length [TEST_LENGTH] --repeat_examples
```

The arguments are explained as follows:
1.  Dataset: The dataset you wish to use
2.  Seed: The random seed, Default : 2
3. Num_shots: The number of examples used in the prompt, Default : 4
4. Example_pool_size: The number of examples in the pool, Default : 8
5. Embedding_dim: The dimension of the embeddings generated, Default : 64
6. Failure_level: The probability with which the confidence bounds fail, Default : 0.05
7. Only_test: If you wish to use only the test set or a part thereof
8. Load_chkpt_idx: The checkpoint index you wish to load
9. Load_data_path: The path to the data you wish to load
10. Comments: Any comments regarding the experiment
11. Random_baseline: If you wish to use perform the random baseline
12. Seperate_pools: If you wish to have separate pools of examples for each slot in the prompt
13. Warmup_length: The number of examples in the warmup set. If you wish to use the entire training set for warmup, set only_test to false
14. Test_length: The number of examples to be used from the test set in case you do not wish to use the entire test set
15. Repeat_examples: If you wish to repeat the examples in the prompt
