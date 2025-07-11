### This is a README dedicated to the Prompt Tuning Experiments in the paper [Efficient Algorithms for Logistic Contextual Slate Bandits with Bandit Feedback](https://arxiv.org/abs/2506.13163)

To run the files, you can use the following command:
```
python3 main.py --dataset [DATASET] --seed [SEED] --num_shots [NUM_SHOTS] --example_pool_size [EXAMPLE_POOL_SIZE] --embedding_dim [EMBEDDING_DIM] --failure_level [FAILURE_LEVEL] --only_test --load_chkpt_idx [LOAD_CHKPT_IDX] --load_data_path [LOAD_DATA_PATH] --comments [COMMENTS] --random_baseline --seperate_pools --warmup_length [WARMUP_LENGTH] --test_length [TEST_LENGTH] --repeat_examples
```


The arguments are explained as follows:
1.  DATASET: The dataset you wish to use
2.  SEED: The random seed, Default : 2
3. NUM_SHOTS: The number of examples used in the prompt, Default : 4
4. EXAMPLE_POOL_SIZE: The number of examples in the pool, Default : 8
5. EMBEDDING_DIM: The dimension of the embeddings generated, Default : 64
6. FAILURE_LEVEL: The probability with which the confidence bounds fail, Default : 0.05
7. ONLY_TEST: If you wish to use only the test set or a part thereof
8. LOAD_CHKPT_IDX: The checkpoint index you wish to load
9. LOAD_DATA_PATH: The path to the data you wish to load
10. COMMENTS: Any comments regarding the experiment
11. RANDOM_BASELINE: If you wish to use perform the random baseline
12. SEPERATE_POOLS: If you wish to have separate pools of examples for each slot in the prompt
13. WARMUP_LENGTH: The number of examples in the warmup set. If you wish to use the entire training set for warmup, set only_test to false
14. TEST_LENGTH: The number of examples to be used from the test set in case you do not wish to use the entire test set (i.e ONLY_TEST = False)
15. REPEAT_EXAMPLES: If you wish to repeat the examples in the prompt

If you find our work useful, please consider citing us:
```
@misc{goyal2025efficientalgorithmslogisticcontextual,
      title={Efficient Algorithms for Logistic Contextual Slate Bandits with Bandit Feedback}, 
      author={Tanmay Goyal and Gaurav Sinha},
      year={2025},
      eprint={2506.13163},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.13163}, 
}
```
