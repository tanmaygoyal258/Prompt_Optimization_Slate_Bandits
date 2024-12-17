import numpy as np
from copy import deepcopy
import json
from data_utils import custom_load_dataset
import argparse
import random
from PromptOptEnv import PromptOptEnv
import os
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type = str , default = "glue/sst2")
    parser.add_argument("--seed", type = int , default = 2)
    parser.add_argument("--num_shots", type = int , default = 4)
    parser.add_argument("--example_pool_size", type = int, default = 16)
    parser.add_argument("--embedding_dim" , type = int , default = 64)
    parser.add_argument("--failure_level" , type = float , default = 0.05)
    parser.add_argument("--only_test" , action = "store_true" , help = "Only run experiment on the testing dataset")
    parser.add_argument("--load_chkpt_idx" , type = int , help = "Which index to start from" , default = -1)
    parser.add_argument("--load_data_path" , type = str , help = "Folder to load data from" , default = None)
    parser.add_argument("--comments" , type = str , help = "Comments for the experiment" , default = None)
    parser.add_argument("--random_baseline" , action = "store_true" , help = "Run random baseline")
    return parser.parse_args()


def main():
    args = parse_args()
    params = {}
    params["dataset"] = args.dataset
    params["seed"] = args.seed
    params["num_shots"] = args.num_shots
    params["example_pool_size"] = args.example_pool_size
    params["embedding_dim"] = args.embedding_dim
    params["failure_level"] = args.failure_level
    params["only_test"] = args.only_test
    params["start_with"] = args.load_chkpt_idx
    params["data_path"] = args.load_data_path
    params["comments"] = args.comments
    params["random_baseline"] = args.random_baseline
    assert params["embedding_dim"] in [64,128,256,512,768] , "Invalid dimensions for embedding. Please choose from [64,128,256,512,768]"

    print("Loading Dataset..")
    all_train_sentences, all_train_labels, all_test_sentences, all_test_labels = custom_load_dataset(params)
    print("Dataset has been loaded..")

    # Set seed
    np.random.seed(params['seed'])  

    number_labels = len(params["label_dict"])

    # initializing a count dictionary to zero for all possible labels
    number_dict = {x:0 for x in params['label_dict'].keys()}

    # create the example pool with equal number of examples for each class
    example_pool_sentences = []
    example_pool_labels = []
    example_idx = []

    # in case the number of examples is not divisible by the number of classes
    remainder_sentences = params['example_pool_size'] % number_labels 

    sampled_idx = np.random.choice(len(all_train_sentences), 10*params["example_pool_size"] , replace=False)
    for i , idx in enumerate(sampled_idx):
        if number_dict[all_train_labels[idx]] < int((params['example_pool_size'] - remainder_sentences)/number_labels):
            example_pool_sentences.append(deepcopy(all_train_sentences[idx]))
            example_pool_labels.append(deepcopy(all_train_labels[idx]))
            example_idx.append(idx)
            number_dict[all_train_labels[idx]] += 1

            if sum(number_dict.values()) == params['example_pool_size'] - remainder_sentences:
                sentences_added = 0
                if remainder_sentences > 0 :
                    # add the remaining examples to the example pool
                    for idx in sampled_idx[i+1:]:
                        example_pool_sentences.append(deepcopy(all_train_sentences[idx]))
                        example_pool_labels.append(deepcopy(all_train_labels[idx]))
                        example_idx.append(idx)
                        sentences_added += 1
                        if sentences_added == remainder_sentences:
                            break
            
            if len(example_pool_sentences) == params['example_pool_size']:
                break

    print("Example Pool has been created with length {}".format(len(example_pool_sentences)))

    testing_sentences = []
    testing_labels = []

    if not params["only_test"]:
        for idx in range(len(all_train_sentences)):
            if idx not in example_idx:
                testing_sentences.append(deepcopy(all_train_sentences[idx]))
                testing_labels.append(deepcopy(all_train_labels[idx]))

    testing_sentences += all_test_sentences

    random.seed(params['seed'])
    random.shuffle(testing_sentences)
    testing_labels = [s['label'] for s in testing_sentences]

    print("The remaining sentences have been created with length {}".format(len(testing_sentences)))
    
    # we relabel the indices of example sentences and testing sentences for convinience
    example_pool_sentences_relabeled = [{'sentence' : s['sentence'] , 'label' : s['label'] , 'idx' : i} for i , s in enumerate(example_pool_sentences)]
    testing_sentences_relabeled = [{'sentence' : s['sentence'] , 'label' : s['label'] , 'idx' : i} for i , s in enumerate(testing_sentences)]

    if params["data_path"] is None:
        now = datetime.now()
        timestamp = now.strftime("%d-%m_%H-%M")
        data_path = "{}_result".format(params["dataset"])
        data_path = data_path.replace('/' , '_')
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        data_path_with_timestamp = data_path + "/" + timestamp
        if not os.path.exists(data_path_with_timestamp):
            os.makedirs(data_path_with_timestamp)
    else:
        data_path_with_timestamp = params["data_path"]

    # dump the json
    with open(data_path_with_timestamp + "/configs.json" , "w") as f:
        json.dump(params , f)

    env = PromptOptEnv(params , example_pool_sentences_relabeled, example_pool_labels, testing_sentences_relabeled, testing_labels , data_path_with_timestamp)
    rewards = env.run_algorithm()

if __name__ == "__main__":
    main()
    