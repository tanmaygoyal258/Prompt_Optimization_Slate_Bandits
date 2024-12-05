import numpy as np
from copy import deepcopy
import torch
from data_utils import custom_load_dataset
import argparse
import random
from PromptOptEnv import PromptOptEnv
# from promptsource.templates import DatasetTemplates
# from env import make_env
# from utils import random_sampling

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type = str , default = "glue/sst2")
    parser.add_argument("--seed", type = int , default = 2)
    parser.add_argument("--num_shots", type = int , default = 4)
    parser.add_argument("--example_pool_size", type = int, default = 16)
    parser.add_argument("--embedding_dim" , type = int , default = 64)
    return parser.parse_args()


def main():
    args = parse_args()
    params = {}
    params["dataset"] = args.dataset
    params["seed"] = args.seed
    params["num_shots"] = args.num_shots
    params["example_pool_size"] = args.example_pool_size
    params["embedding_dim"] = args.embedding_dim
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
    # print("Example sentences are: ", example_pool_sentences)

    testing_sentences = []
    testing_labels = []

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

    env = PromptOptEnv(params , example_pool_sentences_relabeled, example_pool_labels, testing_sentences_relabeled, testing_labels)

    # few_shot_train_sentences = []
    # few_shot_train_labels = []
    
    # # initializing a count dictionary to zero for all possible labels
    # number_dict = {x:0 for x in params['label_dict'].keys()} 
    
    # # randomly sampling 100 training samples and labels
    # hundred_train_sentences, hundred_train_labels = random_sampling(all_train_sentences, all_train_labels, 100)
    
    # # ensure that total training samples dont exceed total and number of samples per class dont exceed total/#classes
    # # where total = params['example_pool_size']
    # for train_sentence, train_label in zip(hundred_train_sentences, hundred_train_labels):
    #     if number_dict[train_label] < int(params['example_pool_size']/len(number_dict.values())):
    #         few_shot_train_sentences.append(deepcopy(train_sentence))
    #         few_shot_train_labels.append(deepcopy(train_label))
    #         number_dict[train_label] += 1
    #     if sum(number_dict.values()) == params['example_pool_size']:
    #         break
    # train_sentences, train_labels = few_shot_train_sentences, few_shot_train_labels

    # # same as current prompt sentences and current prompt labels
    # prompt_sentence_pool, prompt_label_pool = train_sentences[:params['num_shots']], train_labels[:params['num_shots']] 

    # add_prompt_sentence_pool, add_prompt_label_pool = train_sentences[params['num_shots']:], train_labels[params['num_shots']:]

    # if len(all_train_sentences) > 100:
        
    #     all_prompt_sentence_pool, all_prompt_label_pool = random_sampling(all_train_sentences, all_train_labels, 100)
        
    #     if params['sub_sample']:
    #         few_shot_train_sentences = []
    #         few_shot_train_labels = []
    #         number_dict = {x:0 for x in params['label_dict'].keys()}
    #         hundred_train_sentences, hundred_train_labels = random_sampling(all_train_sentences, all_train_labels, 1000)
    #         for train_sentence, train_label in zip(hundred_train_sentences, hundred_train_labels):
    #             if number_dict[train_label] < 16:
    #                 few_shot_train_sentences.append(deepcopy(train_sentence))
    #                 few_shot_train_labels.append(deepcopy(train_label))
    #                 number_dict[train_label] += 1
    #             if sum(number_dict.values()) == 16 * len(number_dict.values()):
    #                 break
    #         all_train_sentences, all_train_labels = few_shot_train_sentences, few_shot_train_labels        
    # else:
    #     all_prompt_sentence_pool, all_prompt_label_pool = all_train_sentences, all_train_labels

    # print("Prompt_Sentence_Pool ", len(prompt_sentence_pool))
    # print("Add_Prompt_sentence_pool ", len(add_prompt_sentence_pool))
    # print("All prompt Sentence Pool", len(all_prompt_sentence_pool))
    # print("All_train_sentences", len(all_train_sentences))

    # # env = make_env(params, params, prompt_sentence_pool, prompt_label_pool, all_prompt_sentence_pool, all_prompt_label_pool, add_prompt_sentence_pool, add_prompt_label_pool, train_sentences, train_labels, max_steps, num_processes, obs_size, entropy_coef=0.0, loss_type=params['rew_type'], verbalizer=params['verbalizer'], evaluate=True):
    # env = make_env(params, prompt_sentence_pool, prompt_label_pool, all_prompt_sentence_pool, all_prompt_label_pool, add_prompt_sentence_pool, add_prompt_label_pool, all_train_sentences, all_train_labels, entropy_coef=0.0, verbalizer = True, evaluate=True)

    


if __name__ == "__main__":
    main()
    