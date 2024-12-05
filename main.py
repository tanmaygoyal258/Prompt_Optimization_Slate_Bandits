import numpy as np
from copy import deepcopy
# from promptsource.templates import DatasetTemplates
import torch
# from env import make_env
from data_utils import custom_load_dataset
# from utils import random_sampling
from arguments import parse_args

def main():
    args = parse_args()
    params = {}
    params["dataset"] = args.dataset
    params["seed"] = args.seed
    params["num_shots"] = args.num_shots
    params["example_pool_size"] = args.example_pool_size
    print(args.example_pool_size)

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

    # in case the number of examples is not divisible by the number of classes
    remainder_sentences = params['example_pool_size'] % number_labels 
    print("Remainder sentences", params["example_pool_size"] , number_labels, remainder_sentences) 
    sampled_idx = np.random.choice(len(all_train_sentences), 10*params["example_pool_size"] , replace=False)
    for i , idx in enumerate(sampled_idx):
        print(number_dict)
        if number_dict[all_train_labels[idx]] < int((params['example_pool_size'] - remainder_sentences)/number_labels):
            example_pool_sentences.append(deepcopy(all_train_sentences[idx]))
            example_pool_labels.append(deepcopy(all_train_labels[idx]))
            number_dict[all_train_labels[idx]] += 1

            if sum(number_dict.values()) == params['example_pool_size'] - remainder_sentences:
                sentences_added = 0
                # add the remaining examples to the example pool
                for idx in sampled_idx[i+1:]:
                    example_pool_sentences.append(deepcopy(all_train_sentences[idx]))
                    example_pool_labels.append(deepcopy(all_train_labels[idx]))
                    sentences_added += 1
                    if sentences_added == remainder_sentences:
                        print("Length of Example pool now is {}".format(len(example_pool_sentences)))
                        break
            
            if len(example_pool_sentences) == params['example_pool_size']:
                break

    print("Example Pool has been created with length {}".format(len(example_pool_sentences)))
    print("Example sentences are: ", example_pool_sentences)


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
    