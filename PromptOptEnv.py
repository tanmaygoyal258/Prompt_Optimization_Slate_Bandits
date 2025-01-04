from utils import generate_embeddings #, ChatGPT_eval
import numpy as np
from Slate_GLincb_Prompt_Opt import Slate_GLinCB_Prompt_Opt
from utils import setup_roberta
from tqdm import tqdm
import os
import torch
from sentence_transformers import SentenceTransformer
from time import time
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PromptOptEnv():

    def __init__(self , params , example_pool_sentences, example_pool_labels, testing_sentences, testing_labels , data_path):
        self.prompt_prefix = params["prompt_prefix"]
        self.q_prefix = params["q_prefix"]
        self.a_prefix = params["a_prefix"]
        self.label_dict = params["label_dict"]
        self.inv_label_dict = params["inv_label_dict"]
        self.num_shots = params["num_shots"]
        self.example_pool_size = params["example_pool_size"]
        self.embedding_dim = params["embedding_dim"]
        self.failure_level = params["failure_level"]
        self.start_with = params["start_with"]
        self.param_norm_ub = 1
        self.data_path = data_path
        self.random_baseline = params["random_baseline"]
        self.seperate_pools = params["seperate_pools"]
        self.outfile = open(data_path + "/prompts_chosen.txt" , "a+")


        self.chosen_examples = []
        self.example_pool = example_pool_sentences
        self.example_pool_labels = example_pool_labels
        
        if not self.random_baseline:
            # load the embedding model and create the embeddings
            self.embedding_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
            self.example_embeddings = {s['idx'] : generate_embeddings(s['sentence'] , self.embedding_dim , self.embedding_model) for s in example_pool_sentences}

            # appending the labels' embeddings to the embeddings
            for k in self.example_embeddings.keys():
                self.example_embeddings[k] = np.hstack([self.example_embeddings[k] , generate_embeddings(self.label_dict[self.example_pool_labels[k]][0] , self.embedding_dim , self.embedding_model)])
            
            # creating the inverse dictionary
            # we have to create tuples since lists are not hashable
            self.inv_example_embeddings = {}
            for k , v in self.example_embeddings.items():
                self.inv_example_embeddings[tuple(v)] = k

        self.queries = testing_sentences
        self.query_labels = testing_labels

        # setting up the roberta model
        self.model, self.tokenizer = setup_roberta()

        # loading the rewards
        self.rewards = [] if self.start_with < 0 else np.load(self.data_path + "/parameters_{}/rewards_array.npy".format(self.start_with)).tolist()

        # setting up the algorithm
        self.alg = Slate_GLinCB_Prompt_Opt(self.num_shots , self.example_pool_size , self.embedding_dim , self.failure_level , self.param_norm_ub , self.start_with , self.data_path , len(self.queries))


    def construct_prompt(self , query_idx):
        '''
        function to create the prompt for a given query
        '''
        prompt = self.prompt_prefix
        for i in range(self.num_shots):
            prompt += self.q_prefix + self.chosen_examples[i]['sentence']
            prompt += "\n"
            prompt += self.a_prefix + self.label_dict[self.chosen_examples[i]['label']][0]
            prompt += "\n"
        prompt += self.q_prefix + self.queries[query_idx]['sentence']
        prompt += "\n"
        prompt += self.a_prefix + ' <mask>'
        return prompt

    def construct_arms(self , query_idx):
        '''
        function to construct the arms which concatenates the query and example embeddings
        '''
        query = self.queries[query_idx]['sentence']
        query_embedding = generate_embeddings(query , self.embedding_dim , self.embedding_model)
        self.arm_set = [np.hstack([query_embedding , example_embedding]) for example_embedding in self.example_embeddings.values()]

    def construct_random_pools(self):
        '''
        function to break the example pool randomly into subsets for each slot
        '''
        ordering = np.random.permutation(self.example_pool_size).reshape(self.num_shots , -1)
        self.arm_set =  [[self.arm_set[i] for i in ordering[j]] for j in range(self.num_shots)]

    def run_algorithm(self):
        '''
        runs the algorithm
        '''
        for time_idx in tqdm(range(self.start_with+1 , len(self.queries))):
            assert time_idx == self.queries[time_idx]["idx"]

            self.chosen_examples = []
            if not self.random_baseline:
                
                # construct the armset which uses embedding of queries and all examples
                self.construct_arms(time_idx)   

                # if seperate_pools, then construct the random pools
                if self.seperate_pools:
                    self.construct_random_pools()

                # pull the arm
                chosen_examples_indices = self.alg.pull(self.arm_set , self.seperate_pools)
                chosen_examples = [self.arm_set[idx] for idx in chosen_examples_indices] if not self.seperate_pools else [self.arm_set[i][idx] for i,idx in enumerate(chosen_examples_indices)]

                # find the chosen examples
                for query_plus_embedding in chosen_examples:
                    embedding = query_plus_embedding[self.embedding_dim:]
                    self.chosen_examples.append(self.example_pool[self.inv_example_embeddings[tuple(embedding)]])

            else:
                chosen_examples_indices = np.random.choice(len(self.example_pool) , self.num_shots , replace=False)
                self.chosen_examples = [self.example_pool[idx] for idx in chosen_examples_indices]

            # construct the prompt
            self.outfile.write("Iteration {}\n".format(time_idx + 1))
            prompt = self.construct_prompt(time_idx)
            self.outfile.write(prompt + "\n") 
            
            # obtain the answer from roberta
            response = self.get_prediction(prompt)
            self.outfile.write("Predicted Answer: {}\n".format(response))
            self.outfile.write("Correct Answer: {} \n".format(self.label_dict[self.query_labels[time_idx]][0] + "\n"))

            # get the score from GPT judge
            score = ChatGPT_eval(response , self.label_dict[self.query_labels[time_idx]][0])
            # score = 1 if response.strip() == self.label_dict[self.query_labels[time_idx]][0].strip() else 0
            # print(score)
            self.rewards.append(score)
            self.outfile.write("Reward: {}\n".format(score))
            self.outfile.write("Accuracy {:.2f}\n".format(np.sum(self.rewards)/len(self.rewards) * 100))
            self.outfile.write("\n")

            # update the parameters
            if not self.random_baseline:
                self.alg.update_parameters(chosen_examples , score)

                # delete the previous results
                if os.path.exists(self.data_path + "/parameters_{}".format(time_idx-1)):
                    for filename in os.listdir(self.data_path + "/parameters_{}".format(time_idx-1)):
                        os.remove(os.path.join(self.data_path + "/parameters_{}".format(time_idx-1) , filename))
                    os.rmdir(self.data_path + "/parameters_{}".format(time_idx-1))

                # save the new results
                time_folder = self.data_path + "/parameters_{}".format(time_idx)
                os.makedirs(self.data_path + "/parameters_{}".format(time_idx))
                np.save(time_folder + "/rewards_array", np.array(self.rewards))
                np.save(time_folder + "/vtilde_matrix", self.alg.vtilde_matrix)
                np.save(time_folder + "/vtilde_matrix_inv", self.alg.vtilde_matrix_inv)
                np.save(time_folder + "/v_matrices", self.alg.v_matrices)
                np.save(time_folder + "/v_matrices_inv", self.alg.v_matrices_inv)
                np.save(time_folder + "/theta", self.alg.theta)
                np.save(time_folder + "/conf_radius", self.alg.conf_radius)
                np.save(time_folder + "/cum_loss", self.alg.cum_loss)
                np.save(time_folder + "/ctr", self.alg.ctr)

        # saving the final reward array
        np.save(self.data_path + "/rewards_array_final" , np.array(self.rewards))
        return self.rewards

    def get_prediction(self , prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if inputs['input_ids'].shape[1] > 512:
            # print("Token length exceeds 512. Truncating to 512")
            inputs['input_ids'] = inputs['input_ids'][:, -512:]
            inputs['attention_mask'] = inputs['attention_mask'][:, -512:]
        
        with torch.no_grad():
            logits = self.model(**inputs).logits

        # retrieve index of <mask>
        mask_token_index = (inputs.input_ids == self.tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        return self.tokenizer.decode(predicted_token_id)