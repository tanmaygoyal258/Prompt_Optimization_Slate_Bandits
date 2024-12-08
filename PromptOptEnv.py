from utils import generate_embeddings , ChatGPT_eval
import numpy as np
from Slate_GLincb_Prompt_Opt import Slate_GLinCB_Prompt_Opt
from utils import setup_roberta
import torch
from tqdm import tqdm
import os
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
        self.param_norm_ub = self.embedding_dim # extremely loose upper bound
        self.data_path = data_path

        self.example_embeddings = {s['idx'] : generate_embeddings(s['sentence'] , self.embedding_dim) for s in example_pool_sentences}
        self.inv_example_embeddings = {}
        for k , v in self.example_embeddings.items():
            self.inv_example_embeddings[tuple(v)] = k

        self.chosen_examples = []
        self.example_pool = example_pool_sentences
        self.example_pool_labels = example_pool_labels

        self.queries = testing_sentences
        self.query_labels = testing_labels

        self.model, self.tokenizer = setup_roberta()

        self.rewards = []

        # TODO: param_norm_ub
        self.alg = Slate_GLinCB_Prompt_Opt(self.num_shots , self.example_pool_size , self.embedding_dim , self.failure_level , self.param_norm_ub)

    def construct_prompt(self , query_idx):
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
        query = self.queries[query_idx]['sentence']
        query_embedding = generate_embeddings(query , self.embedding_dim)
        self.arm_set = [np.hstack([query_embedding , example_embedding]) for example_embedding in self.example_embeddings.values()]

    def run_algorithm(self):
        for time_idx in tqdm(range(len(self.queries))):
            assert time_idx == self.queries[time_idx]["idx"]

            # construct the armset which uses embedding of queries and all examples
            self.construct_arms(time_idx)   

            # pull the arm
            chosen_examples_embeddings = self.alg.pull(self.arm_set)

            # find the chosen examples
            for query_plus_embedding in chosen_examples_embeddings:
                embedding = query_plus_embedding[self.embedding_dim:]
                self.chosen_examples.append(self.example_pool[self.inv_example_embeddings[tuple(embedding)]])

            # construct the prompt
            prompt = self.construct_prompt(time_idx)
            # print(prompt) 
            
            # obtain the answer from roberta
            response = self.get_prediction(prompt)

            # get the score from GPT judge
            score = ChatGPT_eval(response , self.label_dict[self.query_labels[time_idx]][0])
            # print(score)
            self.rewards.append(score)

            # update the parameters
            self.alg.update_parameters(chosen_examples_embeddings , score)
            
            
            if time_idx >= 0:# and time_idx % 100 == 0:
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

        np.save(self.data_path + "/rewards_array_final" , np.array(self.rewards))
        return self.rewards

    def get_prediction(self , prompt):
        prompt = [prompt]
        label_idx = [self.tokenizer.convert_tokens_to_ids('\u0120'+label) for label in self.inv_label_dict.keys()]
        input_ids = self.tokenizer.batch_encode_plus(prompt, return_tensors="pt").to(self.model.device)
        
        if input_ids['input_ids'].shape[1] > 512:
            input_ids['input_ids'] = input_ids['input_ids'][:, -512:]
            input_ids['attention_mask'] = input_ids['attention_mask'][:, -512:]
        mask_ids = (input_ids['input_ids'] == 50264)[0].nonzero(as_tuple=True)[0]

        # they want the probs of the top tokens
        # we are left padding, so we need to adjust the position IDs
        attention_mask = (input_ids['input_ids'] != 50256).float()
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        with torch.no_grad():
            outputs = self.model(**input_ids, output_hidden_states=True)

        logits = outputs.logits.detach().cpu()

        # get the top tokens and probs for the generated l tokens
        probs = torch.softmax(logits[:, mask_ids.cpu()].float(), dim=2).cpu()
        top_tokens = torch.cat([torch.tensor(np.array(label_idx)).to(probs.device).unsqueeze(0).unsqueeze(0) for _ in range(probs.shape[0])], dim=0)
        top_probs = probs[:, :, torch.tensor(np.array(label_idx)).to(probs.device)]

        top_log_probs = torch.log(top_probs)
        total_sequences = input_ids['input_ids']

        return_json = {}
        choices = []
        curr_json = {}

        # text is just the optional context and next l tokens
        for batch_id in range(len(prompt)):
            curr_json['text'] = self.tokenizer.decode(total_sequences[batch_id][mask_ids], skip_special_tokens=True)
            curr_json['logprobs'] = {}
            curr_json['logprobs']['top_logprobs'] = []
            curr_json['logprobs']['token_logprobs'] = []
            curr_json['logprobs']['tokens'] = []
            # cutoff the -1 here because the probs are shifted one over for LMs
            for current_element_top_log_probs, current_element_top_tokens in zip(top_log_probs[batch_id], top_tokens[batch_id]):
                # tokens is a list of the top token at each position
                curr_json['logprobs']['tokens'].append(self.tokenizer.decode([current_element_top_tokens[0]]))
                # token_logprobs is a list of the logprob of the top token at each position
                curr_json['logprobs']['token_logprobs'].append(current_element_top_log_probs[0].item())
                # top_logprobs is a list of dicts for the top K tokens. with each entry being {'token_name': log_prob}
                temp = {}
                for log_prob, token in zip(current_element_top_log_probs, current_element_top_tokens):
                    temp[self.tokenizer.decode(token.item())] = log_prob.item()
                curr_json['logprobs']['top_logprobs'].append(temp)

            choices.append(curr_json)

        return_json['choices'] = choices
        return return_json["choices"][0]["logprobs"]["tokens"][0].strip()