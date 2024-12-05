

class PromptOptEnv():

    def __init__(self , params , example_pool_sentences, example_pool_labels, testing_sentences, testing_labels):
        self.prompt_prefix = params["prompt_prefix"]
        self.q_prefix = params["q_prefix"]
        self.a_prefix = params["a_prefix"]
        self.label_dict = params["label_dict"]
        self.inv_label_dict = params["inv_label_dict"]
        self.num_shots = params["num_shots"]
        self.example_pool_size = params["example_pool_size"]
        
        self.current_prompt_examples = example_pool_sentences[:self.num_shots]
        self.current_prompt_labels = example_pool_labels[:self.num_shots]


        self.example_pool = example_pool_sentences[self.num_shots:]
        self.example_pool_labels = example_pool_labels[self.num_shots:]

        self.queries = testing_sentences
        self.query_labels = testing_labels

    def construct_prompt(self , query_idx):
        prompt = self.prompt_prefix
        for i in range(self.num_shots):
            prompt += self.q_prefix + self.current_prompt_examples[i]['sentence']
            prompt += "\n"
            prompt += self.a_prefix + self.label_dict[self.current_prompt_labels[i]][0]
            prompt += "\n"
        prompt += self.q_prefix + self.queries[query_idx]['sentence']
        prompt += "\n"
        prompt += self.a_prefix + ' <mask>'
        return prompt

    def play(self):
        print(self.construct_prompt(0))
        