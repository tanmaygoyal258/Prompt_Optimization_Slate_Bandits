import torch.nn.functional as F
from ChatGPT import ChatGPT
from copy import deepcopy
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel, RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, AutoTokenizer
import torch
from scipy.linalg import sqrtm

# RATING_TEMPLATE = "Please provide a rating between 0 and 1 about the semantic similarity between '[Target]' and '[Response]'. Provide the rating only and NOTHING else."
RATING_TEMPLATE = "Please provide a rating which is either 0 or 1 about the semantic similarity between '[Target]' and '[Response]'. Provide the rating only and NOTHING else. The rating has to be either 0 or 1, NOTHING else."

def generate_embeddings(text: str , dim: int = 64 , model = None):
    assert dim in [64,128,256,512,768] , "Invalid Dimensions. Please choose from [64,128,256,512,768]"
    matryoshka_dim = dim
    model = model.eval()
    with torch.no_grad():
        embeddings = model.encode([text], convert_to_tensor=True)
    embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
    embeddings = embeddings[:, :matryoshka_dim]
    embeddings = F.normalize(embeddings, p=2, dim=1)
    embeddings = embeddings.cpu().numpy().reshape(-1)
    return embeddings

def ChatGPT_eval(response: str, target: str) -> float:

    evaluatorLLM = ChatGPT()
    reward = 0
    prompt = RATING_TEMPLATE.replace("[Target]", target).replace("[Response]", response)
    while True:
        try:
            reward = float(evaluatorLLM.get_response(prompt = prompt, n = 1)[0])
            if reward >= 0 and reward <=1:
                return reward
        except:
            print("Try LLM evaluation again...")

def sigmoid(x):
    if x >= 0:
        return 1 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))

def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def weighted_norm(x, A):
    return np.sqrt(np.dot(x, np.dot(A, x)))

def gaussian_sample_ellipsoid(center, design, radius):
    dim = len(center)
    sample = np.random.normal(0, 1, (dim,))
    res = np.real_if_close(center + np.linalg.solve(sqrtm(design), sample) * radius)
    return res

def setup_roberta():
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
    model = RobertaForMaskedLM.from_pretrained("FacebookAI/roberta-base")
    print("Finished Model Setup")
    return model , tokenizer

def random_equal_sampling(sentences , labels , num_labels , num_points):
    number_dict = {x:0 for x in range(num_labels)}

    # create the example pool with equal number of examples for each class
    sentences_sampled = []
    labels_sampled = []
    idx_sampled = []
    # in case the number of examples is not divisible by the number of classes
    remainder_sentences = num_points % num_labels 

    sampled_idx = np.random.choice(len(sentences), 10*num_points , replace=False)
    for i , idx in enumerate(sampled_idx):
        if number_dict[labels[idx]] < int((num_points - remainder_sentences)/num_labels):
            sentences_sampled.append(deepcopy(sentences[idx]))
            labels_sampled.append(deepcopy(labels[idx]))
            idx_sampled.append(idx)
            number_dict[labels[idx]] += 1

            if sum(number_dict.values()) == num_points - remainder_sentences:
                sentences_added = 0
                if remainder_sentences > 0 :
                    # add the remaining examples to the example pool
                    for idx in sampled_idx[i+1:]:
                        sentences.append(deepcopy(sentences[idx]))
                        labels.append(deepcopy(labels[idx]))
                        idx_sampled.append(idx)
                        sentences_added += 1
                        if sentences_added == remainder_sentences:
                            break
            
            if len(sentences) == num_points:
                return sentences_sampled , labels_sampled , idx_sampled

    return sentences_sampled , labels_sampled , idx_sampled