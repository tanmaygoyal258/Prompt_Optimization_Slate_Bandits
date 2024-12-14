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
    x = np.float128(x)
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
    # config = RobertaConfig.from_pretrained("roberta-large")
    # roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    # roberta_model = RobertaForMaskedLM.from_pretrained("roberta-large", config=config)# , device_map = "auto")
    # roberta_model.eval().to('cuda:'+str(int(1)))

    # print("Finished Model Setup")
    # return roberta_model, roberta_tokenizer

    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
    model = RobertaForMaskedLM.from_pretrained("FacebookAI/roberta-base")
    print("Finished Model Setup")
    return model , tokenizer