import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from ChatGPT import ChatGPT
from copy import deepcopy
import numpy as np

RATING_TEMPLATE = "Please provide a rating between 0 and 1 about the semantic similarity between '[Target]' and '[Response]'. Provide the rating only and NOTHING else."


def generate_embeddings(text: str , dim: int = 64):
    assert dim in [64,128,256,512,768] , "Invalid Dimensions. Please choose from [64,128,256,512,768]"
    matryoshka_dim = dim
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    embeddings = model.encode([text], convert_to_tensor=True)
    embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
    embeddings = embeddings[:, :matryoshka_dim]
    embeddings = F.normalize(embeddings, p=2, dim=1)
    embeddings = embeddings.cpu().numpy().reshape(-1)
    return embeddings

# generate_embeddings("Hello World")

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

# generate_embeddings("Hello World")
# print(ChatGPT_eval("Abra-ka-dabra" , "Abra"))

# def random_sampling(sentences, labels, num):
#     """randomly sample subset of the training pairs"""
#     assert len(sentences) == len(labels)
#     if num > len(labels):
#         assert False, f"you tried to randomly sample {num}, which is more than the total size of the pool {len(labels)}"
#     idxs = np.random.choice(len(labels), size=num, replace=False)
#     selected_sentences = [sentences[i] for i in idxs]
#     selected_labels = [labels[i] for i in idxs]
#     return deepcopy(selected_sentences), deepcopy(selected_labels)

# def chunks(lst, n):
#     """Yield successive n-sized chunks from lst."""
#     for i in range(0, len(lst), n):
#         yield lst[i:i + n]

# def chunk_size_helper(params):
#     # Set the batch size (the size of the chunks determines the batch size). Default to 4 for GPT-2 and 20 for OpenAI if
#     # no batch size is specified.
#     bs = params['bs']
#     if bs is None:
#         if 'gpt2' in params['model']:
#             return 1
#         else:
#             assert params['model'] in ['ada', 'babbage', 'curie', 'davinci', 'ada-beta', 'babbage-beta', 'curie-beta', 'davinci-beta']
#             return 20
#     else:
#         return bs