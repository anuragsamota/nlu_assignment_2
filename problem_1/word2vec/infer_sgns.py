import pickle
import torch
import torch.nn.functional as F
from sgns import SkipGramModel

# setting project base path
directory_base_path ="./word2vec"
sgns_model_path = directory_base_path+"/models/sgns/skipgram_model.pth"

# loading vocabulary
with open(directory_base_path+"/models/sgns/vocab/word_to_index.pkl", "rb") as f:
    word_to_index = pickle.load(f)

with open(directory_base_path+"/models/sgns/vocab/index_to_word.pkl", "rb") as f:
    index_to_word = pickle.load(f)

vocab_size = len(word_to_index)



# loading model
embedding_dim = 50  # must match training

model = SkipGramModel(vocab_size, embedding_dim)
model.load_state_dict(torch.load(sgns_model_path, map_location="cpu"))
model.eval()


# extracting embeddings
embeddings = model.input_embeddings.weight.data



def find_similar_words(word, top_k=5):
    
    if word not in word_to_index:
        return "Word not in vocabulary"

    word_index = word_to_index[word]
    word_vector = embeddings[word_index]

    similarities = F.cosine_similarity(
        word_vector.unsqueeze(0), 
        embeddings
    )

    top_indices = torch.topk(similarities, top_k + 1).indices

    similar_words = []

    for idx in top_indices:
        idx = idx.item()
        if idx != word_index:
            similar_words.append(index_to_word[idx])

    return similar_words[:top_k]




# Test examples
print(find_similar_words("indian"))
print(find_similar_words("computer"))
print(find_similar_words("tech"))
print(find_similar_words("iit"))
print(find_similar_words("jodhpur"))