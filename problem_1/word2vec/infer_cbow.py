import torch
import torch.nn as nn
import pickle
from cbow import CBOWModel, CBOWNegativeSamplingModel



directory_base_path ="./word2vec"

cbow_model_path = directory_base_path+"/models/cbow/cbow_model_7_200_5.pth"

# loading vocabulary
with open(directory_base_path+"/models/cbow/vocab/word_to_index.pkl", "rb") as f:
    word_to_index = pickle.load(f)

with open(directory_base_path+"/models/cbow/vocab/index_to_word.pkl", "rb") as f:
    index_to_word = pickle.load(f)



embedding_dim = 200  # this must be same as training
vocab_size = len(word_to_index)

state_dict = torch.load(cbow_model_path, map_location="cpu")

# Load model type based on checkpoint keys.
if "linear.weight" in state_dict:
    model = CBOWModel(vocab_size, embedding_dim)
else:
    model = CBOWNegativeSamplingModel(vocab_size, embedding_dim)

model.load_state_dict(state_dict)
model.eval()



# inference function
def predict_word(context_words):
    indices = [word_to_index[w] for w in context_words]

    context_tensor = torch.LongTensor(indices).unsqueeze(0)

    with torch.no_grad():
        if isinstance(model, CBOWModel):
            output = model(context_tensor)
        else:
            context_vector = model.embedding(context_tensor).mean(dim=1)
            output = torch.matmul(context_vector, model.output_embeddings.weight.t())

    predicted_index = torch.argmax(output, dim=1).item()
    predicted_word = index_to_word[predicted_index]

    return predicted_word



# Test Exmples
print(predict_word(["btech"]))
print(predict_word(["department", "computer"]))
print(predict_word(["student","campus" ,"environment" ]))