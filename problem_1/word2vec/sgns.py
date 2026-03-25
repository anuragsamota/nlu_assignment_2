import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import pickle


#setting base directory
directory_base_path ="./word2vec"

# Getting corpus
file_path = "./corpus/corpus.txt"

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()


# preprocessing corpus
words = text.split()
words = words[:5000]
words = [w.strip() for w in words if w.strip() != ""]


# building vocab
unique_words = list(set(words))

word_to_index = {}
index_to_word = {}

for i, word in enumerate(unique_words):
    word_to_index[word] = i
    index_to_word[i] = word

vocab_size = len(unique_words)

# Saving vocabulary to files
with open(directory_base_path+"/models/sgns/vocab/word_to_index.pkl", "wb") as f:
    pickle.dump(word_to_index, f)

with open(directory_base_path+"/models/sgns/vocab/index_to_word.pkl", "wb") as f:
    pickle.dump(index_to_word, f)




def sample_negative_words(context_index, vocab_size, num_samples):
    negatives = []

    while len(negatives) < num_samples:
        rand_word = random.randint(0, vocab_size - 1)

        if rand_word != context_index:
            negatives.append(rand_word)

    return negatives



# Skip gram model implementation

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, target_word, context_word, negative_words):

        target_vector = self.input_embeddings(target_word)
        context_vector = self.output_embeddings(context_word)

        # Positive score
        pos_score = torch.sum(target_vector * context_vector, dim=1)
        pos_loss = F.logsigmoid(pos_score)

        # Negative samples
        neg_vectors = self.output_embeddings(negative_words)
        neg_score = torch.bmm(neg_vectors, target_vector.unsqueeze(2)).squeeze(2)
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1)

        loss = -(pos_loss + neg_loss)

        return loss.mean()
    



def train_skipgram_model(window_size=2, embedding_dim=50, epochs=4, learning_rate=0.01 , negative_samples=5):
    # training data generation
    training_data = []

    for i in range(len(words)):
        target_word = word_to_index[words[i]]

        for j in range(i - window_size, i + window_size + 1):
            if j == i or j < 0 or j >= len(words):
                continue

            context_word = word_to_index[words[j]]
            training_data.append((target_word, context_word))



    # Training the model
    model = SkipGramModel(vocab_size, embedding_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0

        for target, context in training_data:

            target_tensor = torch.LongTensor([target]).to(device)
            context_tensor = torch.LongTensor([context]).to(device)

            negatives = sample_negative_words(context, vocab_size, negative_samples)
            negative_tensor = torch.LongTensor(negatives).unsqueeze(0).to(device)

            loss = model(target_tensor, context_tensor, negative_tensor)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(training_data)
        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.6f}")


    # Saving the model
    torch.save(model.state_dict(), directory_base_path + "/models/sgns/skipgram_model_"+str(window_size)+"_"+str(embedding_dim)+"_"+str(negative_samples)+".pth")


