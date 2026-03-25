import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import random


# reading corpus
file_path = "./corpus/corpus.txt"
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()



words = text.split()

# reducing word size from original corpus
words = words[:20000]

# handling extra spaces and newline if any
words = [w.strip() for w in words if w.strip() != ""]



# building vocabulary
unique_words = list(set(words))

word_to_index = {}
index_to_word = {}

for i, word in enumerate(unique_words):
    word_to_index[word] = i
    index_to_word[i] = word

vocab_size = len(unique_words)


# base directory path to reference for model and vocabulary with respect to project root directory.
directory_base_path ="./word2vec"


# Saving vocabulary to files
with open(directory_base_path+"/models/cbow/vocab/word_to_index.pkl", "wb") as f:
    pickle.dump(word_to_index, f)

with open(directory_base_path+"/models/cbow/vocab/index_to_word.pkl", "wb") as f:
    pickle.dump(index_to_word, f)




# Model implementation
class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, context_words, target_word, negative_words):
        context_embeds = self.embedding(context_words)
        context_vector = context_embeds.mean(dim=1)

        target_vector = self.output_embeddings(target_word)
        pos_score = torch.sum(context_vector * target_vector, dim=1)
        pos_loss = F.logsigmoid(pos_score)

        neg_vectors = self.output_embeddings(negative_words)
        neg_score = torch.bmm(neg_vectors, context_vector.unsqueeze(2)).squeeze(2)
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1)

        loss = -(pos_loss + neg_loss)
        return loss.mean()

# negative function sampling for CBOW
def sample_negative_words(target_index, vocab_size, num_samples):
    negatives = []

    while len(negatives) < num_samples:
        rand_word = random.randint(0, vocab_size - 1)

        if rand_word != target_index:
            negatives.append(rand_word)

    return negatives





def train_cbow_model(window_size=2, embedding_dim=50, epochs=20, learning_rate=0.01, negative_samples=5):
    # training data for Cbow
    training_data = []

    for i in range(len(words)):
        context_words = []

        for j in range(i - window_size, i + window_size + 1):
            if j == i or j < 0 or j >= len(words):
                continue
            context_words.append(word_to_index[words[j]])

        target_word = word_to_index[words[i]]

        if len(context_words) > 0:
            training_data.append((context_words, target_word))


    # Checking gpu before training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)



    # Model training with CBOW + negative sampling objective.
    model = CBOWModel(vocab_size, embedding_dim)

    # use gpu if available
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0

        for context, target in training_data:

            context_tensor = torch.LongTensor(context).unsqueeze(0).to(device)
            target_tensor = torch.LongTensor([target]).to(device)
            negatives = sample_negative_words(target, vocab_size, negative_samples)
            negative_tensor = torch.LongTensor(negatives).unsqueeze(0).to(device)

            loss = model(context_tensor, target_tensor, negative_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # Saving model to file
    torch.save(
        model.state_dict(),
        directory_base_path+"/models/cbow/cbow_model_"+str(window_size)+"_"+str(embedding_dim)+"_"+str(negative_samples)+".pth"
    )


