from sgns import train_skipgram_model

# Skipgram takes longer to train so less epochs are used compared to CBOW
Epochs = 4

Learning_rate = 0.02



# Training Skipgram with embedding dimension 100
train_skipgram_model(epochs=Epochs, learning_rate=Learning_rate , embedding_dim=100)

# Training with embedding dimension 200
train_skipgram_model(embedding_dim=200, epochs=Epochs, learning_rate=Learning_rate)

# Training with embedding dimension 300
train_skipgram_model(embedding_dim=300, epochs=Epochs, learning_rate=Learning_rate)

# Training with window size 5 and embedding dimension 100
train_skipgram_model(window_size=5, embedding_dim=100, epochs=Epochs, learning_rate=Learning_rate)

# Training with window size 5 and embedding dimension 200
train_skipgram_model(window_size=5, embedding_dim=200, epochs=Epochs, learning_rate=Learning_rate)

# Training with window size 5 and embedding dimension 300
train_skipgram_model(window_size=5, embedding_dim=300, epochs=Epochs, learning_rate=Learning_rate)


# Negative sampling with 10 negative samples
# Trainig Skipgram with embedding dimension 100
train_skipgram_model(embedding_dim=100,negative_samples=10 , epochs=Epochs, learning_rate=Learning_rate)

# Training with embedding dimension 200
train_skipgram_model(embedding_dim=200 , negative_samples=10, epochs=Epochs, learning_rate=Learning_rate)

# Training with embedding dimension 300
train_skipgram_model(embedding_dim=300, negative_samples=10, epochs=Epochs, learning_rate=Learning_rate)

# Training with window size 5 and embedding dimension 100
train_skipgram_model(window_size=5, embedding_dim=100 , negative_samples=10, epochs=Epochs, learning_rate=Learning_rate)

# Training with window size 5 and embedding dimension 200
train_skipgram_model(window_size=5, embedding_dim=200, negative_samples=10, epochs=Epochs, learning_rate=Learning_rate)

# Training with window size 5 and embedding dimension 300
train_skipgram_model(window_size=5, embedding_dim=300, epochs=Epochs, negative_samples=10, learning_rate=Learning_rate)