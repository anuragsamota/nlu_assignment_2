from cbow import train_cbow_model


Epochs = 7
Learning_rate = 0.02

# Trainig CBOW with embedding dimension 100
train_cbow_model(epochs=Epochs , learning_rate=Learning_rate, embedding_dim=100 , negative_samples=5)

# Training with embedding dimension 200
train_cbow_model(epochs=Epochs, learning_rate=Learning_rate,embedding_dim=200 , negative_samples=5)

# Training with embedding dimension 300
train_cbow_model(epochs=Epochs,learning_rate=Learning_rate, embedding_dim=300, negative_samples=5)


# Training with window size 5 and embedding dimension 100
train_cbow_model(epochs=Epochs,learning_rate=Learning_rate,window_size=5, embedding_dim=100 , negative_samples=5)

# Training with window size 5 and embedding dimension 100
train_cbow_model(epochs=Epochs, learning_rate=Learning_rate, window_size=5, embedding_dim=200, negative_samples=5)

# Training with window size 5 and embedding dimension 200
train_cbow_model(epochs=Epochs, learning_rate=Learning_rate, window_size=5, embedding_dim=300, negative_samples=5)



# Negative sampling with 10 negative samples

# Trainig CBOW with default settings
train_cbow_model(epochs=Epochs , learning_rate=Learning_rate , embedding_dim=100 , negative_samples=10)

# Training withembedding dimension 200
train_cbow_model(epochs=Epochs, learning_rate=Learning_rate,embedding_dim=200 , negative_samples=10)

# Training with embedding dimension 300
train_cbow_model(epochs=Epochs,learning_rate=Learning_rate, embedding_dim=300, negative_samples=10)


# Training with window size 5 and embedding dimension 100
train_cbow_model(epochs=Epochs,learning_rate=Learning_rate,window_size=5, embedding_dim=100 , negative_samples=10)

# Training with window size 5 and embedding dimension 200
train_cbow_model(epochs=Epochs, learning_rate=Learning_rate, window_size=5, embedding_dim=200, negative_samples=10)

# Training with window size 5 and embedding dimension 300
train_cbow_model(epochs=Epochs, learning_rate=Learning_rate, window_size=5, embedding_dim=300, negative_samples=10)