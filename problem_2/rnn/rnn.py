import torch.nn as nn

class VanillaRNN(nn.Module):
    def __init__(self, vocab_size, embed_size=64, hidden_size=128):
        super().__init__()
        # Maps token ids to dense character embeddings.
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Processes sequence left-to-right using hidden_size state.
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        # Projects each timestep hidden state to vocabulary logits.
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        return self.fc(out)