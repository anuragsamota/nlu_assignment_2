import torch.nn as nn
import torch

class AttentionRNN(nn.Module):

    def __init__(self, vocab_size, embed_size=64, hidden_size=128):
        super().__init__()
        # character embeddings used as recurrent inputs.
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        # Learns per-feature attention scores over timesteps.
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)

        # Softmax on timestep axis gives attention weights for each feature.
        weights = torch.softmax(self.attn(out), dim=1)
        context = out * weights

        return self.fc(context)