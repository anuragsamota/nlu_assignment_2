import torch.nn as nn

class BLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size=64, hidden_size=128):
        super().__init__()
        # Character embedding table.
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # BiLSTM reads sequence in both directions; output width doubles.
        self.lstm = nn.LSTM(embed_size, hidden_size,
                            batch_first=True, bidirectional=True)
        # Combines forward/backward features into vocabulary logits.
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        return self.fc(out)