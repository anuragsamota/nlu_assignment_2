import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader

from vocab import build_vocab, START, END, PAD
from dataset import NameDataset, make_pad_collate_fn
from engine import train

from rnn_attention import AttentionRNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# Loading and preparing data
with open("./TrainingNames.txt") as f:
    names = f.read().splitlines()

char_to_idx, idx_to_char = build_vocab(names)
vocab_size = len(char_to_idx)

# creating dataset and dataloader with padding
dataset = NameDataset(names, char_to_idx, START, END)
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    pin_memory=device.type == "cuda",
    collate_fn=make_pad_collate_fn(char_to_idx[PAD]),
)

# train model
model = AttentionRNN(vocab_size).to(device)

train(model, dataloader, device, vocab_size, lr=0.0005, pad_idx=char_to_idx[PAD])

# save trained model
torch.save(model.state_dict(), "./models/rnn_attention.pt")