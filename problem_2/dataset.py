import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


# character level encoding names dataset with start and end tokens
class NameDataset(Dataset):
    def __init__(self, names, char_to_idx, START, END):
        self.names = names
        self.char_to_idx = char_to_idx
        self.START = START
        self.END = END

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        encoded = [self.char_to_idx[ch] for ch in name]
        x = [self.char_to_idx[self.START]] + encoded
        y = encoded + [self.char_to_idx[self.END]]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)



# padding variable length sequences to batch size
def make_pad_collate_fn(pad_idx):
    def collate_fn(batch):
        x_batch, y_batch = zip(*batch)
        x_padded = pad_sequence(x_batch, batch_first=True, padding_value=pad_idx)
        y_padded = pad_sequence(y_batch, batch_first=True, padding_value=pad_idx)
        return x_padded, y_padded

    return collate_fn