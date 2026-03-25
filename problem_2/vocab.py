# special tokens
START = "<s>"
END = "</s>"
PAD = "<pad>"


# building character level vocab
def build_vocab(names):
    chars = sorted(list(set("".join(names))))
    chars = [PAD, START, END] + chars

    # Bidirectional mappings between characters and index
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}

    return char_to_idx, idx_to_char