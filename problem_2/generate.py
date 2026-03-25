import torch

# models predicts one character at a time this function combines them to generate one name for all types of model in this problem i.e blstm , rnn , rnn_attention.
def generate_name(model, char_to_idx, idx_to_char, START, END, device):
    model.eval()
    x = torch.tensor([[char_to_idx[START]]]).to(device)
    name = ""

    # max chars 20
    for _ in range(20):
        # forward pass to get logits for next character
        output = model(x)
        
        # convert logits to probability distribution
        probs = torch.softmax(output[0, -1], dim=0)

        # sample next character from probability distribution
        idx = torch.multinomial(probs, 1).item()

        # stop on end token
        if idx == char_to_idx[END]:
            break

        name += idx_to_char[idx]
        
        # appending sampled character to sequence for next iteration
        x = torch.cat([x, torch.tensor([[idx]]).to(device)], dim=1)

    return name