import torch


# training function for all types of model in this problem.
def train(model, dataloader, device, vocab_size, epochs=50, lr=0.001, pad_idx=None):
    # Ignore padding tokens in loss calculation if specified
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx) if pad_idx else torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            output = model(x)
            loss = criterion(output.view(-1, vocab_size), y.view(-1))

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch}: {total_loss:.4f}")