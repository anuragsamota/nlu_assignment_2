import csv
import os
import random
import torch

from vocab import build_vocab, START, END
from generate import generate_name
from rnn.rnn import VanillaRNN
from blstm.blstm import BLSTM
from rnn_attention.rnn_attention import AttentionRNN



NUM_SAMPLES = 60 # names in samples
RANDOM_SEED = 42
OUTPUT_CSV_PATH = "./evaluation_summary.csv"
OUTPUT_SAMPLES_DIR = "./samples"


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# hyperparameters getting
def get_recurrent_hparams(model):
    if hasattr(model, "rnn"):
        return model.rnn.hidden_size, model.rnn.num_layers
    if hasattr(model, "lstm"):
        return model.lstm.hidden_size, model.lstm.num_layers
    raise AttributeError(f"Unable to locate recurrent module on {type(model).__name__}")


def get_model_specs(vocab_size):
    return {
        "RNN": {
            "model": VanillaRNN(vocab_size),
            "checkpoint": "./models/rnn.pt",
            "learning_rate": 0.001,
        },
        "BLSTM": {
            "model": BLSTM(vocab_size),
            "checkpoint": "./models/blstm.pt",
            "learning_rate": 0.001,
        },
        "RNN_Attention": {
            "model": AttentionRNN(vocab_size),
            "checkpoint": "./models/rnn_attention.pt",
            "learning_rate": 0.0005,
        },
    }


def generate_samples(model, char_to_idx, idx_to_char, device, num_samples):
    return [
        generate_name(model, char_to_idx, idx_to_char, START, END, device)
        for _ in range(num_samples)
    ]


def compute_metrics(samples, train_name_set):
    novelty = len([name for name in samples if name not in train_name_set]) / len(samples)
    diversity = len(set(samples)) / len(samples)
    return novelty, diversity


def print_comparison_table(rows):
    headers = ["Model", "Params", "Hidden", "Layers", "LR", "Novelty", "Diversity"]
    table = []

    for row in rows:
        table.append([
            row["model"],
            str(row["params"]),
            str(row["hidden_size"]),
            str(row["layers"]),
            f"{row['learning_rate']:.4g}",
            f"{row['novelty']:.4f}",
            f"{row['diversity']:.4f}",
        ])

    widths = [len(h) for h in headers]
    for record in table:
        for i, value in enumerate(record):
            widths[i] = max(widths[i], len(value))

    def fmt(record):
        return " | ".join(value.ljust(widths[i]) for i, value in enumerate(record))

    print("Model Evaluation Comparison")
    print(fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for record in table:
        print(fmt(record))


def save_csv(rows, csv_path):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model",
            "params",
            "hidden_size",
            "layers",
            "learning_rate",
            "novelty",
            "diversity",
        ])
        for row in rows:
            writer.writerow([
                row["model"],
                row["params"],
                row["hidden_size"],
                row["layers"],
                row["learning_rate"],
                row["novelty"],
                row["diversity"],
            ])


def save_samples(samples_by_model, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for model_name, samples in samples_by_model.items():
        safe_name = model_name.lower().replace(" ", "_")
        sample_path = os.path.join(output_dir, f"{safe_name}_samples.txt")

        with open(sample_path, "w") as f:
            f.write("\n".join(samples))
            f.write("\n")




## Evaluation

# Set random seeds for reproducibility
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# checking if gpu is available or else falling back to cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# Loading training names and building vocab
with open("TrainingNames.txt") as f:
    train_names = f.read().splitlines()

char_to_idx, idx_to_char = build_vocab(train_names)
vocab_size = len(char_to_idx)
train_name_set = set(train_names)

# Collecting evaluation results for all Models
rows = []
samples_by_model = {}
model_specs = get_model_specs(vocab_size)

# Evaluating each model with the same sample for proper comparison
for model_name, spec in model_specs.items():
    model = spec["model"].to(device)
    checkpoint_path = spec["checkpoint"]

    # Loads pre-trained weights from checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found for {model_name}: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    # Generate names and compute metrics such as novelty and diversity
    samples = generate_samples(model, char_to_idx, idx_to_char, device, NUM_SAMPLES)
    novelty, diversity = compute_metrics(samples, train_name_set)
    hidden_size, layers = get_recurrent_hparams(model)

    # storing results for this model
    rows.append({
        "model": model_name,
        "params": count_trainable_parameters(model),
        "hidden_size": hidden_size,
        "layers": layers,
        "learning_rate": spec["learning_rate"],
        "novelty": novelty,
        "diversity": diversity,
    })
    samples_by_model[model_name] = samples

print_comparison_table(rows)


# Save results to files
save_csv(rows, OUTPUT_CSV_PATH)
save_samples(samples_by_model, OUTPUT_SAMPLES_DIR)