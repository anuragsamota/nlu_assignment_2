import pickle
import re
from pathlib import Path



import numpy as np
import torch


# settings
BASE_PATH = "./word2vec"
MODEL_SELECTION = "both"  # choose: cbow, sgns, both
TOP_K = 5

QUERY_WORDS = ["research", "student", "phd"]
ANALOGY_TASKS = [
    ("ug", "btech", "pg"),
    ("computer", "science", "mechanical"),
    ("student", "campus", "faculty"),
]





def normalize_rows(matrix):
    # Normalize rows so cosine similarity becomes a dot product.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-12, norms)
    return matrix / norms


def safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_hparams_from_filename(model_name, file_name):
    # Extract window, dimension, and negative sample counts from filenames.
    stem = Path(file_name).stem.lower()
    info = {
        "embedding_dim": None,
        "window_size": None,
        "negative_samples": None,
    }

    patterns = [
        r"(?:cbow|skipgram)_model_(?P<window>\d+)_(?P<dim>\d+)(?:_(?P<neg>\d+))?$",
        r"(?:cbow|sgns|skipgram)_w(?P<window>\d+)_d(?P<dim>\d+)(?:_n(?P<neg>\d+))?$",
        r"(?:cbow|sgns|skipgram).*?dim(?P<dim>\d+).*?window(?P<window>\d+)(?:.*?neg(?:ative)?(?P<neg>\d+))?",
    ]

    for pattern in patterns:
        match = re.match(pattern, stem)
        if not match:
            continue
        groups = match.groupdict()
        info["embedding_dim"] = safe_int(groups.get("dim"))
        info["window_size"] = safe_int(groups.get("window"))
        info["negative_samples"] = safe_int(groups.get("neg"))
        break

    if info["embedding_dim"] is None and info["window_size"] is None:
        parts = stem.split("_")
        numeric_parts = [safe_int(part) for part in parts if safe_int(part) is not None]
        if len(numeric_parts) >= 2:
            info["window_size"] = numeric_parts[0]
            info["embedding_dim"] = numeric_parts[1]
        if len(numeric_parts) >= 3:
            info["negative_samples"] = numeric_parts[2]

    return info


def load_vocab(base_path, model_subdir):
    vocab_dir = Path(base_path) / "models" / model_subdir / "vocab"
        # Shared vocab is stored under the model subdir.
    with open(vocab_dir / "word_to_index.pkl", "rb") as f:
        word_to_index = pickle.load(f)
    with open(vocab_dir / "index_to_word.pkl", "rb") as f:
        index_to_word_raw = pickle.load(f)

    index_to_word = {}
    for key, value in index_to_word_raw.items():
        try:
            index_to_word[int(key)] = value
        except (TypeError, ValueError):
            index_to_word[key] = value

    return word_to_index, index_to_word


def discover_checkpoints(base_path, model_name):
    subdir = "cbow" if model_name == "CBOW" else "sgns"
        # Collect model checkpoints for the selected architecture.
    model_dir = Path(base_path) / "models" / subdir
    if not model_dir.exists():
        return []

    prefixes = ("cbow",) if model_name == "CBOW" else ("skipgram", "sgns")

    files = []
    for path in sorted(model_dir.glob("*.pth")):
        if path.stem.lower().startswith(prefixes):
            files.append(path)
    return files


def get_embedding_matrix(model_name, state_dict):
    key = "embedding.weight" if model_name == "CBOW" else "input_embeddings.weight"
    if key not in state_dict:
        raise KeyError(f"Missing key '{key}' in checkpoint.")
    return state_dict[key].detach().cpu().numpy()


def load_model_checkpoints(base_path, model_name):
    model_subdir = "cbow" if model_name == "CBOW" else "sgns"
    word_to_index, index_to_word = load_vocab(base_path, model_subdir)
    files = discover_checkpoints(base_path, model_name)

    loaded = []
    for file_path in files:
        state_dict = torch.load(file_path, map_location="cpu", weights_only=True)
        matrix = get_embedding_matrix(model_name, state_dict)
        matrix = normalize_rows(matrix)

        loaded.append(
            {
                "model": model_name,
                "checkpoint_name": file_path.name,
                "word_to_index": word_to_index,
                "index_to_word": index_to_word,
                "embeddings": matrix,
            }
        )

    return loaded


def get_top_neighbors(word, top_k, word_to_index, index_to_word, embeddings):
    if word not in word_to_index:
        return []

    idx = word_to_index[word]
    query_vec = embeddings[idx]
    sims = embeddings @ query_vec
    sims[idx] = -1.0

    top_indices = np.argsort(sims)[-top_k:][::-1]
    return [(index_to_word[i], float(sims[i])) for i in top_indices]


def solve_analogy(word_a, word_b, word_c, top_k, word_to_index, index_to_word, embeddings):
    for token in [word_a, word_b, word_c]:
        if token not in word_to_index:
            return []

    vec = embeddings[word_to_index[word_b]] - embeddings[word_to_index[word_a]] + embeddings[word_to_index[word_c]]
    vec_norm = np.linalg.norm(vec)
    if vec_norm == 0:
        return []

    vec = vec / vec_norm
    sims = embeddings @ vec
    for token in [word_a, word_b, word_c]:
        sims[word_to_index[token]] = -1.0

    top_indices = np.argsort(sims)[-top_k:][::-1]
    return [(index_to_word[i], float(sims[i])) for i in top_indices]


def evaluate_checkpoint(checkpoint):
    word_to_index = checkpoint["word_to_index"]
        # Compute mean top-1 cosine scores for neighbors and analogies.
    index_to_word = checkpoint["index_to_word"]
    embeddings = checkpoint["embeddings"]
    neighbor_top1_scores = []
    analogy_top1_scores = []
    neighbors_by_query = {}
    analogies_by_query = {}

    for word in QUERY_WORDS:
        neighbors = get_top_neighbors(word, TOP_K, word_to_index, index_to_word, embeddings)
        neighbors_by_query[word] = neighbors
        if neighbors:
            neighbor_top1_scores.append(neighbors[0][1])

    for word_a, word_b, word_c in ANALOGY_TASKS:
        answers = solve_analogy(word_a, word_b, word_c, TOP_K, word_to_index, index_to_word, embeddings)
        analogies_by_query[(word_a, word_b, word_c)] = answers
        if answers:
            analogy_top1_scores.append(answers[0][1])

    mean_neighbor = float(np.mean(neighbor_top1_scores)) if neighbor_top1_scores else float("nan")
    mean_analogy = float(np.mean(analogy_top1_scores)) if analogy_top1_scores else float("nan")

    return {
        "model": checkpoint["model"],
        "checkpoint": checkpoint["checkpoint_name"],
        "mean_neighbor_top1": mean_neighbor,
        "mean_analogy_top1": mean_analogy,
        "neighbors_by_query": neighbors_by_query,
        "analogies_by_query": analogies_by_query,
    }

def print_best_analogies(summary_rows):
    print("\nTask-3 Analogies (Best Models)")
    print("=" * 31)
    for row in summary_rows:
        print(f"\n{row['model']} - {row['checkpoint']}")
        for word_a, word_b, word_c in ANALOGY_TASKS:
            answers = row["analogies_by_query"].get((word_a, word_b, word_c), [])
            formatted = ", ".join([f"{tok} ({score:.4f})" for tok, score in answers]) or "No results"
            print(f"{word_a}:{word_b}::{word_c}:? -> {formatted}")


def print_best_neighbors(summary_rows):
    print("\nTask-3 Nearest Neighbors (Best Models)")
    print("=" * 41)
    for row in summary_rows:
        print(f"\n{row['model']} - {row['checkpoint']}")
        for query_word in QUERY_WORDS:
            neighbors = row["neighbors_by_query"].get(query_word, [])
            formatted = ", ".join([f"{tok} ({score:.4f})" for tok, score in neighbors]) or "No results"
            print(f"{query_word}: {formatted}")


def get_selected_model_names():
    option = MODEL_SELECTION.lower().strip()
    if option == "cbow":
        return ["CBOW"]
    if option == "sgns":
        return ["SGNS"]
    return ["CBOW", "SGNS"]



print("Word2Vec Hyperparameter Experiment Runner")
    # Load checkpoints, evaluate, then print best examples per model.
print(f"Query words     : {', '.join(QUERY_WORDS)}")
print(f"Analogy count   : {len(ANALOGY_TASKS)}")

all_checkpoints = []
for model_name in get_selected_model_names():
    loaded = load_model_checkpoints(BASE_PATH, model_name)
    if not loaded:
        print(f"\nNo checkpoints found for {model_name} under {BASE_PATH}/models")
    all_checkpoints.extend(loaded)

if not all_checkpoints:
    raise FileNotFoundError("No checkpoint files found. Train models first and save .pth files.")

summary_rows = []
for checkpoint in all_checkpoints:
    summary_rows.append(evaluate_checkpoint(checkpoint))

best_by_model = {}
for row in summary_rows:
    combined = (row["mean_neighbor_top1"] + row["mean_analogy_top1"]) / 2
    current = best_by_model.get(row["model"])
    if current is None:
        best_by_model[row["model"]] = row
        continue
    current_combined = (current["mean_neighbor_top1"] + current["mean_analogy_top1"]) / 2
    if combined > current_combined:
        best_by_model[row["model"]] = row

compact_rows = list(best_by_model.values())
print_best_neighbors(compact_rows)
print_best_analogies(compact_rows)
