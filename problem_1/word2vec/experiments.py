import os
import pickle
import re
from pathlib import Path



import numpy as np
import torch


# settings
BASE_PATH = "./word2vec"
MODEL_SELECTION = "both"  # choose: cbow, sgns, both
TOP_K = 5
PRINT_DETAILED = True
REPORT_OUTPUT = "./report_word2vec_experiments.md"

QUERY_WORDS = ["research", "student", "phd", "exam"]
ANALOGY_TASKS = [
    ("ug", "btech", "pg"),
    ("computer", "science", "mechanical"),
    ("student", "campus", "faculty"),
]









def normalize_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-12, norms)
    return matrix / norms


def safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_hparams_from_filename(model_name, file_name):
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
        state_dict = torch.load(file_path, map_location="cpu")
        matrix = get_embedding_matrix(model_name, state_dict)
        matrix = normalize_rows(matrix)

        parsed = parse_hparams_from_filename(model_name, file_path.name)
        if parsed["embedding_dim"] is None:
            parsed["embedding_dim"] = int(matrix.shape[1])

        loaded.append(
            {
                "model": model_name,
                "checkpoint_name": file_path.name,
                "checkpoint_path": str(file_path),
                "word_to_index": word_to_index,
                "index_to_word": index_to_word,
                "embeddings": matrix,
                "vocab_size": len(word_to_index),
                "embedding_dim": parsed["embedding_dim"],
                "window_size": parsed["window_size"],
                "negative_samples": parsed["negative_samples"],
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


def print_ranked(title, items):
    print("\n" + title)
    print("-" * len(title))
    if not items:
        print("No results.")
        return

    for rank, (token, score) in enumerate(items, start=1):
        print(f"{rank:>2}. {token:<20} score={score:.4f}")


def evaluate_checkpoint(checkpoint):
    word_to_index = checkpoint["word_to_index"]
    index_to_word = checkpoint["index_to_word"]
    embeddings = checkpoint["embeddings"]

    missing_queries = [word for word in QUERY_WORDS if word not in word_to_index]
    neighbor_top1_scores = []
    analogy_top1_scores = []

    if PRINT_DETAILED:
        print("\n" + "=" * 72)
        print(f"Model checkpoint: {checkpoint['checkpoint_name']}")
        print("=" * 72)
        print(f"Model            : {checkpoint['model']}")
        print(f"Vocab size       : {checkpoint['vocab_size']}")
        print(f"Embedding dim    : {checkpoint['embedding_dim']}")
        print(f"Window size      : {checkpoint['window_size']}")
        print(f"Negative samples : {checkpoint['negative_samples']}")

    for word in QUERY_WORDS:
        neighbors = get_top_neighbors(word, TOP_K, word_to_index, index_to_word, embeddings)
        if neighbors:
            neighbor_top1_scores.append(neighbors[0][1])
        if PRINT_DETAILED:
            print_ranked(f"Nearest for '{word}'", neighbors)

    for word_a, word_b, word_c in ANALOGY_TASKS:
        answers = solve_analogy(word_a, word_b, word_c, TOP_K, word_to_index, index_to_word, embeddings)
        if answers:
            analogy_top1_scores.append(answers[0][1])
        if PRINT_DETAILED:
            expression = f"{word_a} : {word_b} :: {word_c} : ?"
            print_ranked(expression, answers)

    mean_neighbor = float(np.mean(neighbor_top1_scores)) if neighbor_top1_scores else float("nan")
    mean_analogy = float(np.mean(analogy_top1_scores)) if analogy_top1_scores else float("nan")

    return {
        "model": checkpoint["model"],
        "checkpoint": checkpoint["checkpoint_name"],
        "embedding_dim": checkpoint["embedding_dim"],
        "window_size": checkpoint["window_size"],
        "negative_samples": checkpoint["negative_samples"],
        "vocab_size": checkpoint["vocab_size"],
        "queries_covered": len(QUERY_WORDS) - len(missing_queries),
        "queries_total": len(QUERY_WORDS),
        "mean_neighbor_top1": mean_neighbor,
        "mean_analogy_top1": mean_analogy,
        "missing_queries": ",".join(missing_queries) if missing_queries else "-",
    }


def sort_summary_row(row):
    window = row["window_size"] if row["window_size"] is not None else 10**9
    neg = row["negative_samples"] if row["negative_samples"] is not None else 10**9
    return (row["model"], row["embedding_dim"], window, neg, row["checkpoint"])


def format_value(value):
    if isinstance(value, float) and np.isnan(value):
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_summary_table(summary_rows):
    print("\nOverall Comparison")
    print("=" * 18)
    print(
        "{:<6} {:<35} {:>5} {:>7} {:>7} {:>10} {:>10}".format(
            "Model", "Checkpoint", "Dim", "Window", "Neg", "NN@1(avg)", "ANA@1(avg)"
        )
    )
    print("-" * 90)

    for row in sorted(summary_rows, key=sort_summary_row):
        print(
            "{:<6} {:<35} {:>5} {:>7} {:>7} {:>10} {:>10}".format(
                row["model"],
                row["checkpoint"][:35],
                row["embedding_dim"],
                row["window_size"] if row["window_size"] is not None else "NA",
                row["negative_samples"] if row["negative_samples"] is not None else "NA",
                format_value(row["mean_neighbor_top1"]),
                format_value(row["mean_analogy_top1"]),
            )
        )


def write_markdown_report(summary_rows, output_path):
    lines = []
    lines.append("# Word2Vec Hyperparameter Comparison")
    lines.append("")
    lines.append("## Experiment Setup")
    lines.append("")
    lines.append(f"- Query words: {', '.join(QUERY_WORDS)}")
    lines.append("- Analogy tasks: " + "; ".join([f"{a}:{b}:{c}" for a, b, c in ANALOGY_TASKS]))
    lines.append("")
    lines.append("## Formal Comparison Table")
    lines.append("")
    lines.append(
        "| Model | Checkpoint | Embedding Dim | Window Size | Negative Samples | "
        "Vocab Size | Query Coverage | Mean Top-1 Neighbor Cosine | Mean Top-1 Analogy Cosine | Missing Query Words |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")

    for row in sorted(summary_rows, key=sort_summary_row):
        coverage = f"{row['queries_covered']}/{row['queries_total']}"
        lines.append(
            "| {model} | {checkpoint} | {dim} | {window} | {neg} | {vocab} | {coverage} | {neighbor} | {analogy} | {missing} |".format(
                model=row["model"],
                checkpoint=row["checkpoint"],
                dim=row["embedding_dim"],
                window=row["window_size"] if row["window_size"] is not None else "NA",
                neg=row["negative_samples"] if row["negative_samples"] is not None else "NA",
                vocab=row["vocab_size"],
                coverage=coverage,
                neighbor=format_value(row["mean_neighbor_top1"]),
                analogy=format_value(row["mean_analogy_top1"]),
                missing=row["missing_queries"],
            )
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Higher cosine values usually mean stronger semantic alignment.")
    lines.append("- NA means that setting was not found in file naming.")

    # os.makedirs(str(Path(output_path).parent), exist_ok=True)
    # with open(output_path, "w", encoding="utf-8") as f:
    #     f.write("\n".join(lines) + "\n")


def get_selected_model_names():
    option = MODEL_SELECTION.lower().strip()
    if option == "cbow":
        return ["CBOW"]
    if option == "sgns":
        return ["SGNS"]
    return ["CBOW", "SGNS"]


def run():
    print("Word2Vec Hyperparameter Experiment Runner")
    print("==========================================")
    print(f"Base path       : {BASE_PATH}")
    print(f"Model selection : {MODEL_SELECTION}")
    print(f"Top-k           : {TOP_K}")
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

    print_summary_table(summary_rows)
    write_markdown_report(summary_rows, REPORT_OUTPUT)
    print(f"\nFormal report written to: {REPORT_OUTPUT}")


if __name__ == "__main__":
    run()
