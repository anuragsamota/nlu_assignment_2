from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from experiments import (
    evaluate_checkpoint,
    load_model_checkpoints,
)


def load_top_words(corpus_path: Path, top_k: int):
    # Simple frequency-based vocabulary slice for plotting.
    text = corpus_path.read_text(encoding="utf-8", errors="ignore")
    tokens = [tok for tok in text.split() if tok.strip()]
    counts = Counter(tokens)
    return [word for word, _count in counts.most_common(top_k)]


def select_best_checkpoint(checkpoints):
    # Pick the checkpoint with the highest mean neighbor/analogy score.
    best_checkpoint = None
    best_eval = None
    best_score = None

    for checkpoint in checkpoints:
        result = evaluate_checkpoint(checkpoint)
        combined = (result["mean_neighbor_top1"] + result["mean_analogy_top1"]) / 2
        if best_checkpoint is None or combined > best_score:
            best_checkpoint = checkpoint
            best_eval = result
            best_score = combined

    return best_checkpoint, best_eval, best_score


def run_pca(vectors: np.ndarray):
    # PCA via SVD for 2D projection.
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    _, _s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T
    return centered @ components


def plot_embeddings(points, words, title, output_path, annotate_k):
    plt.figure(figsize=(10, 7))
    plt.scatter(points[:, 0], points[:, 1], s=18, alpha=0.7)

    for idx, word in enumerate(words[:annotate_k]):
        plt.annotate(word, (points[idx, 0], points[idx, 1]), fontsize=8)

    plt.title(title)
    plt.xlabel("Dim-1")
    plt.ylabel("Dim-2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()



base_path = Path("./word2vec")
corpus_path = Path("./corpus/corpus.txt")
top_k = 100
annotate_k = 20
output_dir = base_path / "visualizations"
output_dir.mkdir(parents=True, exist_ok=True)

top_words = load_top_words(corpus_path, top_k)

results = []
for model_name in ("CBOW", "SGNS"):
    checkpoints = load_model_checkpoints(base_path, model_name)
    if not checkpoints:
        print(f"No checkpoints found for {model_name}.")
        continue

    best_checkpoint, best_eval, best_score = select_best_checkpoint(checkpoints)
    if best_checkpoint is None or best_eval is None:
        print(f"No valid checkpoint found for {model_name}.")
        continue

    word_to_index = best_checkpoint["word_to_index"]
    words = [w for w in top_words if w in word_to_index]
    vectors = best_checkpoint["embeddings"][[word_to_index[w] for w in words]]

    results.append(
        {
            "model": model_name,
            "checkpoint": best_eval["checkpoint"],
            "words": words,
            "vectors": vectors,
            "score": best_score,
        }
    )

if not results:
    print("No models available for visualization.")
    exit(0)

for item in results:
    model = item["model"]
    checkpoint = item["checkpoint"]
    words = item["words"]
    vectors = item["vectors"]

    pca_points = run_pca(vectors)
    title = f"{model} PCA - {checkpoint}"
    output_path = output_dir / f"{model.lower()}_pca.png"
    plot_embeddings(pca_points, words, title, output_path, annotate_k)

# Minimal summary output for repeatable runs.
print(f"Models compared: {', '.join([r['model'] for r in results])}")
print(f"Output folder: {output_dir}")
