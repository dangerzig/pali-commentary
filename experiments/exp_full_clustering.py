#!/usr/bin/env python3
"""Cross-piṭaka clustering experiment.

Loads all canon, commentary, subcommentary, and paracanonical texts from
the Sutta, Vinaya, and Abhidhamma piṭakas, computes word-frequency features,
and runs k-means clustering with k=2 through k=6 to explore whether
additional literary registers emerge beyond the canon/commentary distinction.

Also runs PCA for visualisation.
"""

import csv
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

# Paths
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw-full"
LABELS_CSV = DATA_DIR / "LABELS.csv"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results-full"

# Pāli tokenisation (matches data_loader.py)
PALI_TOKEN_RE = re.compile(r'[^a-zA-ZāīūṭḍṇṅñṃḷĀĪŪṬḌṆṄÑṂḶ]+')


def tokenize(text: str) -> list[str]:
    text = text.replace('ṁ', 'ṃ')
    tokens = PALI_TOKEN_RE.split(text.lower())
    return [t for t in tokens if t]


def load_corpus():
    """Load all texts with their labels."""
    # Read label map
    label_map = {}
    with open(LABELS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_map[row["filename"]] = row

    documents = []
    for filepath in sorted(DATA_DIR.glob("*.txt")):
        fname = filepath.name
        if fname not in label_map:
            continue

        info = label_map[fname]
        text = filepath.read_text(encoding="utf-8")
        tokens = tokenize(text)

        if len(tokens) < 100:
            continue

        documents.append({
            "filename": fname,
            "label": info["label"],
            "pitaka": info["pitaka"],
            "text_name": info["text_name"],
            "tokens": tokens,
            "word_counts": Counter(tokens),
            "n_tokens": len(tokens),
        })

    return documents


def build_feature_matrix(documents, n_features=1000):
    """Build word-frequency feature matrix (relative frequencies of top N words)."""
    # Aggregate word counts across all documents
    total_counts = Counter()
    for doc in documents:
        total_counts.update(doc["word_counts"])

    # Top N most common words
    top_words = [w for w, _ in total_counts.most_common(n_features)]

    # Feature matrix: relative frequencies
    X = np.zeros((len(documents), len(top_words)))
    for i, doc in enumerate(documents):
        n = doc["n_tokens"]
        for j, word in enumerate(top_words):
            X[i, j] = doc["word_counts"].get(word, 0) / n

    return X, top_words


def binary_label(label):
    """Convert to binary canon/non-canon.

    Paracanonical texts are treated as non-canon for ground truth,
    consistent with data_loader.binary_labels(). Their classification
    as canonical by the clustering is then a finding, not an assumption.
    """
    return "canon" if label == "canon" else "non-canon"


def run_kmeans(X, k, random_state=42):
    """Run k-means and return (cluster_assignments, inertia)."""
    km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    clusters = km.fit_predict(X)
    return clusters, km.inertia_


def evaluate_k2_vs_binary(clusters, labels):
    """Check how well k=2 clusters align with canon/non-canon split."""
    binary = [binary_label(l) for l in labels]
    unique_binary = sorted(set(binary))

    # Try both cluster-to-label assignments and pick the better one
    best_accuracy = 0
    for mapping in [(0, 1), (1, 0)]:
        predicted = [unique_binary[mapping[c]] for c in clusters]
        acc = sum(p == t for p, t in zip(predicted, binary)) / len(binary)
        best_accuracy = max(best_accuracy, acc)

    return best_accuracy


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading corpus...")
    documents = load_corpus()
    print(f"  {len(documents)} documents loaded")

    # Summary
    label_counts = Counter(d["label"] for d in documents)
    pitaka_counts = Counter(d["pitaka"] for d in documents)
    print(f"  Labels: {dict(label_counts)}")
    print(f"  Piṭakas: {dict(pitaka_counts)}")

    print("\nBuilding feature matrix (top 1000 words)...")
    X, top_words = build_feature_matrix(documents, n_features=1000)
    print(f"  Feature matrix: {X.shape}")

    # Standardise for PCA and clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    labels = [d["label"] for d in documents]
    pitakas = [d["pitaka"] for d in documents]
    names = [d["text_name"] for d in documents]
    filenames = [d["filename"] for d in documents]

    # ── PCA ──
    print("\nRunning PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    var_explained = pca.explained_variance_ratio_

    # ── K-means for k=2..6 ──
    print("\nRunning k-means clustering...")
    results = {}
    for k in range(2, 7):
        clusters, inertia = run_kmeans(X_scaled, k)

        if k == 2:
            accuracy = evaluate_k2_vs_binary(clusters, labels)
            print(f"  k={k}: accuracy vs canon/non-canon = {accuracy:.1%}")
        else:
            binary_numeric = [0 if binary_label(l) == "canon" else 1 for l in labels]
            ari = adjusted_rand_score(binary_numeric, clusters)
            print(f"  k={k}: ARI vs binary = {ari:.3f}, inertia = {inertia:.1f}")

        results[k] = {
            "clusters": clusters,
            "inertia": inertia,
        }

    # ── Plot 1: PCA coloured by label ──
    fig, ax = plt.subplots(figsize=(12, 8))
    label_styles = {
        "canon": ("tab:blue", "o", "Canon"),
        "commentary": ("tab:red", "s", "Commentary"),
        "subcommentary": ("tab:orange", "^", "Sub-commentary"),
        "paracanonical": ("tab:green", "D", "Paracanonical"),
    }
    for label, (color, marker, display) in label_styles.items():
        mask = [l == label for l in labels]
        if not any(mask):
            continue
        idx = [i for i, m in enumerate(mask) if m]
        ax.scatter(X_pca[idx, 0], X_pca[idx, 1],
                   c=color, marker=marker, s=40, alpha=0.7, label=display)
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%} variance)")
    ax.set_title("Full corpus: PCA by label")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "pca_by_label.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved pca_by_label.png")

    # ── Plot 2: PCA coloured by piṭaka ──
    fig, ax = plt.subplots(figsize=(12, 8))
    pitaka_styles = {
        "sutta": ("tab:blue", "o", "Sutta"),
        "vinaya": ("tab:green", "s", "Vinaya"),
        "abhidhamma": ("tab:purple", "^", "Abhidhamma"),
    }
    for pitaka, (color, marker, display) in pitaka_styles.items():
        mask = [p == pitaka for p in pitakas]
        if not any(mask):
            continue
        idx = [i for i, m in enumerate(mask) if m]
        ax.scatter(X_pca[idx, 0], X_pca[idx, 1],
                   c=color, marker=marker, s=40, alpha=0.7, label=display)
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%} variance)")
    ax.set_title("Full corpus: PCA by piṭaka")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "pca_by_pitaka.png", dpi=150)
    plt.close(fig)
    print(f"Saved pca_by_pitaka.png")

    # ── Plot 3: PCA coloured by piṭaka AND label (shape=label, colour=piṭaka) ──
    fig, ax = plt.subplots(figsize=(14, 9))
    label_markers = {
        "canon": "o",
        "commentary": "s",
        "subcommentary": "^",
        "paracanonical": "D",
    }
    pitaka_colors = {
        "sutta": "tab:blue",
        "vinaya": "tab:green",
        "abhidhamma": "tab:purple",
    }
    for label, marker in label_markers.items():
        for pitaka, color in pitaka_colors.items():
            mask = [(l == label and p == pitaka)
                    for l, p in zip(labels, pitakas)]
            if not any(mask):
                continue
            idx = [i for i, m in enumerate(mask) if m]
            ax.scatter(X_pca[idx, 0], X_pca[idx, 1],
                       c=color, marker=marker, s=50, alpha=0.7,
                       label=f"{pitaka.title()} {label}")
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%} variance)")
    ax.set_title("Full corpus: PCA by piṭaka and label")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "pca_combined.png", dpi=150)
    plt.close(fig)
    print(f"Saved pca_combined.png")

    # ── Plot 4: k-means results for k=2..6 ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Elbow plot
    ax = axes[0, 0]
    ks = sorted(results.keys())
    inertias = [results[k]["inertia"] for k in ks]
    ax.plot(ks, inertias, 'bo-')
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow plot")

    # Cluster plots for k=2..6
    cmap = plt.colormaps["tab10"]
    for idx, k in enumerate(range(2, 7)):
        ax = axes[(idx + 1) // 3, (idx + 1) % 3]
        clusters = results[k]["clusters"]
        for c in range(k):
            mask = clusters == c
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=[cmap(c)], s=30, alpha=0.6, label=f"Cluster {c}")
        ax.set_xlabel(f"PC1")
        ax.set_ylabel(f"PC2")
        ax.set_title(f"k-means, k={k}")
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "kmeans_k2_to_k6.png", dpi=150)
    plt.close(fig)
    print(f"Saved kmeans_k2_to_k6.png")

    # ── Analysis: what's in each cluster for k=3 and k=4? ──
    for k in [2, 3, 4, 5]:
        clusters = results[k]["clusters"]
        print(f"\n{'='*60}")
        print(f"k={k} cluster composition:")
        print(f"{'='*60}")
        for c in range(k):
            mask = clusters == c
            cluster_labels = [labels[i] for i in range(len(labels)) if mask[i]]
            cluster_pitakas = [pitakas[i] for i in range(len(pitakas)) if mask[i]]
            cluster_names = [names[i] for i in range(len(names)) if mask[i]]

            label_dist = Counter(cluster_labels)
            pitaka_dist = Counter(cluster_pitakas)

            print(f"\n  Cluster {c} ({sum(mask)} texts):")
            print(f"    Labels: {dict(label_dist)}")
            print(f"    Piṭakas: {dict(pitaka_dist)}")
            if sum(mask) <= 15:
                for name in cluster_names:
                    print(f"      - {name}")

    # ── Save detailed results ──
    with open(RESULTS_DIR / "cluster_assignments.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header = ["filename", "text_name", "label", "pitaka"] + [f"k{k}" for k in range(2, 7)]
        writer.writerow(header)
        for i, doc in enumerate(documents):
            row = [doc["filename"], doc["text_name"], doc["label"], doc["pitaka"]]
            row += [str(results[k]["clusters"][i]) for k in range(2, 7)]
            writer.writerow(row)

    print(f"\nSaved cluster_assignments.csv")


if __name__ == "__main__":
    main()
