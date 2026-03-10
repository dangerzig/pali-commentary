#!/usr/bin/env python3
"""
Experiment 0: Reproduce the original 2021 paper baseline.

Method:
  - Tokenize raw VRI texts
  - Compute word frequencies per document
  - Select top-1000 words by mean frequency across all documents
  - Compute Euclidean distance matrix
  - k-means clustering (k=2)
  - PCA visualization (matching Figure 1 from the paper)

Success criterion: Same 3 misclassifications as original paper
  (s0507m Petavatthu, s0515m Mahaniddesa, s0516m Culaniddesa)

Then repeat with lemmatized word frequencies.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from sklearn.cluster import KMeans

from data_loader import load_raw_documents, binary_labels
from features import word_frequency_matrix, euclidean_distance_matrix
from visualization import plot_pca, plot_dendrogram


def run_kmeans_experiment(documents, feature_matrix, feature_names, tag="raw"):
    """Run k-means (k=2) and report results."""
    print(f"\n{'='*60}")
    print(f"  K-means Clustering ({tag})")
    print(f"{'='*60}")

    n_docs = len(documents)
    labels_true = binary_labels(documents)  # 0=canon, 1=non-canon
    file_ids = [d.file_id for d in documents]
    doc_labels = [d.label for d in documents]

    # Compute distance matrix
    dist_matrix = euclidean_distance_matrix(feature_matrix)

    # K-means with k=2
    # Use multiple initializations for stability
    km = KMeans(n_clusters=2, n_init=50, random_state=42, algorithm='lloyd')
    cluster_labels = km.fit_predict(feature_matrix)

    # Determine which cluster corresponds to canon vs non-canon
    # Count how many canon docs are in each cluster
    canon_in_0 = sum(1 for i in range(n_docs)
                     if labels_true[i] == 0 and cluster_labels[i] == 0)
    canon_in_1 = sum(1 for i in range(n_docs)
                     if labels_true[i] == 0 and cluster_labels[i] == 1)

    if canon_in_1 > canon_in_0:
        # Flip cluster labels
        cluster_labels = 1 - cluster_labels

    # Find misclassifications
    misclassified = []
    for i in range(n_docs):
        if cluster_labels[i] != labels_true[i]:
            misclassified.append(file_ids[i])

    # Report results
    n_correct = sum(1 for i in range(n_docs)
                    if cluster_labels[i] == labels_true[i])
    accuracy = n_correct / n_docs

    print(f"\n  Documents: {n_docs}")
    print(f"  Features: {len(feature_names)} words")
    print(f"  Accuracy: {accuracy:.4f} ({n_correct}/{n_docs})")
    print(f"  Misclassified: {len(misclassified)}")
    for fid in misclassified:
        doc = next(d for d in documents if d.file_id == fid)
        predicted = 'canon' if doc.label != 'canon' else 'commentary'
        print(f"    {fid}: true={doc.label}, predicted={predicted}")

    # Check if we matched the original paper's results
    expected_misclass = {'s0507m', 's0515m', 's0516m'}
    if set(misclassified) == expected_misclass:
        print(f"\n  *** MATCHES ORIGINAL PAPER: same 3 misclassifications ***")
    else:
        print(f"\n  Original paper misclassified: {expected_misclass}")
        print(f"  We misclassified: {set(misclassified)}")

    # PCA visualization
    plot_pca(
        feature_matrix, doc_labels, file_ids,
        title=f'PCA: Top-1000 Word Frequencies ({tag})',
        filename=f'exp0_pca_{tag}.png',
        misclassified=misclassified,
    )

    # Dendrogram
    plot_dendrogram(
        dist_matrix, doc_labels, file_ids,
        title=f'Hierarchical Clustering ({tag})',
        filename=f'exp0_dendrogram_{tag}.png',
    )

    # Top discriminative features
    print(f"\n  Top 20 most discriminative words:")
    labels_arr = np.array(labels_true)
    canon_mean = feature_matrix[labels_arr == 0].mean(axis=0)
    noncanon_mean = feature_matrix[labels_arr == 1].mean(axis=0)
    diffs = noncanon_mean - canon_mean
    top_idx = np.argsort(np.abs(diffs))[::-1][:20]
    print(f"    {'Word':<20} {'Canon freq':>12} {'Non-canon freq':>14} {'Diff':>10}")
    print(f"    {'-'*56}")
    for idx in top_idx:
        print(f"    {feature_names[idx]:<20} {canon_mean[idx]:>12.6f} {noncanon_mean[idx]:>14.6f} {diffs[idx]:>10.6f}")

    return accuracy, misclassified, cluster_labels


def main():
    print("Experiment 0: Reproduce Original Baseline")
    print("=" * 60)

    # Load all Sutta Pitaka documents
    documents = load_raw_documents()
    print(f"\nLoaded {len(documents)} documents:")
    for label in ['canon', 'commentary', 'subcommentary', 'paracanonical']:
        subset = [d for d in documents if d.label == label]
        print(f"  {label}: {len(subset)} docs")

    # Build word frequency matrix (top 1000)
    print("\nBuilding word frequency matrix (top 1000 words)...")
    freq_matrix, feature_names = word_frequency_matrix(documents, top_n=1000)
    print(f"  Matrix shape: {freq_matrix.shape}")
    print(f"  Top 10 words: {feature_names[:10]}")

    # Run k-means on raw features
    accuracy, misclassified, clusters = run_kmeans_experiment(
        documents, freq_matrix, feature_names, tag="raw"
    )

    # --- Now try with lemmatized features if available ---
    from data_loader import load_lemmatized_documents
    docs_with_lemmas = load_lemmatized_documents(raw_docs=documents)
    docs_with_lemma_data = [d for d in docs_with_lemmas if len(d.lemmas) > 0]

    if len(docs_with_lemma_data) > 0:
        print(f"\n\nFound lemmatized data for {len(docs_with_lemma_data)}/{len(documents)} documents")

        if len(docs_with_lemma_data) == len(documents):
            print("Running k-means on lemmatized features...")
            lemma_matrix, lemma_features = word_frequency_matrix(
                documents, top_n=1000, use_lemmas=True
            )
            accuracy_l, misclass_l, clusters_l = run_kmeans_experiment(
                documents, lemma_matrix, lemma_features, tag="lemmatized"
            )
        else:
            print("Not all documents have lemmatized versions; skipping lemmatized experiment.")
    else:
        print("\n\nNo lemmatized data available; skipping lemmatized experiment.")

    print("\n\nExperiment 0 complete.")


if __name__ == '__main__':
    main()
