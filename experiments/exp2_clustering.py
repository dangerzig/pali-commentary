#!/usr/bin/env python3
"""
Experiment 2: Better Clustering Algorithms.

Compares clustering methods using the best features from Experiment 1:
  - K-means (baseline)
  - Spectral clustering
  - Gaussian mixture models
  - Agglomerative clustering (ward, complete, average)
  - HDBSCAN

Uses both word frequency and TF-IDF features.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import load_raw_documents, load_lemmatized_documents
from features import word_frequency_matrix, tfidf_matrix
from clustering import (run_kmeans, run_spectral, run_gmm,
                         run_agglomerative, run_hdbscan)
from visualization import plot_confusion_comparison


def run_all_clustering(documents, feature_matrix, feature_names, tag=""):
    """Run all clustering methods on a feature matrix."""
    results = {}

    methods = [
        ('K-means', lambda: run_kmeans(feature_matrix, documents)),
        ('Spectral', lambda: run_spectral(feature_matrix, documents)),
        ('GMM', lambda: run_gmm(feature_matrix, documents)),
        ('Agglom. (ward)', lambda: run_agglomerative(feature_matrix, documents, 'ward')),
        ('Agglom. (complete)', lambda: run_agglomerative(feature_matrix, documents, 'complete')),
        ('Agglom. (average)', lambda: run_agglomerative(feature_matrix, documents, 'average')),
        ('HDBSCAN', lambda: run_hdbscan(feature_matrix, documents)),
    ]

    print(f"\n{'Method':<25} {'Accuracy':>10} {'Misclassified':>14} {'Details'}")
    print("-" * 70)

    for name, run_fn in methods:
        r = run_fn()
        acc = r.get('accuracy')
        if acc is not None:
            n_mis = r.get('n_misclassified', '?')
            details = ', '.join(r.get('misclassified', []))
            print(f"{name:<25} {acc:>10.4f} {n_mis:>14} {details}")
            results[f'{name} {tag}'.strip()] = acc
        else:
            error = r.get('error', 'unknown error')
            print(f"{name:<25} {'N/A':>10} {'N/A':>14} {error}")

    return results


def main():
    print("Experiment 2: Clustering Algorithm Comparison")
    print("=" * 60)

    documents = load_raw_documents()
    file_ids = [d.file_id for d in documents]
    doc_labels = [d.label for d in documents]
    n_docs = len(documents)
    print(f"Loaded {n_docs} documents\n")

    all_results = {}

    # --- Word frequency top 1000 ---
    print("Features: Word frequency (top 1000)")
    freq_matrix, freq_names = word_frequency_matrix(documents, top_n=1000)
    results = run_all_clustering(documents, freq_matrix, freq_names, "(word freq)")
    all_results.update(results)

    # --- TF-IDF 2000 ---
    print("\n\nFeatures: TF-IDF (top 2000)")
    tfidf_mat, tfidf_names = tfidf_matrix(documents, max_features=2000)
    results = run_all_clustering(documents, tfidf_mat, tfidf_names, "(TF-IDF)")
    all_results.update(results)

    # --- PCA visualization for best method ---
    # Find best method
    best_method = max(all_results, key=all_results.get)
    print(f"\nBest method: {best_method} ({all_results[best_method]:.4f})")

    # --- Lemmatized ---
    print("\n\nChecking for lemmatized data...")
    docs_lemma = load_lemmatized_documents(raw_docs=documents)
    n_with_lemmas = sum(1 for d in docs_lemma if len(d.lemmas) > 0)

    if n_with_lemmas == n_docs:
        print(f"All {n_docs} documents have lemmatized versions.")
        print("\nFeatures: Lemma frequency (top 1000)")
        lemma_matrix, lemma_names = word_frequency_matrix(documents, top_n=1000, use_lemmas=True)
        results = run_all_clustering(documents, lemma_matrix, lemma_names, "(lemma freq)")
        all_results.update(results)

        print("\nFeatures: Lemma TF-IDF (top 2000)")
        lemma_tfidf, lemma_tfidf_names = tfidf_matrix(documents, max_features=2000, use_lemmas=True)
        results = run_all_clustering(documents, lemma_tfidf, lemma_tfidf_names, "(lemma TF-IDF)")
        all_results.update(results)
    else:
        print(f"Only {n_with_lemmas}/{n_docs} have lemmatized data. Skipping.")

    # --- Summary ---
    print("\n\nFull Summary:")
    print(f"{'Method':<40} {'Accuracy':>10}")
    print("-" * 52)
    for method, acc in sorted(all_results.items(), key=lambda x: x[1], reverse=True):
        print(f"{method:<40} {acc:>10.4f}")

    plot_confusion_comparison(all_results,
                               title='Experiment 2: Clustering Method Comparison',
                               filename='exp2_comparison.png')

    print("\nExperiment 2 complete.")


if __name__ == '__main__':
    main()
