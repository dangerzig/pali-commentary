#!/usr/bin/env python3
"""
Experiment 1: Improved Classical Features.

Compares different feature extraction methods:
  - TF-IDF (top 1000, 2000, 5000 words)
  - Character n-grams (3-grams, 4-grams)
  - Lemma frequencies (using lemmatized text)

Each is evaluated using k-means (k=2) clustering accuracy.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from sklearn.cluster import KMeans

from data_loader import load_raw_documents, load_lemmatized_documents, binary_labels
from features import (word_frequency_matrix, tfidf_matrix, char_ngram_matrix,
                       euclidean_distance_matrix)
from clustering import run_kmeans, evaluate_clustering
from visualization import plot_pca, plot_confusion_comparison


def run_feature_experiment(documents, feature_matrix, feature_names, method_name):
    """Run k-means and report results for a given feature set."""
    result = run_kmeans(feature_matrix, documents)
    print(f"  {method_name}: accuracy={result['accuracy']:.4f}, "
          f"misclassified={result['n_misclassified']} {result['misclassified']}")
    return result


def main():
    print("Experiment 1: Feature Comparison")
    print("=" * 60)

    documents = load_raw_documents()
    labels_true = binary_labels(documents)
    file_ids = [d.file_id for d in documents]
    doc_labels = [d.label for d in documents]
    n_docs = len(documents)

    print(f"Loaded {n_docs} documents\n")

    results = {}

    # --- Word frequency baselines ---
    print("Word Frequency Features:")
    for top_n in [500, 1000, 2000]:
        matrix, names = word_frequency_matrix(documents, top_n=top_n)
        r = run_feature_experiment(documents, matrix, names, f"Word freq (top {top_n})")
        results[f'Word freq ({top_n})'] = r['accuracy']

    # --- TF-IDF ---
    print("\nTF-IDF Features:")
    for max_feat in [1000, 2000, 5000]:
        matrix, names = tfidf_matrix(documents, max_features=max_feat)
        r = run_feature_experiment(documents, matrix, names, f"TF-IDF (top {max_feat})")
        results[f'TF-IDF ({max_feat})'] = r['accuracy']

    # --- Character n-grams ---
    print("\nCharacter N-gram Features:")
    for n in [(3, 3), (4, 4), (3, 4)]:
        matrix, names = char_ngram_matrix(documents, n_range=n, max_features=2000)
        label = f"Char {n[0]}-{n[1]}-grams"
        r = run_feature_experiment(documents, matrix, names, label)
        results[label] = r['accuracy']

    # --- PCA visualizations for best methods ---
    print("\nGenerating PCA plots for selected methods...")

    # TF-IDF 2000
    matrix, names = tfidf_matrix(documents, max_features=2000)
    r = run_kmeans(matrix, documents)
    plot_pca(matrix, doc_labels, file_ids,
             title='PCA: TF-IDF (2000 features)',
             filename='exp1_pca_tfidf2000.png',
             misclassified=r['misclassified'])

    # Char 3-4 grams
    matrix, names = char_ngram_matrix(documents, n_range=(3, 4), max_features=2000)
    r = run_kmeans(matrix, documents)
    plot_pca(matrix, doc_labels, file_ids,
             title='PCA: Character 3-4-grams (2000 features)',
             filename='exp1_pca_charngrams.png',
             misclassified=r['misclassified'])

    # --- Lemmatized features if available ---
    print("\nChecking for lemmatized data...")
    docs_lemma = load_lemmatized_documents(raw_docs=documents)
    n_with_lemmas = sum(1 for d in docs_lemma if len(d.lemmas) > 0)

    if n_with_lemmas == n_docs:
        print(f"All {n_docs} documents have lemmatized versions.")
        print("\nLemma Frequency Features:")
        for top_n in [500, 1000, 2000]:
            matrix, names = word_frequency_matrix(documents, top_n=top_n, use_lemmas=True)
            r = run_feature_experiment(documents, matrix, names, f"Lemma freq (top {top_n})")
            results[f'Lemma freq ({top_n})'] = r['accuracy']

        print("\nLemma TF-IDF Features:")
        for max_feat in [1000, 2000]:
            matrix, names = tfidf_matrix(documents, max_features=max_feat, use_lemmas=True)
            r = run_feature_experiment(documents, matrix, names, f"Lemma TF-IDF ({max_feat})")
            results[f'Lemma TF-IDF ({max_feat})'] = r['accuracy']
    else:
        print(f"Only {n_with_lemmas}/{n_docs} documents have lemmatized data.")
        print("Skipping lemmatized feature experiments.")

    # --- Summary comparison ---
    print("\n\nSummary:")
    print(f"{'Method':<30} {'Accuracy':>10}")
    print("-" * 42)
    for method, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{method:<30} {acc:>10.4f}")

    plot_confusion_comparison(results,
                               title='Experiment 1: Feature Comparison (k-means accuracy)',
                               filename='exp1_comparison.png')

    print("\nExperiment 1 complete.")


if __name__ == '__main__':
    main()
