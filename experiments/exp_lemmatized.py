#!/usr/bin/env python3
"""
Run key experiments on lemmatized text to compare with raw word forms.

Tests: word frequency, TF-IDF, k-means, supervised classification.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from data_loader import load_raw_documents, load_lemmatized_documents, binary_labels
from features import word_frequency_matrix, tfidf_matrix
from clustering import run_kmeans, run_spectral, run_gmm
from classification import run_loo_cv, get_logistic_coefficients
from visualization import plot_pca, plot_confusion_comparison, RESULTS_DIR


def main():
    print("Lemmatized Feature Experiments")
    print("=" * 60)

    documents = load_raw_documents()
    documents = load_lemmatized_documents(raw_docs=documents)
    n_with_lemmas = sum(1 for d in documents if len(d.lemmas) > 0)
    print(f"Loaded {len(documents)} documents, {n_with_lemmas} with lemmatized data\n")

    if n_with_lemmas < len(documents):
        print("ERROR: Not all documents have lemmatized data")
        return

    file_ids = [d.file_id for d in documents]
    doc_labels = [d.label for d in documents]
    results = {}

    # --- K-means with lemma frequencies ---
    print("K-means clustering with lemma frequencies:")
    for top_n in [500, 1000, 2000]:
        matrix, names = word_frequency_matrix(documents, top_n=top_n, use_lemmas=True)
        r = run_kmeans(matrix, documents)
        print(f"  Lemma freq (top {top_n}): accuracy={r['accuracy']:.4f}, "
              f"misclassified={r['n_misclassified']} {r['misclassified']}")
        results[f'K-means lemma freq ({top_n})'] = r['accuracy']

    # PCA visualization
    matrix, names = word_frequency_matrix(documents, top_n=1000, use_lemmas=True)
    r = run_kmeans(matrix, documents)
    plot_pca(matrix, doc_labels, file_ids,
             title='PCA: Lemma Frequencies (top 1000)',
             filename='exp_lemma_pca_freq1000.png',
             misclassified=r['misclassified'])

    # --- K-means with lemma TF-IDF ---
    print("\nK-means clustering with lemma TF-IDF:")
    for max_feat in [1000, 2000, 5000]:
        matrix, names = tfidf_matrix(documents, max_features=max_feat, use_lemmas=True)
        r = run_kmeans(matrix, documents)
        print(f"  Lemma TF-IDF ({max_feat}): accuracy={r['accuracy']:.4f}, "
              f"misclassified={r['n_misclassified']} {r['misclassified']}")
        results[f'K-means lemma TF-IDF ({max_feat})'] = r['accuracy']

    # --- Other clustering methods ---
    print("\nOther clustering methods (lemma freq top 1000):")
    matrix, names = word_frequency_matrix(documents, top_n=1000, use_lemmas=True)
    for name, fn in [
        ('Spectral', lambda: run_spectral(matrix, documents)),
        ('GMM', lambda: run_gmm(matrix, documents)),
    ]:
        r = fn()
        print(f"  {name}: accuracy={r['accuracy']:.4f}, "
              f"misclassified={r['n_misclassified']} {r['misclassified']}")
        results[f'{name} (lemma freq)'] = r['accuracy']

    # --- Supervised classification ---
    print("\nSupervised classification (lemma TF-IDF 2000):")
    matrix, names = tfidf_matrix(documents, max_features=2000, use_lemmas=True)
    sup_results = run_loo_cv(matrix, documents)
    for r in sup_results:
        print(f"  {r['method']}: accuracy={r['accuracy']:.4f}, "
              f"misclassified={r['n_misclassified']} {r['misclassified']}")
        results[f"{r['method']} (lemma TF-IDF)"] = r['accuracy']

    # --- Feature importance ---
    print("\nTop discriminative lemmas (LR coefficients):")
    lr_coefs = get_logistic_coefficients(matrix, documents, names, top_n=20)
    for word, coef in lr_coefs:
        direction = "commentary" if coef > 0 else "canon"
        print(f"  {word:<20} {coef:>8.4f}  ({direction})")

    # --- Comparison: raw vs lemmatized ---
    print("\n\nComparison: Raw vs Lemmatized (k-means, top 1000)")
    raw_matrix, raw_names = word_frequency_matrix(documents, top_n=1000, use_lemmas=False)
    r_raw = run_kmeans(raw_matrix, documents)
    lemma_matrix, lemma_names = word_frequency_matrix(documents, top_n=1000, use_lemmas=True)
    r_lemma = run_kmeans(lemma_matrix, documents)
    print(f"  Raw word freq: accuracy={r_raw['accuracy']:.4f}, misclassified={r_raw['misclassified']}")
    print(f"  Lemma freq:    accuracy={r_lemma['accuracy']:.4f}, misclassified={r_lemma['misclassified']}")

    # --- Summary ---
    print("\n\nSummary:")
    print(f"{'Method':<45} {'Accuracy':>10}")
    print("-" * 57)
    for method, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{method:<45} {acc:>10.4f}")

    plot_confusion_comparison(results,
                               title='Lemmatized Feature Experiments',
                               filename='exp_lemma_comparison.png')

    print("\nLemmatized experiments complete.")


if __name__ == '__main__':
    main()
