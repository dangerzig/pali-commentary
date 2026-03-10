#!/usr/bin/env python3
"""
Experiment 3: Supervised Classification.

Uses leave-one-out cross-validation with:
  - Logistic regression (L1, L2)
  - SVM (linear, RBF)
  - Random forest
  - Gradient boosting

Reports accuracy, precision, recall, F1.
Feature importance analysis.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_loader import load_raw_documents, load_lemmatized_documents
from features import word_frequency_matrix, tfidf_matrix
from classification import run_loo_cv, get_feature_importance, get_logistic_coefficients
from visualization import RESULTS_DIR


def print_results_table(results: list[dict], tag: str = ""):
    """Print a formatted results table."""
    print(f"\n{'Method':<30} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Mis':>5} {'Misclassified'}")
    print("-" * 100)
    for r in results:
        mis = ', '.join(r['misclassified']) if r['misclassified'] else 'none'
        print(f"{r['method']:<30} {r['accuracy']:>7.4f} {r['precision']:>7.4f} "
              f"{r['recall']:>7.4f} {r['f1']:>7.4f} {r['n_misclassified']:>5} {mis}")


def plot_feature_importance(features: list[tuple[str, float]],
                            title: str, filename: str):
    """Plot feature importance bar chart."""
    names = [f[0] for f in features[:25]]
    values = [f[1] for f in features[:25]]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#F44336' if v > 0 else '#2196F3' for v in values]
    ax.barh(range(len(names)), values, color=colors, alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Coefficient / Importance')
    ax.set_title(title)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#F44336', label='Commentary indicator'),
                       Patch(facecolor='#2196F3', label='Canon indicator')]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    outpath = RESULTS_DIR / filename
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")


def main():
    print("Experiment 3: Supervised Classification")
    print("=" * 60)

    documents = load_raw_documents()
    n_docs = len(documents)
    print(f"Loaded {n_docs} documents\n")

    # --- TF-IDF features (best from exp1 typically) ---
    print("Features: TF-IDF (top 2000)")
    tfidf_mat, tfidf_names = tfidf_matrix(documents, max_features=2000)

    print("\nRunning LOO-CV on all classifiers...")
    results_tfidf = run_loo_cv(tfidf_mat, documents)
    print_results_table(results_tfidf, "TF-IDF 2000")

    # --- Word frequency features ---
    print("\n\nFeatures: Word frequency (top 1000)")
    freq_mat, freq_names = word_frequency_matrix(documents, top_n=1000)

    print("\nRunning LOO-CV on all classifiers...")
    results_freq = run_loo_cv(freq_mat, documents)
    print_results_table(results_freq, "Word freq 1000")

    # --- Feature importance analysis ---
    print("\n\nFeature Importance Analysis (TF-IDF 2000):")

    print("\n  Top 25 features by Random Forest importance:")
    rf_importance = get_feature_importance(tfidf_mat, documents, tfidf_names, top_n=25)
    for word, imp in rf_importance:
        print(f"    {word:<20} {imp:.4f}")
    plot_feature_importance(rf_importance,
                            'Random Forest Feature Importance (TF-IDF)',
                            'exp3_rf_importance.png')

    print("\n  Top 25 features by Logistic Regression coefficients:")
    lr_coefs = get_logistic_coefficients(tfidf_mat, documents, tfidf_names, top_n=25)
    for word, coef in lr_coefs:
        direction = "commentary" if coef > 0 else "canon"
        print(f"    {word:<20} {coef:>8.4f}  ({direction})")
    plot_feature_importance(lr_coefs,
                            'Logistic Regression Coefficients (TF-IDF)',
                            'exp3_lr_coefficients.png')

    # --- Lemmatized features if available ---
    print("\n\nChecking for lemmatized data...")
    docs_lemma = load_lemmatized_documents(raw_docs=documents)
    n_with_lemmas = sum(1 for d in docs_lemma if len(d.lemmas) > 0)

    if n_with_lemmas == n_docs:
        print(f"All {n_docs} documents have lemmatized versions.")
        print("\nFeatures: Lemma TF-IDF (top 2000)")
        lemma_tfidf, lemma_names = tfidf_matrix(documents, max_features=2000, use_lemmas=True)

        print("\nRunning LOO-CV on all classifiers...")
        results_lemma = run_loo_cv(lemma_tfidf, documents)
        print_results_table(results_lemma, "Lemma TF-IDF 2000")

        print("\n  Top 25 lemma features by LR coefficients:")
        lr_lemma = get_logistic_coefficients(lemma_tfidf, documents, lemma_names, top_n=25)
        for word, coef in lr_lemma:
            direction = "commentary" if coef > 0 else "canon"
            print(f"    {word:<20} {coef:>8.4f}  ({direction})")
        plot_feature_importance(lr_lemma,
                                'Logistic Regression Coefficients (Lemma TF-IDF)',
                                'exp3_lr_coefficients_lemma.png')
    else:
        print(f"Only {n_with_lemmas}/{n_docs} have lemmatized data. Skipping.")

    # --- Summary ---
    print("\n\nSummary of best results by feature type:")
    for tag, results in [("TF-IDF 2000", results_tfidf),
                          ("Word freq 1000", results_freq)]:
        best = max(results, key=lambda r: r['accuracy'])
        print(f"  {tag}: {best['method']} = {best['accuracy']:.4f}")

    print("\nExperiment 3 complete.")


if __name__ == '__main__':
    main()
