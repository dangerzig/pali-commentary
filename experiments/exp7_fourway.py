#!/usr/bin/env python3
"""
Experiment 7: Four-way classification.

Tests whether canon, commentary, subcommentary, and paracanonical texts
can be distinguished using 4-way LOO-CV with LR, SVM, RF, gradient boosting
on TF-IDF (2000) and word frequency (1000) features.

Key question: are commentary and subcommentary separable from each other?
Caveat: paracanonical class has only 4 samples, so classification of that
class will be unreliable.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.decomposition import PCA

from data_loader import load_raw_documents, VRI_NAMES
from features import word_frequency_matrix, tfidf_matrix
from visualization import LABEL_COLORS, RESULTS_DIR


CLASS_ORDER = ['canon', 'commentary', 'subcommentary', 'paracanonical']


def get_fourway_classifiers():
    """Return classifiers for 4-way classification."""
    return {
        'Logistic Regression (L2)': (
            LogisticRegression(solver='lbfgs', C=1.0, max_iter=1000,
                               random_state=42),
            True
        ),
        'SVM (linear)': (
            SVC(kernel='linear', C=1.0, random_state=42),
            True
        ),
        'Random Forest': (
            RandomForestClassifier(n_estimators=100, random_state=42),
            False
        ),
        'Gradient Boosting': (
            GradientBoostingClassifier(n_estimators=100, random_state=42,
                                       max_depth=3),
            False
        ),
    }


def run_fourway_loo_cv(X, labels, documents):
    """Run LOO-CV with 4-way labels."""
    le = LabelEncoder()
    le.fit(CLASS_ORDER)
    y = le.transform(labels)

    loo = LeaveOneOut()
    classifiers = get_fourway_classifiers()
    results = []

    for name, (clf, needs_scaling) in classifiers.items():
        X_work = X.copy()
        if needs_scaling:
            scaler = StandardScaler()
            X_work = scaler.fit_transform(X_work)

        preds = np.zeros(len(y), dtype=int)
        for train_idx, test_idx in loo.split(X_work):
            clf_copy = type(clf)(**clf.get_params())
            clf_copy.fit(X_work[train_idx], y[train_idx])
            preds[test_idx] = clf_copy.predict(X_work[test_idx])

        acc = accuracy_score(y, preds)
        pred_labels = le.inverse_transform(preds)
        misclassified = [(documents[i].file_id, labels[i], pred_labels[i])
                         for i in range(len(y)) if preds[i] != y[i]]

        results.append({
            'method': name,
            'accuracy': acc,
            'predictions': pred_labels,
            'true_labels': labels,
            'misclassified': misclassified,
            'y_true': y,
            'y_pred': preds,
        })

    return results, le


def plot_confusion_matrix(y_true, y_pred, le, method, feature_tag, filename):
    """Plot and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    class_names = le.classes_

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.5, linecolor='gray')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'4-Way Confusion Matrix\n{method} on {feature_tag}')

    plt.tight_layout()
    outpath = RESULTS_DIR / filename
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_fourway_pca(X, labels, documents, filename):
    """PCA plot with 4-way labels."""
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(12, 8))

    for label in CLASS_ORDER:
        mask = [l == label for l in labels]
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        ax.scatter(
            coords[idx, 0], coords[idx, 1],
            c=LABEL_COLORS.get(label, '#999'),
            s=60, alpha=0.7,
            label=f'{label} (n={len(idx)})',
            edgecolors='white', linewidth=0.5,
        )

    # Annotate boundary texts and paracanonical
    boundary_ids = {'s0515m', 's0516m', 's0517m', 's0518m', 's0519m', 's0520m'}
    for i, d in enumerate(documents):
        if d.file_id in boundary_ids or d.label == 'paracanonical':
            ax.annotate(
                d.file_id, (coords[i, 0], coords[i, 1]),
                fontsize=7, fontweight='bold',
                xytext=(5, 5), textcoords='offset points',
            )

    var_exp = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({var_exp[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({var_exp[1]:.1%} variance)')
    ax.set_title('PCA: 4-Way Classification\n(canon / commentary / subcommentary / paracanonical)')
    ax.legend(loc='best')

    plt.tight_layout()
    outpath = RESULTS_DIR / filename
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")


def main():
    print("Experiment 7: Four-Way Classification")
    print("=" * 60)

    # Load all documents with 4-way labels
    documents = load_raw_documents()
    labels = [d.label for d in documents]
    n_docs = len(documents)

    print(f"Loaded {n_docs} documents")
    for cls in CLASS_ORDER:
        n = sum(1 for l in labels if l == cls)
        print(f"  {cls}: {n}")
    print()

    # --- TF-IDF features ---
    print("Features: TF-IDF (top 2000)")
    tfidf_mat, tfidf_names = tfidf_matrix(documents, max_features=2000)
    results_tfidf, le = run_fourway_loo_cv(tfidf_mat, labels, documents)

    for r in results_tfidf:
        print(f"\n  {r['method']}: accuracy = {r['accuracy']:.4f} "
              f"({len(r['misclassified'])} misclassified)")
        if r['misclassified']:
            for fid, true, pred in r['misclassified']:
                print(f"    {fid}: {true} -> {pred}")

    # Best TF-IDF result: full report and confusion matrix
    best_tfidf = max(results_tfidf, key=lambda r: r['accuracy'])
    print(f"\nBest TF-IDF: {best_tfidf['method']}")
    print(classification_report(best_tfidf['y_true'], best_tfidf['y_pred'],
                                target_names=le.classes_))
    plot_confusion_matrix(best_tfidf['y_true'], best_tfidf['y_pred'], le,
                          best_tfidf['method'], 'TF-IDF 2000',
                          'exp7_fourway_confusion_tfidf.png')

    # --- Word frequency features ---
    print("\n\nFeatures: Word frequency (top 1000)")
    freq_mat, freq_names = word_frequency_matrix(documents, top_n=1000)
    results_freq, le = run_fourway_loo_cv(freq_mat, labels, documents)

    for r in results_freq:
        print(f"\n  {r['method']}: accuracy = {r['accuracy']:.4f} "
              f"({len(r['misclassified'])} misclassified)")
        if r['misclassified']:
            for fid, true, pred in r['misclassified']:
                print(f"    {fid}: {true} -> {pred}")

    # Best word freq result
    best_freq = max(results_freq, key=lambda r: r['accuracy'])
    print(f"\nBest Word Freq: {best_freq['method']}")
    print(classification_report(best_freq['y_true'], best_freq['y_pred'],
                                target_names=le.classes_))
    plot_confusion_matrix(best_freq['y_true'], best_freq['y_pred'], le,
                          best_freq['method'], 'Word freq 1000',
                          'exp7_fourway_confusion_freq.png')

    # --- Combined best confusion matrix ---
    # Use whichever is best overall
    all_results = results_tfidf + results_freq
    overall_best = max(all_results, key=lambda r: r['accuracy'])
    plot_confusion_matrix(overall_best['y_true'], overall_best['y_pred'], le,
                          overall_best['method'], 'best features',
                          'exp7_fourway_confusion.png')

    # --- 4-way PCA ---
    plot_fourway_pca(freq_mat, labels, documents, 'exp7_fourway_pca.png')

    # --- Summary table ---
    print("\n\n" + "=" * 60)
    print("SUMMARY: 4-Way Classification Results")
    print("=" * 60)
    print(f"\n{'Features':<20} {'Method':<25} {'Accuracy':>10} {'Misc':>5}")
    print("-" * 62)
    for tag, results in [("TF-IDF 2000", results_tfidf),
                          ("Word freq 1000", results_freq)]:
        for r in results:
            print(f"  {tag:<18} {r['method']:<25} {r['accuracy']:>10.4f} "
                  f"{len(r['misclassified']):>5}")

    print("\nExperiment 7 complete.")


if __name__ == '__main__':
    main()
