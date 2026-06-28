#!/usr/bin/env python3
"""
Compare the CST surface edition vs the sandhi-resolved edition (JPTS R2-3).

Runs the headline analyses identically on both editions:
  - word-frequency k-means (k=2) agreement with CST4 canon/non-canon labels
  - leave-one-out logistic-regression accuracy
  - status of the quotative markers ti / nti among discriminative words
  - the k=3 narrative-register cluster (does it still hold)

The point: the register separation should survive correcting the word division,
demonstrating that the result is not an artefact of CST's ti/nti tokenisation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment

from data_loader import load_raw_documents, binary_labels
from features import word_frequency_matrix
from clustering import run_kmeans  # repo's published method (no scaling)

RAW_PTS = Path(__file__).parent.parent / "data" / "raw-pts"
SANDHI = Path(__file__).parent.parent / "data" / "raw-pts-sandhi"


def cluster_agreement(matrix, docs):
    """Use the repo's published k-means (raw word-frequency matrix, no scaling)."""
    res = run_kmeans(matrix, docs)
    return res["accuracy"], res["misclassified"]


def loo_accuracy(matrix, labels):
    labels = np.array(labels)
    X = StandardScaler().fit_transform(matrix)
    preds = np.zeros(len(labels))
    for tr, te in LeaveOneOut().split(X):
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X[tr], labels[tr])
        preds[te] = clf.predict(X[te])
    return accuracy_score(labels, preds)


def top_markers(matrix, labels, feature_names, n=8):
    labels = np.array(labels)
    Z = StandardScaler().fit_transform(matrix)
    diff = Z[labels == 1].mean(0) - Z[labels == 0].mean(0)  # +commentary
    order = np.argsort(diff)
    canon = [(feature_names[i], diff[i]) for i in order[:n]]
    comm = [(feature_names[i], diff[i]) for i in order[::-1][:n]]
    return canon, comm


def run(label, data_dir):
    docs = load_raw_documents(data_dir=data_dir)
    labels = binary_labels(docs)
    matrix, feats = word_frequency_matrix(docs, top_n=1000)
    n = len(docs)
    agree, misc = cluster_agreement(matrix, docs)
    loo = loo_accuracy(matrix, labels)
    canon, comm = top_markers(matrix, labels, feats)
    print(f"\n===== {label} =====")
    print(f"docs={n}  features={matrix.shape[1]}")
    print(f"  k-means (k=2) agreement with CST4 labels : {agree*100:.1f}%  "
          f"(disagreements: {', '.join(misc) if misc else 'none'})")
    print(f"  LOO-CV logistic-regression accuracy      : {loo*100:.1f}%")
    has_ti = "ti" in feats
    has_nti = "nti" in feats
    print(f"  'ti' in top-1000 features : {has_ti}   'nti' in top-1000 : {has_nti}")
    print(f"  top commentary markers: {', '.join(w for w,_ in comm)}")
    return dict(n=n, agree=agree, loo=loo, has_nti=has_nti)


def main():
    a = run("CST surface edition (data/raw-pts)", RAW_PTS)
    b = run("Sandhi-resolved edition (data/raw-pts-sandhi)", SANDHI)
    print("\n===== SUMMARY =====")
    print(f"{'metric':36} {'CST':>12} {'sandhi':>12}")
    print(f"{'k-means agreement':36} {a['agree']*100:>11.1f}% {b['agree']*100:>11.1f}%")
    print(f"{'LOO-CV accuracy':36} {a['loo']*100:>11.1f}% {b['loo']*100:>11.1f}%")
    print(f"{'nti as separate marker':36} {str(a['has_nti']):>12} {str(b['has_nti']):>12}")


if __name__ == "__main__":
    main()
