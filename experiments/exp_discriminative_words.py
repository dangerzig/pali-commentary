#!/usr/bin/env python3
"""
Produce a fuller ranked list of discriminative words separating canon from
commentary (non-canon), for a JPTS appendix.

Method matches the paper's Figure 1 feature space:
  - word_frequency_matrix(documents, top_n=1000): top-1000 words by mean
    relative frequency, per-document relative frequencies.
  - binary labels: canon=0, non-canon (commentary/subcommentary/paracanonical)=1.
  - StandardScaler on the feature matrix.

We rank by TWO statistics and report both (the L1 estimator used elsewhere in
the repo is too sparse here -- it keeps only ~21 nonzero coefficients -- so it
cannot supply a 30+30 list; we therefore use a dense L2 logistic regression as
the signed-coefficient method, and the standardized mean-frequency difference as
the primary appendix statistic):

  1. STANDARDIZED MEAN-FREQUENCY DIFFERENCE (primary, used for the appendix):
     on the StandardScaler-standardized matrix, mean(class1) - mean(class0).
     This is the standardized-effect-size (Cohen-style) separation of the two
     classes for each word. Positive => marks COMMENTARY, negative => CANON.

  2. L2 LOGISTIC-REGRESSION COEFFICIENT (corroborating): LogisticRegression(
     solver='lbfgs', C=1.0, random_state=42, max_iter=1000) on the standardized
     matrix. coef > 0 => COMMENTARY (class 1), coef < 0 => CANON (class 0).

We report the top 30 canon-marking and top 30 commentary-marking words by the
standardized mean-difference, with each word's L2 coefficient and its mean
relative frequency in each class.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data_loader import load_raw_documents, binary_labels
from features import word_frequency_matrix
from shared import get_data_dir, use_pts_data

TOP_PER_DIRECTION = 30
OUT_CSV = Path.home() / "pali-papers" / "jpts-register" / "discriminative_words_appendix.csv"


def main():
    documents = load_raw_documents(data_dir=get_data_dir())
    labels = np.array(binary_labels(documents))
    print(f"Loaded {len(documents)} documents "
          f"(canon={int((labels==0).sum())}, non-canon={int((labels==1).sum())}) "
          f"[--pts={use_pts_data()}]")

    matrix, feature_names = word_frequency_matrix(documents, top_n=1000)

    # Per-class mean relative frequency (on the raw, unscaled rel-freq matrix)
    mean_canon = matrix[labels == 0].mean(axis=0)
    mean_noncanon = matrix[labels == 1].mean(axis=0)

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(matrix)

    # (1) Standardized mean-difference (primary statistic)
    std_diff = X[labels == 1].mean(axis=0) - X[labels == 0].mean(axis=0)

    # (2) L2 logistic-regression coefficients (dense, corroborating)
    lr = LogisticRegression(solver='lbfgs', C=1.0,
                            random_state=42, max_iter=1000)
    lr.fit(X, labels)
    coefs = lr.coef_[0]

    rows = []
    for j, w in enumerate(feature_names):
        rows.append({
            'word': w,
            'std_diff': float(std_diff[j]),
            'l2_coef': float(coefs[j]),
            'direction': 'commentary' if std_diff[j] > 0 else 'canon',
            'freq_diff_noncanon_minus_canon': float(mean_noncanon[j] - mean_canon[j]),
            'mean_relfreq_canon': float(mean_canon[j]),
            'mean_relfreq_noncanon': float(mean_noncanon[j]),
        })

    # Rank by standardized mean-difference
    canon_marking = sorted([r for r in rows if r['std_diff'] < 0],
                           key=lambda r: r['std_diff'])[:TOP_PER_DIRECTION]
    comm_marking = sorted([r for r in rows if r['std_diff'] > 0],
                          key=lambda r: r['std_diff'], reverse=True)[:TOP_PER_DIRECTION]

    # Write CSV (both directions, with rank)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rank', 'direction', 'word', 'std_mean_diff', 'l2_coef',
                         'freq_diff_noncanon_minus_canon',
                         'mean_relfreq_canon', 'mean_relfreq_noncanon'])
        for rank, r in enumerate(canon_marking, 1):
            writer.writerow([rank, 'canon', r['word'], f"{r['std_diff']:.6f}",
                             f"{r['l2_coef']:.6f}",
                             f"{r['freq_diff_noncanon_minus_canon']:.8f}",
                             f"{r['mean_relfreq_canon']:.8f}",
                             f"{r['mean_relfreq_noncanon']:.8f}"])
        for rank, r in enumerate(comm_marking, 1):
            writer.writerow([rank, 'commentary', r['word'], f"{r['std_diff']:.6f}",
                             f"{r['l2_coef']:.6f}",
                             f"{r['freq_diff_noncanon_minus_canon']:.8f}",
                             f"{r['mean_relfreq_canon']:.8f}",
                             f"{r['mean_relfreq_noncanon']:.8f}"])
    print(f"Wrote CSV: {OUT_CSV}")

    def print_table(title, items):
        print(f"\n{title}")
        print(f"{'#':>3} {'word':<16} {'std_diff':>9} {'l2_coef':>9} "
              f"{'relfreq_canon':>14} {'relfreq_noncanon':>17}")
        print("-" * 74)
        for i, r in enumerate(items, 1):
            print(f"{i:>3} {r['word']:<16} {r['std_diff']:>9.4f} {r['l2_coef']:>9.4f} "
                  f"{r['mean_relfreq_canon']:>14.6f} {r['mean_relfreq_noncanon']:>17.6f}")

    print_table(f"TOP {len(canon_marking)} CANON-MARKING WORDS (most negative std mean-diff)",
                canon_marking)
    print_table(f"TOP {len(comm_marking)} COMMENTARY-MARKING WORDS (most positive std mean-diff)",
                comm_marking)


if __name__ == '__main__':
    main()
