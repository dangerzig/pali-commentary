#!/usr/bin/env python3
"""
Experiment 8: Jātaka position analysis.

Extracts the Jātaka's (s0510m1, s0510m2) position in the feature space:
  - Centroid distances
  - Log distance ratio
  - 5 nearest neighbours
  - Z-scores for 15 key discriminative words

Follows patterns from exp_misclassification_analysis.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from scipy.spatial.distance import cdist

from data_loader import load_raw_documents, binary_labels
from features import word_frequency_matrix, euclidean_distance_matrix

from shared import (MISCLASSIFIED_IDS, DISCRIMINATIVE_WORDS, SHORT_NAMES,
                    get_doc_index, get_data_dir)


def main():
    print("Experiment 8: Jātaka Position Analysis")
    print("=" * 60)

    # Load data
    documents = load_raw_documents(data_dir=get_data_dir())
    labels = binary_labels(documents)
    file_ids = [d.file_id for d in documents]
    print(f"Loaded {len(documents)} documents\n")

    # Build feature matrix
    matrix, feature_names = word_frequency_matrix(documents, top_n=1000)
    print(f"Feature matrix: {matrix.shape[0]} docs x {matrix.shape[1]} features\n")

    # Find canonical Jātaka documents only (VRI: s0513m/s0514m; PTS: ja1-ja6)
    jataka_ids = [d.file_id for d in documents
                  if d.label == 'canon' and (
                      d.file_id.startswith('s0513m') or d.file_id.startswith('s0514m')
                      or d.file_id.startswith('ja'))]
    if not jataka_ids:
        print("ERROR: No Jātaka documents found!")
        return

    print(f"Jātaka documents found: {jataka_ids}")
    for jid in jataka_ids:
        idx = get_doc_index(documents, jid)
        d = documents[idx]
        print(f"  {jid}: {d.name}, label={d.label}, "
              f"tokens={len(d.tokens):,}, vocab={len(d.word_counts):,}")

    # --- Centroid distances ---
    print("\n" + "=" * 60)
    print("CENTROID DISTANCES")
    print("=" * 60)

    misc_set = set(MISCLASSIFIED_IDS)

    # Compute centroids excluding boundary texts
    ref_canon_idx = [i for i, d in enumerate(documents)
                     if labels[i] == 0 and d.file_id not in misc_set]
    ref_noncanon_idx = [i for i, d in enumerate(documents)
                        if labels[i] == 1 and d.file_id not in misc_set]

    canon_centroid = matrix[ref_canon_idx].mean(axis=0).reshape(1, -1)
    noncanon_centroid = matrix[ref_noncanon_idx].mean(axis=0).reshape(1, -1)

    # Distances for all documents
    dist_to_canon = cdist(matrix, canon_centroid, metric='euclidean').flatten()
    dist_to_noncanon = cdist(matrix, noncanon_centroid, metric='euclidean').flatten()
    log_ratio = np.log10(dist_to_noncanon / dist_to_canon)

    # Print Jātaka distances
    print(f"\n{'Text':<35} {'d(canon)':>10} {'d(non-c)':>10} {'log ratio':>10}")
    print("-" * 70)

    for jid in jataka_ids:
        idx = get_doc_index(documents, jid)
        print(f"  {documents[idx].name:<33} "
              f"{dist_to_canon[idx]:>10.5f} {dist_to_noncanon[idx]:>10.5f} "
              f"{log_ratio[idx]:>10.4f}")

    # Compare with boundary texts and class statistics
    canon_log_ratios = [log_ratio[i] for i in ref_canon_idx]
    boundary_log_ratios = []
    for fid in MISCLASSIFIED_IDS:
        idx = get_doc_index(documents, fid)
        if idx is not None:
            boundary_log_ratios.append(float(log_ratio[idx]))

    print(f"\nReference statistics:")
    print(f"  Canon mean log ratio:     {np.mean(canon_log_ratios):>10.4f} "
          f"(sd = {np.std(canon_log_ratios):.4f})")
    print(f"  Canon min log ratio:      {np.min(canon_log_ratios):>10.4f}")
    print(f"  Canon max log ratio:      {np.max(canon_log_ratios):>10.4f}")
    if boundary_log_ratios:
        print(f"  Boundary texts range:     "
              f"{min(boundary_log_ratios):>10.4f} to {max(boundary_log_ratios):.4f}")

    # Rank the Jātaka among all canonical texts by log ratio
    canon_with_jataka = [(documents[i].file_id, log_ratio[i])
                         for i in range(len(documents))
                         if labels[i] == 0]
    canon_with_jataka.sort(key=lambda x: x[1])
    for jid in jataka_ids:
        rank = next(i+1 for i, (fid, _) in enumerate(canon_with_jataka)
                    if fid == jid)
        print(f"  {jid} rank among {len(canon_with_jataka)} canonical texts "
              f"(by log ratio, 1=most canon-like): {rank}")

    # --- 5 Nearest Neighbours ---
    print("\n" + "=" * 60)
    print("5 NEAREST NEIGHBOURS")
    print("=" * 60)

    dist_matrix = euclidean_distance_matrix(matrix)

    for jid in jataka_ids:
        idx = get_doc_index(documents, jid)
        distances = dist_matrix[idx].copy()
        distances[idx] = np.inf

        nn_indices = np.argsort(distances)[:5]
        print(f"\n  {documents[idx].name}:")
        print(f"    {'Neighbor':<35} {'Label':<15} {'Distance':>10}")
        print(f"    {'-'*62}")
        for ni in nn_indices:
            d = documents[ni]
            lbl = 'canon' if labels[ni] == 0 else 'non-canon'
            print(f"    {d.name:<35} {lbl:<15} {distances[ni]:>10.5f}")

        # Purity
        nn_labels = [labels[ni] for ni in nn_indices]
        true_label = labels[idx]
        purity = sum(1 for l in nn_labels if l == true_label) / 5
        print(f"    True-class purity: {purity:.0%}")

    # --- Z-scores for discriminative words ---
    print("\n" + "=" * 60)
    print("DISCRIMINATIVE WORD Z-SCORES")
    print("=" * 60)

    name_to_idx = {w: i for i, w in enumerate(feature_names)}
    available_words = []
    available_indices = []
    for w in DISCRIMINATIVE_WORDS:
        if w in name_to_idx:
            available_words.append(w)
            available_indices.append(name_to_idx[w])

    if available_indices:
        sub_matrix = matrix[:, available_indices]
        means = sub_matrix.mean(axis=0)
        stds = sub_matrix.std(axis=0)
        stds[stds == 0] = 1
        z_matrix = (sub_matrix - means) / stds

        # Print header
        header = f"\n{'Text':<35}"
        for w in available_words:
            header += f" {w:>10}"
        print(header)
        print("-" * (35 + 11 * len(available_words)))

        # Canon and non-canon means
        canon_mean_z = z_matrix[ref_canon_idx].mean(axis=0)
        noncanon_mean_z = z_matrix[ref_noncanon_idx].mean(axis=0)

        print(f"{'Canon mean':<35}", end='')
        for z in canon_mean_z:
            print(f" {z:>10.2f}", end='')
        print()

        print(f"{'Non-canon mean':<35}", end='')
        for z in noncanon_mean_z:
            print(f" {z:>10.2f}", end='')
        print()

        print("-" * (35 + 11 * len(available_words)))

        # Jātaka z-scores
        for jid in jataka_ids:
            idx = get_doc_index(documents, jid)
            print(f"{documents[idx].name:<35}", end='')
            for z in z_matrix[idx]:
                print(f" {z:>10.2f}", end='')
            print()

        # Also show boundary texts for comparison
        print("-" * (35 + 11 * len(available_words)))
        for fid in MISCLASSIFIED_IDS:
            idx = get_doc_index(documents, fid)
            if idx is None:
                continue
            name = SHORT_NAMES.get(fid, fid)
            print(f"{name:<35}", end='')
            for z in z_matrix[idx]:
                print(f" {z:>10.2f}", end='')
            print()

    print("\n" + "=" * 60)
    print("Experiment 8 complete.")
    print("=" * 60)


if __name__ == '__main__':
    main()
