#!/usr/bin/env python3
"""
Generate a figure showing where the Jātaka volumes sit relative to other
canonical texts in the distance-ratio space, illustrating that they are
consistently the canonical texts closest to the canon/commentary boundary.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from data_loader import load_raw_documents, binary_labels
from features import word_frequency_matrix
from visualization import LABEL_COLORS, RESULTS_DIR
from shared import MISCLASSIFIED_IDS, get_data_dir

# Jātaka PTS volume file IDs
JATAKA_IDS = {'ja1', 'ja2', 'ja3', 'ja4', 'ja5', 'ja6'}

JATAKA_NAMES = {
    'ja1': 'J I', 'ja2': 'J II', 'ja3': 'J III',
    'ja4': 'J IV', 'ja5': 'J V', 'ja6': 'J VI',
}


def main():
    documents = load_raw_documents(data_dir=get_data_dir())
    labels = binary_labels(documents)
    labels_arr = np.array(labels)
    matrix, feature_names = word_frequency_matrix(documents, top_n=1000)

    misc_set = set(MISCLASSIFIED_IDS)

    # Compute centroids excluding boundary texts
    ref_canon_idx = [i for i, d in enumerate(documents)
                     if labels[i] == 0 and d.file_id not in misc_set]
    ref_noncanon_idx = [i for i, d in enumerate(documents)
                        if labels[i] == 1 and d.file_id not in misc_set]

    canon_centroid = matrix[ref_canon_idx].mean(axis=0).reshape(1, -1)
    noncanon_centroid = matrix[ref_noncanon_idx].mean(axis=0).reshape(1, -1)

    dist_to_canon = cdist(matrix, canon_centroid, metric='euclidean').flatten()
    dist_to_noncanon = cdist(matrix, noncanon_centroid, metric='euclidean').flatten()
    log_ratio = np.log10(dist_to_noncanon / dist_to_canon)

    # Filter to canonical texts only (label == 0), excluding boundary texts
    canon_indices = [i for i, d in enumerate(documents)
                     if labels[i] == 0 and d.file_id not in misc_set]

    canon_fids = [documents[i].file_id for i in canon_indices]
    canon_ratios = log_ratio[canon_indices]

    # Sort by distance ratio
    sort_order = np.argsort(canon_ratios)
    sorted_fids = [canon_fids[i] for i in sort_order]
    sorted_ratios = canon_ratios[sort_order]

    # Print table
    print(f"\n{'Rank':<6} {'File ID':<12} {'Log ratio':>10}")
    print("-" * 30)
    for rank, (fid, ratio) in enumerate(zip(sorted_fids, sorted_ratios), 1):
        marker = ' <-- Jātaka' if fid in JATAKA_IDS else ''
        print(f"{rank:<6} {fid:<12} {ratio:>10.4f}{marker}")

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(sorted_fids))
    colors = []
    for fid in sorted_fids:
        if fid in JATAKA_IDS:
            colors.append('#E67E22')  # orange for Jātaka
        else:
            colors.append(LABEL_COLORS['canon'])

    bars = ax.bar(x, sorted_ratios, color=colors, edgecolor='gray',
                  linewidth=0.3)

    # Label Jātaka bars
    for i, fid in enumerate(sorted_fids):
        if fid in JATAKA_IDS:
            ax.annotate(JATAKA_NAMES[fid], (x[i], sorted_ratios[i]),
                        fontsize=8, fontweight='bold', rotation=45,
                        ha='left', va='bottom',
                        xytext=(2, 3), textcoords='offset points')

    ax.set_ylabel('log\u2081\u2080(d_noncanon / d_canon)')
    ax.set_title('Distance ratio for canonical texts\n'
                 '(lower values = closer to the commentary boundary)')
    ax.set_xticks([])

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=LABEL_COLORS['canon'], label='Other canonical texts'),
        Patch(facecolor='#E67E22', label='Jātaka volumes'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    plt.tight_layout()
    outpath = RESULTS_DIR / 'exp_jataka_distance.png'
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {outpath}")


if __name__ == '__main__':
    main()
