#!/usr/bin/env python3
"""
Misclassification analysis: why six boundary texts are consistently misclassified.

Canon texts predicted as non-canon (exegetical/hermeneutical style):
  s0515m (Mahaniddesa)    - commentary on Atthakavagga
  s0516m (Culaniddesa)    - commentary on Parayanavagga
  s0517m (Nettippakarana) - hermeneutical treatise
  s0519m (Petakopadesa)   - methodological treatise

Paracanonical texts predicted as canon (sutta-like style):
  s0518m (Milindapanha)   - philosophical dialogue
  s0520m (Suttasangaha)   - sutta-style compilation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

from data_loader import load_raw_documents, binary_labels
from features import word_frequency_matrix, euclidean_distance_matrix
from visualization import LABEL_COLORS, RESULTS_DIR

from shared import (MISCLASSIFIED_IDS, CANON_AS_NONCANON, NONCANON_AS_CANON,
                    DISCRIMINATIVE_WORDS, SHORT_NAMES, get_doc_index,
                    get_data_dir)


def section_1_document_statistics(documents, labels):
    """Compare basic text statistics of misclassified texts vs. class means."""
    print("\n" + "=" * 70)
    print("SECTION 1: Document Statistics")
    print("=" * 70)

    misc_set = set(MISCLASSIFIED_IDS)

    # Compute statistics for all documents
    stats = []
    for d in documents:
        tokens = d.tokens
        n_tokens = len(tokens)
        vocab = set(tokens)
        n_types = len(vocab)
        ttr = n_types / n_tokens if n_tokens > 0 else 0
        hapax = sum(1 for w, c in d.word_counts.items() if c == 1)
        hapax_ratio = hapax / n_types if n_types > 0 else 0
        mean_word_len = np.mean([len(t) for t in tokens]) if tokens else 0
        stats.append({
            'file_id': d.file_id,
            'label': d.label,
            'binary': 0 if d.label == 'canon' else 1,
            'n_tokens': n_tokens,
            'n_types': n_types,
            'ttr': ttr,
            'hapax_ratio': hapax_ratio,
            'mean_word_len': mean_word_len,
            'is_misc': d.file_id in misc_set,
        })

    stat_names = ['n_tokens', 'n_types', 'ttr', 'hapax_ratio', 'mean_word_len']
    stat_labels = ['Token count', 'Vocabulary size', 'Type-token ratio',
                   'Hapax ratio', 'Mean word length']

    # Compute class means excluding the 6 boundary texts
    ref_canon = [s for s in stats if s['binary'] == 0 and not s['is_misc']]
    ref_noncanon = [s for s in stats if s['binary'] == 1 and not s['is_misc']]

    canon_means = {k: np.mean([s[k] for s in ref_canon]) for k in stat_names}
    noncanon_means = {k: np.mean([s[k] for s in ref_noncanon]) for k in stat_names}

    # Print table
    header = f"{'Text':<18} {'Label':<14} {'Tokens':>10} {'Vocab':>8} {'TTR':>7} {'Hapax':>7} {'WordLen':>8}"
    print(header)
    print("-" * len(header))

    # Print class means first
    print(f"{'Canon mean':<18} {'(reference)':<14} "
          f"{canon_means['n_tokens']:>10,.0f} {canon_means['n_types']:>8,.0f} "
          f"{canon_means['ttr']:>7.4f} {canon_means['hapax_ratio']:>7.4f} "
          f"{canon_means['mean_word_len']:>8.2f}")
    print(f"{'Non-canon mean':<18} {'(reference)':<14} "
          f"{noncanon_means['n_tokens']:>10,.0f} {noncanon_means['n_types']:>8,.0f} "
          f"{noncanon_means['ttr']:>7.4f} {noncanon_means['hapax_ratio']:>7.4f} "
          f"{noncanon_means['mean_word_len']:>8.2f}")
    print("-" * len(header))

    misc_stats = [s for s in stats if s['is_misc']]
    for s in misc_stats:
        direction = "C->NC" if s['file_id'] in CANON_AS_NONCANON else "NC->C"
        print(f"{SHORT_NAMES[s['file_id']]:<18} {s['label']:<7} {direction:<6} "
              f"{s['n_tokens']:>10,} {s['n_types']:>8,} "
              f"{s['ttr']:>7.4f} {s['hapax_ratio']:>7.4f} "
              f"{s['mean_word_len']:>8.2f}")

    # Bar chart comparing misclassified texts to class means
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))

    for ax, sname, slabel in zip(axes, stat_names, stat_labels):
        # Bars: canon mean, non-canon mean, then each misclassified text
        bar_labels = ['Canon\nmean', 'Non-canon\nmean']
        bar_values = [canon_means[sname], noncanon_means[sname]]
        bar_colors = [LABEL_COLORS['canon'], LABEL_COLORS['commentary']]

        for s in misc_stats:
            bar_labels.append(s['file_id'])
            bar_values.append(s[sname])
            if s['file_id'] in CANON_AS_NONCANON:
                bar_colors.append('#90CAF9')  # light blue (canon misclassified)
            else:
                bar_colors.append('#A5D6A7')  # light green (paracanonical misclassified)

        x = np.arange(len(bar_labels))
        ax.bar(x, bar_values, color=bar_colors, edgecolor='gray', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, fontsize=7, rotation=45, ha='right')
        ax.set_title(slabel, fontsize=10)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 4))

    plt.suptitle('Document Statistics: Misclassified Texts vs. Class Means', fontsize=13)
    plt.tight_layout()
    outpath = RESULTS_DIR / 'exp_misclass_doc_statistics.png'
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")


def section_2_keyword_heatmap(documents, labels, matrix, feature_names):
    """Heatmap of discriminative word frequencies, z-scored."""
    print("\n" + "=" * 70)
    print("SECTION 2: Key Word Frequency Profiles")
    print("=" * 70)

    misc_set = set(MISCLASSIFIED_IDS)

    # Find column indices for discriminative words
    name_to_idx = {w: i for i, w in enumerate(feature_names)}
    available_words = []
    available_indices = []
    for w in DISCRIMINATIVE_WORDS:
        if w in name_to_idx:
            available_words.append(w)
            available_indices.append(name_to_idx[w])
        else:
            # Try without diacritics
            for fn in feature_names:
                if fn.startswith(w[:4]):
                    available_words.append(fn)
                    available_indices.append(name_to_idx[fn])
                    break

    if not available_indices:
        print("  WARNING: No discriminative words found in feature set")
        return

    sub_matrix = matrix[:, available_indices]

    # Z-score across all documents
    means = sub_matrix.mean(axis=0)
    stds = sub_matrix.std(axis=0)
    stds[stds == 0] = 1
    z_matrix = (sub_matrix - means) / stds

    # Build rows: canon mean, non-canon mean, then 6 texts
    ref_canon_idx = [i for i, d in enumerate(documents)
                     if labels[i] == 0 and d.file_id not in misc_set]
    ref_noncanon_idx = [i for i, d in enumerate(documents)
                        if labels[i] == 1 and d.file_id not in misc_set]

    canon_mean_z = z_matrix[ref_canon_idx].mean(axis=0)
    noncanon_mean_z = z_matrix[ref_noncanon_idx].mean(axis=0)

    misc_indices = [get_doc_index(documents, fid) for fid in MISCLASSIFIED_IDS]

    row_data = np.vstack([
        canon_mean_z,
        noncanon_mean_z,
        z_matrix[misc_indices],
    ])
    row_labels = ['Canon mean', 'Non-canon mean'] + \
                 [f"{SHORT_NAMES[fid]} ({fid})" for fid in MISCLASSIFIED_IDS]

    # Print table
    print(f"\n{'Text':<35}", end='')
    for w in available_words:
        print(f" {w:>10}", end='')
    print()
    print("-" * (35 + 11 * len(available_words)))
    for i, rl in enumerate(row_labels):
        print(f"{rl:<35}", end='')
        for j in range(len(available_words)):
            print(f" {row_data[i, j]:>10.2f}", end='')
        print()

    # Heatmap
    fig, ax = plt.subplots(figsize=(14, 6))
    vmax = np.abs(row_data).max()
    sns.heatmap(row_data, ax=ax, cmap='RdBu_r', center=0,
                vmin=-vmax, vmax=vmax,
                xticklabels=available_words, yticklabels=row_labels,
                annot=True, fmt='.1f', annot_kws={'fontsize': 8},
                linewidths=0.5, linecolor='gray')
    ax.set_title('Discriminative Word Z-Scores\n(blue = canon-like, red = commentary-like)',
                 fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)

    # Add horizontal lines to separate means from texts
    ax.axhline(y=2, color='black', linewidth=2)

    plt.tight_layout()
    outpath = RESULTS_DIR / 'exp_misclass_heatmap.png'
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")


def section_3_feature_space(documents, labels, matrix, feature_names):
    """Distance to class centroids and PCA with centroid markers."""
    print("\n" + "=" * 70)
    print("SECTION 3: Feature Space Positioning")
    print("=" * 70)

    misc_set = set(MISCLASSIFIED_IDS)
    labels_arr = np.array(labels)

    # Compute centroids (excluding misclassified texts)
    ref_canon_idx = [i for i, d in enumerate(documents)
                     if labels[i] == 0 and d.file_id not in misc_set]
    ref_noncanon_idx = [i for i, d in enumerate(documents)
                        if labels[i] == 1 and d.file_id not in misc_set]

    canon_centroid = matrix[ref_canon_idx].mean(axis=0).reshape(1, -1)
    noncanon_centroid = matrix[ref_noncanon_idx].mean(axis=0).reshape(1, -1)

    # Distances to centroids for all documents
    dist_to_canon = cdist(matrix, canon_centroid, metric='euclidean').flatten()
    dist_to_noncanon = cdist(matrix, noncanon_centroid, metric='euclidean').flatten()

    # Log distance ratio: positive = closer to non-canon, negative = closer to canon
    log_ratio = np.log10(dist_to_noncanon / dist_to_canon)

    # Print table for misclassified texts
    print(f"\n{'Text':<35} {'d(canon)':>10} {'d(non-c)':>10} {'log ratio':>10} {'True':>7}")
    print("-" * 82)
    for fid in MISCLASSIFIED_IDS:
        idx = get_doc_index(documents, fid)
        true_label = 'canon' if labels[idx] == 0 else 'non-can'
        print(f"{SHORT_NAMES[fid] + ' (' + fid + ')':<35} "
              f"{dist_to_canon[idx]:>10.5f} {dist_to_noncanon[idx]:>10.5f} "
              f"{log_ratio[idx]:>10.4f} {true_label:>7}")

    # --- Distance ratio chart ---
    fig, ax = plt.subplots(figsize=(14, 6))

    # Sort by log ratio
    sort_idx = np.argsort(log_ratio)
    sorted_ids = [documents[i].file_id for i in sort_idx]
    sorted_ratios = log_ratio[sort_idx]
    sorted_labels = labels_arr[sort_idx]

    colors = []
    for i, fid in enumerate(sorted_ids):
        if fid in misc_set:
            colors.append('#FFD700')  # gold for misclassified
        elif sorted_labels[i] == 0:
            colors.append(LABEL_COLORS['canon'])
        else:
            colors.append(LABEL_COLORS['commentary'])

    x = np.arange(len(sorted_ids))
    bars = ax.bar(x, sorted_ratios, color=colors, edgecolor='gray', linewidth=0.3)
    ax.axhline(y=0, color='black', linewidth=1, linestyle='-')
    ax.set_ylabel('log₁₀(d_noncanon / d_canon)')
    ax.set_title('Distance Ratio to Class Centroids\n'
                 '(negative = closer to canon, positive = closer to non-canon)')

    # Label misclassified bars
    for i, fid in enumerate(sorted_ids):
        if fid in misc_set:
            ax.annotate(fid, (x[i], sorted_ratios[i]),
                        fontsize=7, fontweight='bold', rotation=90,
                        ha='center', va='bottom' if sorted_ratios[i] >= 0 else 'top',
                        xytext=(0, 3 if sorted_ratios[i] >= 0 else -3),
                        textcoords='offset points')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=LABEL_COLORS['canon'], label='Canon'),
        Patch(facecolor=LABEL_COLORS['commentary'], label='Non-canon'),
        Patch(facecolor='#FFD700', label='Misclassified'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    ax.set_xticks([])

    plt.tight_layout()
    outpath = RESULTS_DIR / 'exp_misclass_distance_ratios.png'
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")

    # --- PCA with centroids ---
    pca = PCA(n_components=2)
    coords = pca.fit_transform(matrix)

    # Project centroids
    canon_centroid_2d = pca.transform(canon_centroid).flatten()
    noncanon_centroid_2d = pca.transform(noncanon_centroid).flatten()

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot all points
    for lbl_name, lbl_val in [('canon', 0), ('commentary', 1)]:
        if lbl_val == 1:
            # All non-canon labels
            mask = labels_arr == 1
            label_display = 'non-canon'
        else:
            mask = labels_arr == 0
            label_display = 'canon'

        non_misc_mask = mask & np.array([d.file_id not in misc_set for d in documents])
        ax.scatter(coords[non_misc_mask, 0], coords[non_misc_mask, 1],
                   c=LABEL_COLORS.get(lbl_name, '#999'),
                   alpha=0.5, s=40, label=label_display)

    # Plot misclassified texts prominently
    for fid in MISCLASSIFIED_IDS:
        idx = get_doc_index(documents, fid)
        color = LABEL_COLORS['canon'] if labels[idx] == 0 else LABEL_COLORS['paracanonical']
        ax.scatter(coords[idx, 0], coords[idx, 1],
                   c=color, s=120, edgecolors='black', linewidth=2, zorder=10)
        ax.annotate(SHORT_NAMES[fid], (coords[idx, 0], coords[idx, 1]),
                    fontsize=8, fontweight='bold',
                    xytext=(8, 8), textcoords='offset points')

        # Draw lines to both centroids
        ax.plot([coords[idx, 0], canon_centroid_2d[0]],
                [coords[idx, 1], canon_centroid_2d[1]],
                color=LABEL_COLORS['canon'], alpha=0.3, linewidth=1, linestyle='--')
        ax.plot([coords[idx, 0], noncanon_centroid_2d[0]],
                [coords[idx, 1], noncanon_centroid_2d[1]],
                color=LABEL_COLORS['commentary'], alpha=0.3, linewidth=1, linestyle='--')

    # Plot centroids
    ax.scatter(*canon_centroid_2d, marker='X', s=300, c=LABEL_COLORS['canon'],
               edgecolors='black', linewidth=2, zorder=20, label='Canon centroid')
    ax.scatter(*noncanon_centroid_2d, marker='X', s=300, c=LABEL_COLORS['commentary'],
               edgecolors='black', linewidth=2, zorder=20, label='Non-canon centroid')

    var_exp = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({var_exp[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({var_exp[1]:.1%} variance)')
    ax.set_title('PCA: Misclassified Texts and Class Centroids')
    ax.legend(loc='best', fontsize=9)

    plt.tight_layout()
    outpath = RESULTS_DIR / 'exp_misclass_pca_centroids.png'
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")


def section_4_decision_decomposition(documents, labels, matrix, feature_names):
    """Per-feature contribution from L1 logistic regression for each misclassified text."""
    print("\n" + "=" * 70)
    print("SECTION 4: Decision Decomposition (L1 Logistic Regression)")
    print("=" * 70)

    labels_arr = np.array(labels)

    # Train L1 LR on all data with StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(matrix)

    lr = LogisticRegression(solver='liblinear', C=1.0, l1_ratio=1.0,
                            random_state=42, max_iter=1000)
    lr.fit(X_scaled, labels_arr)

    coefs = lr.coef_[0]
    intercept = lr.intercept_[0]

    # 2x3 panel figure
    n_panels = len(MISCLASSIFIED_IDS)
    ncols = min(3, n_panels)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
    axes = np.array(axes).flatten() if n_panels > 1 else [axes]

    for panel_idx, fid in enumerate(MISCLASSIFIED_IDS):
        ax = axes[panel_idx]
        idx = get_doc_index(documents, fid)
        true_label = 'canon' if labels[idx] == 0 else 'non-canon'

        # Per-feature contribution: coef_j * x_scaled_ij
        x_scaled = X_scaled[idx]
        contributions = coefs * x_scaled

        # Total decision score
        decision = contributions.sum() + intercept

        # Top 10 pushing toward each class
        # Positive contribution -> non-canon, negative -> canon
        sorted_idx = np.argsort(contributions)

        # Top 10 toward canon (most negative) and top 10 toward non-canon (most positive)
        top_canon = sorted_idx[:10]
        top_noncanon = sorted_idx[-10:][::-1]

        # Combine and sort for display
        show_idx = np.concatenate([top_canon, top_noncanon])
        show_contributions = contributions[show_idx]
        show_words = [feature_names[i] for i in show_idx]

        # Sort by contribution value for the bar chart
        sort_order = np.argsort(show_contributions)
        show_contributions = show_contributions[sort_order]
        show_words = [show_words[i] for i in sort_order]

        colors = ['#2196F3' if c < 0 else '#F44336' for c in show_contributions]
        ax.barh(range(len(show_words)), show_contributions, color=colors,
                edgecolor='gray', linewidth=0.3)
        ax.set_yticks(range(len(show_words)))
        ax.set_yticklabels(show_words, fontsize=7)
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.set_xlabel('Contribution to decision', fontsize=8)

        predicted = 'non-canon' if decision > 0 else 'canon'
        ax.set_title(f"{SHORT_NAMES[fid]} ({fid})\n"
                     f"True: {true_label}, Score: {decision:.2f} -> {predicted}",
                     fontsize=9)

        # Print top contributors
        print(f"\n  {SHORT_NAMES[fid]} ({fid}) - true: {true_label}, "
              f"decision score: {decision:.3f}")
        print(f"    Top words pushing toward canon (blue):")
        for i in top_canon[:5]:
            print(f"      {feature_names[i]:<20} contribution={contributions[i]:>8.4f}  "
                  f"(coef={coefs[i]:.4f}, x_scaled={x_scaled[i]:.4f})")
        print(f"    Top words pushing toward non-canon (red):")
        for i in top_noncanon[:5]:
            print(f"      {feature_names[i]:<20} contribution={contributions[i]:>8.4f}  "
                  f"(coef={coefs[i]:.4f}, x_scaled={x_scaled[i]:.4f})")

    for i in range(n_panels, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Decision Decomposition: Words Driving (Mis)classification\n'
                 '(blue = toward canon, red = toward non-canon)', fontsize=13)
    plt.tight_layout()
    outpath = RESULTS_DIR / 'exp_misclass_decision_decomp.png'
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: {outpath}")


def section_5_nearest_neighbors(documents, labels, matrix):
    """Find 5 nearest neighbors of each misclassified text."""
    print("\n" + "=" * 70)
    print("SECTION 5: Nearest Neighbor Analysis")
    print("=" * 70)

    dist_matrix = euclidean_distance_matrix(matrix)
    labels_arr = np.array(labels)

    k = 5
    results = {}

    print(f"\n{'Boundary text':<30} {'Neighbor':<12} {'Label':<10} {'Distance':>10}")
    print("-" * 65)

    for fid in MISCLASSIFIED_IDS:
        idx = get_doc_index(documents, fid)
        true_label = labels[idx]
        distances = dist_matrix[idx].copy()
        distances[idx] = np.inf  # exclude self

        nn_indices = np.argsort(distances)[:k]
        nn_ids = [documents[i].file_id for i in nn_indices]
        nn_labels = [labels[i] for i in nn_indices]
        nn_dists = [distances[i] for i in nn_indices]

        # Class purity: fraction sharing the true label
        purity = sum(1 for l in nn_labels if l == true_label) / k

        results[fid] = {
            'neighbors': nn_ids,
            'labels': nn_labels,
            'distances': nn_dists,
            'purity': purity,
        }

        for i in range(k):
            prefix = f"  {SHORT_NAMES[fid]}" if i == 0 else " " * (len(SHORT_NAMES[fid]) + 2)
            lbl_str = 'canon' if nn_labels[i] == 0 else 'non-can'
            print(f"{prefix:<30} {nn_ids[i]:<12} {lbl_str:<10} {nn_dists[i]:>10.5f}")

        print(f"{'':>30} Purity (true-class neighbors): {purity:.0%}")
        print()

    # Figure: show neighbor relationships
    n_panels = len(MISCLASSIFIED_IDS)
    ncols = min(3, n_panels)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.5 * nrows))
    axes = np.array(axes).flatten() if n_panels > 1 else [axes]

    for panel_idx, fid in enumerate(MISCLASSIFIED_IDS):
        ax = axes[panel_idx]
        r = results[fid]
        true_lbl = labels[get_doc_index(documents, fid)]

        y_pos = np.arange(k)
        bar_colors = [LABEL_COLORS['canon'] if l == 0 else LABEL_COLORS['commentary']
                      for l in r['labels']]

        ax.barh(y_pos, r['distances'], color=bar_colors, edgecolor='gray', linewidth=0.5)
        ax.set_yticks(y_pos)
        nn_display = []
        for nid, nl in zip(r['neighbors'], r['labels']):
            tag = 'C' if nl == 0 else 'NC'
            nn_display.append(f"{nid} [{tag}]")
        ax.set_yticklabels(nn_display, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Euclidean distance', fontsize=8)

        true_str = 'canon' if true_lbl == 0 else 'non-canon'
        ax.set_title(f"{SHORT_NAMES[fid]}\n(true: {true_str}, purity: {r['purity']:.0%})",
                     fontsize=9)

    # Hide unused axes
    for i in range(n_panels, len(axes)):
        axes[i].set_visible(False)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=LABEL_COLORS['canon'], label='Canon neighbor'),
        Patch(facecolor=LABEL_COLORS['commentary'], label='Non-canon neighbor'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=10)

    plt.suptitle('5 Nearest Neighbors of Misclassified Texts', fontsize=13)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    outpath = RESULTS_DIR / 'exp_misclass_neighbors.png'
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")


def section_6_vocabulary_overlap(documents, labels):
    """Decompose vocabulary into canon-exclusive, non-canon-exclusive, shared, and unique."""
    print("\n" + "=" * 70)
    print("SECTION 6: Vocabulary Overlap Analysis")
    print("=" * 70)

    misc_set = set(MISCLASSIFIED_IDS)

    # Build reference vocabularies excluding the 6 boundary texts
    canon_vocab = set()
    noncanon_vocab = set()
    for d in documents:
        if d.file_id in misc_set:
            continue
        vocab = set(d.word_counts.keys())
        if d.label == 'canon':
            canon_vocab |= vocab
        else:
            noncanon_vocab |= vocab

    shared_vocab = canon_vocab & noncanon_vocab
    exclusive_canon = canon_vocab - noncanon_vocab
    exclusive_noncanon = noncanon_vocab - canon_vocab

    print(f"\nReference vocabulary sizes (excluding boundary texts):")
    print(f"  Canon-exclusive words:     {len(exclusive_canon):>8,}")
    print(f"  Non-canon-exclusive words: {len(exclusive_noncanon):>8,}")
    print(f"  Shared words:              {len(shared_vocab):>8,}")

    # For each misclassified text, decompose its vocabulary
    print(f"\n{'Text':<30} {'Canon-excl':>11} {'NonCan-excl':>12} {'Shared':>8} {'Unique':>8} {'Total':>8}")
    print("-" * 80)

    decomp = {}
    for fid in MISCLASSIFIED_IDS:
        idx = get_doc_index(documents, fid)
        doc_vocab = set(documents[idx].word_counts.keys())
        total = len(doc_vocab)

        in_canon_excl = len(doc_vocab & exclusive_canon)
        in_noncanon_excl = len(doc_vocab & exclusive_noncanon)
        in_shared = len(doc_vocab & shared_vocab)
        in_unique = len(doc_vocab - canon_vocab - noncanon_vocab)

        decomp[fid] = {
            'canon_excl': in_canon_excl / total,
            'noncanon_excl': in_noncanon_excl / total,
            'shared': in_shared / total,
            'unique': in_unique / total,
            'total': total,
        }

        true_label = documents[idx].label
        print(f"  {SHORT_NAMES[fid]:<28} "
              f"{in_canon_excl:>5} ({in_canon_excl/total:>5.1%}) "
              f"{in_noncanon_excl:>5} ({in_noncanon_excl/total:>5.1%}) "
              f"{in_shared:>5} ({in_shared/total:>5.1%}) "
              f"{in_unique:>5} ({in_unique/total:>5.1%}) "
              f"{total:>8}")

    # Stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(MISCLASSIFIED_IDS))
    width = 0.6

    canon_excl_vals = [decomp[fid]['canon_excl'] for fid in MISCLASSIFIED_IDS]
    noncanon_excl_vals = [decomp[fid]['noncanon_excl'] for fid in MISCLASSIFIED_IDS]
    shared_vals = [decomp[fid]['shared'] for fid in MISCLASSIFIED_IDS]
    unique_vals = [decomp[fid]['unique'] for fid in MISCLASSIFIED_IDS]

    bottom1 = np.array(canon_excl_vals)
    bottom2 = bottom1 + np.array(noncanon_excl_vals)
    bottom3 = bottom2 + np.array(shared_vals)

    ax.bar(x, canon_excl_vals, width, label='Canon-exclusive', color='#2196F3')
    ax.bar(x, noncanon_excl_vals, width, bottom=bottom1,
           label='Non-canon-exclusive', color='#F44336')
    ax.bar(x, shared_vals, width, bottom=bottom2, label='Shared', color='#9E9E9E')
    ax.bar(x, unique_vals, width, bottom=bottom3, label='Unique to text', color='#FFD700')

    bar_labels = [f"{SHORT_NAMES[fid]}\n({fid})" for fid in MISCLASSIFIED_IDS]
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=9)
    ax.set_ylabel('Fraction of vocabulary')
    ax.set_title('Vocabulary Composition of Misclassified Texts')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    outpath = RESULTS_DIR / 'exp_misclass_vocab_overlap.png'
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")


def main():
    print("Misclassification Analysis")
    print("=" * 70)
    print("Analyzing why 6 boundary texts are consistently misclassified\n")

    # Load data
    documents = load_raw_documents(data_dir=get_data_dir())
    labels = binary_labels(documents)
    file_ids = [d.file_id for d in documents]
    print(f"Loaded {len(documents)} documents "
          f"({sum(1 for l in labels if l == 0)} canon, "
          f"{sum(1 for l in labels if l == 1)} non-canon)")

    # Filter boundary text lists to those present in current data
    global MISCLASSIFIED_IDS, CANON_AS_NONCANON, NONCANON_AS_CANON
    found = set(d.file_id for d in documents)
    missing = [fid for fid in MISCLASSIFIED_IDS if fid not in found]
    if missing:
        print(f"  Note: {missing} not in current data (split into PTS volumes)")
    MISCLASSIFIED_IDS = [fid for fid in MISCLASSIFIED_IDS if fid in found]
    CANON_AS_NONCANON = [fid for fid in CANON_AS_NONCANON if fid in found]
    NONCANON_AS_CANON = [fid for fid in NONCANON_AS_CANON if fid in found]
    print()

    # Build feature matrix (top 1000 word frequencies)
    matrix, feature_names = word_frequency_matrix(documents, top_n=1000)
    print(f"Feature matrix: {matrix.shape[0]} docs x {matrix.shape[1]} features\n")

    # Run all sections
    section_1_document_statistics(documents, labels)
    section_2_keyword_heatmap(documents, labels, matrix, feature_names)
    section_3_feature_space(documents, labels, matrix, feature_names)
    section_4_decision_decomposition(documents, labels, matrix, feature_names)
    section_5_nearest_neighbors(documents, labels, matrix)
    section_6_vocabulary_overlap(documents, labels)

    print("\n" + "=" * 70)
    print("Misclassification analysis complete.")
    print(f"All figures saved to: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
