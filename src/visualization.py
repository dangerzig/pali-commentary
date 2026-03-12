"""
Visualization functions for Pāli text classification experiments.

PCA and UMAP plots, colored by label, with misclassification markers.
"""

from typing import Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path

def _get_results_dir() -> Path:
    """Return results dir, respecting --pts flag."""
    import sys
    base = Path(__file__).parent.parent
    if '--pts' in sys.argv:
        d = base / "results-pts"
    else:
        d = base / "results"
    d.mkdir(exist_ok=True)
    return d

RESULTS_DIR = _get_results_dir()

# Color scheme for document types
LABEL_COLORS = {
    'canon': '#2196F3',          # blue
    'commentary': '#F44336',      # red
    'subcommentary': '#FF9800',   # orange
    'paracanonical': '#4CAF50',   # green
}

LABEL_MARKERS = {
    'canon': 'o',
    'commentary': 's',
    'subcommentary': '^',
    'paracanonical': 'D',
}


def plot_pca(feature_matrix: np.ndarray,
             labels: list[str],
             file_ids: list[str],
             title: str = 'PCA of Document Features',
             filename: str = 'pca.png',
             misclassified: Optional[list[str]] = None,
             annotate_all: bool = False,
             figsize: tuple = (12, 8)) -> str:
    """Create a PCA scatter plot colored by document label.

    Args:
        feature_matrix: (n_docs, n_features) array.
        labels: Document labels (e.g., 'canon', 'commentary').
        file_ids: Document identifiers for annotations.
        title: Plot title.
        filename: Output filename (saved to results/).
        misclassified: List of file_ids to highlight.
        annotate_all: If True, label all points.
        figsize: Figure size.

    Returns:
        Path to saved figure.
    """
    pca = PCA(n_components=2)
    coords = pca.fit_transform(feature_matrix)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot each label type
    unique_labels = sorted(set(labels))
    for label in unique_labels:
        mask = [l == label for l in labels]
        idx = np.where(mask)[0]
        ax.scatter(
            coords[idx, 0], coords[idx, 1],
            c=LABEL_COLORS.get(label, '#999999'),
            marker=LABEL_MARKERS.get(label, 'o'),
            s=60, alpha=0.8,
            label=label,
            edgecolors='white', linewidth=0.5,
        )

    # Highlight misclassified points
    if misclassified:
        for i, fid in enumerate(file_ids):
            if fid in misclassified:
                ax.scatter(
                    coords[i, 0], coords[i, 1],
                    facecolors='none', edgecolors='black',
                    s=200, linewidth=2, zorder=10,
                )
                ax.annotate(
                    fid, (coords[i, 0], coords[i, 1]),
                    fontsize=8, fontweight='bold',
                    xytext=(5, 5), textcoords='offset points',
                )

    # Annotate all points if requested
    if annotate_all:
        for i, fid in enumerate(file_ids):
            if misclassified and fid in misclassified:
                continue  # already annotated
            ax.annotate(
                fid, (coords[i, 0], coords[i, 1]),
                fontsize=6, alpha=0.6,
                xytext=(3, 3), textcoords='offset points',
            )

    var_explained = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({var_explained[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({var_explained[1]:.1%} variance)')
    ax.set_title(title)
    ax.legend(loc='best')

    plt.tight_layout()
    outpath = RESULTS_DIR / filename
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")
    return str(outpath)


def plot_umap(feature_matrix: np.ndarray,
              labels: list[str],
              file_ids: list[str],
              title: str = 'UMAP of Document Features',
              filename: str = 'umap.png',
              misclassified: Optional[list[str]] = None,
              n_neighbors: int = 15,
              min_dist: float = 0.1,
              figsize: tuple = (12, 8)) -> str:
    """Create a UMAP scatter plot colored by document label."""
    import umap

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                         random_state=42)
    coords = reducer.fit_transform(feature_matrix)

    fig, ax = plt.subplots(figsize=figsize)

    unique_labels = sorted(set(labels))
    for label in unique_labels:
        mask = [l == label for l in labels]
        idx = np.where(mask)[0]
        ax.scatter(
            coords[idx, 0], coords[idx, 1],
            c=LABEL_COLORS.get(label, '#999999'),
            marker=LABEL_MARKERS.get(label, 'o'),
            s=60, alpha=0.8,
            label=label,
            edgecolors='white', linewidth=0.5,
        )

    if misclassified:
        for i, fid in enumerate(file_ids):
            if fid in misclassified:
                ax.scatter(
                    coords[i, 0], coords[i, 1],
                    facecolors='none', edgecolors='black',
                    s=200, linewidth=2, zorder=10,
                )
                ax.annotate(
                    fid, (coords[i, 0], coords[i, 1]),
                    fontsize=8, fontweight='bold',
                    xytext=(5, 5), textcoords='offset points',
                )

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title(title)
    ax.legend(loc='best')

    plt.tight_layout()
    outpath = RESULTS_DIR / filename
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")
    return str(outpath)


def plot_dendrogram(distance_matrix: np.ndarray,
                    labels: list[str],
                    file_ids: list[str],
                    title: str = 'Hierarchical Clustering Dendrogram',
                    filename: str = 'dendrogram.png',
                    method: str = 'ward',
                    figsize: tuple = (16, 8)) -> str:
    """Plot a dendrogram from a distance matrix."""
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform

    # Convert to condensed form if square
    if distance_matrix.shape[0] == distance_matrix.shape[1]:
        condensed = squareform(distance_matrix)
    else:
        condensed = distance_matrix

    Z = linkage(condensed, method=method)

    fig, ax = plt.subplots(figsize=figsize)

    display_labels = file_ids

    dendrogram(
        Z,
        labels=display_labels,
        leaf_rotation=90,
        leaf_font_size=7,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_ylabel('Distance')

    plt.tight_layout()
    outpath = RESULTS_DIR / filename
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")
    return str(outpath)


def plot_confusion_comparison(results: dict,
                               title: str = 'Method Comparison',
                               filename: str = 'comparison.png',
                               figsize: tuple = (10, 6)) -> str:
    """Plot a comparison bar chart of method accuracies.

    Args:
        results: Dict mapping method name -> accuracy (0-1).
        title: Plot title.
        filename: Output filename.

    Returns:
        Path to saved figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    methods = list(results.keys())
    accuracies = [results[m] for m in methods]

    bars = ax.barh(methods, accuracies, color='steelblue', alpha=0.8)
    ax.set_xlim(0.8, 1.01)
    ax.set_xlabel('Accuracy')
    ax.set_title(title)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f'{acc:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    outpath = RESULTS_DIR / filename
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")
    return str(outpath)
