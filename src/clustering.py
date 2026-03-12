"""
Clustering methods for Pāli text classification.

Wraps various clustering algorithms and evaluates them against ground truth.
"""

import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, adjusted_rand_score

from data_loader import Document, binary_labels


def align_cluster_labels(cluster_labels: np.ndarray,
                         true_labels: np.ndarray) -> np.ndarray:
    """Align binary cluster labels to match true labels (handle label permutation).

    Since clustering doesn't know which cluster is "canon" vs "commentary",
    we try both permutations and pick the one with higher accuracy.

    Only works for binary (0/1) labels. For multi-class, use Hungarian matching.
    """
    cluster_labels = np.array(cluster_labels)
    true_labels = np.array(true_labels)

    acc_original = accuracy_score(true_labels, cluster_labels)
    acc_flipped = accuracy_score(true_labels, 1 - cluster_labels)

    if acc_flipped > acc_original:
        return 1 - cluster_labels
    return cluster_labels


def evaluate_clustering(cluster_labels: np.ndarray,
                        documents: list[Document]) -> dict:
    """Evaluate clustering results against ground truth.

    Returns dict with accuracy, misclassified file_ids, ARI.
    """
    true_labels = np.array(binary_labels(documents))
    aligned = align_cluster_labels(cluster_labels, true_labels)

    misclassified = []
    for i, doc in enumerate(documents):
        if aligned[i] != true_labels[i]:
            misclassified.append(doc.file_id)

    return {
        'accuracy': accuracy_score(true_labels, aligned),
        'ari': adjusted_rand_score(true_labels, aligned),
        'misclassified': misclassified,
        'n_misclassified': len(misclassified),
        'cluster_labels': aligned,
    }


def run_kmeans(feature_matrix: np.ndarray, documents: list[Document],
               n_init: int = 50) -> dict:
    """K-means clustering (k=2)."""
    km = KMeans(n_clusters=2, n_init=n_init, random_state=42, algorithm='lloyd')
    labels = km.fit_predict(feature_matrix)
    result = evaluate_clustering(labels, documents)
    result['method'] = 'K-means'
    return result


def run_spectral(feature_matrix: np.ndarray, documents: list[Document]) -> dict:
    """Spectral clustering (k=2)."""
    sc = SpectralClustering(n_clusters=2, random_state=42, affinity='rbf',
                            n_init=20)
    labels = sc.fit_predict(feature_matrix)
    result = evaluate_clustering(labels, documents)
    result['method'] = 'Spectral'
    return result


def run_gmm(feature_matrix: np.ndarray, documents: list[Document]) -> dict:
    """Gaussian mixture model (2 components)."""
    gmm = GaussianMixture(n_components=2, random_state=42, n_init=10)
    labels = gmm.fit_predict(feature_matrix)
    result = evaluate_clustering(labels, documents)
    result['method'] = 'GMM'
    return result


def run_agglomerative(feature_matrix: np.ndarray, documents: list[Document],
                      linkage: str = 'ward') -> dict:
    """Agglomerative clustering with specified linkage."""
    ac = AgglomerativeClustering(n_clusters=2, linkage=linkage)
    labels = ac.fit_predict(feature_matrix)
    result = evaluate_clustering(labels, documents)
    result['method'] = f'Agglomerative ({linkage})'
    return result


def run_hdbscan(feature_matrix: np.ndarray, documents: list[Document],
                min_cluster_size: int = 5) -> dict:
    """HDBSCAN density-based clustering."""
    try:
        import hdbscan
    except ImportError:
        return {
            'method': 'HDBSCAN',
            'accuracy': None,
            'error': 'hdbscan not installed',
        }

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(feature_matrix)

    # HDBSCAN may assign -1 for noise; treat noise as misclassified
    n_noise = (labels == -1).sum()
    n_clusters = len(set(labels) - {-1})

    if n_clusters < 2:
        return {
            'method': 'HDBSCAN',
            'accuracy': None,
            'error': f'Found {n_clusters} clusters (need 2), {n_noise} noise points',
            'n_noise': n_noise,
        }

    # For evaluation, assign noise points to whichever cluster is closest
    true_labels = np.array(binary_labels(documents))
    noise_mask = labels == -1
    if noise_mask.any():
        from sklearn.neighbors import KNeighborsClassifier
        non_noise = ~noise_mask
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(feature_matrix[non_noise], labels[non_noise])
        labels[noise_mask] = knn.predict(feature_matrix[noise_mask])

    result = evaluate_clustering(labels, documents)
    result['method'] = 'HDBSCAN'
    result['n_noise'] = int(n_noise)
    result['n_clusters'] = n_clusters
    return result
