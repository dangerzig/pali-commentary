"""
Feature extraction for Pāli text classification.

Supports:
- Word frequency features (top-N by mean frequency)
- TF-IDF features
- Character n-gram features
- Lemma frequency features
- Euclidean distance matrices
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import pdist, squareform

from data_loader import Document


def word_frequency_matrix(documents: list[Document],
                          top_n: int = 1000,
                          use_lemmas: bool = False) -> tuple[np.ndarray, list[str]]:
    """Build a document-word frequency matrix using top-N words.

    Selects the top-N words by mean relative frequency across documents,
    matching the approach in the original paper.

    Args:
        documents: List of Document objects.
        top_n: Number of top words to use as features.
        use_lemmas: If True, use lemma counts instead of word counts.

    Returns:
        (matrix, feature_names): matrix is (n_docs, top_n), feature_names
        is the list of words corresponding to columns.
    """
    if use_lemmas:
        counters = [d.lemma_counts for d in documents]
    else:
        counters = [d.word_counts for d in documents]

    # Get all unique words
    all_words = set()
    for c in counters:
        all_words.update(c.keys())

    # Compute relative frequencies per document
    doc_totals = [sum(c.values()) for c in counters]
    rel_freqs = {}
    for word in all_words:
        freqs = [c[word] / total if total > 0 else 0
                 for c, total in zip(counters, doc_totals)]
        rel_freqs[word] = np.mean(freqs)

    # Select top-N by mean relative frequency
    sorted_words = sorted(rel_freqs.keys(), key=lambda w: rel_freqs[w], reverse=True)
    top_words = sorted_words[:top_n]

    # Build matrix
    matrix = np.zeros((len(documents), len(top_words)))
    for i, (c, total) in enumerate(zip(counters, doc_totals)):
        for j, word in enumerate(top_words):
            matrix[i, j] = c[word] / total if total > 0 else 0

    return matrix, top_words


def tfidf_matrix(documents: list[Document],
                 max_features: int = 1000,
                 use_lemmas: bool = False,
                 ngram_range: tuple = (1, 1),
                 analyzer: str = 'word') -> tuple[np.ndarray, list[str]]:
    """Build a TF-IDF matrix from documents.

    Args:
        documents: List of Document objects.
        max_features: Maximum number of features.
        use_lemmas: If True, use lemmas instead of raw tokens.
        ngram_range: N-gram range (e.g., (1,1) for unigrams, (3,4) for 3-4 char grams).
        analyzer: 'word' or 'char' or 'char_wb'.

    Returns:
        (matrix, feature_names): TF-IDF matrix and feature names.
    """
    if use_lemmas:
        texts = [' '.join(d.lemmas) for d in documents]
    else:
        texts = [' '.join(d.tokens) for d in documents]

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        analyzer=analyzer,
        lowercase=True,
    )
    matrix = vectorizer.fit_transform(texts).toarray()
    feature_names = vectorizer.get_feature_names_out().tolist()

    return matrix, feature_names


def char_ngram_matrix(documents: list[Document],
                      n_range: tuple = (3, 3),
                      max_features: int = 1000,
                      use_lemmas: bool = False) -> tuple[np.ndarray, list[str]]:
    """Build a character n-gram frequency matrix.

    Args:
        documents: List of Document objects.
        n_range: Range of n-gram sizes, e.g., (3, 3) or (3, 4).
        max_features: Max features to keep.
        use_lemmas: If True, use lemma text.

    Returns:
        (matrix, feature_names)
    """
    return tfidf_matrix(
        documents,
        max_features=max_features,
        use_lemmas=use_lemmas,
        ngram_range=n_range,
        analyzer='char_wb',
    )


def euclidean_distance_matrix(feature_matrix: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix.

    Args:
        feature_matrix: (n_docs, n_features) array.

    Returns:
        (n_docs, n_docs) symmetric distance matrix.
    """
    return squareform(pdist(feature_matrix, metric='euclidean'))


def cosine_distance_matrix(feature_matrix: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine distance matrix."""
    return squareform(pdist(feature_matrix, metric='cosine'))


def get_top_discriminative_words(feature_matrix: np.ndarray,
                                  feature_names: list[str],
                                  labels: list[int],
                                  top_n: int = 50) -> list[tuple[str, float]]:
    """Find words most discriminative between two classes.

    Uses the difference in mean frequency between classes.

    Args:
        feature_matrix: (n_docs, n_features) array.
        feature_names: Feature names.
        labels: Binary labels (0/1).
        top_n: Number of top features to return.

    Returns:
        List of (word, score) tuples, sorted by absolute score.
        Positive score = more frequent in class 1, negative = class 0.
    """
    labels = np.array(labels)
    class0 = feature_matrix[labels == 0].mean(axis=0)
    class1 = feature_matrix[labels == 1].mean(axis=0)
    diffs = class1 - class0

    indices = np.argsort(np.abs(diffs))[::-1][:top_n]
    return [(feature_names[i], diffs[i]) for i in indices]
