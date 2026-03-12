"""Tests for src/features.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features import (
    word_frequency_matrix, tfidf_matrix, char_ngram_matrix,
    euclidean_distance_matrix, cosine_distance_matrix,
    get_top_discriminative_words,
)
from data_loader import binary_labels


class TestWordFrequencyMatrix:
    def test_shape(self, sample_documents):
        matrix, names = word_frequency_matrix(sample_documents, top_n=5)
        assert matrix.shape == (4, 5)
        assert len(names) == 5

    def test_normalized(self, sample_documents):
        matrix, _ = word_frequency_matrix(sample_documents, top_n=5)
        # Each row should sum to <= 1 (relative frequencies of top words)
        for i in range(matrix.shape[0]):
            assert matrix[i].sum() <= 1.0 + 1e-9

    def test_nonnegative(self, sample_documents):
        matrix, _ = word_frequency_matrix(sample_documents, top_n=5)
        assert (matrix >= 0).all()


    def test_top_n_exceeds_vocab(self, sample_documents):
        matrix, names = word_frequency_matrix(sample_documents, top_n=10000)
        # When top_n exceeds vocab, matrix width should match actual vocab
        assert matrix.shape[0] == 4
        assert matrix.shape[1] == len(names)
        assert len(names) < 10000  # actual vocab is smaller


class TestTfidfMatrix:
    def test_shape(self, sample_documents):
        matrix, names = tfidf_matrix(sample_documents, max_features=5)
        assert matrix.shape[0] == 4
        assert matrix.shape[1] <= 5
        assert len(names) == matrix.shape[1]

    def test_nonnegative(self, sample_documents):
        matrix, _ = tfidf_matrix(sample_documents, max_features=5)
        assert (matrix >= 0).all()


class TestCharNgramMatrix:
    def test_shape(self, sample_documents):
        matrix, names = char_ngram_matrix(sample_documents, n_range=(3, 3),
                                           max_features=20)
        assert matrix.shape[0] == 4
        assert matrix.shape[1] <= 20
        assert len(names) == matrix.shape[1]

    def test_ngram_length(self, sample_documents):
        _, names = char_ngram_matrix(sample_documents, n_range=(3, 3),
                                      max_features=20)
        # char_wb ngrams include space boundaries, so some may be shorter
        for name in names:
            assert len(name) <= 3


class TestEuclideanDistanceMatrix:
    def test_symmetric(self, sample_feature_matrix):
        dist = euclidean_distance_matrix(sample_feature_matrix)
        np.testing.assert_array_almost_equal(dist, dist.T)

    def test_diagonal_zero(self, sample_feature_matrix):
        dist = euclidean_distance_matrix(sample_feature_matrix)
        np.testing.assert_array_almost_equal(np.diag(dist), 0)

    def test_shape(self, sample_feature_matrix):
        dist = euclidean_distance_matrix(sample_feature_matrix)
        n = sample_feature_matrix.shape[0]
        assert dist.shape == (n, n)

    def test_nonnegative(self, sample_feature_matrix):
        dist = euclidean_distance_matrix(sample_feature_matrix)
        assert (dist >= -1e-10).all()


class TestCosineDistanceMatrix:
    def test_range(self, sample_feature_matrix):
        dist = cosine_distance_matrix(sample_feature_matrix)
        assert (dist >= -1e-10).all()
        assert (dist <= 2.0 + 1e-10).all()

    def test_symmetric(self, sample_feature_matrix):
        dist = cosine_distance_matrix(sample_feature_matrix)
        np.testing.assert_array_almost_equal(dist, dist.T)


class TestGetTopDiscriminativeWords:
    def test_returns_correct_count(self, sample_documents):
        matrix, names = word_frequency_matrix(sample_documents, top_n=10)
        labels = binary_labels(sample_documents)
        result = get_top_discriminative_words(matrix, names, labels, top_n=5)
        assert len(result) == 5

    def test_sorted_by_abs_value(self, sample_documents):
        matrix, names = word_frequency_matrix(sample_documents, top_n=10)
        labels = binary_labels(sample_documents)
        result = get_top_discriminative_words(matrix, names, labels, top_n=5)
        scores = [abs(s) for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_returns_tuples(self, sample_documents):
        matrix, names = word_frequency_matrix(sample_documents, top_n=10)
        labels = binary_labels(sample_documents)
        result = get_top_discriminative_words(matrix, names, labels, top_n=3)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], (float, np.floating))
