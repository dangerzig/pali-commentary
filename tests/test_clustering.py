"""Tests for src/clustering.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clustering import (align_cluster_labels, evaluate_clustering, run_kmeans,
                        run_spectral, run_gmm, run_agglomerative)
from data_loader import RAW_DATA_DIR


class TestAlignClusterLabels:
    def test_already_aligned(self):
        true = np.array([0, 0, 1, 1])
        pred = np.array([0, 0, 1, 1])
        result = align_cluster_labels(pred, true)
        np.testing.assert_array_equal(result, np.array([0, 0, 1, 1]))

    def test_flipped(self):
        true = np.array([0, 0, 1, 1])
        pred = np.array([1, 1, 0, 0])
        result = align_cluster_labels(pred, true)
        np.testing.assert_array_equal(result, np.array([0, 0, 1, 1]))

    def test_partial_match(self):
        true = np.array([0, 0, 1, 1, 0])
        pred = np.array([0, 0, 1, 0, 0])  # 4/5 correct
        result = align_cluster_labels(pred, true)
        # Should keep original since 4/5 > 1/5
        np.testing.assert_array_equal(result, pred)


class TestEvaluateClustering:
    def test_perfect(self, sample_documents):
        perfect_labels = np.array([0, 0, 1, 1])
        result = evaluate_clustering(perfect_labels, sample_documents)
        assert result['accuracy'] == 1.0
        assert result['ari'] == 1.0
        assert result['misclassified'] == []
        assert result['n_misclassified'] == 0

    def test_perfect_flipped(self, sample_documents):
        # Flipped labels should also give perfect score after alignment
        flipped_labels = np.array([1, 1, 0, 0])
        result = evaluate_clustering(flipped_labels, sample_documents)
        assert result['accuracy'] == 1.0

    def test_imperfect(self, sample_documents):
        bad_labels = np.array([0, 1, 1, 1])
        result = evaluate_clustering(bad_labels, sample_documents)
        assert result['accuracy'] < 1.0
        assert result['n_misclassified'] > 0

    def test_result_keys(self, sample_documents):
        labels = np.array([0, 0, 1, 1])
        result = evaluate_clustering(labels, sample_documents)
        for key in ('accuracy', 'ari', 'misclassified', 'n_misclassified',
                     'cluster_labels'):
            assert key in result


class TestRunKmeans:
    @pytest.mark.skipif(
        not RAW_DATA_DIR.exists() or not list(RAW_DATA_DIR.glob('s*.txt')),
        reason="VRI data directory not available"
    )
    def test_returns_result(self):
        from data_loader import load_raw_documents
        from features import word_frequency_matrix

        docs = load_raw_documents(include_labels=['canon', 'commentary'])
        matrix, _ = word_frequency_matrix(docs, top_n=100)
        result = run_kmeans(matrix, docs)

        assert 'method' in result
        assert result['method'] == 'K-means'
        assert 'accuracy' in result
        assert 0 <= result['accuracy'] <= 1


class TestOtherClusteringMethods:
    @pytest.mark.skipif(
        not RAW_DATA_DIR.exists() or not list(RAW_DATA_DIR.glob('s*.txt')),
        reason="VRI data directory not available"
    )
    def test_spectral(self):
        from data_loader import load_raw_documents
        from features import word_frequency_matrix

        docs = load_raw_documents(include_labels=['canon', 'commentary'])
        matrix, _ = word_frequency_matrix(docs, top_n=100)
        result = run_spectral(matrix, docs)
        assert result['method'] == 'Spectral'
        assert 0 <= result['accuracy'] <= 1

    @pytest.mark.skipif(
        not RAW_DATA_DIR.exists() or not list(RAW_DATA_DIR.glob('s*.txt')),
        reason="VRI data directory not available"
    )
    def test_gmm(self):
        from data_loader import load_raw_documents
        from features import word_frequency_matrix

        docs = load_raw_documents(include_labels=['canon', 'commentary'])
        matrix, _ = word_frequency_matrix(docs, top_n=100)
        result = run_gmm(matrix, docs)
        assert result['method'] == 'GMM'
        assert 0 <= result['accuracy'] <= 1

    @pytest.mark.skipif(
        not RAW_DATA_DIR.exists() or not list(RAW_DATA_DIR.glob('s*.txt')),
        reason="VRI data directory not available"
    )
    def test_agglomerative(self):
        from data_loader import load_raw_documents
        from features import word_frequency_matrix

        docs = load_raw_documents(include_labels=['canon', 'commentary'])
        matrix, _ = word_frequency_matrix(docs, top_n=100)
        result = run_agglomerative(matrix, docs)
        assert 'Agglomerative' in result['method']
        assert 0 <= result['accuracy'] <= 1
