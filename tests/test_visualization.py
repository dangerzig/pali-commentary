"""Tests for src/visualization.py."""

import sys
import os
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from visualization import RESULTS_DIR, plot_pca, plot_dendrogram, plot_confusion_comparison


class TestResultsDir:
    def test_exists(self):
        assert RESULTS_DIR.exists()
        assert RESULTS_DIR.is_dir()


class TestPlotPca:
    def test_creates_file(self, sample_feature_matrix, tmp_path):
        # Temporarily override RESULTS_DIR via filename path
        outfile = 'test_pca_output.png'
        labels = ['canon', 'canon', 'commentary', 'commentary']
        file_ids = ['s0101m', 's0201m', 's0101a', 's0201a']

        path = plot_pca(
            sample_feature_matrix, labels, file_ids,
            title='Test PCA', filename=outfile,
        )
        assert Path(path).exists()
        # Clean up
        os.remove(path)

    def test_with_misclassified(self, sample_feature_matrix, tmp_path):
        outfile = 'test_pca_misc.png'
        labels = ['canon', 'canon', 'commentary', 'commentary']
        file_ids = ['s0101m', 's0201m', 's0101a', 's0201a']

        path = plot_pca(
            sample_feature_matrix, labels, file_ids,
            title='Test PCA', filename=outfile,
            misclassified=['s0201m'],
        )
        assert Path(path).exists()
        os.remove(path)


class TestPlotDendrogram:
    def test_creates_file(self, sample_feature_matrix):
        from features import euclidean_distance_matrix
        dist = euclidean_distance_matrix(sample_feature_matrix)
        labels = ['canon', 'canon', 'commentary', 'commentary']
        file_ids = ['s0101m', 's0201m', 's0101a', 's0201a']

        outfile = 'test_dendrogram.png'
        path = plot_dendrogram(
            dist, labels, file_ids,
            title='Test Dendrogram', filename=outfile,
        )
        assert Path(path).exists()
        os.remove(path)


class TestPlotConfusionComparison:
    def test_creates_file(self):
        outfile = 'test_comparison.png'
        results = {'K-means': 0.95, 'SVM': 0.98, 'RF': 0.97}

        path = plot_confusion_comparison(
            results, title='Test', filename=outfile,
        )
        assert Path(path).exists()
        os.remove(path)
