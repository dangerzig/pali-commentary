"""Tests for src/classification.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from classification import get_classifiers, run_loo_cv, get_feature_importance
from data_loader import binary_labels, RAW_DATA_DIR


class TestGetClassifiers:
    def test_returns_dict(self):
        classifiers = get_classifiers()
        assert isinstance(classifiers, dict)
        expected_keys = {
            'Logistic Regression (L1)',
            'Logistic Regression (L2)',
            'SVM (linear)',
            'SVM (RBF)',
            'Random Forest',
            'Gradient Boosting',
        }
        assert set(classifiers.keys()) == expected_keys

    def test_needs_scaling(self):
        classifiers = get_classifiers()
        # LR and SVM need scaling
        assert classifiers['Logistic Regression (L1)'][1] is True
        assert classifiers['Logistic Regression (L2)'][1] is True
        assert classifiers['SVM (linear)'][1] is True
        assert classifiers['SVM (RBF)'][1] is True
        # Tree-based methods don't
        assert classifiers['Random Forest'][1] is False
        assert classifiers['Gradient Boosting'][1] is False

    def test_classifier_objects(self):
        classifiers = get_classifiers()
        for name, (clf, _) in classifiers.items():
            assert hasattr(clf, 'fit')
            assert hasattr(clf, 'predict')


class TestRunLooCv:
    @pytest.mark.skipif(
        not RAW_DATA_DIR.exists() or not list(RAW_DATA_DIR.glob('s*.txt')),
        reason="VRI data directory not available"
    )
    def test_returns_metrics(self):
        from data_loader import load_raw_documents
        from features import word_frequency_matrix

        docs = load_raw_documents(include_labels=['canon', 'commentary'])
        matrix, _ = word_frequency_matrix(docs, top_n=100)
        results = run_loo_cv(matrix, docs, classifier_name='Random Forest')

        assert len(results) == 1
        r = results[0]
        for key in ('method', 'accuracy', 'precision', 'recall', 'f1',
                     'misclassified', 'predictions'):
            assert key in r
        assert 0 <= r['accuracy'] <= 1
        assert r['method'] == 'Random Forest'

    @pytest.mark.skipif(
        not RAW_DATA_DIR.exists() or not list(RAW_DATA_DIR.glob('s*.txt')),
        reason="VRI data directory not available"
    )
    def test_single_classifier_filter(self):
        from data_loader import load_raw_documents
        from features import word_frequency_matrix

        docs = load_raw_documents(include_labels=['canon', 'commentary'])
        matrix, _ = word_frequency_matrix(docs, top_n=100)
        results = run_loo_cv(matrix, docs, classifier_name='Random Forest')
        assert len(results) == 1
        assert results[0]['method'] == 'Random Forest'


class TestGetFeatureImportance:
    @pytest.mark.skipif(
        not RAW_DATA_DIR.exists() or not list(RAW_DATA_DIR.glob('s*.txt')),
        reason="VRI data directory not available"
    )
    def test_returns_tuples(self):
        from data_loader import load_raw_documents
        from features import word_frequency_matrix

        docs = load_raw_documents(include_labels=['canon', 'commentary'])
        matrix, names = word_frequency_matrix(docs, top_n=100)
        result = get_feature_importance(matrix, docs, names, top_n=10)

        assert len(result) == 10
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)
