"""
Supervised classification methods for Pāli text classification.

Logistic regression, SVM, random forest, gradient boosting.
Evaluated with leave-one-out cross-validation.
"""

from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_loader import Document, binary_labels


def get_classifiers() -> dict[str, tuple]:
    """Return dict of classifier name -> (classifier, needs_scaling)."""
    return {
        'Logistic Regression (L1)': (
            LogisticRegression(solver='liblinear', C=1.0, penalty='l1',
                               random_state=42, max_iter=1000),
            True
        ),
        'Logistic Regression (L2)': (
            LogisticRegression(solver='lbfgs', C=1.0,
                               random_state=42, max_iter=1000),
            True
        ),
        'SVM (linear)': (
            SVC(kernel='linear', C=1.0, random_state=42),
            True
        ),
        'SVM (RBF)': (
            SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
            True
        ),
        'Random Forest': (
            RandomForestClassifier(n_estimators=100, random_state=42),
            False
        ),
        'Gradient Boosting': (
            GradientBoostingClassifier(n_estimators=100, random_state=42,
                                        max_depth=3),
            False
        ),
    }


def run_loo_cv(feature_matrix: np.ndarray,
               documents: list[Document],
               classifier_name: Optional[str] = None,
               cv=None) -> list[dict]:
    """Run cross-validation for all (or one) classifier(s).

    Args:
        feature_matrix: (n_docs, n_features) array.
        documents: List of Document objects.
        classifier_name: If provided, only run this classifier.
        cv: Cross-validation splitter. Defaults to LeaveOneOut().
            Use StratifiedKFold(10) for faster results on large feature sets.

    Returns:
        List of result dicts with accuracy, precision, recall, F1,
        and misclassified file_ids.
    """
    true_labels = np.array(binary_labels(documents))
    file_ids = [d.file_id for d in documents]
    if cv is None:
        cv = LeaveOneOut()

    classifiers = get_classifiers()
    if classifier_name:
        classifiers = {classifier_name: classifiers[classifier_name]}

    results = []
    for name, (clf, needs_scaling) in classifiers.items():
        if needs_scaling:
            estimator = make_pipeline(StandardScaler(), clf)
        else:
            estimator = clf

        predictions = cross_val_predict(estimator, feature_matrix, true_labels, cv=cv)

        misclassified = [file_ids[i] for i in range(len(documents))
                         if predictions[i] != true_labels[i]]

        result = {
            'method': name,
            'accuracy': accuracy_score(true_labels, predictions),
            'precision': precision_score(true_labels, predictions, zero_division=0),
            'recall': recall_score(true_labels, predictions, zero_division=0),
            'f1': f1_score(true_labels, predictions, zero_division=0),
            'misclassified': misclassified,
            'n_misclassified': len(misclassified),
            'predictions': predictions,
        }
        results.append(result)

    return results


def get_feature_importance(feature_matrix: np.ndarray,
                           documents: list[Document],
                           feature_names: list[str],
                           top_n: int = 30) -> list[tuple[str, float]]:
    """Get feature importance from a trained random forest.

    Trains on all data (not LOO) since we want overall feature importance.

    Returns:
        List of (feature_name, importance) tuples, sorted by importance.
    """
    true_labels = np.array(binary_labels(documents))

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(feature_matrix, true_labels)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    return [(feature_names[i], importances[i]) for i in indices]


def get_logistic_coefficients(feature_matrix: np.ndarray,
                               documents: list[Document],
                               feature_names: list[str],
                               top_n: int = 30) -> list[tuple[str, float]]:
    """Get top features by logistic regression coefficient magnitude.

    Positive coef = predictive of class 1 (non-canon).
    Negative coef = predictive of class 0 (canon).
    """
    true_labels = np.array(binary_labels(documents))

    scaler = StandardScaler()
    X = scaler.fit_transform(feature_matrix)

    lr = LogisticRegression(solver='liblinear', C=1.0, penalty='l1',
                             random_state=42, max_iter=1000)
    lr.fit(X, true_labels)

    coefs = lr.coef_[0]
    indices = np.argsort(np.abs(coefs))[::-1][:top_n]

    return [(feature_names[i], coefs[i]) for i in indices]
