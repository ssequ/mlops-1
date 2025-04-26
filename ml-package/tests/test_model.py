import pytest
import numpy as np
from sklearn.datasets import make_classification

from ml_package.model import NaiveBinaryClassifier


def _make_dataset(n_major: int, n_minor: int, random_state: int | None = None):
    X_major, _ = make_classification(
        n_samples=n_major,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        n_clusters_per_class=1,
        weights=[1.0],
        flip_y=0.0,
        random_state=random_state,
    )
    X_minor, _ = make_classification(
        n_samples=n_minor,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        n_clusters_per_class=1,
        weights=[1.0],
        flip_y=0.0,
        random_state=random_state + 1 if random_state is not None else None,
    )
    X = np.vstack([X_major, X_minor])
    y = np.concatenate([
        np.zeros(n_major, dtype=int),
        np.ones(n_minor, dtype=int),
    ])
    return X, y


def test_predicts_majority_class():
    X, y = _make_dataset(n_major=30, n_minor=10, random_state=42)
    clf = NaiveBinaryClassifier().fit(X, y)

    X_new = np.random.randn(5, X.shape[1])
    preds = clf.predict(X_new)

    assert np.all(preds == 0), "All predictions should be 0 (majority class)"


def test_switches_with_different_majority():
    X, y = _make_dataset(n_major=5, n_minor=15, random_state=0)
    clf = NaiveBinaryClassifier().fit(X, y)

    X_new = np.random.randn(3, X.shape[1])
    preds = clf.predict(X_new)

    assert np.all(preds == 1), "All predictions should be 1 (new majority class)"


def test_predict_proba_shape_and_values():
    X, y = _make_dataset(n_major=12, n_minor=8, random_state=1)
    clf = NaiveBinaryClassifier().fit(X, y)

    X_new = np.random.randn(4, X.shape[1])
    probs = clf.predict_proba(X_new)

    assert probs.shape == (4, 2)
    majority_index = int(clf.most_common_ == clf.classes_[1])
    minority_index = 1 - majority_index
    assert np.allclose(probs[:, majority_index], clf.majority_proba_)
    assert np.allclose(probs[:, minority_index], 1 - clf.majority_proba_)


def test_raises_if_not_fitted():
    clf = NaiveBinaryClassifier()
    with pytest.raises(Exception):
        clf.predict(np.zeros((2, 3)))

