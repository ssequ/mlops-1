from collections import Counter

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin # type: ignore[import-untyped]
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted # type: ignore[import-untyped]
from sklearn.utils.multiclass import unique_labels # type: ignore[import-untyped]


class NaiveBinaryClassifier(BaseEstimator, ClassifierMixin):
    """A trivial binary classifier that predicts the majority class.

    The classifier determines the most frequent label in *y* during
    :py:meth:`fit` and always returns that label in :py:meth:`predict`.
    Probability estimates returned by :py:meth:`predict_proba` are
    constant for all samples and equal to the observed class
    frequencies.
    """

    def fit(self, X: ArrayLike, y: ArrayLike) -> "NaiveBinaryClassifier":
        """Fit the classifier.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training vectors (ignored apart from basic validation).
        y : ArrayLike of shape (n_samples,)
            Target labels. Exactly **two** distinct labels are required.

        Returns
        -------
        self
            Fitted estimator.
        """
        X, y = check_X_y(X, y, y_numeric="auto")

        self.classes_ = unique_labels(y)
        if self.classes_.size != 2:
            raise ValueError(
                "NaiveBinaryClassifier supports exactly two distinct labels; "
                f"got {self.classes_.size}."
            )

        counts = Counter(y) # type: ignore[arg-type]
        self.most_common_, majority_count = counts.most_common(1)[0]
        self.majority_proba_ = majority_count / len(y) # type: ignore[arg-type]

        return self

    
    def predict(self, X: ArrayLike) -> np.ndarray:  
        """Return constant predictions – the majority label.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        ndarray of shape (n_samples,)
            All‑constant label predictions.
        """
        check_is_fitted(self, ["classes_", "most_common_", "majority_proba_"])
        X = check_array(X, accept_sparse=True)

        return np.full(shape=(X.shape[0],), fill_value=self.most_common_, dtype=object) # type: ignore[union-attr]

    def predict_proba(self, X: ArrayLike) -> np.ndarray:  
        """Return constant probability estimates.

        The probability for the majority class is *majority_proba_*; for the
        minority class it is *1 − majority_proba_*.
        """
        check_is_fitted(self, ["classes_", "most_common_", "majority_proba_"])
        X = check_array(X, accept_sparse=True)

        n_samples = X.shape[0] # type: ignore[union-attr]
        probs = np.empty((n_samples, 2), dtype=float)

        majority_index = int(self.most_common_ == self.classes_[1])
        minority_index = 1 - majority_index

        probs[:, majority_index] = self.majority_proba_
        probs[:, minority_index] = 1.0 - self.majority_proba_
        return probs

