"""
knn_scratch.py
--------------
K-Nearest Neighbors (KNN) classifier implemented from scratch using
NumPy only.

MATHEMATICAL INTUITION
=======================
KNN makes NO assumption about the functional form of the relationship
between features and target (unlike linear/logistic regression, which
assume a linear decision boundary). Instead, it is a purely
"instance-based" / "memory-based" learner:

    To classify a new point x:
    1. Compute the distance from x to every point in the training set
    2. Find the K closest training points (the "neighbors")
    3. Predict the MAJORITY class among those K neighbors
       (for probability estimates: the fraction of neighbors in class 1)

There is no "training" in the traditional sense - fit() just stores the
data. All the computational work happens at prediction time, which is
KNN's key weakness at scale (it's O(n) per prediction, vs O(1) for a
linear model's dot product).

DISTANCE METRIC: Euclidean distance
-------------------------------------
    d(x, x') = sqrt( sum_i (x_i - x'_i)^2 )

This is why feature scaling is CRITICAL for KNN (more so than for
linear/logistic regression): if one feature has a much larger numeric
range than another, it will dominate the distance calculation and the
smaller-range feature becomes almost irrelevant. We standardize all
features before using KNN (see preprocessing.py).

CHOOSING K
-----------
- Small K (e.g. K=1): very sensitive to noise, can overfit (the decision
  boundary follows individual training points closely)
- Large K: smoother decision boundary, but can underfit / blur the
  boundary between classes, and is slower to compute
- K is typically chosen via cross-validation (we do this in this project)
- K is usually chosen odd for binary classification to avoid ties
"""

import numpy as np
from collections import Counter


class KNNClassifierScratch:
    def __init__(self, k: int = 5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        # KNN is "lazy" - fitting just means storing the data
        self.X_train = X
        self.y_train = y
        return self

    def _euclidean_distances(self, x: np.ndarray) -> np.ndarray:
        """Vectorized Euclidean distance from a single point x to all training points."""
        return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns P(class=1) for each row, based on the fraction of the
        k nearest neighbors that belong to class 1."""
        probs = np.zeros(len(X))
        for i, x in enumerate(X):
            distances = self._euclidean_distances(x)
            k_nearest_idx = np.argsort(distances)[: self.k]
            k_nearest_labels = self.y_train[k_nearest_idx]
            probs[i] = np.mean(k_nearest_labels)
        return probs

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


def cross_validate_k(X: np.ndarray, y: np.ndarray, k_values, n_folds: int = 5, random_state: int = 42):
    """
    Simple from-scratch k-fold cross-validation to select the best K.
    Returns a dict {k: mean_cv_accuracy}.
    """
    rng = np.random.RandomState(random_state)
    n = len(X)
    indices = rng.permutation(n)
    fold_size = n // n_folds
    folds = [indices[i * fold_size: (i + 1) * fold_size] for i in range(n_folds)]
    # put any remainder into the last fold
    remainder = indices[n_folds * fold_size:]
    if len(remainder) > 0:
        folds[-1] = np.concatenate([folds[-1], remainder])

    results = {}
    for k in k_values:
        accuracies = []
        for fold_idx in range(n_folds):
            test_idx = folds[fold_idx]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fold_idx])

            model = KNNClassifierScratch(k=k)
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[test_idx])
            acc = np.mean(preds == y[test_idx])
            accuracies.append(acc)

        results[k] = np.mean(accuracies)

    return results


if __name__ == "__main__":
    # smoke test: two well-separated 2D Gaussian blobs - KNN should
    # achieve near-perfect accuracy
    rng = np.random.RandomState(0)
    class0 = rng.randn(100, 2) + np.array([0, 0])
    class1 = rng.randn(100, 2) + np.array([5, 5])
    X = np.vstack([class0, class1])
    y = np.array([0] * 100 + [1] * 100)

    model = KNNClassifierScratch(k=5)
    model.fit(X, y)
    preds = model.predict(X)
    acc = np.mean(preds == y)
    print(f"Smoke test accuracy on well-separated blobs (k=5): {acc:.3f}")

    cv_results = cross_validate_k(X, y, k_values=[1, 3, 5, 7, 9])
    print("Cross-validation results:", cv_results)
