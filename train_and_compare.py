"""
train_and_compare.py
---------------------
Main experiment script: selects the best K via from-scratch cross-
validation, trains the from-scratch KNN and sklearn's KNeighborsClassifier
on the same train/test split, and reports a fair comparison.
"""

import sys
import os
import time
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(__file__))

from preprocessing import load_data, get_train_test_split, check_data_quality
from knn_scratch import KNNClassifierScratch, cross_validate_k
from evaluate import full_report, print_report

from sklearn.neighbors import KNeighborsClassifier as SklearnKNN


def run():
    df = load_data()
    print("Data quality check:")
    for k, v in check_data_quality(df).items():
        print(f"  {k}: {v}")
    print()

    X_train, X_test, y_train, y_test, feature_names, scaler, medians = get_train_test_split(df)

    # ---- select K via from-scratch cross-validation ----
    print("Running 5-fold cross-validation to select K...")
    k_candidates = [1, 3, 5, 7, 9, 11, 13, 15, 19, 25]
    cv_results = cross_validate_k(X_train, y_train, k_candidates, n_folds=5)
    for k, acc in cv_results.items():
        print(f"  k={k:2d} | mean CV accuracy = {acc:.4f}")
    best_k = max(cv_results, key=cv_results.get)
    print(f"\nBest K selected: {best_k} (CV accuracy = {cv_results[best_k]:.4f})\n")

    results = []

    # ---- from-scratch KNN with best K ----
    model = KNNClassifierScratch(k=best_k)
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)
    predict_time = time.time() - t0

    report = full_report(y_test, y_pred, y_scores, label=f"KNN (k={best_k}, from scratch)")
    report["train_time_sec"] = train_time
    report["predict_time_sec"] = predict_time
    print_report(report)
    print(f"  Train time: {train_time:.5f}s | Predict time: {predict_time:.4f}s\n")
    results.append(report)

    # ---- sklearn baseline (same K) ----
    sk_model = SklearnKNN(n_neighbors=best_k)
    t0 = time.time()
    sk_model.fit(X_train, y_train)
    sk_train_time = time.time() - t0

    t0 = time.time()
    y_pred_sk = sk_model.predict(X_test)
    y_scores_sk = sk_model.predict_proba(X_test)[:, 1]
    sk_predict_time = time.time() - t0

    sk_report = full_report(y_test, y_pred_sk, y_scores_sk, label=f"KNN (k={best_k}, sklearn)")
    sk_report["train_time_sec"] = sk_train_time
    sk_report["predict_time_sec"] = sk_predict_time
    print_report(sk_report)
    print(f"  Train time: {sk_train_time:.5f}s | Predict time: {sk_predict_time:.4f}s\n")
    results.append(sk_report)

    # ---- summary table ----
    results_df = pd.DataFrame(results)[
        ["label", "accuracy", "precision", "recall", "f1_score", "roc_auc", "train_time_sec", "predict_time_sec"]
    ]
    print("\n=== SUMMARY TABLE ===")
    print(results_df.to_string(index=False))

    results_df.to_csv(
        os.path.join(os.path.dirname(__file__), "..", "reports", "model_comparison_results.csv"),
        index=False,
    )

    cv_df = pd.DataFrame(list(cv_results.items()), columns=["k", "cv_accuracy"])
    cv_df.to_csv(
        os.path.join(os.path.dirname(__file__), "..", "reports", "cv_k_selection_results.csv"),
        index=False,
    )

    return results_df, cv_results, best_k, X_train, X_test, y_train, y_test, feature_names, df


if __name__ == "__main__":
    run()
