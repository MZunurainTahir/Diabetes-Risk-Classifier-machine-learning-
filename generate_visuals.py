"""
generate_visuals.py
--------------------
Generates all charts: EDA plots (including missing-value impact),
correlation heatmap, the K-selection curve (with overfitting
demonstration), confusion matrices, ROC curves, and a 3D KNN
neighborhood visualization.

All figures are saved to reports/figures/.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.append(os.path.dirname(__file__))

from preprocessing import load_data, clean_missing_values, get_train_test_split, ZERO_AS_MISSING_COLS
from knn_scratch import KNNClassifierScratch, cross_validate_k
from evaluate import confusion_matrix_scratch, roc_auc_scratch

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120


def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_target_distribution(df):
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["Outcome"].value_counts().sort_index()
    labels = ["No Diabetes (0)", "Diabetes (1)"]
    ax.bar(labels, counts.values, color=["#4C72B0", "#C44E52"])
    ax.set_title("Target Class Distribution")
    ax.set_ylabel("Number of Patients")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 5, str(v), ha="center", fontweight="bold")
    save(fig, "01_target_distribution.png")


def plot_missing_value_impact(df):
    """Shows the % of biologically-impossible zeros per column before cleaning."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    pcts = [100 * (df[c] == 0).sum() / len(df) for c in ZERO_AS_MISSING_COLS]
    bars = ax.bar(ZERO_AS_MISSING_COLS, pcts, color="#C44E52")
    ax.set_ylabel("% of rows with value = 0 (encoded missing)")
    ax.set_title("Encoded Missing Values by Column\n(biologically impossible zeros)")
    ax.set_xticklabels(ZERO_AS_MISSING_COLS, rotation=20, ha="right")
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width()/2, pct + 1, f"{pct:.1f}%", ha="center", fontweight="bold")
    save(fig, "02_missing_value_impact.png")


def plot_correlation_heatmap(df_clean):
    fig, ax = plt.subplots(figsize=(8, 7))
    corr = df_clean.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, square=True)
    ax.set_title("Feature Correlation Heatmap (after cleaning missing values)")
    save(fig, "03_correlation_heatmap.png")


def plot_glucose_bmi_by_outcome(df_clean):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.histplot(data=df_clean, x="Glucose", hue="Outcome", kde=True, ax=axes[0], palette=["#4C72B0", "#C44E52"])
    axes[0].set_title("Glucose Distribution by Outcome")
    sns.histplot(data=df_clean, x="BMI", hue="Outcome", kde=True, ax=axes[1], palette=["#4C72B0", "#C44E52"])
    axes[1].set_title("BMI Distribution by Outcome")
    fig.tight_layout()
    save(fig, "04_glucose_bmi_distribution.png")


def plot_k_selection_curve(cv_results, X_train, y_train, X_test, y_test):
    """
    Shows CV accuracy across K (for selecting K) AND the train-vs-test
    accuracy gap across K (to visually demonstrate overfitting at low K).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ks = list(cv_results.keys())
    accs = list(cv_results.values())
    axes[0].plot(ks, accs, marker="o", color="#4C72B0")
    best_k = max(cv_results, key=cv_results.get)
    axes[0].axvline(best_k, color="#C44E52", linestyle="--", alpha=0.7, label=f"Best K={best_k}")
    axes[0].set_xlabel("K (number of neighbors)")
    axes[0].set_ylabel("Mean 5-fold CV Accuracy")
    axes[0].set_title("Cross-Validation: Selecting K")
    axes[0].legend()

    k_range = [1, 3, 5, 7, 9, 11, 13, 15, 19, 25, 35, 51]
    train_accs, test_accs = [], []
    for k in k_range:
        m = KNNClassifierScratch(k=k)
        m.fit(X_train, y_train)
        train_accs.append(np.mean(m.predict(X_train) == y_train))
        test_accs.append(np.mean(m.predict(X_test) == y_test))

    axes[1].plot(k_range, train_accs, marker="o", label="Train accuracy", color="#55A868")
    axes[1].plot(k_range, test_accs, marker="o", label="Test accuracy", color="#C44E52")
    axes[1].set_xlabel("K (number of neighbors)")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Overfitting at Low K: Train vs Test Accuracy")
    axes[1].legend()

    fig.tight_layout()
    save(fig, "05_k_selection_and_overfitting.png")


def plot_confusion_matrices(cm_dict):
    fig, axes = plt.subplots(1, len(cm_dict), figsize=(5 * len(cm_dict), 4.5))
    if len(cm_dict) == 1:
        axes = [axes]
    for ax, (label, cm) in zip(axes, cm_dict.items()):
        tn, fp, fn, tp = cm
        matrix = np.array([[tn, fp], [fn, tp]])
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
        ax.set_title(label, fontsize=10)
    fig.tight_layout()
    save(fig, "06_confusion_matrices.png")


def plot_roc_curves(y_test, scores_dict):
    from sklearn.metrics import roc_curve, auc
    fig, ax = plt.subplots(figsize=(6, 6))
    for label, scores in scores_dict.items():
        fpr, tpr, _ = roc_curve(y_test, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves: From-Scratch vs sklearn KNN")
    ax.legend(fontsize=9)
    save(fig, "07_roc_curves.png")


def plot_3d_knn_neighborhood(X_train, y_train, feature_names, best_k):
    """
    3D visualization of KNN's core idea using 3 clinically relevant
    features: Glucose, BMI, Age. Highlights one query point and its K
    nearest neighbors to make the "vote among neighbors" mechanism visible.
    """
    idx_glucose = feature_names.index("Glucose")
    idx_bmi = feature_names.index("BMI")
    idx_age = feature_names.index("Age")

    X_vis = X_train[:, [idx_glucose, idx_bmi, idx_age]]

    rng = np.random.RandomState(7)
    query_idx = rng.choice(len(X_vis))
    query_point = X_vis[query_idx]

    distances = np.sqrt(np.sum((X_vis - query_point) ** 2, axis=1))
    k_nearest_idx = np.argsort(distances)[: best_k + 1]  # includes itself at distance 0
    k_nearest_idx = k_nearest_idx[k_nearest_idx != query_idx][:best_k]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # all points, colored by class, faded
    ax.scatter(X_vis[:, 0], X_vis[:, 1], X_vis[:, 2], c=y_train, cmap="coolwarm",
               s=20, alpha=0.25, edgecolor="none")

    # highlight the k nearest neighbors
    ax.scatter(X_vis[k_nearest_idx, 0], X_vis[k_nearest_idx, 1], X_vis[k_nearest_idx, 2],
               c=y_train[k_nearest_idx], cmap="coolwarm", s=90, edgecolor="black", linewidth=1.2,
               label=f"{best_k} Nearest Neighbors")

    # highlight the query point
    ax.scatter(*query_point, c="lime", s=250, marker="*", edgecolor="black", linewidth=1.5,
               label="Query point")

    ax.set_xlabel("Glucose (standardized)")
    ax.set_ylabel("BMI (standardized)")
    ax.set_zlabel("Age (standardized)")
    ax.set_title(f"KNN Mechanism: Query Point and its {best_k} Nearest Neighbors")
    ax.legend(loc="upper left", fontsize=8)
    save(fig, "08_3d_knn_neighborhood.png")


if __name__ == "__main__":
    df = load_data()
    df_clean_full = clean_missing_values(df)

    print("Generating EDA plots...")
    plot_target_distribution(df)
    plot_missing_value_impact(df)
    plot_correlation_heatmap(df_clean_full.fillna(df_clean_full.median()))
    plot_glucose_bmi_by_outcome(df_clean_full.fillna(df_clean_full.median()))

    print("\nRunning models for result-based plots...")
    from train_and_compare import run
    results_df, cv_results, best_k, X_train, X_test, y_train, y_test, feature_names, df2 = run()

    plot_k_selection_curve(cv_results, X_train, y_train, X_test, y_test)

    from sklearn.neighbors import KNeighborsClassifier as SklearnKNN

    m = KNNClassifierScratch(k=best_k)
    m.fit(X_train, y_train)
    preds = m.predict(X_test)
    scores = m.predict_proba(X_test)
    tp, tn, fp, fn = confusion_matrix_scratch(y_test, preds)

    sk_model = SklearnKNN(n_neighbors=best_k)
    sk_model.fit(X_train, y_train)
    sk_preds = sk_model.predict(X_test)
    sk_scores = sk_model.predict_proba(X_test)[:, 1]
    sk_tp, sk_tn, sk_fp, sk_fn = confusion_matrix_scratch(y_test, sk_preds)

    cm_dict = {
        f"KNN (k={best_k}, scratch)": (tn, fp, fn, tp),
        f"KNN (k={best_k}, sklearn)": (sk_tn, sk_fp, sk_fn, sk_tp),
    }
    scores_dict = {
        f"Scratch (k={best_k})": scores,
        f"sklearn (k={best_k})": sk_scores,
    }

    plot_confusion_matrices(cm_dict)
    plot_roc_curves(y_test, scores_dict)
    plot_3d_knn_neighborhood(X_train, y_train, feature_names, best_k)

    print("\nAll figures generated successfully.")
