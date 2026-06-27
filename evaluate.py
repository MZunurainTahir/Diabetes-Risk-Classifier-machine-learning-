"""
evaluate.py
-----------
Evaluation metrics for binary classification, implemented from scratch
(no sklearn.metrics), validated against sklearn for correctness.
"""

import numpy as np


def confusion_matrix_scratch(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp, tn, fp, fn


def accuracy_scratch(y_true, y_pred) -> float:
    return np.mean(y_true == y_pred)


def precision_scratch(y_true, y_pred) -> float:
    tp, tn, fp, fn = confusion_matrix_scratch(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall_scratch(y_true, y_pred) -> float:
    tp, tn, fp, fn = confusion_matrix_scratch(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_scratch(y_true, y_pred) -> float:
    p = precision_scratch(y_true, y_pred)
    r = recall_scratch(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def roc_auc_scratch(y_true, y_scores) -> float:
    pos_scores = y_scores[y_true == 1]
    neg_scores = y_scores[y_true == 0]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float("nan")
    diff = pos_scores[:, None] - neg_scores[None, :]
    wins = np.sum(diff > 0) + 0.5 * np.sum(diff == 0)
    return wins / (len(pos_scores) * len(neg_scores))


def full_report(y_true, y_pred, y_scores=None, label: str = "Model") -> dict:
    tp, tn, fp, fn = confusion_matrix_scratch(y_true, y_pred)
    report = {
        "label": label,
        "accuracy": accuracy_scratch(y_true, y_pred),
        "precision": precision_scratch(y_true, y_pred),
        "recall": recall_scratch(y_true, y_pred),
        "f1_score": f1_scratch(y_true, y_pred),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }
    if y_scores is not None:
        report["roc_auc"] = roc_auc_scratch(y_true, y_scores)
    return report


def print_report(report: dict):
    print(f"--- {report['label']} ---")
    print(f"  Accuracy : {report['accuracy']:.4f}")
    print(f"  Precision: {report['precision']:.4f}")
    print(f"  Recall   : {report['recall']:.4f}")
    print(f"  F1 Score : {report['f1_score']:.4f}")
    if "roc_auc" in report:
        print(f"  ROC-AUC  : {report['roc_auc']:.4f}")
    print(f"  Confusion: TP={report['tp']} TN={report['tn']} FP={report['fp']} FN={report['fn']}")
