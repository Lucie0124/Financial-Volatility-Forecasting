"""Classification metrics with explicit Stress labels and denominators."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

CLASS_LABELS = ["Normal", "Elevated", "Stress"]


def compute_classification_metrics(y_true, y_pred):
    """Keep all three labels in macro-F1; absent Stress recall is undefined.

    Undefined precision/F1 use sklearn's zero convention. Stress support and
    alert counts make these conventions inspectable in exported results.
    """
    actual, predicted = np.asarray(y_true), np.asarray(y_pred)
    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("Regime labels must be one-dimensional, without dates.")
    if len(actual) == 0 or len(actual) != len(predicted):
        raise ValueError("Actual and predicted labels must have equal non-zero length.")
    if not set(actual).union(predicted).issubset(CLASS_LABELS):
        raise ValueError("Regime labels must be Normal, Elevated or Stress.")
    stress_support = int((actual == "Stress").sum())
    return {
        "accuracy": accuracy_score(actual, predicted),
        "macro_f1": f1_score(
            actual, predicted, labels=CLASS_LABELS, average="macro", zero_division=0
        ),
        "stress_precision": float(precision_score(
            actual, predicted, labels=["Stress"], average=None, zero_division=0
        )[0]),
        "stress_recall": float(recall_score(
            actual, predicted, labels=["Stress"], average=None, zero_division=0
        )[0]) if stress_support else float("nan"),
        "stress_f1": float(f1_score(
            actual, predicted, labels=["Stress"], average=None, zero_division=0
        )[0]),
        "stress_support": stress_support,
        "predicted_stress_count": int((predicted == "Stress").sum()),
    }


def compute_stress_probability_metrics(y_true, probability):
    """Return ROC-AUC and average precision, not trapezoidal PR-AUC.

    ROC-AUC is undefined without both outcomes. Average precision is undefined
    when there are no positive observations. Export these cases as missing.
    """
    actual = np.asarray(y_true)
    scores = np.asarray(probability, dtype=float)
    if actual.ndim != 1 or scores.ndim != 1 or len(actual) != len(scores):
        raise ValueError("Stress scores must align with one-dimensional labels.")
    if len(actual) == 0 or not np.isfinite(scores).all():
        raise ValueError("Stress scores must be non-empty and finite.")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Stress probabilities must lie between zero and one.")
    stress = actual == "Stress"
    return {
        "stress_roc_auc": roc_auc_score(stress, scores)
        if stress.any() and (~stress).any() else float("nan"),
        "stress_average_precision": average_precision_score(stress, scores)
        if stress.any() else float("nan"),
    }
