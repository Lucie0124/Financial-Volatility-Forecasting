"""Reusable classification evaluation logic for the regime_classification notebook"""

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score 

CLASS_LABELS = ["Normal", "Elevated", "Stress"]


def compute_classification_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0),
        "stress_precision": precision_score(y_true, y_pred, labels=["Stress"], average=None, zero_division=0)[0],
        "stress_recall": recall_score(y_true, y_pred, labels=["Stress"], average=None, zero_division=0)[0],
        "stress_f1": f1_score(y_true, y_pred, labels=["Stress"], average=None, zero_division=0)[0],
    }