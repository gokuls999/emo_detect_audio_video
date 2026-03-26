from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, hamming_loss, mean_absolute_error, precision_recall_fscore_support


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"precision_weighted": float(precision), "recall_weighted": float(recall), "f1_weighted": float(f1), "f1_macro": float(macro_f1)}


def multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "mAP": float(average_precision_score(y_true, y_prob, average="macro")),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "subset_accuracy": float((y_true == y_pred).all(axis=1).mean()),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else 0.0
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"pearson_r": pearson, "mae": mae}


def affect_instability_index(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.abs(np.diff(values)).mean())
