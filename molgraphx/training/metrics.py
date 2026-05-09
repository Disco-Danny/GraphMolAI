from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import f1_score, mean_squared_error, roc_auc_score


def compute_metrics(task_type: str, y_true: np.ndarray, predictions: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.float32).reshape(-1)
    if task_type == "classification":
        return {
            "roc_auc": float(roc_auc_score(y_true, predictions)),
            "f1": float(f1_score(y_true, (predictions >= threshold).astype(np.int64), zero_division=0)),
        }
    return {"rmse": float(math.sqrt(mean_squared_error(y_true, predictions)))}


def best_f1_threshold(y_true: np.ndarray, predictions: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.float32).reshape(-1)
    thresholds = np.linspace(0.2, 0.8, 25, dtype=np.float32)
    best_threshold, best_score = 0.5, -1.0
    for threshold in thresholds:
        score = f1_score(y_true, (predictions >= threshold).astype(np.int64), zero_division=0)
        if score > best_score:
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold
