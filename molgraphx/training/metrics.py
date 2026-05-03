from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def compute_metrics(task_type: str, y_true: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.float32).reshape(-1)

    if task_type == "classification":
        binary_predictions = (predictions >= 0.5).astype(np.int64)
        try:
            roc_auc = float(roc_auc_score(y_true, predictions))
        except ValueError:
            roc_auc = float("nan")
        metrics = {
            "roc_auc": roc_auc,
            "accuracy": float(accuracy_score(y_true, binary_predictions)),
        }
    else:
        metrics = {
            "rmse": rmse(y_true, predictions),
            "mae": float(mean_absolute_error(y_true, predictions)),
        }
    return metrics


def format_metrics(metrics: dict[str, float]) -> str:
    return ", ".join(f"{name}={value:.4f}" for name, value in metrics.items())
