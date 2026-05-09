from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

from molgraphx.training.metrics import compute_metrics


@dataclass(slots=True)
class BaselineArtifacts:
    metrics: dict[str, float]
    predictions: np.ndarray
    targets: np.ndarray


class ECFPBaseline:
    def __init__(self, task_type: str, random_seed: int = 42) -> None:
        self.task_type = task_type
        self.model = (
            LogisticRegression(max_iter=3000, solver="liblinear", class_weight="balanced", C=0.5, random_state=random_seed)
            if task_type == "classification"
            else Ridge(alpha=1.0, random_state=random_seed)
        )

    def _extract_xy(self, dataset: list) -> tuple[np.ndarray, np.ndarray]:
        x = np.vstack([graph.fingerprint.numpy() for graph in dataset]).astype(np.float32)
        y = np.array([float(graph.y.item()) for graph in dataset], dtype=np.float32)
        return x, y

    def fit(self, train_dataset: list) -> None:
        x_train, y_train = self._extract_xy(train_dataset)
        self.model.fit(x_train, y_train)

    def predict(self, dataset: list) -> np.ndarray:
        x, _ = self._extract_xy(dataset)
        return self.model.predict_proba(x)[:, 1] if self.task_type == "classification" else self.model.predict(x)

    def evaluate(self, dataset: list) -> BaselineArtifacts:
        _, y_true = self._extract_xy(dataset)
        predictions = self.predict(dataset)
        return BaselineArtifacts(metrics=compute_metrics(self.task_type, y_true, predictions), predictions=predictions, targets=y_true)
