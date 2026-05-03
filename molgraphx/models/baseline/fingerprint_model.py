from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

from molgraphx.features.fingerprint import batch_smiles_to_ecfp
from molgraphx.training.metrics import compute_metrics


@dataclass(slots=True)
class BaselineArtifacts:
    metrics: dict[str, float]
    predictions: np.ndarray
    targets: np.ndarray


class FingerprintBaselineModel:
    """ECFP-based classical baseline for molecular property prediction."""

    def __init__(self, task_type: str, radius: int = 2, n_bits: int = 2048, random_seed: int = 42) -> None:
        self.task_type = task_type
        self.radius = radius
        self.n_bits = n_bits
        self.random_seed = random_seed
        if task_type == "classification":
            self.model = LogisticRegression(
                max_iter=2000,
                solver="liblinear",
                class_weight="balanced",
                random_state=random_seed,
            )
        else:
            self.model = Ridge(alpha=1.0, random_state=random_seed)

    def _extract_xy(self, dataset: list) -> tuple[np.ndarray, np.ndarray]:
        smiles = [graph.smiles for graph in dataset]
        x = batch_smiles_to_ecfp(smiles, radius=self.radius, n_bits=self.n_bits)
        y = np.array([float(graph.y.item()) for graph in dataset], dtype=np.float32)
        return x, y

    def fit(self, train_dataset: list) -> None:
        x_train, y_train = self._extract_xy(train_dataset)
        self.model.fit(x_train, y_train)

    def predict(self, dataset: list) -> np.ndarray:
        x, _ = self._extract_xy(dataset)
        if self.task_type == "classification":
            return self.model.predict_proba(x)[:, 1]
        return self.model.predict(x)

    def evaluate(self, dataset: list) -> BaselineArtifacts:
        _, y_true = self._extract_xy(dataset)
        predictions = self.predict(dataset)
        metrics = compute_metrics(self.task_type, y_true, predictions)
        return BaselineArtifacts(metrics=metrics, predictions=predictions, targets=y_true)
