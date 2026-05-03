from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch_geometric.data import Batch

from molgraphx.training.metrics import compute_metrics, format_metrics


@dataclass(slots=True)
class TrainingArtifacts:
    history: dict[str, list[float]]
    best_epoch: int
    best_val_loss: float
    metrics: dict[str, float]
    predictions: np.ndarray
    targets: np.ndarray


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        task_type: str,
        device: torch.device,
        learning_rate: float,
        weight_decay: float,
        max_epochs: int,
        patience: int,
        checkpoint_path: Path,
        standardize_targets: bool = False,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        scheduler_patience: int = 5,
        grad_clip_norm: float | None = 5.0,
    ) -> None:
        self.model = model
        self.task_type = task_type
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.standardize_targets = standardize_targets and task_type == "regression"
        self.target_mean = float(target_mean)
        self.target_std = float(target_std) if abs(float(target_std)) > 1e-8 else 1.0
        self.grad_clip_norm = grad_clip_norm

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=scheduler_patience,
            min_lr=1e-6,
        )
        self.loss_fn = nn.BCEWithLogitsLoss() if task_type == "classification" else nn.MSELoss()

    def _move_batch(self, batch: Batch) -> Batch:
        return batch.to(self.device)

    def _targets_from_batch(self, batch: Batch) -> torch.Tensor:
        return batch.y.view(-1).float()

    def _predict_from_batch(self, batch: Batch) -> torch.Tensor:
        return self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

    def _targets_for_loss(self, targets: torch.Tensor) -> torch.Tensor:
        if not self.standardize_targets:
            return targets
        return (targets - self.target_mean) / self.target_std

    def _predictions_for_metrics(self, outputs: torch.Tensor) -> torch.Tensor:
        if self.task_type == "classification":
            return torch.sigmoid(outputs)
        if self.standardize_targets:
            return outputs * self.target_std + self.target_mean
        return outputs

    def _run_epoch(self, loader, training: bool) -> tuple[float, dict[str, float]]:
        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_examples = 0
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []

        for batch in loader:
            batch = self._move_batch(batch)
            y_true = self._targets_from_batch(batch)

            if training:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(training):
                logits = self._predict_from_batch(batch)
                loss = self.loss_fn(logits, self._targets_for_loss(y_true))
                if training:
                    loss.backward()
                    if self.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.optimizer.step()

            batch_size = y_true.size(0)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size

            y_pred = self._predictions_for_metrics(logits).detach().cpu().numpy()
            predictions.append(y_pred)
            targets.append(y_true.detach().cpu().numpy())

        concatenated_predictions = np.concatenate(predictions) if predictions else np.array([])
        concatenated_targets = np.concatenate(targets) if targets else np.array([])
        epoch_loss = total_loss / max(total_examples, 1)
        metrics = compute_metrics(self.task_type, concatenated_targets, concatenated_predictions)
        return epoch_loss, metrics

    def train(self, train_loader, val_loader) -> dict[str, list[float]]:
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0

        for epoch in range(1, self.max_epochs + 1):
            train_loss, train_metrics = self._run_epoch(train_loader, training=True)
            val_loss, val_metrics = self._run_epoch(val_loader, training=False)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} | {format_metrics(train_metrics)} | "
                f"val_loss={val_loss:.4f} | {format_metrics(val_metrics)}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.patience:
                print(f"Early stopping triggered at epoch {epoch}. Best epoch: {best_epoch}.")
                break

            self.scheduler.step(val_loss)

        if self.checkpoint_path.exists():
            self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))

        history["best_epoch"] = [float(best_epoch)]
        history["best_val_loss"] = [float(best_val_loss)]
        return history

    def evaluate(self, loader) -> TrainingArtifacts:
        loss, metrics = self._run_epoch(loader, training=False)
        predictions, targets = self.predict(loader)
        return TrainingArtifacts(
            history={},
            best_epoch=0,
            best_val_loss=loss,
            metrics=metrics,
            predictions=predictions,
            targets=targets,
        )

    def predict(self, loader) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        all_predictions: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []

        with torch.no_grad():
            for batch in loader:
                batch = self._move_batch(batch)
                y_true = self._targets_from_batch(batch)
                outputs = self._predict_from_batch(batch)
                outputs = self._predictions_for_metrics(outputs)

                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(y_true.cpu().numpy())

        predictions = np.concatenate(all_predictions) if all_predictions else np.array([])
        targets = np.concatenate(all_targets) if all_targets else np.array([])
        return predictions, targets

    @staticmethod
    def plot_history(history: dict[str, list[float]], output_path: Path) -> None:
        plt.figure(figsize=(7, 4))
        plt.plot(history["train_loss"], label="Train loss")
        plt.plot(history["val_loss"], label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Curves")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()

    @staticmethod
    def save_history(history: dict[str, list[float]], output_path: Path) -> None:
        serializable = {key: value for key, value in history.items()}
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(serializable, handle, indent=2)
