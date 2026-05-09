from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast

from molgraphx.training.metrics import best_f1_threshold, compute_metrics


@dataclass(slots=True)
class TrainArtifacts:
    history: dict[str, list[float]]
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
        target_mean: float = 0.0,
        target_std: float = 1.0,
    ) -> None:
        self.model = model.to(device)
        self.task_type = task_type
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.target_mean = float(target_mean)
        self.target_std = float(target_std) if abs(float(target_std)) > 1e-8 else 1.0
        self.use_amp = device.type == "cuda"
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-6)
        self.loss_fn = nn.MSELoss() if task_type == "regression" else None
        self.scaler = GradScaler(enabled=self.use_amp)
        self.metric_name = "roc_auc" if task_type == "classification" else "rmse"
        self.metric_sign = 1.0 if task_type == "classification" else -1.0
        self.threshold = 0.5

    def _loss_targets(self, y: torch.Tensor) -> torch.Tensor:
        return y if self.task_type == "classification" else (y - self.target_mean) / self.target_std

    def _metric_outputs(self, y: torch.Tensor) -> torch.Tensor:
        if self.task_type == "classification":
            return torch.sigmoid(y)
        return y * self.target_std + self.target_mean

    def _classification_loss(self, logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.65, gamma: float = 1.5) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, prob, 1.0 - prob)
        weight = torch.where(targets > 0.5, torch.full_like(targets, alpha), torch.full_like(targets, 1.0 - alpha))
        return (weight * (1.0 - pt).pow(gamma) * bce).mean()

    def _forward(self, batch) -> torch.Tensor:
        fingerprint = batch.fingerprint
        if fingerprint.dim() == 1:
            fingerprint = fingerprint.view(batch.num_graphs, -1)
        return self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, fingerprint)

    def _run_epoch(self, loader, training: bool) -> tuple[float, dict[str, float]]:
        self.model.train(training)
        losses, preds, targets = 0.0, [], []
        total = 0
        for batch in loader:
            batch = batch.to(self.device)
            y = batch.y.view(-1).float()
            if training:
                self.optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                if self.use_amp:
                    with autocast():
                        out = self._forward(batch)
                        loss = self.loss_fn(out, self._loss_targets(y)) if self.task_type == "regression" else self._classification_loss(out, y)
                else:
                    out = self._forward(batch)
                    loss = self.loss_fn(out, self._loss_targets(y)) if self.task_type == "regression" else self._classification_loss(out, y)
                if training:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            losses += float(loss.item()) * y.size(0)
            total += y.size(0)
            preds.append(self._metric_outputs(out).detach().cpu().numpy())
            targets.append(y.detach().cpu().numpy())
        pred = np.concatenate(preds)
        true = np.concatenate(targets)
        threshold = self.threshold if not training else 0.5
        return losses / max(total, 1), compute_metrics(self.task_type, true, pred, threshold=threshold)

    def train(self, train_loader, val_loader) -> dict[str, list[float]]:
        history = {"train_loss": [], "val_loss": [], "epoch_time": [], "val_metric": []}
        best_score, bad = -float("inf"), 0
        for epoch in range(self.max_epochs):
            start = time.perf_counter()
            train_loss, _ = self._run_epoch(train_loader, True)
            val_loss, val_metrics = self._run_epoch(val_loader, False)
            epoch_time = time.perf_counter() - start
            history["epoch_time"].append(epoch_time)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_metric"].append(val_metrics[self.metric_name])
            if self.task_type == "classification":
                self.threshold = best_f1_threshold(*self.predict(val_loader))
            score = self.metric_sign * float(val_metrics[self.metric_name])
            if score > best_score:
                best_score = score
                bad = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
            else:
                bad += 1
            self.scheduler.step(val_loss)
            if (epoch + 1) % max(1, self.max_epochs // 10) == 0 or bad >= self.patience:
                metric_val = float(val_metrics[self.metric_name])
                print(f"  Epoch {epoch+1}/{self.max_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, {self.metric_name}={metric_val:.4f}, time={epoch_time:.2f}s")
            if bad >= self.patience:
                print(f"  Early stopping at epoch {epoch+1} (patience={self.patience})")
                break
        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device, weights_only=True))
        return history

    def evaluate(self, loader) -> TrainArtifacts:
        true, pred = self.predict(loader)
        return TrainArtifacts(history={}, metrics=compute_metrics(self.task_type, true, pred, threshold=self.threshold), predictions=pred, targets=true)

    def predict(self, loader) -> tuple[np.ndarray, np.ndarray]:
        preds, targets = [], []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                out = self._metric_outputs(self._forward(batch))
                preds.append(out.cpu().numpy())
                targets.append(batch.y.view(-1).cpu().numpy())
        return np.concatenate(targets), np.concatenate(preds)

    @staticmethod
    def save_history(history: dict[str, list[float]], output_path: Path) -> None:
        output_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
