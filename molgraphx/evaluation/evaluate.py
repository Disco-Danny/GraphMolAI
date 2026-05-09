from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve


def save_results(results: dict[str, dict[str, float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Model", "ROC-AUC/RMSE", "F1", "Params", "Epoch Time"])
        for model, metrics in results.items():
            writer.writerow([model, metrics.get("roc_auc", metrics.get("rmse", "")), metrics.get("f1", ""), metrics["params"], metrics["epoch_time"]])


def save_predictions(rows: list[dict[str, object]], output_dir: Path) -> None:
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "index", "prediction", "target"])
        for row in rows:
            writer.writerow([row["model"], row["index"], row["prediction"], row["target"]])


def plot_loss(history: dict[str, list[float]], output_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_bar(results: dict[str, dict[str, float]], metric: str, output_path: Path, lower_is_better: bool) -> None:
    models = list(results)
    values = [results[model][metric] for model in models]
    plt.figure(figsize=(6, 4))
    plt.bar(models, values, color=["#4C78A8", "#E45756"])
    plt.ylabel(metric)
    plt.title("lower better" if lower_is_better else "higher better")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_roc(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray, labels: tuple[str, str], output_path: Path) -> None:
    plt.figure(figsize=(5, 5))
    for y_pred, label in zip((y_pred_a, y_pred_b), labels):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        plt.plot(fpr, tpr, label=label)
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def write_case_study(output_dir: Path, dataset_name: str, task_type: str, results: dict[str, dict[str, float]], top_atom: dict[str, object] | None) -> None:
    baseline = results["ecfp_lr"]
    gat = results["hybrid_gat"]
    score_name = "ROC-AUC" if task_type == "classification" else "RMSE"
    score_a = gat.get("roc_auc", gat.get("rmse"))
    score_b = baseline.get("roc_auc", baseline.get("rmse"))
    lines = [
        "# MolGraphX Case Study",
        "",
        "## Problem Statement",
        f"Test if a small hybrid ECFP + GAT model beats ECFP alone on {dataset_name}.",
        "",
        "## Methodology",
        "- Baseline: ECFP + Logistic Regression",
        "- Hybrid: 2-layer GAT + ECFP fusion",
        "- Split: train / val / test",
        "- Training: mixed precision, early stopping, deterministic seed",
        "",
        "## Benchmark Table",
        "",
        "| Model | ROC-AUC/RMSE | F1 | Params | Epoch Time |",
        "|---|---:|---:|---:|---:|",
        f"| ecfp_lr | {baseline.get('roc_auc', baseline.get('rmse')):.4f} | {baseline.get('f1', '')} | {baseline['params']:.0f} | {baseline['epoch_time']:.4f} |",
        f"| hybrid_gat | {gat.get('roc_auc', gat.get('rmse')):.4f} | {gat.get('f1', '')} | {gat['params']:.0f} | {gat['epoch_time']:.4f} |",
        "",
        "## Efficiency Comparison",
        f"- baseline latency ms: {baseline['latency_ms']:.4f}",
        f"- hybrid latency ms: {gat['latency_ms']:.4f}",
        "",
        "## Interpretability Insight",
        f"- sample attention image: outputs/attention/sample.png",
        f"- top atom: {top_atom['atom_symbol']} index {top_atom['atom_index']} score {top_atom['importance']:.4f}" if top_atom else "- no atom explanation",
        "",
        "## Conclusion",
        f"- baseline {score_name}: {score_b:.4f}",
        f"- hybrid {score_name}: {score_a:.4f}",
        "- compare accuracy vs added compute",
    ]
    (output_dir / "case_study.md").write_text("\n".join(lines), encoding="utf-8")
