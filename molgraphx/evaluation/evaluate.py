from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def render_metrics_table(results: dict[str, dict[str, float]]) -> str:
    metric_names = sorted({metric for model_metrics in results.values() for metric in model_metrics})
    if not metric_names:
        return "No metrics available."

    widths = {"model": max(5, max(len(name) for name in results))}
    for metric_name in metric_names:
        widths[metric_name] = max(len(metric_name), 10)

    header = " | ".join(["Model".ljust(widths["model"])] + [metric.ljust(widths[metric]) for metric in metric_names])
    separator = "-+-".join(["-" * widths["model"]] + ["-" * widths[metric] for metric in metric_names])

    rows = [header, separator]
    for model_name, model_metrics in results.items():
        values = [f"{model_metrics.get(metric, np.nan):.4f}".ljust(widths[metric]) for metric in metric_names]
        rows.append(" | ".join([model_name.ljust(widths["model"])] + values))
    return "\n".join(rows)


def save_results(
    results: dict[str, dict[str, float]],
    output_dir: Path,
    dataset_name: str,
    suffix: str = "",
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{dataset_name.lower()}_metrics{suffix}.json"
    csv_path = output_dir / f"{dataset_name.lower()}_metrics{suffix}.csv"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    metric_names = sorted({metric for model_metrics in results.values() for metric in model_metrics})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", *metric_names])
        for model_name, model_metrics in results.items():
            writer.writerow([model_name, *[model_metrics.get(metric, "") for metric in metric_names]])

    return json_path, csv_path


def load_results(json_path: Path) -> dict[str, dict[str, float]]:
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {str(model): {str(metric): float(value) for metric, value in metrics.items()} for model, metrics in data.items()}


def print_sample_predictions(model_name: str, predictions: np.ndarray, targets: np.ndarray, max_items: int = 5) -> None:
    print(f"\nSample predictions from {model_name}:")
    for index, (prediction, target) in enumerate(zip(predictions[:max_items], targets[:max_items]), start=1):
        print(f"  {index}. prediction={float(prediction):.4f} | target={float(target):.4f}")


def save_sample_predictions(
    model_name: str,
    predictions: np.ndarray,
    targets: np.ndarray,
    output_path: Path,
    max_items: int = 20,
) -> None:
    rows = []
    for prediction, target in zip(predictions[:max_items], targets[:max_items]):
        rows.append({"prediction": float(prediction), "target": float(target)})

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump({"model": model_name, "samples": rows}, handle, indent=2)


def aggregate_seed_results(seed_results: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in seed_results:
        model_name = str(row["model"])
        metrics = row["metrics"]
        if not isinstance(metrics, dict):
            continue
        grouped.setdefault(model_name, {})
        for metric_name, value in metrics.items():
            grouped[model_name].setdefault(metric_name, []).append(float(value))

    summary: dict[str, dict[str, float]] = {}
    for model_name, metrics in grouped.items():
        summary[model_name] = {}
        for metric_name, values in metrics.items():
            array = np.asarray(values, dtype=np.float32)
            summary[model_name][f"{metric_name}_mean"] = float(np.nanmean(array))
            summary[model_name][f"{metric_name}_std"] = float(np.nanstd(array, ddof=1)) if len(array) > 1 else 0.0
    return summary


def save_multiseed_results(
    seed_results: list[dict[str, object]],
    summary: dict[str, dict[str, float]],
    output_dir: Path,
    dataset_name: str,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = output_dir / f"{dataset_name.lower()}_multiseed_runs.csv"
    summary_csv_path = output_dir / f"{dataset_name.lower()}_multiseed_summary.csv"
    summary_json_path = output_dir / f"{dataset_name.lower()}_multiseed_summary.json"

    metric_names = sorted(
        {
            metric_name
            for row in seed_results
            if isinstance(row["metrics"], dict)
            for metric_name in row["metrics"]
        }
    )
    with runs_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "model", *metric_names])
        for row in seed_results:
            metrics = row["metrics"] if isinstance(row["metrics"], dict) else {}
            writer.writerow([row["seed"], row["model"], *[metrics.get(metric, "") for metric in metric_names]])

    summary_metric_names = sorted({metric for model_metrics in summary.values() for metric in model_metrics})
    with summary_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", *summary_metric_names])
        for model_name, model_metrics in summary.items():
            writer.writerow([model_name, *[model_metrics.get(metric, "") for metric in summary_metric_names]])

    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return runs_path, summary_csv_path, summary_json_path


def plot_model_comparison(
    results: dict[str, dict[str, float]],
    output_path: Path,
    primary_metric: str,
    lower_is_better: bool,
) -> None:
    model_names = list(results)
    values = [results[model].get(primary_metric, np.nan) for model in model_names]

    plt.figure(figsize=(8, 4.5))
    colors = ["#5B8DEF" if model != "mpnnpp" else "#18A999" for model in model_names]
    plt.bar(model_names, values, color=colors)
    plt.ylabel(primary_metric.replace("_", " ").upper())
    title_suffix = "lower is better" if lower_is_better else "higher is better"
    plt.title(f"Model comparison ({title_suffix})")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def generate_metric_plots(
    results: dict[str, dict[str, float]],
    output_dir: Path,
    prefix: str,
    task_type: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = sorted({metric for model_metrics in results.values() for metric in model_metrics})
    models = list(results)
    lower_is_better = {"rmse", "mae", "rmse_mean", "mae_mean", "rmse_std", "mae_std"}

    for metric in metrics:
        values = [results[model].get(metric, np.nan) for model in models]
        plt.figure(figsize=(8, 4.5))
        plt.bar(models, values, color=["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"][: len(models)])
        plt.ylabel(metric.upper())
        plt.title(f"{metric} by model")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}_{metric}.png", dpi=220)
        plt.close()

    matrix = np.array([[results[model].get(metric, np.nan) for metric in metrics] for model in models], dtype=float)
    score = matrix.copy()
    for idx, metric in enumerate(metrics):
        column = score[:, idx]
        finite = np.isfinite(column)
        if not finite.any():
            continue
        min_v, max_v = np.nanmin(column), np.nanmax(column)
        if max_v - min_v < 1e-12:
            score[finite, idx] = 1.0
        else:
            norm = (column - min_v) / (max_v - min_v)
            score[finite, idx] = 1.0 - norm if metric in lower_is_better else norm

    plt.figure(figsize=(1.6 * len(metrics) + 2, 0.7 * len(models) + 2))
    plt.imshow(score, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
    plt.colorbar(label="normalized score")
    plt.xticks(range(len(metrics)), metrics, rotation=20, ha="right")
    plt.yticks(range(len(models)), models)
    plt.title(f"{task_type} metric heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_heatmap.png", dpi=220)
    plt.close()

    ranks = np.zeros_like(matrix)
    for idx, metric in enumerate(metrics):
        order = np.argsort(matrix[:, idx])
        if metric not in lower_is_better:
            order = order[::-1]
        for rank, model_idx in enumerate(order, start=1):
            ranks[model_idx, idx] = rank

    plt.figure(figsize=(8, 4.5))
    for idx, model in enumerate(models):
        plt.plot(metrics, ranks[idx], marker="o", label=model)
    plt.gca().invert_yaxis()
    plt.ylabel("rank")
    plt.title("Model ranks across metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_ranks.png", dpi=220)
    plt.close()
