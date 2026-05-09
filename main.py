from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from rdkit import RDLogger

from molgraphx.data.loaders.dataset import load_dataset_bundle, set_random_seed
from molgraphx.evaluation.evaluate import plot_bar, plot_loss, plot_roc, save_predictions, save_results, write_case_study
from molgraphx.interpretability.visualize_attention import visualize_attention
from molgraphx.models.baseline.fingerprint_model import ECFPBaseline
from molgraphx.models.gnn import HybridGAT
from molgraphx.training.metrics import best_f1_threshold, compute_metrics
from molgraphx.training.trainer import Trainer
from molgraphx.utils.config import ExperimentConfig, ensure_output_dirs, normalize_dataset_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="BBBP")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def regression_stats(dataset: list) -> tuple[float, float]:
    values = torch.tensor([float(graph.y.item()) for graph in dataset], dtype=torch.float32)
    return float(values.mean().item()), float(values.std(unbiased=False).item())


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def latency_ms_baseline(model: ECFPBaseline, dataset: list) -> float:
    start = time.perf_counter()
    model.predict(dataset)
    return (time.perf_counter() - start) * 1000.0 / max(len(dataset), 1)


def latency_ms_gnn(trainer: Trainer, loader, device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    trainer.evaluate(loader)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / max(len(loader.dataset), 1)


def tune_baseline_threshold(model: ECFPBaseline, val_dataset: list) -> float:
    _, y_true = model._extract_xy(val_dataset)
    predictions = model.predict(val_dataset)
    return best_f1_threshold(y_true, predictions)


def main() -> None:
    args = parse_args()
    RDLogger.DisableLog("rdApp.*")
    dataset_name = normalize_dataset_name(args.dataset)
    output_dirs = ensure_output_dirs()
    config = ExperimentConfig(dataset_name=dataset_name, batch_size=args.batch_size, max_epochs=args.epochs, random_seed=args.seed)
    set_random_seed(config.random_seed)
    
    print(f"\n===== MolGraphX Training =====")
    print(f"Dataset: {dataset_name}")
    print(f"Config: hidden_dim={config.hidden_dim}, heads={config.heads}, dropout={config.dropout}")
    print(f"Training epochs: {config.max_epochs}, batch_size: {config.batch_size}")
    print(f"Loading dataset...")
    
    bundle = load_dataset_bundle(config)
    print(f"✓ Dataset loaded: {len(bundle.train_dataset)} train, {len(bundle.val_dataset)} val, {len(bundle.test_dataset)} test")
    print(f"  Node features: {bundle.num_node_features}, Edge features: {bundle.num_edge_features}, FP dim: {bundle.fingerprint_dim}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Device: {device}")
    
    train_loader, val_loader, test_loader = bundle.build_loaders(config.batch_size)

    print(f"\n--- Training Baseline (ECFP + LR) ---")
    baseline = ECFPBaseline(bundle.task_type, random_seed=config.random_seed)
    baseline.fit(bundle.train_dataset)
    baseline_artifacts = baseline.evaluate(bundle.test_dataset)
    baseline_metrics = dict(baseline_artifacts.metrics)
    if bundle.task_type == "classification":
        baseline_metrics = compute_metrics(bundle.task_type, baseline_artifacts.targets, baseline_artifacts.predictions, threshold=tune_baseline_threshold(baseline, bundle.val_dataset))
    baseline_metrics["params"] = 0.0
    baseline_metrics["epoch_time"] = 0.0
    baseline_metrics["latency_ms"] = latency_ms_baseline(baseline, bundle.test_dataset)
    print(f"✓ Baseline complete - ROC-AUC: {baseline_metrics.get('roc_auc', baseline_metrics.get('rmse', 0)):.4f}")

    print(f"\n--- Training Hybrid ECFP + GAT ---")
    target_mean, target_std = regression_stats(bundle.train_dataset)
    hybrid = HybridGAT(
        num_node_features=bundle.num_node_features,
        num_edge_features=bundle.num_edge_features,
        fingerprint_dim=bundle.fingerprint_dim,
        hidden_dim=config.hidden_dim,
        heads=config.heads,
        dropout=config.dropout,
    )
    trainer = Trainer(
        model=hybrid,
        task_type=bundle.task_type,
        device=device,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_epochs=config.max_epochs,
        patience=config.patience,
        checkpoint_path=output_dirs["checkpoints"] / f"{dataset_name.lower()}_hybrid_gat.pt",
        target_mean=target_mean,
        target_std=target_std,
    )
    print(f"Training started... (patience={config.patience})")
    history = trainer.train(train_loader, val_loader)
    print(f"✓ Training complete after {len(history['epoch_time'])} epochs")
    
    print(f"\n--- Evaluating Hybrid ECFP + GAT ---")
    hybrid_artifacts = trainer.evaluate(test_loader)
    hybrid_metrics = dict(hybrid_artifacts.metrics)
    hybrid_metrics["params"] = float(count_parameters(hybrid))
    hybrid_metrics["epoch_time"] = float(sum(history["epoch_time"]) / max(len(history["epoch_time"]), 1))
    hybrid_metrics["latency_ms"] = latency_ms_gnn(trainer, test_loader, device)
    print(f"✓ Hybrid evaluation complete - ROC-AUC: {hybrid_metrics.get('roc_auc', hybrid_metrics.get('rmse', 0)):.4f}")

    results = {"ecfp_lr": baseline_metrics, "hybrid_gat": hybrid_metrics}
    prediction_rows = []
    for model_name, artifacts in [("ecfp_lr", baseline_artifacts), ("hybrid_gat", hybrid_artifacts)]:
        for idx, (prediction, target) in enumerate(zip(artifacts.predictions, artifacts.targets)):
            prediction_rows.append({"model": model_name, "index": idx, "prediction": float(prediction), "target": float(target)})

    print(f"\n--- Generating Outputs ---")
    top_atom = None
    if bundle.test_dataset:
        print(f"Visualizing attention...")
        scores, _ = visualize_attention(hybrid, bundle.test_dataset[0], output_dirs["attention"] / "sample.png", device)
        best_idx = int(max(range(len(scores)), key=lambda idx: scores[idx]))
        from rdkit import Chem

        mol = Chem.MolFromSmiles(bundle.test_dataset[0].smiles)
        top_atom = {"atom_index": best_idx, "atom_symbol": mol.GetAtomWithIdx(best_idx).GetSymbol(), "importance": scores[best_idx]}
        print(f"✓ Top atom: {top_atom['atom_symbol']} (index {top_atom['atom_index']}, score {top_atom['importance']:.4f})")

    save_results(results, output_dirs["root"])
    save_predictions(prediction_rows, output_dirs["root"])
    plot_loss(history, output_dirs["plots"] / "hybrid_gat_loss.png")
    score_metric = "roc_auc" if bundle.task_type == "classification" else "rmse"
    plot_bar(results, score_metric, output_dirs["plots"] / f"{dataset_name.lower()}_score.png", bundle.task_type == "regression")
    plot_bar(results, "params", output_dirs["plots"] / f"{dataset_name.lower()}_params.png", True)
    plot_bar(results, "epoch_time", output_dirs["plots"] / f"{dataset_name.lower()}_epoch_time.png", True)
    plot_bar(results, "latency_ms", output_dirs["plots"] / f"{dataset_name.lower()}_latency.png", True)
    if bundle.task_type == "classification":
        plot_roc(
            baseline_artifacts.targets,
            baseline_artifacts.predictions,
            hybrid_artifacts.predictions,
            ("ecfp_lr", "hybrid_gat"),
            output_dirs["plots"] / "roc.png",
        )
    write_case_study(output_dirs["root"], dataset_name, bundle.task_type, results, top_atom)
    print(f"✓ All outputs saved to outputs/")

    print(f"\n===== FINAL RESULTS =====")
    print(f"Model: Hybrid ECFP + GAT")
    print(f"Dataset: {dataset_name}")
    if bundle.task_type == "classification":
        print(f"ROC-AUC: {hybrid_metrics['roc_auc']:.4f}")
        print(f"F1: {hybrid_metrics['f1']:.4f}")
    else:
        print(f"RMSE: {hybrid_metrics['rmse']:.4f}")
    print(f"Epoch Time: {hybrid_metrics['epoch_time']:.4f}s")
    print(f"Params: {hybrid_metrics['params']:.0f}")
    print(f"Latency: {hybrid_metrics['latency_ms']:.4f}ms")
    print(f"=============================\n")


if __name__ == "__main__":
    main()
