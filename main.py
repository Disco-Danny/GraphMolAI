from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from molgraphx.data.loaders.dataset import load_dataset_bundle, set_random_seed
from molgraphx.evaluation.evaluate import (
    aggregate_seed_results,
    plot_model_comparison,
    print_sample_predictions,
    render_metrics_table,
    save_multiseed_results,
    save_results,
    save_sample_predictions,
)
from molgraphx.interpretability.atom_importance import (
    compute_atom_importance,
    format_atom_importance,
    save_atom_importance,
)
from molgraphx.models.baseline.fingerprint_model import FingerprintBaselineModel
from molgraphx.models.gnn import AttentionGNN, MPNN, MPNNPlusPlus
from molgraphx.training.trainer import Trainer
from molgraphx.utils.config import ExperimentConfig, ensure_output_dirs, normalize_dataset_name


GNN_VARIANTS = {
    "mpnn": {
        "display_name": "MPNN",
        "artifact_key": "mpnn",
        "result_key": "mpnn",
        "class": MPNN,
    },
    "attention": {
        "display_name": "Attention GNN",
        "artifact_key": "attention_gnn",
        "result_key": "attention_gnn",
        "class": AttentionGNN,
    },
    "mpnnpp": {
        "display_name": "MPNN++",
        "artifact_key": "mpnnpp",
        "result_key": "mpnnpp",
        "class": MPNNPlusPlus,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Molecular property prediction with fingerprints and GNNs.")
    parser.add_argument("--model", choices=["baseline", "gnn", "both"], default="both")
    parser.add_argument(
        "--gnn-type",
        choices=["mpnn", "attention", "mpnnpp", "all"],
        default="all",
        help="GNN variant to train when --model is gnn or both.",
    )
    parser.add_argument("--dataset", type=str, default="ESOL", help="Dataset to use: ESOL, BBBP, or HIV.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument(
        "--no-standardize-targets",
        action="store_true",
        help="Disable target standardization for regression GNN training.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Comma-separated seeds for paper-style repeated experiments, e.g. 0,1,2,3,4.",
    )
    parser.add_argument("--fingerprint-bits", type=int, default=2048)
    parser.add_argument("--fingerprint-radius", type=int, default=2)
    return parser.parse_args()


def parse_seeds(args: argparse.Namespace) -> list[int]:
    if not args.seeds.strip():
        return [args.seed]
    return [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]


def primary_metric_for_task(task_type: str) -> tuple[str, bool]:
    if task_type == "classification":
        return "roc_auc", False
    return "rmse", True


def regression_target_stats(dataset: list) -> tuple[float, float]:
    targets = torch.tensor([float(graph.y.item()) for graph in dataset], dtype=torch.float32)
    return float(targets.mean().item()), float(targets.std(unbiased=False).item())


def run_baseline(
    config: ExperimentConfig,
    dataset_bundle,
    directories: dict[str, Path],
    run_suffix: str = "",
) -> tuple[dict[str, float], object]:
    print("\nTraining fingerprint baseline...")
    baseline = FingerprintBaselineModel(
        task_type=config.task_type,
        radius=config.fingerprint_radius,
        n_bits=config.fingerprint_n_bits,
        random_seed=config.random_seed,
    )
    baseline.fit(dataset_bundle.train_dataset)
    artifacts = baseline.evaluate(dataset_bundle.test_dataset)
    print(f"Baseline metrics: {artifacts.metrics}")
    print_sample_predictions("Fingerprint baseline", artifacts.predictions, artifacts.targets)
    save_sample_predictions(
        "Fingerprint baseline",
        artifacts.predictions,
        artifacts.targets,
        directories["results"] / f"{config.dataset_name.lower()}_baseline_samples{run_suffix}.json",
    )
    return artifacts.metrics, artifacts


def build_gnn_model(config: ExperimentConfig, dataset_bundle, variant: str) -> torch.nn.Module:
    model_class = GNN_VARIANTS[variant]["class"]
    return model_class(
        num_node_features=dataset_bundle.num_node_features,
        num_edge_features=dataset_bundle.num_edge_features,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        output_dim=1,
    )


def resolve_gnn_variants(gnn_type: str) -> list[str]:
    if gnn_type == "all":
        return ["mpnn", "attention", "mpnnpp"]
    return [gnn_type]


def run_gnn_variant(
    config: ExperimentConfig,
    dataset_bundle,
    directories: dict[str, Path],
    variant: str,
    run_suffix: str = "",
) -> tuple[dict[str, float], object]:
    variant_info = GNN_VARIANTS[variant]
    display_name = str(variant_info["display_name"])
    artifact_key = str(variant_info["artifact_key"])

    print(f"\nTraining {display_name}...")
    train_loader, val_loader, test_loader = dataset_bundle.build_loaders(config.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    target_mean, target_std = regression_target_stats(dataset_bundle.train_dataset)

    set_random_seed(config.random_seed)
    model = build_gnn_model(config, dataset_bundle, variant)
    trainer = Trainer(
        model=model,
        task_type=config.task_type,
        device=device,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_epochs=config.max_epochs,
        patience=config.patience,
        checkpoint_path=directories["checkpoints"] / f"{config.dataset_name.lower()}_{artifact_key}{run_suffix}.pt",
        standardize_targets=config.standardize_targets,
        target_mean=target_mean,
        target_std=target_std,
        scheduler_patience=config.scheduler_patience,
        grad_clip_norm=config.grad_clip_norm,
    )

    history = trainer.train(train_loader, val_loader)
    trainer.plot_history(
        history,
        directories["plots"] / f"{config.dataset_name.lower()}_{artifact_key}_training_curve{run_suffix}.png",
    )
    trainer.save_history(
        history,
        directories["results"] / f"{config.dataset_name.lower()}_{artifact_key}_history{run_suffix}.json",
    )
    if variant == "mpnn" and not run_suffix:
        trainer.plot_history(history, directories["plots"] / f"{config.dataset_name.lower()}_training_curve.png")
        trainer.save_history(history, directories["results"] / f"{config.dataset_name.lower()}_history.json")

    artifacts = trainer.evaluate(test_loader)
    print(f"{display_name} metrics: {artifacts.metrics}")
    print_sample_predictions(display_name, artifacts.predictions, artifacts.targets)
    save_sample_predictions(
        display_name,
        artifacts.predictions,
        artifacts.targets,
        directories["results"] / f"{config.dataset_name.lower()}_{artifact_key}_samples{run_suffix}.json",
    )
    if variant == "mpnn" and not run_suffix:
        save_sample_predictions(
            "GNN",
            artifacts.predictions,
            artifacts.targets,
            directories["results"] / f"{config.dataset_name.lower()}_gnn_samples.json",
        )

    explanation_target = dataset_bundle.test_dataset[0]
    report = compute_atom_importance(trainer.model, explanation_target, device=device, task_type=config.task_type)
    print(f"\nInterpretability sample ({display_name})")
    print(f"SMILES: {explanation_target.smiles}")
    print(format_atom_importance(report))
    save_atom_importance(
        report,
        directories["interpretability"] / f"{config.dataset_name.lower()}_{artifact_key}_atom_importance{run_suffix}.json",
    )
    if variant == "mpnn" and not run_suffix:
        save_atom_importance(
            report,
            directories["interpretability"] / f"{config.dataset_name.lower()}_atom_importance.json",
        )
    return artifacts.metrics, artifacts


def run_single_experiment(
    config: ExperimentConfig,
    args: argparse.Namespace,
    directories: dict[str, Path],
    run_suffix: str = "",
) -> dict[str, dict[str, float]]:
    set_random_seed(config.random_seed)
    print(f"Loading dataset: {config.dataset_name}")
    dataset_bundle = load_dataset_bundle(config)
    print(
        f"Loaded {len(dataset_bundle.train_dataset)} train / "
        f"{len(dataset_bundle.val_dataset)} val / "
        f"{len(dataset_bundle.test_dataset)} test molecules."
    )
    print(f"Task type: {config.task_type}")

    results: dict[str, dict[str, float]] = {}

    if args.model in {"baseline", "both"}:
        baseline_metrics, _ = run_baseline(config, dataset_bundle, directories, run_suffix)
        results["baseline"] = baseline_metrics

    if args.model in {"gnn", "both"}:
        for variant in resolve_gnn_variants(args.gnn_type):
            gnn_metrics, _ = run_gnn_variant(config, dataset_bundle, directories, variant, run_suffix)
            results[str(GNN_VARIANTS[variant]["result_key"])] = gnn_metrics

    return results


def main() -> None:
    args = parse_args()
    dataset_name = normalize_dataset_name(args.dataset)

    config = ExperimentConfig(
        dataset_name=dataset_name,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
        patience=args.patience,
        scheduler_patience=args.scheduler_patience,
        grad_clip_norm=args.grad_clip_norm,
        standardize_targets=not args.no_standardize_targets,
        random_seed=args.seed,
        fingerprint_n_bits=args.fingerprint_bits,
        fingerprint_radius=args.fingerprint_radius,
    )

    directories = ensure_output_dirs(config)
    seeds = parse_seeds(args)

    if len(seeds) > 1:
        seed_results: list[dict[str, object]] = []
        for seed in seeds:
            print(f"\n=== Seed {seed} ===")
            seed_config = replace(config, random_seed=seed)
            results = run_single_experiment(seed_config, args, directories, run_suffix=f"_seed_{seed}")
            for model_name, metrics in results.items():
                seed_results.append({"seed": seed, "model": model_name, "metrics": metrics})

        summary = aggregate_seed_results(seed_results)
        print("\nMulti-seed summary")
        print(render_metrics_table(summary))
        runs_path, summary_csv_path, summary_json_path = save_multiseed_results(
            seed_results,
            summary,
            directories["results"],
            config.dataset_name,
        )
        primary_metric, lower_is_better = primary_metric_for_task(config.task_type)
        plot_model_comparison(
            summary,
            directories["plots"] / f"{config.dataset_name.lower()}_multiseed_comparison.png",
            f"{primary_metric}_mean",
            lower_is_better,
        )
        print(f"\nSaved multi-seed runs to {runs_path}.")
        print(f"Saved multi-seed summary to {summary_csv_path} and {summary_json_path}.")
        return

    results = run_single_experiment(config, args, directories)

    if results:
        print("\nComparison")
        print(render_metrics_table(results))
        json_path, csv_path = save_results(results, directories["results"], config.dataset_name)
        primary_metric, lower_is_better = primary_metric_for_task(config.task_type)
        plot_model_comparison(
            results,
            directories["plots"] / f"{config.dataset_name.lower()}_comparison.png",
            primary_metric,
            lower_is_better,
        )
        print(f"\nSaved metrics to {json_path} and {csv_path}.")


if __name__ == "__main__":
    main()
