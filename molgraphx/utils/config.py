from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DATASET_TASKS = {
    "ESOL": {
        "task_type": "regression",
        "target_column": "measured log solubility in mols per litre",
        "raw_filename": "delaney-processed.csv",
    },
    "BBBP": {
        "task_type": "classification",
        "target_column": "p_np",
        "raw_filename": "BBBP.csv",
    },
    "HIV": {
        "task_type": "classification",
        "target_column": "HIV_active",
        "raw_filename": "HIV.csv",
    },
}


@dataclass(slots=True)
class ExperimentConfig:
    dataset_name: str = "ESOL"
    batch_size: int = 64
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 50
    patience: int = 10
    scheduler_patience: int = 5
    grad_clip_norm: float | None = 5.0
    standardize_targets: bool = True
    test_size: float = 0.2
    val_ratio: float = 0.1
    random_seed: int = 42
    fingerprint_radius: int = 2
    fingerprint_n_bits: int = 2048
    root_dir: str = "artifacts"

    @property
    def task_type(self) -> str:
        return DATASET_TASKS[self.dataset_name]["task_type"]

    @property
    def output_root(self) -> Path:
        return Path(self.root_dir) / self.dataset_name.lower()

    @property
    def data_root(self) -> Path:
        return Path("data") / self.dataset_name.lower()


def normalize_dataset_name(name: str) -> str:
    normalized = name.strip().upper()
    if normalized not in DATASET_TASKS:
        supported = ", ".join(DATASET_TASKS)
        raise ValueError(f"Unsupported dataset '{name}'. Supported datasets: {supported}.")
    return normalized


def ensure_output_dirs(config: ExperimentConfig) -> dict[str, Path]:
    output_root = config.output_root
    directories = {
        "root": output_root,
        "checkpoints": output_root / "checkpoints",
        "plots": output_root / "plots",
        "results": output_root / "results",
        "interpretability": output_root / "interpretability",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories
