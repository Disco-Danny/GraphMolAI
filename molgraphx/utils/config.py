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
}


@dataclass(slots=True)
class ExperimentConfig:
    dataset_name: str = "BBBP"
    batch_size: int = 64
    hidden_dim: int = 32
    heads: int = 2
    dropout: float = 0.3
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 120
    patience: int = 10
    test_size: float = 0.2
    val_ratio: float = 0.1
    random_seed: int = 42
    fingerprint_radius: int = 2
    fingerprint_n_bits: int = 2048

    @property
    def task_type(self) -> str:
        return DATASET_TASKS[self.dataset_name]["task_type"]

    @property
    def data_root(self) -> Path:
        return Path("data") / self.dataset_name.lower()


def normalize_dataset_name(name: str) -> str:
    normalized = name.strip().upper()
    if normalized not in DATASET_TASKS:
        raise ValueError("Supported datasets: ESOL, BBBP.")
    return normalized


def ensure_output_dirs() -> dict[str, Path]:
    root = Path("outputs")
    paths = {
        "root": root,
        "plots": root / "plots",
        "attention": root / "attention",
        "checkpoints": root / "checkpoints",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
