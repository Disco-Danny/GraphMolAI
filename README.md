# MolGraphX

MolGraphX is a production-style molecular property prediction project that compares classical fingerprint baselines against Graph Neural Networks on MoleculeNet datasets. It downloads data automatically, trains both model families, evaluates them with task-appropriate metrics, and produces a simple atom-level interpretability report for GNN predictions.

## Features

- Automatic MoleculeNet dataset download for `ESOL`, `BBBP`, and `HIV`
- RDKit-based graph featurization with atom and bond descriptors
- ECFP fingerprint baseline
- Logistic regression for classification datasets and Ridge regression for ESOL regression
- Message Passing Neural Network (MPNN) implemented with PyTorch Geometric
- Attention GNN with edge-aware graph attention layers
- MPNN++ with residual edge-conditioned message passing and multi-pool readout
- Early stopping, checkpointing, saved training curves, and persisted metrics
- Gradient-based atom importance for a sample test molecule

## Project Structure

```text
molgraphx/
├── data/
│   └── loaders/
│       └── dataset.py
├── evaluation/
│   └── evaluate.py
├── features/
│   └── fingerprint.py
├── interpretability/
│   └── atom_importance.py
├── models/
│   ├── baseline/
│   │   └── fingerprint_model.py
│   └── gnn/
│       ├── attention_gnn.py
│       ├── mpnn.py
│       └── mpnnpp.py
├── training/
│   ├── metrics.py
│   └── trainer.py
└── utils/
    └── config.py
main.py
requirements.txt
```

## Setup

1. Create and activate a Python 3.10+ environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

For NVIDIA GPU training on Windows, use the CUDA environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --no-cache-dir -r requirements-cuda.txt
```

## How To Run

Train both the baseline and the GNN on the default `ESOL` dataset:

```bash
python main.py
```

By default, this compares all four research models:

- Fingerprint baseline
- MPNN
- Attention GNN
- MPNN++

Train only the baseline:

```bash
python main.py --model baseline --dataset ESOL
```

Train only the GNN on a classification dataset:

```bash
python main.py --model gnn --dataset BBBP
```

Train a single GNN variant:

```bash
python main.py --model gnn --dataset ESOL --gnn-type mpnn
python main.py --model gnn --dataset ESOL --gnn-type attention
python main.py --model gnn --dataset ESOL --gnn-type mpnnpp
```

Train both models on HIV with custom hyperparameters:

```bash
python main.py --model both --dataset HIV --gnn-type all --epochs 75 --batch-size 128 --hidden-dim 256
```

Run an optimized MPNN++ experiment:

```bash
python main.py --model gnn --dataset ESOL --gnn-type mpnnpp --epochs 150 --patience 25 --hidden-dim 256 --num-layers 4 --dropout 0.1 --learning-rate 0.0003
```

If using the project CUDA venv without activating it first:

```powershell
.\.venv\Scripts\python main.py --model gnn --dataset ESOL --gnn-type mpnnpp --epochs 150 --patience 25 --hidden-dim 256 --num-layers 4 --dropout 0.1 --learning-rate 0.0003
```

Run a paper-style repeated experiment across seeds:

```bash
python main.py --model both --dataset ESOL --gnn-type all --seeds 0,1,2,3,4 --epochs 150 --patience 25 --hidden-dim 256 --num-layers 4 --dropout 0.1 --learning-rate 0.0003
```

## Outputs

Artifacts are saved under `artifacts/<dataset_name>/`:

- `checkpoints/`: best GNN model checkpoint
- `plots/`: training curve plots
- `results/`: metrics table in JSON and CSV, saved training history, and sample predictions
- `results/`: multi-seed run logs and mean/std summaries when `--seeds` is used
- `interpretability/`: atom importance scores for a sample test molecule

## Metrics

- `ESOL` uses regression metrics: `RMSE` and `MAE`
- `BBBP` and `HIV` use classification metrics: `ROC-AUC` and `Accuracy`

## Example Workflow

When you run `python main.py`, the project will:

1. Download and prepare the selected MoleculeNet dataset automatically
2. Build RDKit fingerprints for the baseline model
3. Train the baseline model and report test metrics
4. Train MPNN, Attention GNN, and MPNN++ with validation-based early stopping
5. Compare all four models in a printed metrics table
6. Save sample prediction outputs and GNN atom importance scores

## Research Upgrades

- Regression GNNs standardize targets during training and unscale predictions for final metrics.
- Training uses gradient clipping and `ReduceLROnPlateau` scheduling for better stability.
- `--seeds` enables repeated experiments and saves aggregate mean/std tables.
- Comparison plots are saved automatically under `artifacts/<dataset_name>/plots/`.

## Notes

- `torch-geometric` may install binary dependencies automatically from PyPI. If your environment has trouble resolving PyG wheels, use the official PyTorch Geometric install instructions that match your local PyTorch version.
- The classification baseline uses logistic regression. For `ESOL`, the project switches to Ridge regression because logistic regression is classification-only.
- The baseline is fingerprint-based and intentionally simple, while the GNN variants test learned graph structure, attention-based neighborhood weighting, and stronger residual message passing.
