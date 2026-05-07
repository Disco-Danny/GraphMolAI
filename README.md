# GraphMolAI

GraphMolAI is a production-style molecular property prediction project that compares classical fingerprint baselines against Graph Neural Networks on MoleculeNet datasets. It downloads data automatically, trains both model families, evaluates them with task-appropriate metrics, and produces a simple atom-level interpretability report for GNN predictions.

## Features

- Automatic MoleculeNet dataset download for `ESOL`, `BBBP`, and `HIV`
- RDKit-based graph featurization with atom and bond descriptors
- ECFP fingerprint baseline
- Logistic regression for classification datasets and Ridge regression for ESOL regression
- Message Passing Neural Network (MPNN) implemented with PyTorch Geometric
- Attention GNN with edge-aware graph attention layers
- MPNN++ with residual edge-conditioned message passing and multi-pool readout
- GAT model with attention extraction and molecule highlighting
- Early stopping, checkpointing, saved training curves, and persisted metrics
- Gradient-based atom importance for a sample test molecule
- Attention PNG outputs for GAT explanations
- Plot regeneration without retraining

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
│   └── visualize_attention.py
├── models/
│   ├── baseline/
│   │   └── fingerprint_model.py
│   └── gnn/
│       ├── attention_gnn.py
│       ├── gat.py
│       ├── mpnn.py
│       └── mpnnpp.py
├── training/
│   ├── metrics.py
│   └── trainer.py
└── utils/
    └── config.py
main.py
requirements.txt
requirements-cuda.txt
explanation.txt
```

## Quick Start

CPU:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

CUDA GPU:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --no-cache-dir -r requirements-cuda.txt
```

Run default 5-model comparison:

```powershell
.\.venv\Scripts\python main.py --model both --gnn-type all --dataset ESOL
```

## Setup Details

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

## Models Used

- `baseline`: ECFP fingerprint + sklearn
- `mpnn`: message passing GNN
- `attention`: attention-based GNN
- `mpnnpp`: stronger residual MPNN
- `gat`: graph attention network with attention visualization

## Main Commands

Default full comparison, 5 techniques:

```powershell
.\.venv\Scripts\python main.py --model both --gnn-type all --dataset ESOL
```

Run on BBBP:

```powershell
.\.venv\Scripts\python main.py --model both --gnn-type all --dataset BBBP
```

Run only baseline:

```powershell
.\.venv\Scripts\python main.py --model baseline --dataset ESOL
```

Run one model only:

```powershell
.\.venv\Scripts\python main.py --model gnn --gnn-type mpnn --dataset ESOL
.\.venv\Scripts\python main.py --model gnn --gnn-type attention --dataset ESOL
.\.venv\Scripts\python main.py --model gnn --gnn-type mpnnpp --dataset ESOL
.\.venv\Scripts\python main.py --model gnn --gnn-type gat --dataset ESOL
```

Run with GAT explanation PNG:

```powershell
.\.venv\Scripts\python main.py --model gnn --gnn-type gat --dataset ESOL --explain true
.\.venv\Scripts\python main.py --model gnn --gnn-type gat --dataset BBBP --explain true
```

Optimized MPNN++ run:

```powershell
.\.venv\Scripts\python main.py --model gnn --dataset ESOL --gnn-type mpnnpp --epochs 150 --patience 25 --hidden-dim 256 --num-layers 4 --dropout 0.1 --learning-rate 0.0003
```

Multi-seed research run:

```powershell
.\.venv\Scripts\python main.py --model both --dataset ESOL --gnn-type all --seeds 0,1,2,3,4 --epochs 150 --patience 25 --hidden-dim 256 --num-layers 4 --dropout 0.1 --learning-rate 0.0003
```

Regenerate plots only, no retraining:

```powershell
.\.venv\Scripts\python main.py --dataset ESOL --plots-only true
.\.venv\Scripts\python main.py --dataset ESOL --seeds 0,1,2,3,4 --plots-only true
```

## Outputs

Artifacts are saved under `artifacts/<dataset_name>/`:

- `checkpoints/`: best GNN model checkpoint
- `plots/`: training curve plots
- `plots/`: metric bars, heatmap, rank plot, comparison plot
- `results/`: metrics table in JSON and CSV, saved training history, and sample predictions
- `results/`: multi-seed run logs and mean/std summaries when `--seeds` is used
- `interpretability/`: atom importance scores for a sample test molecule
- `outputs/attention/`: GAT attention PNGs

Important files:

- [explanation.txt](</c:/Users/musta/OneDrive/Desktop/DL CASE STUDY/code/explanation.txt>): short plain-language project explanation
- [requirements.txt](</c:/Users/musta/OneDrive/Desktop/DL CASE STUDY/code/requirements.txt>): CPU environment
- [requirements-cuda.txt](</c:/Users/musta/OneDrive/Desktop/DL CASE STUDY/code/requirements-cuda.txt>): CUDA environment

## Metrics

- `ESOL` uses regression metrics: `RMSE` and `MAE`
- `BBBP` and `HIV` use classification metrics: `ROC-AUC` and `Accuracy`

## Example Workflow

When you run `python main.py`, the project will:

1. Download and prepare the selected MoleculeNet dataset automatically
2. Build RDKit fingerprints for the baseline model
3. Train the baseline model and report test metrics
4. Train MPNN, Attention GNN, MPNN++, and GAT with validation-based early stopping
5. Compare all five models in a printed metrics table
6. Save sample prediction outputs, plots, and interpretability outputs

## Research Upgrades

- Regression GNNs standardize targets during training and unscale predictions for final metrics.
- Training uses gradient clipping and `ReduceLROnPlateau` scheduling for better stability.
- `--seeds` enables repeated experiments and saves aggregate mean/std tables.
- Comparison plots are saved automatically under `artifacts/<dataset_name>/plots/`.
- `--explain true` saves molecule attention images for GAT.
- `--plots-only true` rebuilds visual outputs from saved metrics.

## Troubleshooting

If PowerShell blocks venv activation:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

If GPU is installed but not used:

```powershell
.\.venv\Scripts\python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

If VRAM is low, reduce batch size:

```powershell
.\.venv\Scripts\python main.py --model both --gnn-type all --dataset ESOL --batch-size 32
```

## Notes

- `torch-geometric` may install binary dependencies automatically from PyPI. If your environment has trouble resolving PyG wheels, use the official PyTorch Geometric install instructions that match your local PyTorch version.
- The classification baseline uses logistic regression. For `ESOL`, the project switches to Ridge regression because logistic regression is classification-only.
- The baseline is fingerprint-based and intentionally simple, while the GNN variants test learned graph structure, attention weighting, stronger message passing, and explicit GAT explanations.
