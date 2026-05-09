# MolGraphX

Minimal hybrid molecular learning case study.

## Idea

Compare:

- `ecfp_lr`: ECFP fingerprint + Logistic Regression
- `hybrid_gat`: 2-layer GAT + ECFP fusion

Goal:

- strong score
- low extra cost
- attention-based atom explanation

## Architecture

```text
SMILES
 -> graph + ECFP

baseline:
ECFP -> Logistic Regression

hybrid:
graph -> GAT -> GAT -> mean pool
graph embedding + ECFP -> fusion MLP -> prediction
```

## Datasets

- `BBBP`: ROC-AUC, F1
- `ESOL`: RMSE

## Run

### Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Execute

**Blood-Brain Barrier Classification (BBBP)**
```powershell
python -u main.py --dataset BBBP --epochs 80
```

**Solubility Prediction (ESOL)**
```powershell
python -u main.py --dataset ESOL --epochs 80
```

**Custom Configuration**
```powershell
python -u main.py --dataset BBBP --epochs 100 --batch-size 32 --seed 42
```

## Outputs

- `outputs/results.csv` - model metrics comparison
- `outputs/metrics.json` - full benchmark results
- `outputs/predictions.csv` - per-sample predictions
- `outputs/case_study.md` - analysis report
- `outputs/plots/` - visualization charts
- `outputs/attention/sample.png` - attention heatmap
- `outputs/checkpoints/` - saved model weights

## Results Interpretation

### Metrics

**Classification (BBBP):**
- `ROC-AUC`: Receiver Operating Characteristic Area Under Curve (higher is better, max 1.0)
- `F1`: Harmonic mean of precision/recall (higher is better, max 1.0)

**Regression (ESOL):**
- `RMSE`: Root Mean Square Error (lower is better, measures prediction error)

### Model Comparison

| Metric | Meaning | Baseline | Hybrid | Winner |
|--------|---------|----------|--------|--------|
| ROC-AUC/RMSE | Accuracy | Traditional ECFP | Graph Neural Net | Higher/Lower |
| Params | Model complexity | ~0 | ~70K | Lower is simpler |
| Epoch Time | Training speed | - | ~2-4s | Lower is faster |
| Latency | Inference speed | ~0.02ms | ~0.6ms | Lower is faster |

### Interpreting case_study.md

- **Benchmark Table**: Direct comparison of both approaches
- **Efficiency Comparison**: Speed vs accuracy tradeoff
- **Interpretability Insight**: Top atom identified by attention mechanism

## Reproducibility

- deterministic seed (default: 42)
- AMP (Automatic Mixed Precision) on CUDA
- early stopping with patience=10
- stratified train/val/test split
- fixed hyperparameters: hidden_dim=32, heads=2, dropout=0.3

## Attention Visualization

`outputs/attention/sample.png` shows:
- Molecule structure
- Atom importance scores (darker red = higher importance)
- Top-scoring atom identified by GAT attention weights
