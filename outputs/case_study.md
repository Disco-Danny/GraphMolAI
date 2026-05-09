# MolGraphX Case Study

## Problem Statement
Test if a small hybrid ECFP + GAT model beats ECFP alone on ESOL.

## Methodology
- Baseline: ECFP + Logistic Regression
- Hybrid: 2-layer GAT + ECFP fusion
- Split: train / val / test
- Training: mixed precision, early stopping, deterministic seed

## Benchmark Table

| Model | ROC-AUC/RMSE | F1 | Params | Epoch Time |
|---|---:|---:|---:|---:|
| ecfp_lr | 1.2456 |  | 0 | 0.0000 |
| hybrid_gat | 1.2131 |  | 69665 | 2.0932 |

## Efficiency Comparison
- baseline latency ms: 0.0197
- hybrid latency ms: 0.7069

## Interpretability Insight
- sample attention image: outputs/attention/sample.png
- top atom: C index 0 score 1.0000

## Conclusion
- baseline RMSE: 1.2456
- hybrid RMSE: 1.2131
- compare accuracy vs added compute