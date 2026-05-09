# MolGraphX Case Study

## Problem Statement
Test if a small hybrid ECFP + GAT model beats ECFP alone on BBBP.

## Methodology
- Baseline: ECFP + Logistic Regression
- Hybrid: 2-layer GAT + ECFP fusion
- Split: train / val / test
- Training: mixed precision, early stopping, deterministic seed

## Benchmark Table

| Model | ROC-AUC/RMSE | F1 | Params | Epoch Time |
|---|---:|---:|---:|---:|
| ecfp_lr | 0.9216 | 0.9355828220858896 | 0 | 0.0000 |
| hybrid_gat | 0.9229 | 0.9353383458646617 | 69665 | 4.8277 |

## Efficiency Comparison
- baseline latency ms: 0.0456
- hybrid latency ms: 0.3695

## Interpretability Insight
- sample attention image: outputs/attention/sample.png
- top atom: N index 13 score 1.0000

## Conclusion
- baseline ROC-AUC: 0.9216
- hybrid ROC-AUC: 0.9229
- compare accuracy vs added compute