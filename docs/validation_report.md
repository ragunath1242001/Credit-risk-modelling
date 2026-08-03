# Validation report

`python scripts/train_pd.py` writes the empirical data-quality contract and logs ROC AUC, Gini, KS, Brier score, confusion matrix, seed, dataset version, and challenger comparison. The holdout is stratified; no temporal backtest is claimed.

