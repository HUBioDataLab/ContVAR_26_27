# src/scripts/03_baseline_eval.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error
from scipy.stats import spearmanr
import argparse

"""
python src/scripts/03_baseline_eval.py \
  --data data/processed/tpmt_dataset.npz
  """
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to .npz dataset")
    args = parser.parse_args()

    data = np.load(args.data, allow_pickle=True)

    X = data["X"]       
    y_bin = data["y_bin"]    
    y_score = data["y_score"]  

    print("[INFO] X:", X.shape, "y_bin:", y_bin.shape, "y_score:", y_score.shape)


    X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.2, random_state=42, stratify=y_bin
    )

    Baseline_Model = LogisticRegression(max_iter=3000)
    Baseline_Model.fit(X_train, y_train)

    probs = Baseline_Model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    ap = average_precision_score(y_test, probs)

    print(f"[Binary probe] ROC-AUC = {auc:.4f} | AP = {ap:.4f}")


    X_train, X_test, y_train, y_test = train_test_split(
        X, y_score, test_size=0.2, random_state=42
    )

    reg = Ridge(alpha=1.0)
    reg.fit(X_train, y_train)

    preds = reg.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    spearman = spearmanr(y_test, preds).correlation

    print(f"[Regression probe] MSE = {mse:.4f} | Spearman = {spearman:.4f}")


if __name__ == "__main__":
    main()


"""
RESULTS:
[INFO] X: (100, 1280) y_bin: (100,) y_score: (100,)
[Binary probe] ROC-AUC = 0.7000 | AP = 0.7533
[Regression probe] MSE = 0.1440 | Spearman = 0.2872

"""