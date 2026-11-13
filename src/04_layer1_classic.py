import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from src.layer1_classic import MODEL_NAMES, fit_predict_oof

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'


def main():
    train = pd.read_parquet(DATA_DIR / 'train_features.parquet')
    test = pd.read_parquet(DATA_DIR / 'test_features.parquet')
    folds_df = pd.read_csv(ROOT / 'folds.csv')
    train = train.merge(folds_df, on='id', how='left')

    y = train['is_fraud'].astype(int)
    folds = train['fold'].values
    feature_cols = [c for c in test.columns if c != 'id']
    X, X_test = train[feature_cols], test[feature_cols]

    (ROOT / 'oof').mkdir(exist_ok=True)
    (ROOT / 'test_preds').mkdir(exist_ok=True)

    results = {}
    for model_name in MODEL_NAMES:
        start = time.time()
        oof, test_pred = fit_predict_oof(model_name, X, y, X_test, folds)
        elapsed = time.time() - start
        auc = roc_auc_score(y, oof)
        pr_auc = average_precision_score(y, oof)
        np.save(ROOT / 'oof' / f'{model_name}.npy', oof)
        np.save(ROOT / 'test_preds' / f'{model_name}.npy', test_pred)
        results[model_name] = oof
        print(f'{model_name}: OOF AUC={auc:.4f} PR-AUC={pr_auc:.4f} ({elapsed:.1f}s)')

    corr = pd.DataFrame(results).corr()
    print('\nOOF prediction correlation matrix:')
    print(corr.to_string())


if __name__ == '__main__':
    main()
