import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.layer1_classic import fit_predict_oof
from src.optuna_tune import tune_model

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
TUNED_MODELS = ['lightgbm', 'xgboost', 'catboost', 'rf', 'extratrees']


def trial_budget() -> int:
    hw_path = ROOT / 'hardware.json'
    if not hw_path.exists():
        return 30
    try:
        hw = json.loads(hw_path.read_text())
    except Exception:
        return 30
    if not hw.get('cuda_available'):
        return 20
    return 50


def main():
    train = pd.read_parquet(DATA_DIR / 'train_features.parquet')
    test = pd.read_parquet(DATA_DIR / 'test_features.parquet')
    folds_df = pd.read_csv(ROOT / 'folds.csv')
    train = train.merge(folds_df, on='id', how='left')

    y = train['is_fraud'].astype(int)
    folds = train['fold'].values
    feature_cols = [c for c in test.columns if c != 'id']
    X, X_test = train[feature_cols], test[feature_cols]

    (ROOT / 'params').mkdir(exist_ok=True)
    n_trials = trial_budget()
    print(f'Optuna trial budget per model: {n_trials}')

    exp_rows = []
    for model_name in TUNED_MODELS:
        default_oof_path = ROOT / 'oof' / f'{model_name}.npy'
        default_auc = roc_auc_score(y, np.load(default_oof_path)) if default_oof_path.exists() else np.nan

        start = time.time()
        result = tune_model(model_name, X, y, folds, n_trials=n_trials)
        (ROOT / 'params' / f'{model_name}.json').write_text(json.dumps(result['params'], indent=2))

        if result['cv_auc'] >= (default_auc if not np.isnan(default_auc) else -np.inf):
            oof, test_pred = fit_predict_oof(model_name, X, y, X_test, folds, params=result['params'])
            np.save(ROOT / 'oof' / f'{model_name}.npy', oof)
            np.save(ROOT / 'test_preds' / f'{model_name}.npy', test_pred)
            kept_params = result['params']
        else:
            kept_params = 'default (tuning did not improve CV AUC)'

        elapsed = time.time() - start
        print(f'{model_name}: default AUC={default_auc:.4f} tuned AUC={result["cv_auc"]:.4f} ({elapsed:.1f}s)')
        exp_rows.append({
            'exp_id': f'optuna_{model_name}', 'change': f'Optuna tuning ({n_trials} trials)',
            'feature_set': 'all_features', 'params': json.dumps(kept_params) if isinstance(kept_params, dict) else kept_params,
            'cv_auc_mean': result['cv_auc'], 'cv_auc_std': np.nan, 'cv_pr_auc': np.nan,
            'runtime_s': round(elapsed, 1), 'kept': isinstance(kept_params, dict),
        })

    exp_path = ROOT / 'experiments.csv'
    exp_df = pd.DataFrame(exp_rows)
    if exp_path.exists():
        exp_df = pd.concat([pd.read_csv(exp_path), exp_df], ignore_index=True)
    exp_df.to_csv(exp_path, index=False)
    print(f'Wrote {exp_path}')


if __name__ == '__main__':
    main()
