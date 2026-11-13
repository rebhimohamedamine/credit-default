import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.blend import optimize_blend_weights, build_submission
from src.layer1_classic import fit_predict_oof

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
N_SEEDS = 10


def seed_averaged_lightgbm(X, y, X_test, folds):
    params_path = ROOT / 'params' / 'lightgbm.json'
    base_params = json.loads(params_path.read_text()) if params_path.exists() else {}

    oof_accum = np.zeros(len(X))
    test_accum = np.zeros(len(X_test))
    for seed in range(N_SEEDS):
        params = dict(base_params)
        params['random_state'] = seed
        oof, test_pred = fit_predict_oof('lightgbm', X, y, X_test, folds, params=params)
        oof_accum += oof
        test_accum += test_pred
    return oof_accum / N_SEEDS, test_accum / N_SEEDS


def best_single_model_name(oof_dir):
    train = pd.read_parquet(DATA_DIR / 'train_features.parquet')
    folds_df = pd.read_csv(ROOT / 'folds.csv')
    train = train.merge(folds_df, on='id', how='left')
    y = train['is_fraud'].astype(int)

    best_name, best_auc = None, -1
    for path in oof_dir.glob('*.npy'):
        if path.stem.startswith('stack'):
            continue
        auc = roc_auc_score(y, np.load(path))
        if auc > best_auc:
            best_name, best_auc = path.stem, auc
    return best_name, best_auc


def main():
    train = pd.read_parquet(DATA_DIR / 'train_features.parquet')
    test = pd.read_parquet(DATA_DIR / 'test_features.parquet')
    folds_df = pd.read_csv(ROOT / 'folds.csv')
    train = train.merge(folds_df, on='id', how='left')

    y = train['is_fraud'].astype(int)
    folds = train['fold'].values
    feature_cols = [c for c in test.columns if c != 'id']
    X, X_test = train[feature_cols], test[feature_cols]

    seed_oof, seed_test_pred = seed_averaged_lightgbm(X, y, X_test, folds)
    np.save(ROOT / 'oof' / 'lightgbm_10seed.npy', seed_oof)
    np.save(ROOT / 'test_preds' / 'lightgbm_10seed.npy', seed_test_pred)
    print(f'10-seed LightGBM OOF AUC: {roc_auc_score(y, seed_oof):.4f}')

    stack_name = 'stack_calibrated' if (ROOT / 'oof' / 'stack_calibrated.npy').exists() else 'stack'
    best_name, best_auc = best_single_model_name(ROOT / 'oof')
    print(f'Best single Layer-1 model: {best_name} (AUC={best_auc:.4f})')

    blend_inputs = {
        'stack': np.load(ROOT / 'oof' / f'{stack_name}.npy'),
        'best_single': np.load(ROOT / 'oof' / f'{best_name}.npy'),
        'seed_avg_lightgbm': seed_oof,
    }
    weights = optimize_blend_weights(blend_inputs, y)
    print(f'Blend weights: {weights}')

    blend_oof = sum(w * pd.DataFrame(blend_inputs).rank(pct=True)[name] for name, w in weights.items())
    blend_auc = roc_auc_score(y, blend_oof)
    print(f'Blend OOF AUC: {blend_auc:.4f} (best single model: {best_auc:.4f})')
    if blend_auc < best_auc:
        print('WARNING: blend does not beat the best single model on CV. Shipping it anyway per the plan, '
              'but this should be called out in REPORT.md.')

    test_inputs = {
        'stack': np.load(ROOT / 'test_preds' / f'{stack_name}.npy'),
        'best_single': np.load(ROOT / 'test_preds' / f'{best_name}.npy'),
        'seed_avg_lightgbm': seed_test_pred,
    }
    test_ranks = pd.DataFrame(test_inputs).rank(pct=True)
    blend_test_pred = sum(w * test_ranks[name] for name, w in weights.items())

    sample_sub_path = DATA_DIR / 'sample_submission.csv'
    sample_sub = pd.read_csv(sample_sub_path) if sample_sub_path.exists() else pd.DataFrame(
        {'id': test['id'], 'is_fraud': 0.0})
    submission = build_submission(test['id'], blend_test_pred.values, sample_sub)

    assert len(submission) == len(test), 'row count mismatch'
    pred_fraud_rate = submission['is_fraud'].mean()
    train_fraud_rate = y.mean()
    print(f'Predicted test fraud rate: {pred_fraud_rate:.4%} (train fraud rate: {train_fraud_rate:.4%})')

    submission.to_csv(ROOT / 'submission.csv', index=False)
    print(f"Wrote {ROOT / 'submission.csv'}")


if __name__ == '__main__':
    main()
