import importlib.util
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from src.layer1_foundation import MAX_CONTEXT_SIZE, fit_predict_oof_foundation
from src.features import FEATURE_BLOCKS

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'

RAW_COLS = [
    'amount_usd', 'hours_since_last_txn', 'txn_count_last_24h', 'distance_from_home_km',
    'card_age_months', 'customer_age', 'account_balance_usd', 'cvv_retry_count',
    'velocity_score', 'time_of_day_hour', 'day_of_week', 'merchant_risk_score',
    'prior_disputes', 'merchant_category', 'card_type', 'auth_method', 'channel',
    'device_type', 'is_foreign_transaction', 'is_new_merchant', 'used_vpn',
    'ip_country_mismatch', 'billing_shipping_mismatch', 'is_ai_generated_scam_attempt',
]


def log_experiment(exp_rows, exp_id, change, feature_set, cv_auc, cv_pr, runtime_s, kept):
    exp_rows.append({
        'exp_id': exp_id, 'change': change, 'feature_set': feature_set, 'params': '{}',
        'cv_auc_mean': cv_auc, 'cv_auc_std': np.nan, 'cv_pr_auc': cv_pr,
        'runtime_s': round(runtime_s, 1), 'kept': kept,
    })


def available_foundation_models() -> list[str]:
    models = []
    if importlib.util.find_spec('tabicl'):
        models.append('tabicl')
    if not importlib.util.find_spec('tabpfn'):
        print('tabpfn not installed - skipping it. See REPORT.md for why.')
    elif not os.environ.get('TABPFN_TOKEN'):
        # TabPFN weights are gated behind a Prior Labs account: first use needs a
        # browser login + license acceptance, then a TABPFN_TOKEN to skip that prompt.
        # Without it, .fit() blocks forever waiting for interactive input on a
        # non-interactive run (e.g. a Kaggle notebook), so skip rather than hang.
        print('tabpfn installed but TABPFN_TOKEN is not set - skipping it to avoid '
              'blocking on the interactive license prompt. See '
              'https://docs.priorlabs.ai/how-to-access-gated-models to authorize, '
              'then set TABPFN_TOKEN in the environment.')
    else:
        models.append('tabpfn')
    return models


def main():
    train = pd.read_parquet(DATA_DIR / 'train_features.parquet')
    test = pd.read_parquet(DATA_DIR / 'test_features.parquet')
    folds_df = pd.read_csv(ROOT / 'folds.csv')
    train = train.merge(folds_df, on='id', how='left')

    kept_blocks = set()
    exp_path = ROOT / 'experiments.csv'
    if exp_path.exists():
        exp_df = pd.read_csv(exp_path)
        for _, row in exp_df[exp_df['exp_id'].str.startswith('block_', na=False)].iterrows():
            if row['kept']:
                kept_blocks.add(row['exp_id'].replace('block_', ''))
    engineered_cols = [c for block in kept_blocks for c in FEATURE_BLOCKS.get(block, []) if c in test.columns]

    y = train['is_fraud'].astype(int)
    folds = train['fold'].values

    exp_rows = []
    for model_name in available_foundation_models():
        best = None
        for variant_name, cols in [('raw_only', RAW_COLS), ('raw_plus_engineered', RAW_COLS + engineered_cols)]:
            cols = [c for c in cols if c in train.columns]
            start = time.time()
            oof, test_pred = fit_predict_oof_foundation(
                model_name, train[cols], y, test[cols], folds,
                max_context_size=MAX_CONTEXT_SIZE.get(model_name))
            elapsed = time.time() - start
            auc = roc_auc_score(y, oof)
            pr = average_precision_score(y, oof)
            print(f'{model_name} [{variant_name}]: OOF AUC={auc:.4f} PR-AUC={pr:.4f} ({elapsed:.1f}s)')
            log_experiment(exp_rows, f'{model_name}_{variant_name}', f'{model_name} input variant',
                            variant_name, auc, pr, elapsed, kept=False)
            if best is None or auc > best[0]:
                best = (auc, oof, test_pred, variant_name)

        best_auc, best_oof, best_test_pred, best_variant = best
        for row in exp_rows:
            if row['exp_id'] == f'{model_name}_{best_variant}':
                row['kept'] = True

        np.save(ROOT / 'oof' / f'{model_name}.npy', best_oof)
        np.save(ROOT / 'test_preds' / f'{model_name}.npy', best_test_pred)
        print(f'{model_name}: kept variant={best_variant} (AUC={best_auc:.4f})')

    exp_df = pd.DataFrame(exp_rows)
    if exp_path.exists():
        exp_df = pd.concat([pd.read_csv(exp_path), exp_df], ignore_index=True)
    exp_df.to_csv(exp_path, index=False)
    print(f'Wrote {exp_path}')


if __name__ == '__main__':
    main()
