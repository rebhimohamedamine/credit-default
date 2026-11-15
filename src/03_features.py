import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.clean import CAT_COLS
from src.features import (add_features, write_folds, BLOCK_ORDER, FEATURE_BLOCKS,
                           ABLATION_BLOCKS, add_target_encoding, filter_to_kept_blocks)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'


def build_features():
    train = pd.read_parquet(DATA_DIR / 'train_clean.parquet')
    test = pd.read_parquet(DATA_DIR / 'test_clean.parquet')

    folds_df = write_folds(train, path=str(ROOT / 'folds.csv'))
    train, test = add_target_encoding(train, test, folds_df['fold'].values, CAT_COLS)

    train = train.assign(_is_test=0)
    test = test.assign(_is_test=1, is_fraud=np.nan)
    combined = pd.concat([train, test], ignore_index=True)

    combined = add_features(combined, blocks=list(BLOCK_ORDER))

    train_out = combined[combined['_is_test'] == 0].drop(columns=['_is_test']).reset_index(drop=True)
    test_out = combined[combined['_is_test'] == 1].drop(columns=['_is_test', 'is_fraud']).reset_index(drop=True)

    # Digit-key columns (digit_keys block) need their own, heavier-smoothed target encoding
    # (synthetic-data decimal-digit groups are tiny -- many are singleton-supported), so this
    # runs as a second OOF pass after the block pipeline rather than inside add_features.
    train_out, test_out = add_target_encoding(train_out, test_out, folds_df['fold'].values,
                                               FEATURE_BLOCKS['digit_keys'], smoothing=100.0)

    return train_out, test_out


def write_feature_parquets(train_out, test_out):
    train_out.to_parquet(DATA_DIR / 'train_features.parquet', index=False)
    test_out.to_parquet(DATA_DIR / 'test_features.parquet', index=False)
    print(f'Wrote {DATA_DIR / "train_features.parquet"} ({len(train_out)} rows, {train_out.shape[1]} cols)')
    print(f'Wrote {DATA_DIR / "test_features.parquet"} ({len(test_out)} rows, {test_out.shape[1]} cols)')


# Default-param LightGBM trees are deep enough to overfit this dataset's synthetic
# generation noise (independent write-ups on this exact competition measured ~0.002 AUC
# lost to deeper trees), which can swamp a genuinely small feature's signal in a single-fit
# ablation check. Capping depth/leaves here is the test harness, not the final model config.
SHALLOW_PARAMS = {'max_depth': 4, 'num_leaves': 8}


def _cv_auc_pr(X, y, folds, params=None):
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, average_precision_score

    fold_aucs, fold_prs = [], []
    for fold in sorted(set(folds)):
        tr_idx, va_idx = folds != fold, folds == fold
        model = lgb.LGBMClassifier(random_state=42, verbose=-1, **{**SHALLOW_PARAMS, **(params or {})})
        model.fit(X.loc[tr_idx], y.loc[tr_idx])
        pred = model.predict_proba(X.loc[va_idx])[:, 1]
        fold_aucs.append(roc_auc_score(y.loc[va_idx], pred))
        fold_prs.append(average_precision_score(y.loc[va_idx], pred))
    return np.array(fold_aucs), np.array(fold_prs)


def _encode_for_lightgbm(df):
    df = df.copy()
    for col in df.select_dtypes(include=['string', 'object', 'category']).columns:
        df[col] = df[col].astype('category').cat.codes
    return df


def run_ablation(train_df):
    folds_df = pd.read_csv(ROOT / 'folds.csv')
    train_df = train_df.merge(folds_df, on='id', how='left')
    y = train_df['is_fraud'].astype(int)
    folds = train_df['fold'].values
    raw_cols = [c for c in train_df.columns
                if c not in {'id', 'is_fraud', 'fold'} and
                not any(c in cols for cols in FEATURE_BLOCKS.values())]

    exp_rows = []
    kept_blocks = []

    def log_experiment(exp_id, change, feature_cols, kept, start_time, aucs, prs):
        exp_rows.append({
            'exp_id': exp_id, 'change': change, 'feature_set': ','.join(feature_cols[:5]) + '...',
            'params': 'default', 'cv_auc_mean': aucs.mean(), 'cv_auc_std': aucs.std(),
            'cv_pr_auc': prs.mean(), 'runtime_s': round(time.time() - start_time, 1), 'kept': kept,
        })

    start = time.time()
    baseline_aucs, baseline_prs = _cv_auc_pr(_encode_for_lightgbm(train_df[raw_cols]), y, folds)
    log_experiment('baseline', 'raw cleaned columns only', raw_cols, True, start, baseline_aucs, baseline_prs)
    print(f'baseline AUC {baseline_aucs.mean():.4f} +- {baseline_aucs.std():.4f}')

    for block in ABLATION_BLOCKS:
        block_cols = [c for c in FEATURE_BLOCKS[block] if c in train_df.columns]
        if not block_cols:
            continue
        start = time.time()
        feature_cols = raw_cols + block_cols
        aucs, prs = _cv_auc_pr(_encode_for_lightgbm(train_df[feature_cols]), y, folds)
        delta = aucs - baseline_aucs
        kept = bool((delta > 0).sum() >= 4 and delta.mean() > 0)
        if kept:
            kept_blocks.append(block)
        log_experiment(f'block_{block}', f'add {block} block', feature_cols, kept, start, aucs, prs)
        print(f'{block}: AUC {aucs.mean():.4f} (delta {delta.mean():+.5f}), kept={kept}')

    exp_df = pd.DataFrame(exp_rows)
    exp_path = ROOT / 'experiments.csv'
    if exp_path.exists():
        exp_df = pd.concat([pd.read_csv(exp_path), exp_df], ignore_index=True)
    exp_df.to_csv(exp_path, index=False)
    print(f'Wrote {exp_path}')
    return kept_blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablate', action='store_true')
    args = parser.parse_args()

    train_out, test_out = build_features()
    if args.ablate:
        kept_blocks = run_ablation(train_out)
        dropped_blocks = [b for b in ABLATION_BLOCKS if b not in kept_blocks]
        if dropped_blocks:
            print(f'Dropping blocks that failed the ablation gate: {dropped_blocks}')
        train_out = filter_to_kept_blocks(train_out, kept_blocks)
        test_out = filter_to_kept_blocks(test_out, kept_blocks)

    write_feature_parquets(train_out, test_out)


if __name__ == '__main__':
    main()
