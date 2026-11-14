import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from src.stack import (
    fit_meta_model, cross_fit_isotonic_calibration, fit_final_calibrator,
    simple_mean_meta, rank_average_meta, logit_average_meta,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
TOP_N_RAW_FEATURES = 10


def load_oof_test_pred():
    oof_dir, test_dir = ROOT / 'oof', ROOT / 'test_preds'
    model_names = sorted(p.stem for p in oof_dir.glob('*.npy'))
    oof = {name: np.load(oof_dir / f'{name}.npy') for name in model_names}
    test_pred = {name: np.load(test_dir / f'{name}.npy') for name in model_names
                 if (test_dir / f'{name}.npy').exists()}
    return oof, test_pred, model_names


def _prepare_feature_frame(df: pd.DataFrame, feat_names: list[str],
                            categories: dict[str, pd.Index] | None = None) -> pd.DataFrame:
    # LightGBM's sklearn wrapper rejects raw object/string dtypes outright; it only
    # accepts int/float/bool or pandas 'category'. Casting to 'category' here (with a
    # categories set shared between train and test) lets LightGBM use its native
    # categorical handling and keeps train-fit and test-predict codes consistent.
    out = df[feat_names].copy()
    for col in out.select_dtypes(include=['object', 'category', 'string']).columns:
        cats = categories[col] if categories else pd.Index(out[col].astype(str).unique())
        out[col] = pd.Categorical(out[col].astype(str), categories=cats)
    return out


def top_raw_features(n: int = TOP_N_RAW_FEATURES) -> list[str]:
    params_path = ROOT / 'params' / 'lightgbm.json'
    train = pd.read_parquet(DATA_DIR / 'train_features.parquet')
    feature_cols = [c for c in train.columns if c not in ('id', 'is_fraud')]
    import lightgbm as lgb
    y = pd.read_parquet(DATA_DIR / 'train_features.parquet')['is_fraud'].astype(int)
    X = train[feature_cols].copy()
    for col in X.select_dtypes(include=['object', 'category', 'string']).columns:
        X[col] = X[col].astype('category').cat.codes
    model = lgb.LGBMClassifier(random_state=42, verbose=-1)
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    return importances.head(n).index.tolist()


def main():
    oof, test_pred, model_names = load_oof_test_pred()
    train = pd.read_parquet(DATA_DIR / 'train_features.parquet')
    test = pd.read_parquet(DATA_DIR / 'test_features.parquet')
    folds_df = pd.read_csv(ROOT / 'folds.csv')
    train = train.merge(folds_df, on='id', how='left')
    y = train['is_fraud'].astype(int)
    folds = train['fold'].values

    oof_matrix = np.column_stack([oof[name] for name in model_names])

    feat_names = top_raw_features()
    cat_cols = train[feat_names].select_dtypes(include=['object', 'category', 'string']).columns
    categories = {col: pd.Index(pd.concat([train[col], test[col]]).astype(str).unique()) for col in cat_cols}
    extra_features = _prepare_feature_frame(train, feat_names, categories)

    stack_oof, stack_model = fit_meta_model(oof_matrix, y, extra_features, folds)
    mean_oof = simple_mean_meta(oof_matrix)
    rank_oof = rank_average_meta(oof_matrix)
    logit_oof = logit_average_meta(oof_matrix)

    candidates = {'stack': stack_oof, 'simple_mean': mean_oof, 'rank_average': rank_oof, 'logit_average': logit_oof}
    for name, pred in candidates.items():
        auc = roc_auc_score(y, pred)
        pr = average_precision_score(y, pred)
        print(f'{name}: OOF AUC={auc:.4f} PR-AUC={pr:.4f}')

    best_name = max(candidates, key=lambda n: roc_auc_score(y, candidates[n]))
    best_oof = candidates[best_name]
    print(f'Best meta-learner: {best_name}')

    brier_before = brier_score_loss(y, best_oof)
    calibrated_oof = cross_fit_isotonic_calibration(best_oof, y, folds)
    brier_after = brier_score_loss(y, calibrated_oof)
    print(f'Brier score before calibration: {brier_before:.5f}, after: {brier_after:.5f}')

    fig, ax = plt.subplots(figsize=(5, 5))
    from sklearn.calibration import calibration_curve
    for label, pred in [('before', best_oof), ('after', calibrated_oof)]:
        frac_pos, mean_pred = calibration_curve(y, pred, n_bins=10, strategy='quantile')
        ax.plot(mean_pred, frac_pos, marker='o', label=label)
    ax.plot([0, 1], [0, 1], 'k--', label='perfect')
    ax.set_xlabel('mean predicted probability'); ax.set_ylabel('fraction of positives')
    ax.set_title('calibration reliability curve'); ax.legend()
    fig.tight_layout()
    (ROOT / 'reports' / 'eda').mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / 'reports' / 'eda' / 'calibration_reliability.png')
    plt.close(fig)

    np.save(ROOT / 'oof' / 'stack.npy', best_oof)
    np.save(ROOT / 'oof' / 'stack_calibrated.npy', calibrated_oof)

    if best_name == 'stack' and all(name in test_pred for name in model_names):
        test_matrix = np.column_stack([test_pred[name] for name in model_names])
        test_meta_X = pd.DataFrame(test_matrix, columns=[f'm{i}' for i in range(test_matrix.shape[1])])
        test_meta_X = pd.concat(
            [test_meta_X, _prepare_feature_frame(test, feat_names, categories).reset_index(drop=True)], axis=1)
        test_stack_pred = stack_model.predict_proba(test_meta_X)[:, 1]
    elif all(name in test_pred for name in model_names):
        test_matrix = np.column_stack([test_pred[name] for name in model_names])
        if best_name == 'simple_mean':
            test_stack_pred = simple_mean_meta(test_matrix)
        elif best_name == 'rank_average':
            test_stack_pred = rank_average_meta(test_matrix)
        else:
            test_stack_pred = logit_average_meta(test_matrix)
    else:
        test_stack_pred = None

    if test_stack_pred is not None:
        np.save(ROOT / 'test_preds' / 'stack.npy', test_stack_pred)
        calibrator = fit_final_calibrator(best_oof, y)
        np.save(ROOT / 'test_preds' / 'stack_calibrated.npy', calibrator.predict(test_stack_pred))

    print(f"Wrote {ROOT / 'oof' / 'stack.npy'}, {ROOT / 'oof' / 'stack_calibrated.npy'}")


if __name__ == '__main__':
    main()
