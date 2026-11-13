import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
MODEL_NAMES = ['lightgbm', 'xgboost', 'catboost', 'mlp', 'logreg', 'rf', 'extratrees']


def _load_device_config() -> dict:
    hw_path = ROOT / 'hardware.json'
    if not hw_path.exists():
        return {'cuda_available': False}
    try:
        return json.loads(hw_path.read_text())
    except Exception:
        return {'cuda_available': False}


def build_preprocessor(numeric_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median', add_indicator=True)),
        ('scale', StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    transformers = []
    if numeric_cols:
        transformers.append(('num', numeric_pipe, numeric_cols))
    if cat_cols:
        transformers.append(('cat', cat_pipe, cat_cols))
    return ColumnTransformer(transformers, remainder='drop')


def smoothed_target_encode(train_series: pd.Series, train_y: pd.Series, apply_series: pd.Series,
                             n_inner_folds: int = 5, smoothing: float = 10.0, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import KFold

    global_mean = train_y.mean()
    train_encoded = np.full(len(train_series), np.nan)

    kf = KFold(n_splits=n_inner_folds, shuffle=True, random_state=seed)
    train_series = train_series.reset_index(drop=True)
    train_y = train_y.reset_index(drop=True)
    for inner_tr_idx, inner_val_idx in kf.split(train_series):
        stats = train_y.iloc[inner_tr_idx].groupby(train_series.iloc[inner_tr_idx]).agg(['mean', 'count'])
        smoothed = (stats['mean'] * stats['count'] + global_mean * smoothing) / (stats['count'] + smoothing)
        train_encoded[inner_val_idx] = train_series.iloc[inner_val_idx].map(smoothed).fillna(global_mean).values

    full_stats = train_y.groupby(train_series).agg(['mean', 'count'])
    full_smoothed = (full_stats['mean'] * full_stats['count'] + global_mean * smoothing) / (full_stats['count'] + smoothing)
    apply_encoded = apply_series.map(full_smoothed).fillna(global_mean).values
    return train_encoded, apply_encoded


def _split_col_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    cat_cols = list(X.select_dtypes(include=['object', 'category', 'string']).columns)
    numeric_cols = [c for c in X.columns if c not in cat_cols]
    return numeric_cols, cat_cols


def _fit_predict_fold_tree(model_name, X_train, y_train, X_val, X_test, cat_cols, device_cfg, params):
    X_train, X_val, X_test = X_train.copy(), X_val.copy(), X_test.copy()
    for col in cat_cols:
        enc_train, enc_val = smoothed_target_encode(X_train[col], y_train, X_val[col])
        _, enc_test = smoothed_target_encode(X_train[col], y_train, X_test[col])
        X_train[col], X_val[col], X_test[col] = enc_train, enc_val, enc_test

    if model_name == 'lightgbm':
        import lightgbm as lgb
        device = 'gpu' if device_cfg.get('cuda_available') else 'cpu'
        model = lgb.LGBMClassifier(random_state=42, verbose=-1, device_type=device, **(params or {}))
    elif model_name == 'xgboost':
        import xgboost as xgb
        device = 'cuda' if device_cfg.get('cuda_available') else 'cpu'
        model = xgb.XGBClassifier(random_state=42, device=device, **(params or {}))
    else:
        raise ValueError(model_name)

    try:
        model.fit(X_train, y_train)
    except Exception:
        if model_name == 'lightgbm':
            model = lgb.LGBMClassifier(random_state=42, verbose=-1, device_type='cpu', **(params or {}))
        else:
            model = xgb.XGBClassifier(random_state=42, device='cpu', **(params or {}))
        model.fit(X_train, y_train)

    return (model.predict_proba(X_val)[:, 1], model.predict_proba(X_test)[:, 1])


def _fit_predict_fold_catboost(X_train, y_train, X_val, X_test, cat_cols, device_cfg, params):
    from catboost import CatBoostClassifier

    task_type = 'GPU' if device_cfg.get('cuda_available') else 'CPU'
    model = CatBoostClassifier(random_state=42, verbose=0, task_type=task_type,
                                cat_features=cat_cols, **(params or {}))
    try:
        model.fit(X_train, y_train)
    except Exception:
        model = CatBoostClassifier(random_state=42, verbose=0, task_type='CPU',
                                    cat_features=cat_cols, **(params or {}))
        model.fit(X_train, y_train)
    return (model.predict_proba(X_val)[:, 1], model.predict_proba(X_test)[:, 1])


def _fit_predict_fold_sklearn(model_name, X_train, y_train, X_val, X_test, numeric_cols, cat_cols, params):
    pre = build_preprocessor(numeric_cols, cat_cols)

    if model_name == 'mlp':
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, random_state=42, **(params or {}))
    elif model_name == 'logreg':
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000, random_state=42, **(params or {}))
    elif model_name == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1, **(params or {}))
    elif model_name == 'extratrees':
        from sklearn.ensemble import ExtraTreesClassifier
        clf = ExtraTreesClassifier(class_weight='balanced', bootstrap=False, random_state=42, n_jobs=-1, **(params or {}))
    else:
        raise ValueError(model_name)

    X_train_t = pre.fit_transform(X_train)
    X_val_t = pre.transform(X_val)
    X_test_t = pre.transform(X_test)
    clf.fit(X_train_t, y_train)
    return (clf.predict_proba(X_val_t)[:, 1], clf.predict_proba(X_test_t)[:, 1])


def fit_predict_oof(model_name: str, X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame,
                     folds: np.ndarray, params: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    if model_name not in MODEL_NAMES:
        raise ValueError(f'unknown model_name: {model_name}')

    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    numeric_cols, cat_cols = _split_col_types(X)
    device_cfg = _load_device_config()

    oof = np.zeros(len(X))
    test_preds = np.zeros((len(set(folds)), len(X_test)))

    for i, fold in enumerate(sorted(set(folds))):
        train_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val = X.iloc[val_idx]

        if model_name in ('lightgbm', 'xgboost'):
            val_pred, test_pred = _fit_predict_fold_tree(
                model_name, X_train, y_train, X_val, X_test, cat_cols, device_cfg, params)
        elif model_name == 'catboost':
            val_pred, test_pred = _fit_predict_fold_catboost(
                X_train, y_train, X_val, X_test, cat_cols, device_cfg, params)
        else:
            val_pred, test_pred = _fit_predict_fold_sklearn(
                model_name, X_train, y_train, X_val, X_test, numeric_cols, cat_cols, params)

        oof[val_idx] = val_pred
        test_preds[i] = test_pred

    return oof, test_preds.mean(axis=0)
