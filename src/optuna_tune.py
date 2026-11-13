import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score

from src.layer1_classic import fit_predict_oof

SEARCH_SPACES = {
    'lightgbm': lambda trial: {
        'num_leaves': trial.suggest_int('num_leaves', 16, 128),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 200),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
    },
    'xgboost': lambda trial: {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
    },
    'catboost': lambda trial: {
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'iterations': trial.suggest_int('iterations', 100, 500),
    },
    'rf': lambda trial: {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 4, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 50),
    },
    'extratrees': lambda trial: {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 4, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 50),
    },
    'logreg': lambda trial: {
        'C': trial.suggest_float('C', 1e-3, 10.0, log=True),
    },
    'mlp': lambda trial: {
        'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
        'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-1, log=True),
    },
}


def tune_model(model_name: str, X, y, folds: np.ndarray, n_trials: int = 50, seed: int = 42) -> dict:
    if model_name not in SEARCH_SPACES:
        raise ValueError(f'no search space defined for {model_name}')

    X_test_dummy = X.iloc[:1]

    def objective(trial):
        params = SEARCH_SPACES[model_name](trial)
        oof, _ = fit_predict_oof(model_name, X, y, X_test_dummy, folds, params=params)
        return roc_auc_score(y, oof)

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return {'params': study.best_params, 'cv_auc': study.best_value}
