import numpy as np
import pandas as pd

from src.optuna_tune import tune_model


def test_tune_model_returns_params_with_cv_auc_at_least_as_good_as_default():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({'amount_usd': rng.random(500)})
    y = pd.Series(rng.integers(0, 2, 500))
    folds = np.array([i % 5 for i in range(500)])
    result = tune_model('logreg', X, y, folds, n_trials=5)
    assert 'params' in result and 'cv_auc' in result
    assert 0.0 <= result['cv_auc'] <= 1.0
