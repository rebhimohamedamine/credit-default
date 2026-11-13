import numpy as np
import pandas as pd

from src.layer1_classic import build_preprocessor, fit_predict_oof


def test_logreg_preprocessor_has_no_nan_after_transform():
    df = pd.DataFrame({'amount_usd': [1.0, np.nan, 3.0], 'channel': ['POS', 'Online', None]})
    pre = build_preprocessor(numeric_cols=['amount_usd'], cat_cols=['channel'])
    out = pre.fit_transform(df)
    assert not np.isnan(out).any()


def test_fit_predict_oof_returns_arrays_matching_input_length():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({'amount_usd': rng.random(100)})
    y = pd.Series((X['amount_usd'] > 0.5).astype(int))
    Xte = pd.DataFrame({'amount_usd': rng.random(20)})
    folds = rng.integers(0, 5, size=100)
    oof, test_pred = fit_predict_oof('logreg', X, y, Xte, folds)
    assert oof.shape[0] == 100 and test_pred.shape[0] == 20
    assert np.isfinite(oof).all() and np.isfinite(test_pred).all()


def test_fit_predict_oof_never_trains_fold_on_its_own_validation_rows():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({'amount_usd': np.arange(50, dtype=float)})
    y = pd.Series(rng.integers(0, 2, 50))
    folds = np.array([i % 5 for i in range(50)])
    oof, _ = fit_predict_oof('logreg', X, y, X.iloc[:10], folds)
    assert len(oof) == 50
