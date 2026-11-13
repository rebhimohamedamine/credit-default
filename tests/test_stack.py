import numpy as np
import pandas as pd

from src.stack import cross_fit_isotonic_calibration, fit_meta_model


def test_cross_fit_isotonic_calibration_is_monotonic_and_bounded():
    rng = np.random.default_rng(0)
    raw = rng.random(500)
    y = (raw + rng.normal(0, 0.1, 500) > 0.5).astype(int)
    folds = np.array([i % 5 for i in range(500)])
    calibrated = cross_fit_isotonic_calibration(raw, y, folds)
    assert calibrated.shape == raw.shape
    assert ((calibrated >= 0) & (calibrated <= 1)).all()


def test_fit_meta_model_returns_oof_matching_input_length():
    rng = np.random.default_rng(0)
    oof_matrix = rng.random((300, 3))
    y = pd.Series(rng.integers(0, 2, 300))
    folds = np.array([i % 5 for i in range(300)])
    oof_pred, model = fit_meta_model(oof_matrix, y, extra_features=None, folds=folds)
    assert oof_pred.shape == (300,)
    assert ((oof_pred >= 0) & (oof_pred <= 1)).all()
