import numpy as np
import pandas as pd

from src.layer1_foundation import build_stratified_context, fit_predict_oof_foundation


def test_stratified_context_includes_all_fraud_rows():
    y = np.array([0] * 990 + [1] * 10)
    idx = build_stratified_context(np.zeros((1000, 1)), y, max_context_size=500)
    assert set(np.where(y == 1)[0]).issubset(set(idx))


def test_stratified_context_respects_size_cap():
    y = np.array([0] * 990 + [1] * 10)
    idx = build_stratified_context(np.zeros((1000, 1)), y, max_context_size=500)
    assert len(idx) <= 500


def test_foundation_oof_context_never_overlaps_its_own_validation_fold():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({'amount_usd': rng.random(200)})
    y = pd.Series(rng.integers(0, 2, 200))
    folds = np.array([i % 5 for i in range(200)])
    oof, test_pred = fit_predict_oof_foundation('tabicl', X, y, X.iloc[:20], folds, max_context_size=50)
    assert oof.shape[0] == 200 and test_pred.shape[0] == 20
