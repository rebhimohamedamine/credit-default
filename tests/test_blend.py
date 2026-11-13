import numpy as np
import pandas as pd

from src.blend import optimize_blend_weights, build_submission


def test_optimize_blend_weights_are_nonnegative_and_sum_to_one():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    oof_dict = {'a': rng.random(500), 'b': rng.random(500)}
    weights = optimize_blend_weights(oof_dict, y)
    assert all(w >= 0 for w in weights.values())
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_build_submission_matches_sample_submission_id_order_exactly():
    test_ids = pd.Series([5, 3, 9])
    sample_sub = pd.DataFrame({'id': [5, 3, 9], 'is_fraud': [0.0, 0.0, 0.0]})
    preds = np.array([0.9, 0.1, 0.4])
    sub = build_submission(test_ids, preds, sample_sub)
    assert sub['id'].tolist() == sample_sub['id'].tolist()
    assert sub.columns.tolist() == sample_sub.columns.tolist()
    assert sub['is_fraud'].between(0, 1).all()
    assert not sub['is_fraud'].isna().any()


def test_build_submission_reorders_to_match_template():
    test_ids = pd.Series([3, 9, 5])
    sample_sub = pd.DataFrame({'id': [5, 3, 9], 'is_fraud': [0.0, 0.0, 0.0]})
    preds = np.array([0.1, 0.4, 0.9])
    sub = build_submission(test_ids, preds, sample_sub)
    assert sub['id'].tolist() == [5, 3, 9]
    assert sub.loc[sub['id'] == 5, 'is_fraud'].iloc[0] == 0.9
