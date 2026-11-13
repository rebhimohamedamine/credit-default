import numpy as np
import pandas as pd

from src.features import add_features, FEATURE_BLOCKS


def test_amount_to_balance_handles_zero_and_negative_balance():
    df = pd.DataFrame({'amount_usd': [100.0, 50.0], 'account_balance_usd': [0.0, -20.0]})
    out = add_features(df, blocks=['amount'])
    assert np.isfinite(out['amount_to_balance']).all()
    assert not out['amount_to_balance'].isna().any()


def test_frequency_features_computed_on_combined_train_and_test_not_train_alone():
    train = pd.DataFrame({'device_type': ['iPhone'] * 8 + ['Android Phone'] * 2})
    test = pd.DataFrame({'device_type': ['Android Phone'] * 10})
    combined = pd.concat([train, test], ignore_index=True)
    out = add_features(combined, blocks=['device'])
    assert abs(out['device_type_freq'].iloc[-1] - 12 / 20) < 1e-9


def test_add_features_block_selection_only_adds_requested_columns():
    df = pd.DataFrame({'amount_usd': [10.0], 'account_balance_usd': [100.0]})
    out = add_features(df, blocks=['amount'])
    for col in FEATURE_BLOCKS['velocity']:
        assert col not in out.columns
