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


def test_stat2_sqrt_transforms_and_bins():
    df = pd.DataFrame({
        'amount_usd': [4.0, 9.0, 16.0],
        'distance_from_home_km': [16.0, 25.0, 36.0],
        'hours_since_last_txn': [1.0, 4.0, 9.0],
    })
    out = add_features(df, blocks=['stat2'])
    assert list(out['amount_sqrt']) == [2.0, 3.0, 4.0]
    assert list(out['distance_sqrt']) == [4.0, 5.0, 6.0]
    assert list(out['hours_since_last_txn_sqrt']) == [1.0, 2.0, 3.0]
    assert out['amount_bin'].notna().all()
    assert out['distance_bin'].notna().all()


def test_stat2_bins_tolerate_heavily_duplicated_values():
    # 97 rows share the same amount, which would make plain pd.qcut(..., 10) raise on
    # duplicate bin edges without duplicates='drop'.
    df = pd.DataFrame({'amount_usd': [5.0] * 97 + [10.0, 20.0, 30.0]})
    out = add_features(df, blocks=['stat2'])
    assert out['amount_bin'].notna().all()


def test_rare_cat_collapses_categories_below_threshold():
    df = pd.DataFrame({'device_type': ['common'] * 199 + ['rare']})
    out = add_features(df, blocks=['rare_cat'])
    assert out['device_type_rare'].iloc[-1] == 'Other'
    assert (out['device_type_rare'].iloc[:-1] == 'common').all()


def test_rare_cat_leaves_missing_values_as_missing():
    df = pd.DataFrame({'device_type': ['common'] * 10 + [None]})
    out = add_features(df, blocks=['rare_cat'])
    assert pd.isna(out['device_type_rare'].iloc[-1])


def test_rare_cat_threshold_computed_on_combined_train_and_test_not_train_alone():
    # 'z' is 1/200 = 0.5% of train alone (below the 1% threshold), but test adds 50 more
    # 'z' rows, making it 51/250 = 20.4% of the combined frame (above threshold). If the
    # frequency were computed on train alone, 'z' would wrongly collapse to 'Other'.
    train = pd.DataFrame({'device_type': ['common'] * 199 + ['z']})
    test = pd.DataFrame({'device_type': ['z'] * 50})
    combined = pd.concat([train, test], ignore_index=True)
    out = add_features(combined, blocks=['rare_cat'])
    assert (out.loc[out['device_type'] == 'z', 'device_type_rare'] == 'z').all()


def test_group_agg_count_and_stats_per_group():
    df = pd.DataFrame({
        'channel': ['a', 'a', 'a', 'b', 'b'],
        'amount_usd': [10.0, 20.0, 30.0, 5.0, 7.0],
    })
    out = add_features(df, blocks=['group_agg'])
    assert list(out['count_channel']) == [3, 3, 3, 2, 2]
    assert list(out['amount_usd_median_channel']) == [20.0, 20.0, 20.0, 6.0, 6.0]
    assert list(out['amount_usd_min_channel']) == [10.0, 10.0, 10.0, 5.0, 5.0]
    assert list(out['amount_usd_max_channel']) == [30.0, 30.0, 30.0, 7.0, 7.0]
