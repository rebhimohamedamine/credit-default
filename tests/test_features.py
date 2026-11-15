import numpy as np
import pandas as pd

from src.features import add_features, add_target_encoding, filter_to_kept_blocks, FEATURE_BLOCKS


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


def test_add_target_encoding_oof_excludes_own_fold():
    train = pd.DataFrame({
        'cat': ['a', 'a', 'a', 'b', 'b', 'b'],
        'is_fraud': [1, 1, 0, 0, 0, 1],
    })
    test = pd.DataFrame({'cat': ['a', 'b']})
    folds = np.array([0, 0, 1, 0, 1, 1])
    train_out, _ = add_target_encoding(train, test, folds, cat_cols=['cat'], smoothing=0.0)
    assert train_out.loc[2, 'cat_target_enc'] == 1.0   # fold-1 'a' row encoded from fold-0 'a' rows only (mean 1.0)
    assert train_out.loc[3, 'cat_target_enc'] == 0.5   # fold-0 'b' row encoded from fold-1 'b' rows only (mean 0.5)


def test_add_target_encoding_unseen_test_category_falls_back_to_global_mean():
    train = pd.DataFrame({'cat': ['a', 'a', 'b', 'b'], 'is_fraud': [1, 0, 0, 0]})
    test = pd.DataFrame({'cat': ['unseen']})
    folds = np.array([0, 1, 0, 1])
    _, test_out = add_target_encoding(train, test, folds, cat_cols=['cat'], smoothing=0.0)
    assert test_out['cat_target_enc'].iloc[0] == train['is_fraud'].mean()


def test_filter_to_kept_blocks_drops_columns_of_blocks_not_kept():
    df = pd.DataFrame({'amount_usd': [1.0], 'amount_sqrt': [1.0], 'amount_bin': [0]})
    out = filter_to_kept_blocks(df, kept_blocks=[])
    assert 'amount_sqrt' not in out.columns
    assert 'amount_bin' not in out.columns
    assert 'amount_usd' in out.columns  # raw column, not part of any block, always kept


def test_filter_to_kept_blocks_keeps_columns_of_kept_blocks():
    df = pd.DataFrame({'amount_sqrt': [1.0], 'amount_bin': [0]})
    out = filter_to_kept_blocks(df, kept_blocks=['stat2'])
    assert 'amount_sqrt' in out.columns
    assert 'amount_bin' in out.columns


def test_cross_block_declares_all_its_group_stat_columns():
    df = pd.DataFrame({
        'merchant_category': ['a', 'b'],
        'channel': ['x', 'y'],
        'device_type': ['d1', 'd2'],
        'card_type': ['c1', 'c2'],
        'amount_usd': [10.0, 20.0],
        'distance_from_home_km': [1.0, 2.0],
        'velocity_score': [0.1, 0.2],
        'merchant_risk_score': [0.5, 0.6],
    })
    out = add_features(df, blocks=['cross'])
    added = set(out.columns) - set(df.columns)
    undeclared = added - set(FEATURE_BLOCKS['cross'])
    assert not undeclared, f'cross produced undeclared columns: {undeclared}'


def test_digit_keys_extract_cents_tenths_and_leading_digit():
    df = pd.DataFrame({'amount_usd': [42.86, 7.05, 0.0, -12.34]})
    out = add_features(df, blocks=['digit_keys'])
    assert list(out['amount_usd_cents']) == [86, 5, 0, 34]
    assert list(out['amount_usd_tenths']) == [8, 0, 0, 3]
    assert list(out['amount_usd_leading_digit']) == [4, 7, 0, 1]


def test_digit_keys_only_adds_declared_columns():
    df = pd.DataFrame({'amount_usd': [42.86], 'merchant_risk_score': [0.5]})
    out = add_features(df, blocks=['digit_keys'])
    added = set(out.columns) - set(df.columns)
    assert added == set(FEATURE_BLOCKS['digit_keys']) & added
    assert 'amount_usd_cents' in added and 'merchant_risk_score_cents' in added


def test_digit_decomp_target_enc_columns_are_declared_for_every_digit_key():
    from src.features import _digit_key_cols
    assert FEATURE_BLOCKS['digit_decomp'] == [f'{c}_target_enc' for c in _digit_key_cols()]


def test_pair_interact_builds_all_six_combo_keys():
    df = pd.DataFrame({
        'merchant_category': ['Groceries'],
        'channel': ['POS'],
        'auth_method': ['PIN'],
        'card_type': ['Visa'],
    })
    out = add_features(df, blocks=['pair_interact'])
    assert out['merchant_category_x_channel'].iloc[0] == 'Groceries_POS'
    assert out['auth_method_x_card_type'].iloc[0] == 'PIN_Visa'
    added = set(out.columns) - set(df.columns)
    assert added == set(FEATURE_BLOCKS['pair_interact'])


def test_pair_interact_enc_target_enc_columns_are_declared_for_every_pair_key():
    from src.features import _pair_interact_cols
    assert FEATURE_BLOCKS['pair_interact_enc'] == [f'{c}_target_enc' for c in _pair_interact_cols()]


def test_cross_missing_count_ignores_id_and_is_fraud():
    df = pd.DataFrame({
        'id': [1, 2],
        'is_fraud': [1.0, np.nan],  # row 2 simulates a test-set row concatenated with train
        'amount_usd': [10.0, 20.0],
    })
    out = add_features(df, blocks=['cross'])
    assert list(out['missing_count']) == [0, 0]
    assert 'is_fraud_isna' not in out.columns
