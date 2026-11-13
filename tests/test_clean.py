import pandas as pd

from src.clean import clean, BOOL_COLS, CAT_COLS


def test_clean_casts_bool_strings_to_0_1():
    df = pd.DataFrame({'used_vpn': ['True', 'False', '1', '0', None]})
    for c in BOOL_COLS:
        if c != 'used_vpn':
            df[c] = 'False'
    for c in CAT_COLS:
        df[c] = 'Online'
    out = clean(df)
    assert list(out['used_vpn'].fillna(-1)) == [1.0, 0.0, 1.0, 0.0, -1]


def test_clean_strips_categorical_whitespace():
    df = pd.DataFrame({c: [' Online '] for c in CAT_COLS})
    for c in BOOL_COLS:
        df[c] = 'False'
    out = clean(df)
    assert out['channel'].iloc[0] == 'Online'


def test_clean_does_not_mutate_input():
    df = pd.DataFrame({c: ['False'] for c in BOOL_COLS})
    for c in CAT_COLS:
        df[c] = 'x'
    clean(df)
    assert df['used_vpn'].iloc[0] == 'False'


def test_clean_replaces_sentinels_with_nan_and_flags_them():
    df = pd.DataFrame({'time_of_day_hour': [5, 25, -1, 12]})
    for c in BOOL_COLS:
        df[c] = 'False'
    for c in CAT_COLS:
        df[c] = 'x'
    out = clean(df)
    assert out['time_of_day_hour_was_sentinel'].tolist() == [0, 1, 1, 0]
    assert out['time_of_day_hour'].isna().sum() == 2
