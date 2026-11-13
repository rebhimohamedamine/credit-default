import numpy as np
import pandas as pd

RED_FLAGS = [
    'is_foreign_transaction', 'is_new_merchant', 'used_vpn',
    'ip_country_mismatch', 'billing_shipping_mismatch', 'is_ai_generated_scam_attempt',
]
REMOTE_CHANNELS = ['Online', 'In-App']
STRONG_AUTH = ['3D Secure', 'OTP']

BLOCK_ORDER = ['amount', 'velocity', 'location', 'auth', 'device', 'merchant', 'profile', 'time', 'scam', 'cross']

FEATURE_BLOCKS = {
    'amount': ['amount_log', 'amount_to_balance', 'balance_after', 'drains_account',
               'is_round_amount', 'small_amount_test', 'amount_vs_cat_mean', 'amount_z_in_cat'],
    'velocity': ['txn_rate_24h', 'burst_flag', 'dormant_flag', 'hours_gap_log',
                 'count_over_gap', 'amount_x_velocity', 'spend_24h_proxy', 'velocity_inconsistent'],
    'location': ['distance_log', 'far_flag', 'speed_proxy', 'geo_risk_count',
                 'foreign_x_ipmismatch', 'vpn_x_ipmismatch', 'foreign_but_near', 'far_but_domestic'],
    'auth': ['cvv_retry_any', 'cvv_retry_high', 'is_remote', 'weak_auth_remote',
             'cvv_x_newmerchant', 'cvv_x_amount', 'auth_channel'],
    'device': ['device_channel', 'device_card', 'device_channel_freq', 'auth_channel_freq',
               'merchant_category_freq', 'device_type_freq', 'card_type_freq',
               'rare_device_channel', 'vpn_x_remote'],
    'merchant': ['merchant_risk_x_new', 'merchant_risk_x_amount', 'merchant_risk_bin',
                 'category_channel', 'newmerchant_x_remote'],
    'profile': ['new_card', 'amount_per_card_month', 'new_card_x_amount', 'has_disputes',
                'disputes_per_card_month', 'senior_flag', 'age_bin', 'balance_log', 'low_balance'],
    'time': ['is_night', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend',
             'night_x_foreign', 'night_x_amount', 'night_x_remote'],
    'scam': ['scam_x_newmerchant', 'scam_x_amount', 'scam_with_strong_auth',
             'scam_x_ship_mismatch', 'ship_mismatch_remote_new'],
    'cross': ['red_flag_count', 'red_flags_x_amount', 'missing_count', 'iso_score'],
}


def _add_amount(df):
    if 'amount_usd' not in df.columns or 'account_balance_usd' not in df.columns:
        return df
    df['amount_log'] = np.log1p(df['amount_usd'].clip(lower=0))
    df['amount_to_balance'] = df['amount_usd'] / (df['account_balance_usd'].abs() + 1)
    df['balance_after'] = df['account_balance_usd'] - df['amount_usd']
    df['drains_account'] = (df['amount_to_balance'] >= 0.8).astype(int)
    df['is_round_amount'] = (df['amount_usd'].round(2) % 10 == 0).astype(int)
    df['small_amount_test'] = (df['amount_usd'] <= df['amount_usd'].quantile(0.05)).astype(int)
    if 'merchant_category' in df.columns:
        g = df.groupby('merchant_category')['amount_usd']
        cat_mean = g.transform('mean')
        cat_std = g.transform('std')
        df['amount_vs_cat_mean'] = df['amount_usd'] / (cat_mean + 1e-6)
        df['amount_z_in_cat'] = (df['amount_usd'] - cat_mean) / (cat_std + 1e-6)
    return df


def _add_velocity(df):
    if 'hours_since_last_txn' not in df.columns or 'txn_count_last_24h' not in df.columns:
        return df
    df['txn_rate_24h'] = df['txn_count_last_24h'] / 24
    df['burst_flag'] = (df['hours_since_last_txn'] < 1).astype(int)
    df['dormant_flag'] = (df['hours_since_last_txn'] > 24 * 30).astype(int)
    df['hours_gap_log'] = np.log1p(df['hours_since_last_txn'].clip(lower=0))
    df['count_over_gap'] = df['txn_count_last_24h'] / (df['hours_since_last_txn'] + 1)
    df['velocity_inconsistent'] = ((df['txn_count_last_24h'] == 0) &
                                    (df['hours_since_last_txn'] < 24)).astype(int)
    if 'amount_usd' in df.columns:
        df['spend_24h_proxy'] = df['amount_usd'] * df['txn_count_last_24h']
        if 'velocity_score' in df.columns:
            df['amount_x_velocity'] = df['amount_usd'] * df['velocity_score']
    return df


def _add_location(df):
    if 'distance_from_home_km' not in df.columns:
        return df
    df['distance_log'] = np.log1p(df['distance_from_home_km'].clip(lower=0))
    df['far_flag'] = (df['distance_from_home_km'] > df['distance_from_home_km'].quantile(0.95)).astype(int)
    if 'hours_since_last_txn' in df.columns:
        df['speed_proxy'] = df['distance_from_home_km'] / (df['hours_since_last_txn'] + 1)
    geo_cols = [c for c in ['is_foreign_transaction', 'used_vpn', 'ip_country_mismatch'] if c in df.columns]
    if geo_cols:
        df['geo_risk_count'] = df[geo_cols].sum(axis=1)
    if {'is_foreign_transaction', 'ip_country_mismatch'}.issubset(df.columns):
        df['foreign_x_ipmismatch'] = df['is_foreign_transaction'] * df['ip_country_mismatch']
    if {'used_vpn', 'ip_country_mismatch'}.issubset(df.columns):
        df['vpn_x_ipmismatch'] = df['used_vpn'] * df['ip_country_mismatch']
    if 'is_foreign_transaction' in df.columns:
        df['foreign_but_near'] = ((df['is_foreign_transaction'] == 1) &
                                   (df['distance_from_home_km'] < 50)).astype(int)
        df['far_but_domestic'] = ((df['is_foreign_transaction'] == 0) &
                                   (df['distance_from_home_km'] > 1000)).astype(int)
    return df


def _add_auth(df):
    if 'cvv_retry_count' not in df.columns:
        return df
    df['cvv_retry_any'] = (df['cvv_retry_count'] >= 1).astype(int)
    df['cvv_retry_high'] = (df['cvv_retry_count'] >= 3).astype(int)
    if 'channel' in df.columns:
        df['is_remote'] = df['channel'].isin(REMOTE_CHANNELS).astype(int)
        if 'auth_method' in df.columns:
            df['weak_auth_remote'] = ((df['is_remote'] == 1) &
                                       (~df['auth_method'].isin(STRONG_AUTH))).astype(int)
            df['auth_channel'] = df['auth_method'].astype(str) + '_' + df['channel'].astype(str)
    if 'is_new_merchant' in df.columns:
        df['cvv_x_newmerchant'] = df['cvv_retry_count'] * df['is_new_merchant']
    if 'amount_log' in df.columns:
        df['cvv_x_amount'] = df['cvv_retry_count'] * df['amount_log']
    return df


def _freq_encode(df, col):
    df[f'{col}_freq'] = df[col].map(df[col].value_counts(normalize=True))


def _add_device(df):
    if 'device_type' not in df.columns:
        return df
    _freq_encode(df, 'device_type')
    if 'channel' in df.columns:
        df['device_channel'] = df['device_type'].astype(str) + '_' + df['channel'].astype(str)
        _freq_encode(df, 'device_channel')
        df['rare_device_channel'] = (df['device_channel_freq'] < 0.01).astype(int)
    if 'card_type' in df.columns:
        df['device_card'] = df['device_type'].astype(str) + '_' + df['card_type'].astype(str)
        _freq_encode(df, 'card_type')
    if 'auth_channel' in df.columns:
        _freq_encode(df, 'auth_channel')
    if 'merchant_category' in df.columns:
        _freq_encode(df, 'merchant_category')
    if {'used_vpn', 'is_remote'}.issubset(df.columns):
        df['vpn_x_remote'] = df['used_vpn'] * df['is_remote']
    return df


def _add_merchant(df):
    if 'merchant_risk_score' not in df.columns:
        return df
    if 'is_new_merchant' in df.columns:
        df['merchant_risk_x_new'] = df['merchant_risk_score'] * df['is_new_merchant']
    if 'amount_log' in df.columns:
        df['merchant_risk_x_amount'] = df['merchant_risk_score'] * df['amount_log']
    df['merchant_risk_bin'] = pd.qcut(df['merchant_risk_score'], 10, labels=False, duplicates='drop')
    if {'merchant_category', 'channel'}.issubset(df.columns):
        df['category_channel'] = df['merchant_category'].astype(str) + '_' + df['channel'].astype(str)
    if {'is_new_merchant', 'is_remote'}.issubset(df.columns):
        df['newmerchant_x_remote'] = df['is_new_merchant'] * df['is_remote']
    return df


def _add_profile(df):
    if 'card_age_months' not in df.columns:
        return df
    df['new_card'] = (df['card_age_months'] <= 3).astype(int)
    if 'amount_usd' in df.columns:
        df['amount_per_card_month'] = df['amount_usd'] / (df['card_age_months'] + 1)
    if 'amount_log' in df.columns:
        df['new_card_x_amount'] = df['new_card'] * df['amount_log']
    if 'prior_disputes' in df.columns:
        df['has_disputes'] = (df['prior_disputes'] > 0).astype(int)
        df['disputes_per_card_month'] = df['prior_disputes'] / (df['card_age_months'] + 1)
    if 'customer_age' in df.columns:
        df['senior_flag'] = (df['customer_age'] >= 65).astype(int)
        df['age_bin'] = pd.cut(df['customer_age'], [0, 24, 34, 44, 54, 64, 120], labels=False)
    if 'account_balance_usd' in df.columns:
        df['balance_log'] = np.sign(df['account_balance_usd']) * np.log1p(df['account_balance_usd'].abs())
        df['low_balance'] = (df['account_balance_usd'] <= df['account_balance_usd'].quantile(0.10)).astype(int)
    return df


def _add_time(df):
    if 'time_of_day_hour' not in df.columns or 'day_of_week' not in df.columns:
        return df
    df['is_night'] = df['time_of_day_hour'].between(0, 5).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['time_of_day_hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['time_of_day_hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    if 'is_foreign_transaction' in df.columns:
        df['night_x_foreign'] = df['is_night'] * df['is_foreign_transaction']
    if 'amount_log' in df.columns:
        df['night_x_amount'] = df['is_night'] * df['amount_log']
    if 'is_remote' in df.columns:
        df['night_x_remote'] = df['is_night'] * df['is_remote']
    return df


def _add_scam(df):
    if 'is_ai_generated_scam_attempt' not in df.columns:
        return df
    if 'is_new_merchant' in df.columns:
        df['scam_x_newmerchant'] = df['is_ai_generated_scam_attempt'] * df['is_new_merchant']
    if 'amount_log' in df.columns:
        df['scam_x_amount'] = df['is_ai_generated_scam_attempt'] * df['amount_log']
    if 'auth_method' in df.columns:
        df['scam_with_strong_auth'] = df['is_ai_generated_scam_attempt'] * df['auth_method'].isin(STRONG_AUTH).astype(int)
    if 'billing_shipping_mismatch' in df.columns:
        df['scam_x_ship_mismatch'] = df['is_ai_generated_scam_attempt'] * df['billing_shipping_mismatch']
        if 'is_remote' in df.columns and 'is_new_merchant' in df.columns:
            df['ship_mismatch_remote_new'] = (df['billing_shipping_mismatch'] *
                                               df['is_remote'] * df['is_new_merchant'])
    return df


def _add_cross(df, base_cols):
    present_flags = [c for c in RED_FLAGS if c in df.columns]
    if present_flags:
        df['red_flag_count'] = df[present_flags].sum(axis=1)
        if 'amount_log' in df.columns:
            df['red_flags_x_amount'] = df['red_flag_count'] * df['amount_log']
    df['missing_count'] = df[base_cols].isna().sum(axis=1)
    for col in base_cols:
        if df[col].isna().any():
            df[f'{col}_isna'] = df[col].isna().astype(int)

    group_cols = [c for c in ['merchant_category', 'channel', 'device_type', 'card_type', 'age_bin'] if c in df.columns]
    value_cols = [c for c in ['amount_usd', 'distance_from_home_km', 'velocity_score', 'merchant_risk_score'] if c in df.columns]
    for gcol in group_cols:
        for vcol in value_cols:
            g = df.groupby(gcol)[vcol]
            mean, std = g.transform('mean'), g.transform('std')
            df[f'{vcol}_rel_{gcol}'] = df[vcol] / (mean + 1e-6)
            df[f'{vcol}_z_{gcol}'] = (df[vcol] - mean) / (std + 1e-6)

    numeric_for_iso = df.select_dtypes(include=[np.number]).drop(columns=['is_fraud', 'id'], errors='ignore')
    numeric_for_iso = numeric_for_iso.fillna(numeric_for_iso.median())
    from sklearn.ensemble import IsolationForest
    iso = IsolationForest(n_estimators=300, random_state=42, n_jobs=-1)
    df['iso_score'] = -iso.fit(numeric_for_iso).score_samples(numeric_for_iso)
    return df


_BLOCK_FUNCS = {
    'amount': _add_amount,
    'velocity': _add_velocity,
    'location': _add_location,
    'auth': _add_auth,
    'device': _add_device,
    'merchant': _add_merchant,
    'profile': _add_profile,
    'time': _add_time,
    'scam': _add_scam,
}


def add_features(df: pd.DataFrame, blocks: list[str] | None = None) -> pd.DataFrame:
    if blocks is None:
        blocks = list(BLOCK_ORDER)
    blocks_set = set(blocks)
    base_cols = list(df.columns)
    df = df.copy()
    for name in BLOCK_ORDER:
        if name == 'cross':
            continue
        if name in blocks_set:
            df = _BLOCK_FUNCS[name](df)
    if 'cross' in blocks_set:
        df = _add_cross(df, base_cols)
    return df


def add_polynomial_features(df: pd.DataFrame, top_features: list[str]) -> pd.DataFrame:
    from itertools import combinations
    df = df.copy()
    for a, b in combinations(top_features, 2):
        if a in df.columns and b in df.columns:
            df[f'poly_{a}_x_{b}'] = df[a] * df[b]
    return df


def write_folds(train_df: pd.DataFrame, path: str = 'folds.csv', n_splits: int = 5, seed: int = 42,
                 id_col: str = 'id', target_col: str = 'is_fraud') -> pd.DataFrame:
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_of = np.full(len(train_df), -1, dtype=int)
    for fold, (_, val_idx) in enumerate(skf.split(train_df, train_df[target_col])):
        fold_of[val_idx] = fold
    out = pd.DataFrame({id_col: train_df[id_col].values, 'fold': fold_of})
    out.to_csv(path, index=False)
    return out
