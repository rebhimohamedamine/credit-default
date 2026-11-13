import numpy as np
import pandas as pd

BOOL_COLS = [
    'is_foreign_transaction', 'is_new_merchant', 'used_vpn',
    'ip_country_mismatch', 'billing_shipping_mismatch',
    'is_ai_generated_scam_attempt',
]

CAT_COLS = ['merchant_category', 'card_type', 'auth_method', 'channel', 'device_type']

BOOL_MAP = {'true': 1.0, '1': 1.0, '1.0': 1.0, 'false': 0.0, '0': 0.0, '0.0': 0.0}

SENTINEL_RULES = {
    'time_of_day_hour': lambda s: (s < 0) | (s > 23),
    'day_of_week': lambda s: (s < 0) | (s > 6),
    'customer_age': lambda s: (s < 18) | (s > 100),
    'card_age_months': lambda s: s < 0,
    'amount_usd': lambda s: s <= 0,
    'cvv_retry_count': lambda s: s < 0,
    'distance_from_home_km': lambda s: s < 0,
}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in BOOL_COLS:
        if col not in df.columns:
            continue
        mapped = df[col].astype(str).str.strip().str.lower().map(BOOL_MAP)
        df[col] = mapped

    for col in CAT_COLS:
        if col not in df.columns:
            continue
        df[col] = df[col].astype('string').str.strip()

    for col, rule in SENTINEL_RULES.items():
        if col not in df.columns:
            continue
        mask = rule(df[col]) & df[col].notna()
        if mask.any():
            df[f'{col}_was_sentinel'] = mask.astype(int)
            df.loc[mask, col] = np.nan

    return df


def adversarial_validation_auc(train_df: pd.DataFrame, test_df: pd.DataFrame,
                                 feature_cols: list[str], n_splits: int = 5, seed: int = 42) -> float:
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    combined = pd.concat(
        [train_df[feature_cols].assign(is_test=0), test_df[feature_cols].assign(is_test=1)],
        ignore_index=True,
    )
    X = combined[feature_cols].copy()
    for col in X.columns:
        if X[col].dtype.name in ('string', 'object', 'category'):
            X[col] = X[col].astype('category').cat.codes
    y = combined['is_test'].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        model = lgb.LGBMClassifier(n_estimators=200, random_state=seed, verbose=-1)
        model.fit(X.iloc[train_idx], y[train_idx])
        pred = model.predict_proba(X.iloc[val_idx])[:, 1]
        scores.append(roc_auc_score(y[val_idx], pred))
    return float(np.mean(scores))


def leakage_check(train_df: pd.DataFrame, col: str = 'is_ai_generated_scam_attempt',
                   target: str = 'is_fraud') -> dict:
    from sklearn.feature_selection import mutual_info_classif

    valid = train_df[[col, target]].dropna()
    corr = float(valid[col].corr(valid[target]))
    mi = float(mutual_info_classif(
        valid[[col]].astype(float), valid[target].astype(int),
        discrete_features=True, random_state=42,
    )[0])
    return {'corr': corr, 'mutual_info': mi}
