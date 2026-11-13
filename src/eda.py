import numpy as np
import pandas as pd


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = successes / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    margin = (z * np.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def fraud_rate_with_ci(df: pd.DataFrame, group_col: str, target_col: str = 'is_fraud') -> pd.DataFrame:
    grouped = df.groupby(group_col)[target_col].agg(['sum', 'count'])
    rows = []
    for group, (successes, n) in grouped[['sum', 'count']].itertuples(index=True, name=None):
        ci_low, ci_high = _wilson_ci(int(successes), int(n))
        rows.append({
            group_col: group,
            'n': int(n),
            'fraud_rate': successes / n if n else np.nan,
            'ci_low': ci_low,
            'ci_high': ci_high,
        })
    return pd.DataFrame(rows)


def plot_two_panel(df: pd.DataFrame, group_col: str, target_col: str, out_path, title: str = None):
    import matplotlib.pyplot as plt

    stats = fraud_rate_with_ci(df, group_col, target_col).sort_values(group_col)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(stats[group_col].astype(str), stats['n'])
    ax1.set_title('count'); ax1.tick_params(axis='x', rotation=45)
    ax2.bar(stats[group_col].astype(str), stats['fraud_rate'],
            yerr=[stats['fraud_rate'] - stats['ci_low'], stats['ci_high'] - stats['fraud_rate']])
    ax2.set_title('fraud rate (95% Wilson CI)'); ax2.tick_params(axis='x', rotation=45)
    fig.suptitle(title or group_col)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fraud_rate_by_decile(df: pd.DataFrame, numeric_col: str, target_col: str = 'is_fraud', q: int = 10) -> pd.DataFrame:
    valid = df[[numeric_col, target_col]].dropna()
    valid = valid.assign(_decile=pd.qcut(valid[numeric_col], q, labels=False, duplicates='drop'))
    return fraud_rate_with_ci(valid.rename(columns={'_decile': numeric_col + '_decile'}),
                               numeric_col + '_decile', target_col)


def pair_fraud_rate_heatmap(df: pd.DataFrame, col_a: str, col_b: str, target_col: str = 'is_fraud') -> pd.DataFrame:
    return df.pivot_table(index=col_a, columns=col_b, values=target_col, aggfunc='mean')


def missingness_table(df: pd.DataFrame, target_col: str = 'is_fraud') -> pd.DataFrame:
    rows = []
    for col in df.columns:
        if col == target_col:
            continue
        missing = df[col].isna()
        pct_missing = missing.mean()
        fraud_rate_missing = df.loc[missing, target_col].mean() if missing.any() else np.nan
        fraud_rate_present = df.loc[~missing, target_col].mean()
        rows.append({'column': col, 'pct_missing': pct_missing,
                     'fraud_rate_missing': fraud_rate_missing, 'fraud_rate_present': fraud_rate_present})
    return pd.DataFrame(rows)


def mutual_info_with_target(df: pd.DataFrame, numeric_cols: list[str], target_col: str = 'is_fraud') -> pd.Series:
    from sklearn.feature_selection import mutual_info_classif

    valid = df[numeric_cols + [target_col]].dropna()
    mi = mutual_info_classif(valid[numeric_cols], valid[target_col].astype(int), random_state=42)
    return pd.Series(mi, index=numeric_cols).sort_values(ascending=False)

