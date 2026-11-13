import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.eda import (
    fraud_rate_with_ci, plot_two_panel, fraud_rate_by_decile,
    pair_fraud_rate_heatmap, missingness_table, mutual_info_with_target,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
EDA_DIR = ROOT / 'reports' / 'eda'

HYPOTHESIS_GROUPS = [
    'is_foreign_transaction', 'used_vpn', 'is_new_merchant', 'ip_country_mismatch',
    'billing_shipping_mismatch', 'is_ai_generated_scam_attempt',
    'channel', 'device_type', 'auth_method', 'merchant_category', 'card_type',
]

PAIR_HEATMAPS = [
    ('channel', 'merchant_category'),
    ('device_type', 'channel'),
    ('auth_method', 'channel'),
    ('is_foreign_transaction', 'used_vpn'),
]


def main():
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA_DIR / 'train_clean.parquet')

    # class balance
    fig, ax = plt.subplots(figsize=(4, 4))
    df['is_fraud'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title(f"class balance (fraud rate = {df['is_fraud'].mean():.4%})")
    fig.tight_layout()
    fig.savefig(EDA_DIR / 'class_balance.png')
    plt.close(fig)

    # two-panel plots for each categorical/boolean hypothesis
    for col in HYPOTHESIS_GROUPS:
        if col in df.columns:
            plot_two_panel(df, col, 'is_fraud', EDA_DIR / f'hypothesis_{col}.png', title=col)

    # numeric columns: histogram by class + fraud rate by decile
    numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(['id', 'is_fraud'])
    for col in numeric_cols:
        if df[col].nunique() < 3:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        for cls, label in [(0, 'legit'), (1, 'fraud')]:
            vals = df.loc[df['is_fraud'] == cls, col].dropna()
            ax.hist(vals, bins=50, alpha=0.5, label=label, density=True, log=True)
        ax.set_title(f'{col} by class'); ax.legend()
        fig.tight_layout()
        fig.savefig(EDA_DIR / f'hist_{col}.png')
        plt.close(fig)

        try:
            decile_stats = fraud_rate_by_decile(df, col)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.errorbar(decile_stats.iloc[:, 0], decile_stats['fraud_rate'],
                        yerr=[decile_stats['fraud_rate'] - decile_stats['ci_low'],
                              decile_stats['ci_high'] - decile_stats['fraud_rate']],
                        fmt='o-')
            ax.set_title(f'fraud rate by {col} decile')
            fig.tight_layout()
            fig.savefig(EDA_DIR / f'decile_{col}.png')
            plt.close(fig)
        except ValueError:
            pass

    # hour / day
    if {'time_of_day_hour', 'day_of_week'}.issubset(df.columns):
        plot_two_panel(df, 'time_of_day_hour', 'is_fraud', EDA_DIR / 'hour_of_day.png', title='time_of_day_hour')
        plot_two_panel(df, 'day_of_week', 'is_fraud', EDA_DIR / 'day_of_week.png', title='day_of_week')
        heat = pair_fraud_rate_heatmap(df, 'day_of_week', 'time_of_day_hour')
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(heat, cmap='rocket_r', ax=ax)
        ax.set_title('fraud rate: day_of_week x time_of_day_hour')
        fig.tight_layout()
        fig.savefig(EDA_DIR / 'heatmap_hour_day.png')
        plt.close(fig)

    # pair heatmaps
    for a, b in PAIR_HEATMAPS:
        if {a, b}.issubset(df.columns):
            heat = pair_fraud_rate_heatmap(df, a, b)
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(heat, cmap='rocket_r', ax=ax, annot=False)
            ax.set_title(f'fraud rate: {a} x {b}')
            fig.tight_layout()
            fig.savefig(EDA_DIR / f'pair_{a}_x_{b}.png')
            plt.close(fig)

    # missingness
    miss = missingness_table(df)
    miss.to_csv(EDA_DIR / 'missingness.csv', index=False)

    # correlation + mutual information
    corr = df[numeric_cols].corr(method='spearman')
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, cmap='coolwarm', center=0, ax=ax)
    fig.tight_layout()
    fig.savefig(EDA_DIR / 'spearman_correlation.png')
    plt.close(fig)

    mi = mutual_info_with_target(df, list(numeric_cols))
    fig, ax = plt.subplots(figsize=(8, 10))
    mi.plot(kind='barh', ax=ax)
    ax.set_title('mutual information with is_fraud')
    fig.tight_layout()
    fig.savefig(EDA_DIR / 'mutual_information.png')
    plt.close(fig)
    mi.to_csv(EDA_DIR / 'mutual_information.csv')

    print(f'Wrote figures and tables to {EDA_DIR}')


if __name__ == '__main__':
    main()
