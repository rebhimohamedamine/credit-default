import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.clean import clean, adversarial_validation_auc, leakage_check, CAT_COLS

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'


def describe_raw(df: pd.DataFrame) -> list[str]:
    lines = []
    for col in CAT_COLS:
        lines.append(f'### value_counts: {col}')
        lines.append(df[col].value_counts(dropna=False).to_string())
        lines.append('')
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    lines.append('### numeric min/max')
    lines.append(df[numeric_cols].agg(['min', 'max']).to_string())
    lines.append('')
    return lines


def check_duplicates(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[str]:
    lines = ['### duplicates']
    lines.append(f"duplicated id in train: {train_df['id'].duplicated().sum()}")
    lines.append(f"duplicated id in test: {test_df['id'].duplicated().sum()}")
    lines.append(f"fully duplicated rows in train: {train_df.duplicated().sum()}")
    overlap = set(train_df['id']) & set(test_df['id'])
    lines.append(f"id overlap between train and test: {len(overlap)}")
    lines.append('')
    return lines


def check_semantics(train_df: pd.DataFrame) -> list[str]:
    lines = ['### semantics checks']
    if {'amount_usd', 'account_balance_usd'}.issubset(train_df.columns):
        exceeds = (train_df['amount_usd'] > train_df['account_balance_usd']).mean()
        lines.append(f"amount_usd > account_balance_usd in {exceeds:.2%} of rows "
                      f"(high -> balance_usd likely recorded BEFORE the transaction)")
    if {'day_of_week', 'is_fraud'}.issubset(train_df.columns):
        rate_by_day = train_df.groupby('day_of_week')['is_fraud'].mean()
        lines.append('fraud rate by day_of_week code:')
        lines.append(rate_by_day.to_string())
    lines.append('')
    return lines


def main():
    train_raw = pd.read_csv(DATA_DIR / 'train.csv')
    test_raw = pd.read_csv(DATA_DIR / 'test.csv')

    report = ['# Cleaning summary', '']
    report += describe_raw(train_raw)
    report += check_duplicates(train_raw, test_raw)
    report += check_semantics(train_raw)

    train_clean = clean(train_raw)
    test_clean = clean(test_raw)

    feature_cols = [c for c in test_clean.columns if c != 'id']
    auc = adversarial_validation_auc(train_clean, test_clean, feature_cols)
    report.append(f'### adversarial validation AUC (train vs test): {auc:.4f}')
    report.append('(AUC well above 0.5 => features that drift between train and test; investigate before using them.)')
    report.append('')

    leak = leakage_check(train_clean)
    report.append(f"### is_ai_generated_scam_attempt vs is_fraud: corr={leak['corr']:.4f}, "
                  f"mutual_info={leak['mutual_info']:.4f}")
    report.append('')

    (ROOT / 'reports').mkdir(exist_ok=True)
    (ROOT / 'reports' / 'cleaning_summary.md').write_text('\n'.join(report))

    train_clean.to_parquet(DATA_DIR / 'train_clean.parquet', index=False)
    test_clean.to_parquet(DATA_DIR / 'test_clean.parquet', index=False)

    print(f"Wrote {DATA_DIR / 'train_clean.parquet'} ({len(train_clean)} rows)")
    print(f"Wrote {DATA_DIR / 'test_clean.parquet'} ({len(test_clean)} rows)")
    print(f"Wrote {ROOT / 'reports' / 'cleaning_summary.md'}")


if __name__ == '__main__':
    main()
