import pandas as pd

from src.eda import fraud_rate_with_ci


def test_fraud_rate_with_ci_known_values():
    df = pd.DataFrame({'g': ['a'] * 10 + ['b'] * 10,
                        'is_fraud': [1] * 5 + [0] * 5 + [0] * 10})
    out = fraud_rate_with_ci(df, 'g').set_index('g')
    assert out.loc['a', 'n'] == 10
    assert abs(out.loc['a', 'fraud_rate'] - 0.5) < 1e-9
    assert out.loc['a', 'ci_low'] < 0.5 < out.loc['a', 'ci_high']
    assert out.loc['b', 'fraud_rate'] == 0.0
