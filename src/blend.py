import numpy as np
import pandas as pd


def optimize_blend_weights(oof_dict: dict[str, np.ndarray], y) -> dict[str, float]:
    from scipy.optimize import minimize
    from sklearn.metrics import roc_auc_score

    names = list(oof_dict.keys())
    matrix = np.column_stack([np.asarray(oof_dict[n]) for n in names])
    ranks = pd.DataFrame(matrix).rank(pct=True).values
    y = np.asarray(y)

    def neg_auc(raw_weights):
        w = np.exp(raw_weights)
        w = w / w.sum()
        pred = ranks @ w
        return -roc_auc_score(y, pred)

    x0 = np.zeros(len(names))
    result = minimize(neg_auc, x0, method='Nelder-Mead')
    w = np.exp(result.x)
    w = w / w.sum()
    return dict(zip(names, w.tolist()))


def build_submission(test_ids, preds: np.ndarray, sample_sub_template: pd.DataFrame) -> pd.DataFrame:
    id_col, target_col = sample_sub_template.columns[0], sample_sub_template.columns[1]
    preds = np.clip(np.asarray(preds, dtype=float), 0.0, 1.0)

    assert len(preds) == len(test_ids), 'prediction length does not match test id count'

    sub = pd.DataFrame({id_col: pd.Series(test_ids).values, target_col: preds})
    sub = sub.set_index(id_col).loc[sample_sub_template[id_col].values].reset_index()

    assert sub[id_col].tolist() == sample_sub_template[id_col].tolist(), 'submission id order mismatch'
    assert not sub[target_col].isna().any(), 'NaN found in submission predictions'
    assert sub[target_col].between(0, 1).all(), 'prediction outside [0, 1]'
    return sub
