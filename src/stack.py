import numpy as np
import pandas as pd


def cross_fit_isotonic_calibration(raw_oof: np.ndarray, y, folds: np.ndarray) -> np.ndarray:
    from sklearn.isotonic import IsotonicRegression

    y = np.asarray(y)
    calibrated = np.zeros_like(raw_oof, dtype=float)
    for fold in sorted(set(folds)):
        fit_idx = folds != fold
        apply_idx = folds == fold
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        iso.fit(raw_oof[fit_idx], y[fit_idx])
        calibrated[apply_idx] = iso.predict(raw_oof[apply_idx])
    return calibrated


def fit_final_calibrator(raw_oof: np.ndarray, y):
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso.fit(raw_oof, np.asarray(y))
    return iso


def fit_meta_model(oof_matrix: np.ndarray, y, extra_features: pd.DataFrame | None,
                    folds: np.ndarray, params: dict | None = None) -> tuple[np.ndarray, object]:
    import lightgbm as lgb

    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    meta_X = pd.DataFrame(oof_matrix, columns=[f'm{i}' for i in range(oof_matrix.shape[1])])
    if extra_features is not None:
        meta_X = pd.concat([meta_X, extra_features.reset_index(drop=True)], axis=1)

    default_params = {
        'num_leaves': 7, 'min_child_samples': 100, 'learning_rate': 0.03,
        'n_estimators': 200, 'reg_alpha': 1.0, 'reg_lambda': 1.0,
    }
    default_params.update(params or {})

    oof_pred = np.zeros(len(meta_X))
    models = []
    for fold in sorted(set(folds)):
        train_idx, val_idx = folds != fold, folds == fold
        model = lgb.LGBMClassifier(random_state=42, verbose=-1, **default_params)
        model.fit(meta_X.loc[train_idx], y.loc[train_idx])
        oof_pred[val_idx] = model.predict_proba(meta_X.loc[val_idx])[:, 1]
        models.append(model)

    final_model = lgb.LGBMClassifier(random_state=42, verbose=-1, **default_params)
    final_model.fit(meta_X, y)
    return oof_pred, final_model


def simple_mean_meta(oof_matrix: np.ndarray) -> np.ndarray:
    return oof_matrix.mean(axis=1)


def rank_average_meta(oof_matrix: np.ndarray) -> np.ndarray:
    ranks = pd.DataFrame(oof_matrix).rank(pct=True).values
    return ranks.mean(axis=1)


def logit_average_meta(oof_matrix: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    clipped = np.clip(oof_matrix, eps, 1 - eps)
    logits = np.log(clipped / (1 - clipped))
    return 1 / (1 + np.exp(-logits.mean(axis=1)))
