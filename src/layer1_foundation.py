import numpy as np
import pandas as pd

from src.layer1_classic import _load_device_config

# Measured against the 4 GB (3.9 GB free) GTX 1650 found in hardware.json by Task 0's
# probe (src/00_hardware.py). This is a conservative placeholder until that probe has
# actually been run on this machine with the real feature count (~80-100 cols after
# Task 3) -- see plan Task 6 Step 3. Re-run the probe and update this constant before
# trusting it on the full dataset.
MAX_CONTEXT_SIZE = {'tabicl': 20_000, 'tabpfn': 8_000}

MODEL_CLASSES = {
    'tabicl': ('tabicl', 'TabICLClassifier'),
    'tabpfn': ('tabpfn', 'TabPFNClassifier'),
}

# predict_proba forwards the whole query batch through the model together with the
# context, so its memory scales with query rows too (not just context rows). Chunking
# the query avoids CUDA OOM when predicting on the full ~300k-row test set at once.
PREDICT_BATCH_SIZE = 2_000


def _predict_proba_batched(model, X: pd.DataFrame, batch_size: int = PREDICT_BATCH_SIZE) -> np.ndarray:
    preds = np.empty(len(X))
    for start in range(0, len(X), batch_size):
        chunk = X.iloc[start:start + batch_size]
        preds[start:start + len(chunk)] = model.predict_proba(chunk)[:, 1]
    return preds


def build_stratified_context(X, y, max_context_size: int, seed: int = 42) -> np.ndarray:
    y = np.asarray(y)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng = np.random.default_rng(seed)

    if len(neg_idx) > 0 and len(pos_idx) >= max_context_size:
        # Positives alone would exhaust the budget, leaving zero negatives: a
        # single-class context breaks in-context models (predict_proba collapses
        # to one column). Reserve at least one negative slot. In normal fraud-rate
        # data this branch never triggers since fraud is the rare class.
        pos_keep = rng.choice(pos_idx, size=max_context_size - 1, replace=False)
    else:
        pos_keep = pos_idx

    n_neg_keep = max(0, max_context_size - len(pos_keep))
    if n_neg_keep < len(neg_idx):
        neg_keep = rng.choice(neg_idx, size=n_neg_keep, replace=False)
    else:
        neg_keep = neg_idx

    idx = np.concatenate([pos_keep, neg_keep])
    rng.shuffle(idx)
    return idx


def _prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=['object', 'category', 'string']).columns:
        codes, _ = pd.factorize(df[col], use_na_sentinel=False)
        df[col] = codes.astype(float)
        df.loc[df[col] < 0, col] = np.nan
    return df


def _load_model(model_name: str):
    module_name, class_name = MODEL_CLASSES[model_name]
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def fit_predict_oof_foundation(model_name: str, X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame,
                                 folds: np.ndarray, max_context_size: int | None = None,
                                 seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    if max_context_size is None:
        max_context_size = MAX_CONTEXT_SIZE.get(model_name, 10_000)

    X = _prepare_X(X).reset_index(drop=True)
    X_test = _prepare_X(X_test).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    ModelClass = _load_model(model_name)
    device_cfg = _load_device_config()
    device = 'cuda' if device_cfg.get('cuda_available') else 'cpu'

    oof = np.zeros(len(X))
    test_fold_preds = np.zeros((len(set(folds)), len(X_test)))

    for i, fold in enumerate(sorted(set(folds))):
        train_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]

        context_rel_idx = build_stratified_context(
            X.iloc[train_idx], y.iloc[train_idx].values, max_context_size, seed=seed)
        context_idx = train_idx[context_rel_idx]

        assert set(context_idx).issubset(set(train_idx))
        assert not (set(context_idx) & set(val_idx))

        # kv_cache pre-computes the context's attention state once in fit(), so each
        # chunked predict_proba call below reuses it instead of re-embedding the full
        # context from scratch on every chunk (tabicl only; tabpfn has no such option).
        extra_kwargs = {'kv_cache': True} if model_name == 'tabicl' else {}
        model = ModelClass(device=device, random_state=seed, **extra_kwargs)
        model.fit(X.iloc[context_idx], y.iloc[context_idx])

        oof[val_idx] = _predict_proba_batched(model, X.iloc[val_idx])
        test_fold_preds[i] = _predict_proba_batched(model, X_test)

    return oof, test_fold_preds.mean(axis=0)
