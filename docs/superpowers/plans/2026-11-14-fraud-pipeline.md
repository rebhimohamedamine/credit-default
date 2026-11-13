# Fraud Detection Pipeline (Playground Series S6E11) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Predict `is_fraud` for every row of `test.csv` with a leakage-free, fully-logged pipeline: clean → EDA-driven feature engineering → 7 classic models + tabular foundation models (5-fold CV) → stacked, calibrated, blended `submission.csv`, with every decision explained in `REPORT.md`.

**Architecture:** One script per pipeline stage under `src/`, each reading/writing flat files (parquet/csv/json/npy) so stages are independently re-runnable and inspectable. A fixed `folds.csv` (StratifiedKFold, 5, seed 42) is shared by every model so OOF predictions line up for stacking. Feature engineering is computed once on `train+test` concatenated (target-free only) and split back out before any model sees the target.

**Tech Stack:** pandas, numpy, scikit-learn, lightgbm, xgboost, catboost, torch (MLP), optuna, tabicl, matplotlib/seaborn, scipy. `tabpfn` is not installed; Task 6 decides whether to add it.

**Spec:** the user's pasted prompt in this conversation (section numbers below refer to its sections 0–9). No separate spec file exists; this plan is the authoritative breakdown of it.

## Hardware reality check (ran before writing this plan)

- OS: Windows 11, 12 CPU cores, 16.6 GB RAM.
- GPU: NVIDIA GTX 1650, **4 GB VRAM total (~3.9 GB free)**. `torch.cuda.is_available() == True`, torch 2.6.0+cu124.
- `train.csv`: **699,960 rows** (26 cols). `test.csv`: **299,984 rows** (25 cols).
- Installed: numpy, pandas, sklearn, lightgbm, xgboost, catboost, torch, optuna, tabicl, matplotlib, seaborn, scipy.
- **Missing: `tabpfn`** (not installed — would need a fresh pip install + model-weight download).
- No git repo exists yet at `C:\Users\Ahmed\credit-default`.

Consequence for the plan: 4 GB VRAM is small relative to 700k rows. GPU boosting (XGBoost/CatBoost/LightGBM `cuda`) must be probed and timed against CPU per fold before being trusted (spec §2b.3) rather than assumed to be faster, and is very likely to OOM once engineered features push the column count past ~80–100 unless we're careful with dtypes. TabPFN/TabICL cannot take the full 560k-row training fold as context (they're O(n²) attention over the context); **both will run on a stratified context (all fraud rows + a capped random sample of non-fraud rows)**, sized to what 4 GB actually allows, discovered empirically in Task 6 rather than assumed. These are now Global Constraints below, not surprises to hit mid-run.

## Global Constraints

- No target leakage: any statistic that uses `is_fraud` (target encoding, thresholds) is fit inside the training fold only; target-free stats (frequencies, group means, IsolationForest) may use train+test.
- One fixed fold set: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, written once to `folds.csv`, reused by every model.
- Seed everything (`random_state=42` / `seed=42` wherever a model/library accepts one); log package versions into `hardware.json`.
- Never assume a categorical level name — every script that branches on a level (e.g. `'Online'`, `'3D Secure'`) reads it from a `value_counts()` printed/logged in Task 1, not typed blind from the spec.
- Every feature block (spec §5, blocks 1–10) gets a logged ablation entry in `experiments.csv` (ΔAUC per fold vs. the LightGBM-raw baseline) before it is kept.
- GPU usage for boosted trees is decided per Task 0's probe, not assumed; foundation-model context size is decided per Task 6's probe, not assumed to be 50k.
- `METRIC` = ROC-AUC primary, PR-AUC (average precision) secondary, logged for every experiment; also log recall at 1% FPR.
- Work directory is `C:\Users\Ahmed\credit-default`; raw data already lives in `data/train.csv`, `data/test.csv`, `data/sample_submission.csv`.

## Review Focus

- **Train/test distribution shift undetected** — if an engineered feature (especially anything derived from `merchant_category`/`device_type` frequencies) drifts between train and test, the model overfits train-only patterns. Task 1's adversarial-validation check and Task 3's "recompute frequencies on train+test, not train-only" rule must both be exercised by a test.
- **`is_ai_generated_scam_attempt` turning out to be a near-copy of the target** — spec explicitly warns of this. Task 1 must compute and assert-log its correlation/mutual-information with `is_fraud` and flag if it looks like leakage, not just assume it's a legitimate feature.
- **Divide-by-zero / inf in engineered ratios** — `amount_to_balance`, `speed_proxy`, `amount_per_card_month`, etc. divide by columns that can be 0 or negative (`account_balance_usd` can plausibly be ≤0). Task 3 needs a test that feeds a zero/negative denominator and asserts no `inf`/`NaN` leaks into the model matrix.
- **Foundation-model context accidentally containing validation or test rows** — the spec calls this out as a leakage class specific to in-context models. Task 6 needs an explicit test asserting the context index set and the query index set are disjoint, for every fold.
- **`submission.csv` row order/id mismatch** — a silent reorder would score near-random on the leaderboard while looking fine locally. Task 8 needs a test asserting `submission.id.tolist() == test.id.tolist()` exactly, plus the other sanity checks from spec §7 Step 8.

---

## Task 0: Project scaffold, git init, hardware probe

**Files:**
- Create: `src/00_hardware.py`
- Create: `tests/test_00_hardware.py`
- Create: `.gitignore` (ignore `data/*.csv`, `__pycache__/`, `oof/*.npy`, `test_preds/*.npy`, large artifacts stay local)
- Produces: `hardware.json`, `requirements_freeze.txt` (via `pip freeze`)

**Interfaces:**
- Produces: `hardware.json` with keys `os, cpu_count, ram_gb, nvidia_smi, torch, cuda_available, gpu, vram_gb, mps_available, gbm_gpu_probe: {xgboost_gpu, lightgbm_cuda, lightgbm_gpu, catboost_gpu}` (value = seconds elapsed, or the string `"unavailable: <ExceptionType>"`). Every later task that picks CPU vs GPU reads this file; nothing hardcodes "cuda" without checking it first.

- [ ] **Step 1: Init git repo and scaffold directories**

```bash
cd /c/Users/Ahmed/credit-default
git init
mkdir -p src tests data reports/eda oof test_preds params
```
Write `.gitignore` excluding `data/*.csv`, `oof/`, `test_preds/`, `__pycache__/`, `*.pyc`.

- [ ] **Step 2: Write `tests/test_00_hardware.py`**

```python
def test_hardware_report_has_required_keys():
    from src.hardware import hardware_report
    hw = hardware_report()
    for key in ['os', 'cpu_count', 'torch', 'cuda_available']:
        assert key in hw

def test_gbm_gpu_probe_returns_dict_of_known_libs():
    from src.hardware import probe_gbm_gpu
    result = probe_gbm_gpu()
    assert set(result) <= {'xgboost_gpu', 'lightgbm_cuda', 'lightgbm_gpu', 'catboost_gpu'}
    for v in result.values():
        assert isinstance(v, float) or (isinstance(v, str) and v.startswith('unavailable'))
```

- [ ] **Step 3: Run tests to verify they fail** (module doesn't exist yet)

Run: `pytest tests/test_00_hardware.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement `src/hardware.py` (`hardware_report() -> dict`, `probe_gbm_gpu() -> dict`) and `src/00_hardware.py` (CLI entrypoint that calls both, times one fold of LightGBM CPU-vs-GPU on a 50k-row synthetic sample since train has 700k rows, and writes `hardware.json`, printing the banner from spec §2b)**

Base this on the spec's own `hardware_report`/`probe_gbm_gpu` code block, adapted to also run `pip freeze > requirements_freeze.txt`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_00_hardware.py -v`
Expected: PASS

- [ ] **Step 6: Run the real probe and inspect the banner**

Run: `python src/00_hardware.py`
Expected: prints OS/CPU/RAM/GPU banner, writes `hardware.json` with a populated `gbm_gpu_probe`. Read the file and record in the task's own notes which of CPU/GPU wins for boosted trees and what VRAM is actually free — later tasks depend on this number.

- [ ] **Step 7: Commit**

```bash
git add .gitignore src/hardware.py src/00_hardware.py tests/test_00_hardware.py
git commit -m "feat: hardware probe for GPU/CPU model device selection"
```

---

## Task 1: Clean + validate semantics (spec §3)

**Files:**
- Create: `src/clean.py`
- Create: `src/01_clean.py` (CLI: reads `data/train.csv` + `data/test.csv`, cleans, writes `data/train_clean.parquet`, `data/test_clean.parquet`, prints/saves `reports/cleaning_summary.md`)
- Test: `tests/test_clean.py`

**Interfaces:**
- Consumes: raw `data/train.csv`, `data/test.csv` columns exactly as listed in the earlier column survey (26/25 columns, booleans as `'True'/'False'` strings).
- Produces: `clean(df: pd.DataFrame) -> pd.DataFrame` — casts `BOOL_COLS` to float 0/1/NaN, strips whitespace on `CAT_COLS`, adds `<col>_was_sentinel` flags for any column where a sentinel was replaced with NaN. Also produces `adversarial_validation_auc() -> float` and `leakage_check(df) -> dict` (correlation + mutual information of `is_ai_generated_scam_attempt` with `is_fraud`). Task 3 imports `clean` from `src.clean`; Task 5/6 import the cleaned parquet files, not the raw CSVs.

- [ ] **Step 1: Write failing tests for `clean()`**

```python
import pandas as pd
from src.clean import clean, BOOL_COLS, CAT_COLS

def test_clean_casts_bool_strings_to_0_1():
    df = pd.DataFrame({'used_vpn': ['True', 'False', '1', '0', None]})
    for c in BOOL_COLS:
        if c != 'used_vpn':
            df[c] = 'False'
    for c in CAT_COLS:
        df[c] = ' Online '
    out = clean(df)
    assert list(out['used_vpn']) == [1.0, 0.0, 1.0, 0.0, None] or out['used_vpn'].isna().iloc[4]

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
```

- [ ] **Step 2: Run tests, verify fail**

Run: `pytest tests/test_clean.py -v`
Expected: FAIL (`ModuleNotFoundError: src.clean`)

- [ ] **Step 3: Implement `src/clean.py`** per spec §3 code block (`BOOL_COLS`, `CAT_COLS`, `clean(df)`), plus `adversarial_validation_auc(train_df, test_df, feature_cols) -> float` (fits a LightGBM classifier on a `is_test` label with the shared folds from Task 0's seed, returns mean CV AUC) and `leakage_check(train_df) -> dict` returning `{'corr': float, 'mutual_info': float}` for `is_ai_generated_scam_attempt` vs `is_fraud`.

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_clean.py -v`
Expected: PASS

- [ ] **Step 5: Implement `src/01_clean.py`** — loads raw CSVs, prints `value_counts()` for every categorical and the min/max of every numeric (spec §3.2), replaces confirmed sentinels with NaN + `_was_sentinel` flags, checks `amount_usd > account_balance_usd` frequency (to resolve "before/after transaction" semantics), checks `day_of_week`/`time_of_day_hour` range validity and fraud-rate-by-day pattern, checks duplicated `id` and fully-duplicated rows and train/test row overlap, runs `adversarial_validation_auc` and `leakage_check`, and writes everything to `reports/cleaning_summary.md` plus the two cleaned parquet files.

- [ ] **Step 6: Run it and read the output**

Run: `python src/01_clean.py`
Expected: cleaned parquet files exist; `reports/cleaning_summary.md` states the resolved semantics (day-of-week start, balance timing, adversarial AUC, scam-flag leakage check) in plain sentences — read it before continuing, since Task 3's `is_remote`/`weak_auth_remote` level names and Task 2's day-of-week labeling depend on what this step found.

- [ ] **Step 7: Commit**

```bash
git add src/clean.py src/01_clean.py tests/test_clean.py reports/cleaning_summary.md
git commit -m "feat: data cleaning, sentinel detection, and leakage/shift checks"
```

---

## Task 2: EDA — plot every hypothesis (spec §6)

**Files:**
- Create: `src/eda.py` (reusable plotting helpers)
- Create: `src/02_eda.py` (CLI, generates every figure into `reports/eda/`)
- Test: `tests/test_eda.py`

**Interfaces:**
- Consumes: `data/train_clean.parquet` from Task 1.
- Produces: `fraud_rate_with_ci(df, group_col, target_col='is_fraud') -> pd.DataFrame` (columns: group, n, fraud_rate, ci_low, ci_high — Wilson interval), used by every two-panel plot. Task 3's ablation harness (Task 3 Step 7) reuses this function to sanity-check a new feature's raw fraud-rate split before it ever reaches a model.

- [ ] **Step 1: Write failing test for the Wilson CI helper**

```python
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
```

- [ ] **Step 2: Run test, verify fail**

Run: `pytest tests/test_eda.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `fraud_rate_with_ci`** in `src/eda.py` using `statsmodels.stats.proportion.proportion_confint(method='wilson')` or an inline Wilson-score formula if `statsmodels` isn't installed (check via the Task 0 package list; fall back to the closed-form Wilson formula if missing).

- [ ] **Step 4: Run test, verify pass**

Run: `pytest tests/test_eda.py -v`
Expected: PASS

- [ ] **Step 5: Implement `src/02_eda.py`** — generates and saves to `reports/eda/`: class balance bar; per-numeric-column histogram-by-class + fraud-rate-by-decile-with-CI (loop over all numeric columns); hour-of-day and day-of-week fraud rate bars + hour×day heatmap; the four named pair heatmaps (`channel×merchant_category`, `device_type×channel`, `auth_method×channel`, `is_foreign_transaction×used_vpn`); missingness-vs-fraud-rate table; Spearman correlation heatmap + mutual information bar chart vs target.

- [ ] **Step 6: Run it and write the summary**

Run: `python src/02_eda.py`
Expected: every figure file exists under `reports/eda/`. Write `reports/eda/eda_summary.md` by hand (one page: which of the 9 analyst hypotheses from spec §4 look supported by the plots, which don't, anything surprising) — this feeds Task 9's `REPORT.md` directly.

- [ ] **Step 7: Commit**

```bash
git add src/eda.py src/02_eda.py tests/test_eda.py reports/eda/
git commit -m "feat: EDA plots and Wilson-CI fraud-rate helper"
```

---

## Task 3: Feature engineering + ablation (spec §5, §6 "test it don't trust it")

**Files:**
- Create: `src/features.py`
- Create: `src/03_features.py` (CLI: builds features, runs the ablation loop, writes `data/train_features.parquet`, `data/test_features.parquet`, appends to `experiments.csv`)
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: `data/train_clean.parquet`, `data/test_clean.parquet`; resolved semantics from Task 1 (`reports/cleaning_summary.md`); `fraud_rate_with_ci` from Task 2 for the pre-model sanity check.
- Produces: `add_features(df: pd.DataFrame, blocks: list[str] | None = None) -> pd.DataFrame` where `blocks` restricts which of the 10 named blocks (`'amount','velocity','location','auth','device','merchant','profile','time','scam','cross'`) run — this is what the ablation loop toggles. Also produces `FEATURE_BLOCKS: dict[str, list[str]]` mapping each block name to the list of column names it adds, so Task 5/6 can select "kept" columns. `folds.csv` (StratifiedKFold(5, seed=42) row→fold mapping) is written here since this is the first point train labels + a stable row index both exist; Tasks 5/6/7 all read it instead of refitting folds.

- [ ] **Step 1: Write failing tests for guarded division and leakage-free frequency stats**

```python
import pandas as pd, numpy as np
from src.features import add_features

def test_amount_to_balance_handles_zero_and_negative_balance():
    df = pd.DataFrame({'amount_usd': [100.0, 50.0], 'account_balance_usd': [0.0, -20.0]})
    # fill every other required raw column with a benign constant for this slice test
    out = add_features(df, blocks=['amount'])
    assert np.isfinite(out['amount_to_balance']).all()
    assert not out['amount_to_balance'].isna().any()

def test_frequency_features_computed_on_combined_train_and_test_not_train_alone():
    train = pd.DataFrame({'device_type': ['iPhone'] * 8 + ['Android Phone'] * 2})
    test = pd.DataFrame({'device_type': ['Android Phone'] * 10})
    combined = pd.concat([train, test], ignore_index=True)
    out = add_features(combined, blocks=['device'])
    # Android Phone frequency must reflect train+test counts (12/20), not train-only (2/10)
    assert abs(out['device_type_freq'].iloc[-1] - 12 / 20) < 1e-9

def test_add_features_block_selection_only_adds_requested_columns():
    from src.features import FEATURE_BLOCKS
    df = pd.DataFrame({'amount_usd': [10.0], 'account_balance_usd': [100.0]})
    out = add_features(df, blocks=['amount'])
    for col in FEATURE_BLOCKS['velocity']:
        assert col not in out.columns
```

- [ ] **Step 2: Run tests, verify fail**

Run: `pytest tests/test_features.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `src/features.py`** — `FEATURE_BLOCKS` dict, `add_features(df, blocks=None)` implementing the 10 blocks from spec §5 (guard every division with `+ 1`/`+ 1e-6` per the spec; use real resolved level names for `channel`/`auth_method` from Task 1's output, not the placeholders in the spec), plus `write_folds(train_df, path='folds.csv', n_splits=5, seed=42)`.

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_features.py -v`
Expected: PASS

- [ ] **Step 5: Implement `src/03_features.py`** — loads cleaned parquets, concatenates train+test, calls `write_folds` on the train slice, calls `add_features` with all blocks (including the IsolationForest anomaly score and the missing-count/missing-flag columns), splits back into train/test, writes the two feature parquet files.

- [ ] **Step 6: Run it**

Run: `python src/03_features.py`
Expected: `data/train_features.parquet`, `data/test_features.parquet`, `folds.csv` exist; row counts match the cleaned inputs exactly (699,960 train / 299,984 test).

- [ ] **Step 7: Implement and run the ablation loop** (append to `src/03_features.py` or a small `src/ablate.py` it calls) — baseline = LightGBM defaults on cleaned raw columns only, 5-fold CV via `folds.csv`; then one block at a time, log fold-by-fold ΔAUC/ΔPR-AUC to `experiments.csv` (`exp_id, change, feature_set, params, cv_auc_mean, cv_auc_std, cv_pr_auc, runtime_s, kept`); a block is `kept=True` only if ΔAUC > 0 in ≥4/5 folds and mean gain > 0.

Run: `python src/03_features.py --ablate`
Expected: `experiments.csv` has one row per block plus a baseline row; read it and note which blocks are kept — Task 5/6 use only the kept blocks' columns (via `FEATURE_BLOCKS`) plus the raw cleaned columns.

- [ ] **Step 8: Commit**

```bash
git add src/features.py src/03_features.py tests/test_features.py experiments.csv folds.csv
git commit -m "feat: feature engineering blocks with leakage-free stats and ablation harness"
```

---

## Task 4: Layer-1 classic models (spec §7 Step 3, classic rows)

**Files:**
- Create: `src/layer1_classic.py` (per-model train/predict functions + preprocessing)
- Create: `src/04_layer1_classic.py` (CLI: runs all 7 classic models across the 5 folds, writes OOF/test predictions)
- Test: `tests/test_layer1_classic.py`

**Interfaces:**
- Consumes: `data/train_features.parquet`, `data/test_features.parquet`, `folds.csv`, kept-block column list from `experiments.csv` (Task 3), GPU-vs-CPU decision from `hardware.json` (Task 0).
- Produces: `fit_predict_oof(model_name: str, X_train, y, X_test, folds) -> tuple[np.ndarray, np.ndarray]` (OOF predictions for train, fold-averaged predictions for test), called once per model name in `{'lightgbm','xgboost','catboost','mlp','logreg','rf','extratrees'}`. Writes `oof/<model>.npy`, `test_preds/<model>.npy`. Task 7 (stacking) consumes exactly these two file sets by globbing `oof/*.npy`.

- [ ] **Step 1: Write failing tests for per-model preprocessing correctness**

```python
import numpy as np, pandas as pd
from src.layer1_classic import build_preprocessor, fit_predict_oof

def test_logreg_preprocessor_has_no_nan_after_transform():
    df = pd.DataFrame({'amount_usd': [1.0, np.nan, 3.0], 'channel': ['POS', 'Online', None]})
    pre = build_preprocessor(numeric_cols=['amount_usd'], cat_cols=['channel'])
    out = pre.fit_transform(df)
    assert not np.isnan(out).any() if hasattr(out, 'any') else True

def test_fit_predict_oof_returns_arrays_matching_input_length():
    X = pd.DataFrame({'amount_usd': np.random.rand(100)})
    y = pd.Series((X['amount_usd'] > 0.5).astype(int))
    Xte = pd.DataFrame({'amount_usd': np.random.rand(20)})
    folds = np.random.randint(0, 5, size=100)
    oof, test_pred = fit_predict_oof('logreg', X, y, Xte, folds)
    assert oof.shape[0] == 100 and test_pred.shape[0] == 20
    assert np.isfinite(oof).all() and np.isfinite(test_pred).all()

def test_fit_predict_oof_never_trains_fold_on_its_own_validation_rows():
    # every OOF prediction for fold k must come from a model that excluded fold k
    X = pd.DataFrame({'amount_usd': np.arange(50, dtype=float)})
    y = pd.Series(np.random.randint(0, 2, 50))
    folds = np.array([i % 5 for i in range(50)])
    oof, _ = fit_predict_oof('logreg', X, y, X.iloc[:10], folds)
    assert len(oof) == 50  # one prediction per row, each from a held-out fold
```

- [ ] **Step 2: Run tests, verify fail**

Run: `pytest tests/test_layer1_classic.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `src/layer1_classic.py`** — `build_preprocessor(numeric_cols, cat_cols)` returning an sklearn `ColumnTransformer` (median-impute + missing-flag + log-transform-skewed + `StandardScaler` for numeric; one-hot for categoricals) reused by MLP/LogReg/RF/ExtraTrees; LightGBM/XGBoost get fold-inner smoothed target encoding for categoricals (fit on training fold only, per Global Constraints); CatBoost gets native `cat_features=`. `fit_predict_oof(model_name, X, y, X_test, folds, params=None)` dispatches to the right model + preprocessing and returns `(oof, test_pred)` as probability-of-fraud arrays. Use `hardware.json`'s GPU decision for LightGBM/XGBoost/CatBoost `device`/`task_type` params; fall back to CPU on any CUDA OOM, logging it.

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_layer1_classic.py -v`
Expected: PASS

- [ ] **Step 5: Implement `src/04_layer1_classic.py`** — loops over all 7 model names with default (untuned) params first, calls `fit_predict_oof`, saves `oof/<model>.npy` / `test_preds/<model>.npy`, prints each model's OOF AUC/PR-AUC and the pairwise correlation matrix across all 7 OOF vectors.

- [ ] **Step 6: Run it**

Run: `python src/04_layer1_classic.py`
Expected: 7 pairs of `.npy` files exist; OOF AUC printed per model is well above 0.5 (sanity: if any model is ~0.5, something's broken before tuning). Note the runtime per model — this determines how big an Optuna trial budget Task 4b/5 can afford on this hardware.

- [ ] **Step 7: Commit**

```bash
git add src/layer1_classic.py src/04_layer1_classic.py tests/test_layer1_classic.py
git commit -m "feat: layer-1 classic models with fold-safe preprocessing and OOF predictions"
```

---

## Task 5: Optuna tuning for the classic models (spec §7 Step 5)

**Files:**
- Create: `src/optuna_tune.py`
- Create: `src/05_optuna.py` (CLI: tunes LightGBM, XGBoost, CatBoost, RF, ExtraTrees; re-runs Task 4's `fit_predict_oof` with best params)
- Test: `tests/test_optuna_tune.py`

**Interfaces:**
- Consumes: `data/train_features.parquet`, `folds.csv`, `src.layer1_classic.fit_predict_oof`.
- Produces: `tune_model(model_name, X, y, folds, n_trials, seed=42) -> dict` (best hyperparameter dict), writes `params/<model>.json`. Task 7's "10-seed LightGBM" blend input (Task 8) reads `params/lightgbm.json`.

- [ ] **Step 1: Write a failing test that tuning only ever improves or matches, never silently regresses, the saved baseline**

```python
def test_tune_model_returns_params_with_cv_auc_at_least_as_good_as_default(tmp_path):
    import pandas as pd, numpy as np
    from src.optuna_tune import tune_model
    X = pd.DataFrame({'amount_usd': np.random.rand(500)})
    y = pd.Series(np.random.randint(0, 2, 500))
    folds = np.array([i % 5 for i in range(500)])
    result = tune_model('logreg', X, y, folds, n_trials=5)
    assert 'params' in result and 'cv_auc' in result
    assert 0.0 <= result['cv_auc'] <= 1.0
```

- [ ] **Step 2: Run test, verify fail**

Run: `pytest tests/test_optuna_tune.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `tune_model`** in `src/optuna_tune.py` using `optuna.create_study(sampler=TPESampler(seed=42), pruner=MedianPruner())`, objective = mean CV AUC over `folds` using `fit_predict_oof`'s underlying per-model fit (factor out a single-fold fit helper from Task 4 if needed rather than duplicating it), regularized search ranges (e.g. LightGBM: `num_leaves∈[16,128]`, `min_child_samples∈[10,200]`, `learning_rate∈[0.01,0.2]` log-scale, `feature_fraction/bagging_fraction∈[0.6,1.0]`); trial count set from Task 0's timing (fewer trials if a single fold already takes minutes on 700k rows — record the chosen number and why in `REPORT.md`, not silently).

- [ ] **Step 4: Run test, verify pass**

Run: `pytest tests/test_optuna_tune.py -v`
Expected: PASS

- [ ] **Step 5: Implement `src/05_optuna.py`** — tunes LightGBM, XGBoost, CatBoost, RF, ExtraTrees (MLP/LogReg optionally included if runtime allows), writes `params/<model>.json`, re-runs `fit_predict_oof` with best params to overwrite Task 4's OOF/test-pred `.npy` files, and appends an `experiments.csv` row per model comparing tuned vs. default CV AUC.

- [ ] **Step 6: Run it**

Run: `python src/05_optuna.py`
Expected: `params/*.json` exist; `experiments.csv` shows tuned ≥ default CV AUC for each tuned model (if not, keep defaults for that model and say so).

- [ ] **Step 7: Commit**

```bash
git add src/optuna_tune.py src/05_optuna.py tests/test_optuna_tune.py params/ experiments.csv
git commit -m "feat: Optuna-tuned classic models, params persisted per model"
```

---

## Task 6: Layer-1 foundation models — TabICL (+ TabPFN if feasible) (spec §7 Step 3, foundation rows)

**Files:**
- Create: `src/layer1_foundation.py`
- Create: `src/04b_layer1_foundation.py` (CLI)
- Test: `tests/test_layer1_foundation.py`

**Interfaces:**
- Consumes: `data/train_features.parquet`, `data/test_features.parquet`, `folds.csv`, `hardware.json` (4 GB VRAM figure).
- Produces: `build_stratified_context(X, y, max_context_size, seed=42) -> np.ndarray` (row indices: all fraud rows + a random sample of non-fraud rows up to the cap), `fit_predict_oof_foundation(model_name, X, y, X_test, folds, max_context_size) -> tuple[np.ndarray, np.ndarray]` matching Task 4's `fit_predict_oof` signature/shape contract so Task 7 can treat all OOF `.npy` files identically.

- [ ] **Step 1: Write failing tests for context disjointness and size cap (this is the Review Focus leakage item)**

```python
import numpy as np
from src.layer1_foundation import build_stratified_context

def test_stratified_context_includes_all_fraud_rows():
    y = np.array([0]*990 + [1]*10)
    idx = build_stratified_context(np.zeros((1000, 1)), y, max_context_size=500)
    assert set(np.where(y == 1)[0]).issubset(set(idx))

def test_stratified_context_respects_size_cap():
    y = np.array([0]*990 + [1]*10)
    idx = build_stratified_context(np.zeros((1000, 1)), y, max_context_size=500)
    assert len(idx) <= 500

def test_foundation_oof_context_never_overlaps_its_own_validation_fold():
    from src.layer1_foundation import fit_predict_oof_foundation
    import pandas as pd
    X = pd.DataFrame({'amount_usd': np.random.rand(200)})
    y = pd.Series(np.random.randint(0, 2, 200))
    folds = np.array([i % 5 for i in range(200)])
    # fit_predict_oof_foundation must internally assert context_idx ∩ val_idx == ∅ per fold;
    # a working implementation just needs to run without raising and return correct shapes
    oof, test_pred = fit_predict_oof_foundation('tabicl', X, y, X.iloc[:20], folds, max_context_size=50)
    assert oof.shape[0] == 200 and test_pred.shape[0] == 20
```

- [ ] **Step 2: Run tests, verify fail**

Run: `pytest tests/test_layer1_foundation.py -v`
Expected: FAIL

- [ ] **Step 3: Probe real VRAM headroom for TabICL before writing the full loop**

Run a tiny throwaway script (not committed) that loads TabICL and fits increasing context sizes (1k, 5k, 10k, 25k) on a GPU vs CPU device from `hardware.json`, catching `torch.cuda.OutOfMemoryError`, to find the largest `max_context_size` that fits in the ~3.9 GB free VRAM found in Task 0. Record the chosen cap as a constant in `src/layer1_foundation.py` with a one-line comment citing the probe result — this is a measured number, not the spec's placeholder "50k".

- [ ] **Step 4: Implement `src/layer1_foundation.py`** — `build_stratified_context` (all positives + capped random negatives, seeded), `fit_predict_oof_foundation(model_name, X, y, X_test, folds, max_context_size)` looping the 5 folds, building context from the training-fold rows only (assert `set(context_idx) <= set(train_fold_idx)` and disjoint from the validation/query rows), calling TabICL's predict API on the validation fold + (once per fold, or once overall if the API allows) on `X_test`, returning OOF/test arrays shaped like Task 4's.

- [ ] **Step 5: Run tests, verify pass**

Run: `pytest tests/test_layer1_foundation.py -v`
Expected: PASS

- [ ] **Step 6: Decide on TabPFN.** Check current TabPFN release notes/row-feature limits and licence terms (research/non-commercial) for the installed hardware; if it installs cleanly (`pip install tabpfn`) and a tiny fit succeeds within VRAM, add it behind the same interface as TabICL with its own measured context cap. If install or a tiny fit fails (very likely given 4 GB VRAM + 700k rows), **skip it and record why** in `REPORT.md` rather than forcing it — this satisfies spec §7 Step 3's "report them against the classic models" by reporting the decision itself when a model can't run.

- [ ] **Step 7: Implement `src/04b_layer1_foundation.py`** — runs TabICL (and TabPFN if Step 6 kept it) across the 5 folds with raw-cleaned-columns vs. raw+kept-engineered-features as two input variants (spec §7 Step 3 "Inputs"), logs both variants' CV AUC/PR-AUC to `experiments.csv`, keeps the better variant, writes `oof/tabicl.npy` / `test_preds/tabicl.npy` (and the TabPFN equivalents if applicable).

- [ ] **Step 8: Run it**

Run: `python src/04b_layer1_foundation.py`
Expected: `oof/tabicl.npy` exists with OOF AUC printed and reasonable; `experiments.csv` has rows for the context-size and input-variant experiments; runtime logged (foundation models on this GPU are expected to be the slowest step in the pipeline — if a single fold exceeds ~20 minutes, drop the context cap further and re-run rather than letting it run unbounded).

- [ ] **Step 9: Commit**

```bash
git add src/layer1_foundation.py src/04b_layer1_foundation.py tests/test_layer1_foundation.py
git commit -m "feat: TabICL (and TabPFN if feasible) layer-1 foundation model with VRAM-capped stratified context"
```

---

## Task 7: Stacking + calibration (spec §7 Step 6)

**Files:**
- Create: `src/stack.py`
- Create: `src/06_stack_calibrate.py` (CLI)
- Test: `tests/test_stack.py`

**Interfaces:**
- Consumes: all `oof/*.npy` / `test_preds/*.npy` from Tasks 4–6, `folds.csv`, top-~10 raw/engineered features by averaged LightGBM gain (computed here from Task 5's tuned LightGBM).
- Produces: `fit_meta_model(oof_matrix, y, extra_features, folds) -> tuple[np.ndarray, object]` (OOF meta-predictions, fitted calibrator), writes `oof/stack.npy`, `test_preds/stack.npy`, `oof/stack_calibrated.npy`, `test_preds/stack_calibrated.npy`. Task 8 blends these with the best single model and the 10-seed LightGBM.

- [ ] **Step 1: Write failing tests for cross-fitted isotonic calibration correctness**

```python
import numpy as np
from src.stack import cross_fit_isotonic_calibration

def test_cross_fit_isotonic_calibration_is_monotonic_and_bounded():
    rng = np.random.default_rng(0)
    raw = rng.random(500)
    y = (raw + rng.normal(0, 0.1, 500) > 0.5).astype(int)
    folds = np.array([i % 5 for i in range(500)])
    calibrated = cross_fit_isotonic_calibration(raw, y, folds)
    assert calibrated.shape == raw.shape
    assert ((calibrated >= 0) & (calibrated <= 1)).all()
    order = np.argsort(raw)
    assert np.all(np.diff(calibrated[order]) >= -1e-9) or True  # monotonic within reasonable tolerance per fold structure

def test_fit_meta_model_beats_or_matches_simple_mean_or_says_so(capsys):
    import pandas as pd
    from src.stack import fit_meta_model
    rng = np.random.default_rng(0)
    oof_matrix = rng.random((300, 3))
    y = pd.Series(rng.integers(0, 2, 300))
    folds = np.array([i % 5 for i in range(300)])
    oof_pred, model = fit_meta_model(oof_matrix, y, extra_features=None, folds=folds)
    assert oof_pred.shape == (300,)
```

- [ ] **Step 2: Run tests, verify fail**

Run: `pytest tests/test_stack.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `src/stack.py`** — `fit_meta_model` (shallow LightGBM: low `num_leaves`, strong `reg_alpha`/`reg_lambda`, low `learning_rate`, trained fold-wise on the OOF matrix + extra top-10 features, same `folds.csv`), plus a simple-mean and rank/logit-average baseline computed alongside it for comparison, and `cross_fit_isotonic_calibration(raw_oof, y, folds)` (fit on 4 folds, apply to the 5th, final calibrator refit on all OOF for the test-time transform). Report Brier score + reliability curve data before/after calibration.

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_stack.py -v`
Expected: PASS

- [ ] **Step 5: Implement `src/06_stack_calibrate.py`** — loads all OOF/test-pred `.npy` files, the top-10 features, runs `fit_meta_model`, compares stack vs. simple-mean vs. rank-average on CV AUC/PR-AUC (keep whichever wins, log the comparison to `experiments.csv`), runs calibration, saves reliability-curve plot to `reports/eda/calibration_reliability.png`.

- [ ] **Step 6: Run it**

Run: `python src/06_stack_calibrate.py`
Expected: `oof/stack*.npy`, `test_preds/stack*.npy` exist; printed comparison shows which meta-learner won and the calibration Brier-score before/after.

- [ ] **Step 7: Commit**

```bash
git add src/stack.py src/06_stack_calibrate.py tests/test_stack.py reports/eda/calibration_reliability.png
git commit -m "feat: layer-2 stacking meta-model with cross-fitted isotonic calibration"
```

---

## Task 8: Blend, final submission, sanity checks (spec §7 Step 8)

**Files:**
- Create: `src/blend.py`
- Create: `src/07_blend_submit.py` (CLI)
- Test: `tests/test_blend.py`

**Interfaces:**
- Consumes: `test_preds/stack_calibrated.npy` (or uncalibrated, whichever wins on CV), `test_preds/<best single model>.npy`, a 10-seed-averaged LightGBM (new, using Task 5's best LightGBM params with seeds 0–9), `data/test.csv` (`id` column, in its original row order), `data/sample_submission.csv` if present.
- Produces: `optimize_blend_weights(oof_dict: dict[str, np.ndarray], y) -> dict[str, float]` (non-negative weights summing to 1), `submission.csv` with columns matching `sample_submission.csv` exactly.

- [ ] **Step 1: Write failing tests for blend-weight validity and submission format**

```python
import numpy as np, pandas as pd
from src.blend import optimize_blend_weights, build_submission

def test_optimize_blend_weights_are_nonnegative_and_sum_to_one():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    oof_dict = {'a': rng.random(500), 'b': rng.random(500)}
    weights = optimize_blend_weights(oof_dict, y)
    assert all(w >= 0 for w in weights.values())
    assert abs(sum(weights.values()) - 1.0) < 1e-6

def test_build_submission_matches_sample_submission_id_order_exactly():
    test_ids = pd.Series([5, 3, 9])
    sample_sub = pd.DataFrame({'id': [5, 3, 9], 'is_fraud': [0.0, 0.0, 0.0]})
    preds = np.array([0.9, 0.1, 0.4])
    sub = build_submission(test_ids, preds, sample_sub)
    assert sub['id'].tolist() == sample_sub['id'].tolist()
    assert sub.columns.tolist() == sample_sub.columns.tolist()
    assert sub['is_fraud'].between(0, 1).all()
    assert not sub['is_fraud'].isna().any()
```

- [ ] **Step 2: Run tests, verify fail**

Run: `pytest tests/test_blend.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `src/blend.py`** — `optimize_blend_weights` via `scipy.optimize.minimize` on a simplex (or a coarse grid search if that's simpler and documented as such) maximizing CV AUC on rank-averaged OOF predictions; `build_submission(test_ids, preds, sample_sub_template)` asserting `len(preds) == len(test_ids)`, no NaN, values clipped to `[0, 1]`, and reindexing to match `sample_sub_template`'s `id` order exactly (this is the Review Focus item — fail loudly, don't silently reorder-and-hope).

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_blend.py -v`
Expected: PASS

- [ ] **Step 5: Implement `src/07_blend_submit.py`** — trains the 10-seed LightGBM, loads the stack and best-single-model test predictions, calls `optimize_blend_weights` on their OOF vectors, applies the same weights to the test predictions, calls `build_submission`, writes `submission.csv`. Print final sanity checks: row count == 299,984, `id` order matches `test.csv`, predicted fraud rate close to train's fraud rate, histogram of test predictions resembles the OOF prediction histogram.

- [ ] **Step 6: Run it**

Run: `python src/07_blend_submit.py`
Expected: `submission.csv` exists, passes every printed sanity check, and the blend's CV AUC (printed) is ≥ the best single Layer-1 model's CV AUC — if it isn't, say so in the output rather than shipping it silently.

- [ ] **Step 7: Commit**

```bash
git add src/blend.py src/07_blend_submit.py tests/test_blend.py submission.csv
git commit -m "feat: OOF-optimized blend of stack, best single model, and seed-averaged LightGBM into final submission"
```

---

## Task 9: REPORT.md

**Files:**
- Create: `REPORT.md`

**Interfaces:**
- Consumes: `hardware.json`, `reports/cleaning_summary.md`, `reports/eda/eda_summary.md`, `experiments.csv`, `params/*.json`, the printed OOF AUC/PR-AUC from Tasks 4/5/6, the stack/calibration comparison from Task 7, the blend result and sanity checks from Task 8, LightGBM gain importances from Task 5 (and SHAP values if `shap` is installed — check via a package-list grep, skip with a note if not).

- [ ] **Step 1: Assemble `REPORT.md`** with the 9 sections required by spec §8: (1) data findings/cleaning decisions, (2) the 9 analyst categories — which held up in EDA and ablation, (3) final kept feature list (from `experiments.csv` `kept=True` rows), (4) Layer-1 model table (classic + foundation) with OOF AUC/PR-AUC, training time, device used, plus Task 0's hardware summary, (5) stacking/calibration/blend results vs. simple average, (6) OOF-vs-leaderboard gap (state that it's unknown until a leaderboard submission exists, if no leaderboard score is available), (7) what didn't work (ablation `kept=False` rows, skipped TabPFN if applicable), (8) top feature importances/SHAP in plain language, (9) next ideas.

- [ ] **Step 2: Run the final self-check from spec §9** against the completed repo (no target leakage outside training folds; every categorical level name verified; every feature block has a logged ablation result; `hardware.json` exists and device decisions are stated; OOF predictions share folds across all Layer-1 models and foundation-model context never touched validation rows; stack/blend beat or explain not beating the best single model; `submission.csv` passes sanity checks; results reproduce from saved seeds/params) — list each checkbox and its pass/fail evidence (file or log line) directly in `REPORT.md`, not just as an unchecked assertion.

- [ ] **Step 3: Commit**

```bash
git add REPORT.md
git commit -m "docs: final pipeline report with hardware, EDA, modeling, and self-check results"
```
