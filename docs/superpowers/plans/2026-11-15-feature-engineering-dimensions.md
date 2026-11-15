# Five-Dimension Feature Engineering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the gaps between the five feature-engineering dimensions (statistical, interaction, categorical, aggregation, temporal) requested by the user and what `src/features.py` already implements, then validate the new features through the project's existing ablation gate.

**Architecture:** Add four new, separately-named blocks to the existing `FEATURE_BLOCKS`/`BLOCK_ORDER`/`_BLOCK_FUNCS` pattern in `src/features.py` (`stat2`, `rare_cat`, `group_agg`), plus one special-cased block (`target_enc`) that needs fold/target information and therefore runs as its own function called from `src/03_features.py` rather than through the stateless per-row `add_features` pipeline. A new `ABLATION_BLOCKS` list (= `BLOCK_ORDER` + `['target_enc']`) drives `run_ablation`, so the ablation report gets one row per new block without disturbing `add_features`'s existing block loop.

**Tech Stack:** pandas, numpy, pytest, LightGBM (via existing ablation harness) — no new dependencies.

**Spec:** this plan's design was approved in-conversation (no separate design doc file was written; see the brainstorming summary earlier in this session for the approved short design).

## Global Constraints

- No entity ID column (`customer_id`/`card_id`) exists in the data — any group/aggregation feature must key off existing categorical columns (`merchant_category`, `card_type`, `auth_method`, `channel`, `device_type`, `age_bin`), never a fabricated entity id.
- Target encoding must be computed strictly out-of-fold using `folds.csv`'s fold assignment; a row's encoding must never be derived from its own fold's target values.
- Non-target-derived frequency/group stats are computed on the combined train+test frame, matching the existing convention already covered by `test_frequency_features_computed_on_combined_train_and_test_not_train_alone`.
- New stateless blocks follow the existing `_add_<block>(df) -> df` → `_BLOCK_FUNCS` → `FEATURE_BLOCKS[<block>]` pattern used by `amount`, `velocity`, etc. `target_enc` is the one exception: it is added to `FEATURE_BLOCKS` and `ABLATION_BLOCKS` only, never to `BLOCK_ORDER`/`_BLOCK_FUNCS`.
- Do not push to the remote — commit locally only, per explicit user instruction.
- Do not modify `src/04_layer1_classic.py` through `src/07_blend_submit.py` or any other pipeline stage.

## Review Focus

- OOF target encoding must never use a row's own fold's target values — the smoothing/averaging math must be computed from the other folds only.
- An unseen category in test (never seen in train) must fall back to the global mean for `*_target_enc`, not `NaN`.
- A `NaN` in a categorical column must remain `NaN` after rare-category collapsing (`*_rare`), not be rewritten to `"Other"`.
- The rare-category frequency threshold must be computed on the combined train+test frame, not train alone (consistent with existing frequency-encoding convention).
- `qcut`-based bins (`amount_bin`, `distance_bin`) must tolerate heavily duplicated values via `duplicates='drop'`, the same guard already used for `merchant_risk_bin`.

---

### Task 1: `stat2` block — sqrt transforms and extra quantile bins

**Files:**
- Modify: `src/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Produces: `_add_stat2(df: pd.DataFrame) -> pd.DataFrame`; `FEATURE_BLOCKS['stat2'] = ['amount_sqrt', 'amount_bin', 'distance_sqrt', 'distance_bin', 'hours_since_last_txn_sqrt']`; `_BLOCK_FUNCS['stat2'] = _add_stat2`; `'stat2'` inserted into `BLOCK_ORDER` immediately before `'cross'`.

- [ ] **Step 1: Write the failing test**

```python
def test_stat2_sqrt_transforms_and_bins():
    df = pd.DataFrame({
        'amount_usd': [4.0, 9.0, 16.0],
        'distance_from_home_km': [16.0, 25.0, 36.0],
        'hours_since_last_txn': [1.0, 4.0, 9.0],
    })
    out = add_features(df, blocks=['stat2'])
    assert list(out['amount_sqrt']) == [2.0, 3.0, 4.0]
    assert list(out['distance_sqrt']) == [4.0, 5.0, 6.0]
    assert list(out['hours_since_last_txn_sqrt']) == [1.0, 2.0, 3.0]
    assert out['amount_bin'].notna().all()
    assert out['distance_bin'].notna().all()


def test_stat2_bins_tolerate_heavily_duplicated_values():
    # 97 rows share the same amount, which would make plain pd.qcut(..., 10) raise on
    # duplicate bin edges without duplicates='drop'.
    df = pd.DataFrame({'amount_usd': [5.0] * 97 + [10.0, 20.0, 30.0]})
    out = add_features(df, blocks=['stat2'])
    assert out['amount_bin'].notna().all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_features.py::test_stat2_sqrt_transforms_and_bins -v`
Expected: FAIL (`'stat2'` not a valid block / `KeyError`)

- [ ] **Step 3: Implement `_add_stat2` in `src/features.py`**

Add sqrt of the clipped (`lower=0`) value for `amount_usd`, `distance_from_home_km`, `hours_since_last_txn` (guard each on column presence, same style as `_add_amount`). Add `amount_bin`/`distance_bin` via `pd.qcut(col, 10, labels=False, duplicates='drop')`, guarded the same way. Register the function in `_BLOCK_FUNCS`, the column list in `FEATURE_BLOCKS['stat2']`, and insert `'stat2'` into `BLOCK_ORDER` right before `'cross'`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_features.py::test_stat2_sqrt_transforms_and_bins -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/features.py tests/test_features.py
git commit -m "feat: add stat2 feature block (sqrt transforms, extra quantile bins)"
```

---

### Task 2: `rare_cat` block — collapse rare categories into "Other"

**Files:**
- Modify: `src/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: `CAT_COLS` from `src.clean` (`['merchant_category', 'card_type', 'auth_method', 'channel', 'device_type']`).
- Produces: `_add_rare_cat(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame`; `FEATURE_BLOCKS['rare_cat'] = [f'{c}_rare' for c in CAT_COLS]`; `_BLOCK_FUNCS['rare_cat'] = _add_rare_cat`; `'rare_cat'` inserted into `BLOCK_ORDER` before `'cross'` (after `'stat2'`).

- [ ] **Step 1: Write the failing tests**

```python
def test_rare_cat_collapses_categories_below_threshold():
    df = pd.DataFrame({'device_type': ['common'] * 199 + ['rare']})
    out = add_features(df, blocks=['rare_cat'])
    assert out['device_type_rare'].iloc[-1] == 'Other'
    assert (out['device_type_rare'].iloc[:-1] == 'common').all()


def test_rare_cat_leaves_missing_values_as_missing():
    df = pd.DataFrame({'device_type': ['common'] * 10 + [None]})
    out = add_features(df, blocks=['rare_cat'])
    assert pd.isna(out['device_type_rare'].iloc[-1])


def test_rare_cat_threshold_computed_on_combined_train_and_test_not_train_alone():
    # 'z' is 1/200 = 0.5% of train alone (below the 1% threshold), but test adds 50 more
    # 'z' rows, making it 51/250 = 20.4% of the combined frame (above threshold). If the
    # frequency were computed on train alone, 'z' would wrongly collapse to 'Other'.
    train = pd.DataFrame({'device_type': ['common'] * 199 + ['z']})
    test = pd.DataFrame({'device_type': ['z'] * 50})
    combined = pd.concat([train, test], ignore_index=True)
    out = add_features(combined, blocks=['rare_cat'])
    assert (out.loc[out['device_type'] == 'z', 'device_type_rare'] == 'z').all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_features.py -k rare_cat -v`
Expected: FAIL

- [ ] **Step 3: Implement `_add_rare_cat` in `src/features.py`**

For each column in `CAT_COLS` present in `df`, compute `freq = df[col].value_counts(normalize=True)` on the (combined) frame passed in, then `df[f'{col}_rare'] = df[col].mask(df[col].map(freq) < threshold, 'Other')`. Import `CAT_COLS` from `src.clean` at module level. Register in `_BLOCK_FUNCS`, `FEATURE_BLOCKS['rare_cat']`, and `BLOCK_ORDER`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_features.py -k rare_cat -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/features.py tests/test_features.py
git commit -m "feat: add rare_cat feature block (collapse low-frequency categories)"
```

---

### Task 3: `group_agg` block — count/median/min/max per categorical group

**Files:**
- Modify: `src/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Produces: `_add_group_agg(df: pd.DataFrame) -> pd.DataFrame`; constants `GROUP_AGG_GROUP_COLS = ['merchant_category', 'channel', 'device_type', 'card_type', 'age_bin']`, `GROUP_AGG_VALUE_COLS = ['amount_usd', 'velocity_score']`, `GROUP_AGG_STATS = ['median', 'min', 'max']`; helper `_group_agg_cols() -> list[str]` returning `[f'count_{g}' for g in GROUP_AGG_GROUP_COLS] + [f'{v}_{s}_{g}' for g in GROUP_AGG_GROUP_COLS for v in GROUP_AGG_VALUE_COLS for s in GROUP_AGG_STATS]`; `FEATURE_BLOCKS['group_agg'] = _group_agg_cols()`; `_BLOCK_FUNCS['group_agg'] = _add_group_agg`; `'group_agg'` inserted into `BLOCK_ORDER` before `'cross'` (after `'rare_cat'`).

- [ ] **Step 1: Write the failing test**

```python
def test_group_agg_count_and_stats_per_group():
    df = pd.DataFrame({
        'channel': ['a', 'a', 'a', 'b', 'b'],
        'amount_usd': [10.0, 20.0, 30.0, 5.0, 7.0],
    })
    out = add_features(df, blocks=['group_agg'])
    assert list(out['count_channel']) == [3, 3, 3, 2, 2]
    assert list(out['amount_usd_median_channel']) == [20.0, 20.0, 20.0, 6.0, 6.0]
    assert list(out['amount_usd_min_channel']) == [10.0, 10.0, 10.0, 5.0, 5.0]
    assert list(out['amount_usd_max_channel']) == [30.0, 30.0, 30.0, 7.0, 7.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_features.py::test_group_agg_count_and_stats_per_group -v`
Expected: FAIL

- [ ] **Step 3: Implement `_add_group_agg` in `src/features.py`**

For each group column present in `df` (intersect `GROUP_AGG_GROUP_COLS`), set `count_{gcol}` via `df.groupby(gcol)[gcol].transform('count')`. For each present value column (intersect `GROUP_AGG_VALUE_COLS`), set `{vcol}_{stat}_{gcol}` via `df.groupby(gcol)[vcol].transform(stat)` for `stat` in `GROUP_AGG_STATS`. Register as in Task 1/2.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_features.py::test_group_agg_count_and_stats_per_group -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/features.py tests/test_features.py
git commit -m "feat: add group_agg feature block (count/median/min/max per category group)"
```

---

### Task 4: `target_enc` — leakage-safe out-of-fold target encoding

**Files:**
- Modify: `src/features.py`
- Modify: `src/03_features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: `CAT_COLS` from `src.clean`; `folds` as a `np.ndarray` positionally aligned with `train`'s rows (same convention as `write_folds`'s return value's `'fold'` column).
- Produces: `add_target_encoding(train: pd.DataFrame, test: pd.DataFrame, folds: np.ndarray, cat_cols: list[str], target_col: str = 'is_fraud', smoothing: float = 20.0) -> tuple[pd.DataFrame, pd.DataFrame]`; `FEATURE_BLOCKS['target_enc'] = [f'{c}_target_enc' for c in cat_cols]`; `ABLATION_BLOCKS = BLOCK_ORDER + ['target_enc']` (module-level constant in `src/features.py`). `target_enc` is **not** added to `BLOCK_ORDER` or `_BLOCK_FUNCS`.

- [ ] **Step 1: Write the failing tests**

```python
def test_add_target_encoding_oof_excludes_own_fold():
    train = pd.DataFrame({
        'cat': ['a', 'a', 'a', 'b', 'b', 'b'],
        'is_fraud': [1, 1, 0, 0, 0, 1],
    })
    test = pd.DataFrame({'cat': ['a', 'b']})
    folds = np.array([0, 0, 1, 0, 1, 1])
    train_out, _ = add_target_encoding(train, test, folds, cat_cols=['cat'], smoothing=0.0)
    assert train_out.loc[2, 'cat_target_enc'] == 1.0   # fold-1 'a' row encoded from fold-0 'a' rows only (mean 1.0)
    assert train_out.loc[3, 'cat_target_enc'] == 0.5   # fold-0 'b' row encoded from fold-1 'b' rows only (mean 0.5)


def test_add_target_encoding_unseen_test_category_falls_back_to_global_mean():
    train = pd.DataFrame({'cat': ['a', 'a', 'b', 'b'], 'is_fraud': [1, 0, 0, 0]})
    test = pd.DataFrame({'cat': ['unseen']})
    folds = np.array([0, 1, 0, 1])
    _, test_out = add_target_encoding(train, test, folds, cat_cols=['cat'], smoothing=0.0)
    assert test_out['cat_target_enc'].iloc[0] == train['is_fraud'].mean()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_features.py -k target_encoding -v`
Expected: FAIL (`ImportError`/`AttributeError`, function doesn't exist)

- [ ] **Step 3: Implement `add_target_encoding` in `src/features.py`**

For each `col` in `cat_cols` present in `train`: compute `global_mean = train[target_col].mean()`; for each unique fold value, compute `mean`/`count` grouped by `col` over the rows in the *other* folds, smooth via `(mean * count + global_mean * smoothing) / (count + smoothing)`, map onto that fold's rows, and fill any unmapped (unseen) category with `global_mean`. Assign the full out-of-fold array to `train[f'{col}_target_enc']`. Then compute the same smoothed mean/count over *all* of `train` and map it onto `test[f'{col}_target_enc']`, filling unseen categories with `global_mean`. Return `(train, test)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_features.py -k target_encoding -v`
Expected: PASS

- [ ] **Step 5: Wire into `src/03_features.py`**

In `build_features()`: capture `write_folds(...)`'s return value (`folds_df`), call `train, test = add_target_encoding(train, test, folds_df['fold'].values, CAT_COLS)` right after, before the `_is_test` assignment/concat. Add `from src.clean import CAT_COLS` and update the `from src.features import ...` line to include `ABLATION_BLOCKS, add_target_encoding`. In `run_ablation()`, change the block loop from `for block in BLOCK_ORDER:` to `for block in ABLATION_BLOCKS:`.

- [ ] **Step 6: Run the full feature test suite**

Run: `python -m pytest tests/test_features.py -v`
Expected: PASS (all tests, old and new)

- [ ] **Step 7: Commit**

```bash
git add src/features.py src/03_features.py tests/test_features.py
git commit -m "feat: add leakage-safe out-of-fold target encoding and wire into ablation"
```

---

### Task 5: End-to-end validation against the real ablation gate

**Files:**
- Modify: `.gitignore`

**Interfaces:**
- Consumes: everything produced by Tasks 1–4.
- Produces: `data/train_clean.parquet`, `data/test_clean.parquet`, `data/train_features.parquet`, `data/test_features.parquet`, `folds.csv`, `experiments.csv` (all generated, local-only artifacts).

- [ ] **Step 1: Run cleaning**

Run: `python src/01_clean.py`
Expected: prints `Wrote data/train_clean.parquet ...`, `Wrote data/test_clean.parquet ...`, `Wrote reports/cleaning_summary.md`, no error.

- [ ] **Step 2: Run feature build + ablation**

Run: `python src/03_features.py --ablate`
Expected: prints baseline AUC, then one `AUC ... (delta ..., kept=...)` line per block in `ABLATION_BLOCKS`, including `stat2`, `rare_cat`, `group_agg`, `target_enc`; writes `experiments.csv`, `folds.csv`, `data/train_features.parquet`, `data/test_features.parquet`.

- [ ] **Step 3: Report the new blocks' kept/not-kept verdicts to the user**

Read `experiments.csv`, filter to `exp_id` in `{'block_stat2', 'block_rare_cat', 'block_group_agg', 'block_target_enc'}`, and report each block's `cv_auc_mean`, delta vs. baseline, and `kept` value.

- [ ] **Step 4: Run the full test suite**

Run: `python -m pytest`
Expected: PASS, no regressions in `tests/test_features.py` or any other test file.

- [ ] **Step 5: Keep generated artifacts out of version control**

Add `experiments.csv` and `folds.csv` to `.gitignore` (consistent with the existing `data/*.csv`/`data/*.parquet` pattern — these are machine-local run outputs, not source).

- [ ] **Step 6: Commit**

```bash
git add .gitignore
git commit -m "chore: ignore generated experiments.csv and folds.csv"
```

(No push — per explicit user instruction, work stays local until told otherwise.)
