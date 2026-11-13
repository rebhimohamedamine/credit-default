# Fraud Detection Pipeline — Report

**Status: code complete, not yet executed.** Every stage (`src/00_hardware.py` through
`src/07_blend_submit.py`) is implemented and unit-tested, but per instruction the pipeline
has not been run end-to-end on this machine. The sections below are filled in with what is
already known (hardware, data shape, package availability) and left as explicit TODOs where
real numbers can only come from running the scripts in order:

```
python src/00_hardware.py
python src/01_clean.py
python src/02_eda.py          # then hand-write reports/eda/eda_summary.md
python src/03_features.py --ablate
python src/04_layer1_classic.py
python src/04b_layer1_foundation.py
python src/05_optuna.py
python src/06_stack_calibrate.py
python src/07_blend_submit.py
python -m pytest              # whenever you want to verify correctness
```

## 1. Hardware and data (known now)

- OS: Windows 11, 12 CPU cores, 16.6 GB RAM.
- GPU: NVIDIA GTX 1650, **4 GB VRAM (~3.9 GB free)**. `torch.cuda.is_available() == True` (torch 2.6.0+cu124).
- `train.csv`: 699,960 rows x 26 columns. `test.csv`: 299,984 rows x 25 columns.
- Installed: numpy, pandas, scikit-learn, lightgbm, xgboost, catboost, torch, optuna, tabicl, matplotlib, seaborn, scipy.
- **`tabpfn` is not installed.** `src/04b_layer1_foundation.py` checks for it at runtime and skips it
  with a printed message if missing, rather than failing the pipeline — install it first
  (`pip install tabpfn`) if you want it included, and re-check it actually fits the 4 GB VRAM budget
  before trusting it on the full dataset (very likely it will not without a small stratified context).
- `src/layer1_foundation.py`'s `MAX_CONTEXT_SIZE` (`{'tabicl': 20_000, 'tabpfn': 8_000}`) is a
  **placeholder**, not a measured value — Task 6 Step 3 of the plan calls for probing real VRAM
  headroom with increasing context sizes before trusting these numbers on 700k rows.
- TODO (after running `00_hardware.py`): paste the actual `gbm_gpu_probe` and CPU-vs-GPU timing
  results here, and state which device each boosted-tree model ended up using.

## 2. Data findings and cleaning decisions

TODO after `python src/01_clean.py`: paste the categorical `value_counts()`, the resolved
semantics (day-of-week start convention, whether `account_balance_usd` is pre- or
post-transaction based on the `amount_usd > account_balance_usd` frequency check), sentinel
columns found, duplicate/id-overlap counts, the adversarial-validation train-vs-test AUC, and
the `is_ai_generated_scam_attempt` vs `is_fraud` correlation/mutual-information leakage check
— all written to `reports/cleaning_summary.md` by the script.

## 3. The 9 analyst categories — EDA and ablation results

TODO after `python src/02_eda.py` (hand-fill `reports/eda/eda_summary.md`) and
`python src/03_features.py --ablate` (auto-fills `experiments.csv`). For each of the 9
categories (amount, velocity, location/network, authentication, device/channel, merchant,
card/customer profile, time, scam signals) plus the cross-category block, state whether the
plots showed a CI-separated gap and whether the corresponding feature block was `kept=True`
in `experiments.csv`.

## 4. Final kept feature list

TODO: `grep kept,True experiments.csv` (or filter in pandas) after the ablation run and list
the surviving feature blocks/columns here.

## 5. Layer-1 model table

| Model | OOF AUC | OOF PR-AUC | Training time | Device |
|---|---|---|---|---|
| LightGBM | TODO | TODO | TODO | TODO |
| XGBoost | TODO | TODO | TODO | TODO |
| CatBoost | TODO | TODO | TODO | TODO |
| MLP | TODO | TODO | TODO | CPU |
| Logistic Regression | TODO | TODO | TODO | CPU |
| Random Forest | TODO | TODO | TODO | CPU |
| ExtraTrees | TODO | TODO | TODO | CPU |
| TabICL | TODO | TODO | TODO | TODO |
| TabPFN | not installed — see section 1 | — | — | — |

Fill in from the printed output of `04_layer1_classic.py`, `05_optuna.py`, and
`04b_layer1_foundation.py`.

## 6. Stacking, calibration, and blend vs. simple average

TODO after `python src/06_stack_calibrate.py`: paste which meta-learner won (`stack` vs.
`simple_mean` vs. `rank_average` vs. `logit_average`), the Brier score before/after isotonic
calibration, and the reliability curve (`reports/eda/calibration_reliability.png`). After
`python src/07_blend_submit.py`: paste the blend weights and whether the blend beat the best
single Layer-1 model on OOF AUC (the script prints a warning if it did not).

## 7. OOF-vs-leaderboard gap

Unknown until a submission is scored on the Kaggle leaderboard. TODO: after submitting
`submission.csv`, record the leaderboard score here next to the OOF AUC from section 6 and
note the gap.

## 8. What did not work

TODO: list `kept=False` rows from `experiments.csv` (feature blocks that failed the ablation
gate), any model that lost to default parameters after Optuna tuning (`05_optuna.py` output),
and whether TabPFN had to be skipped for VRAM reasons.

## 9. Top feature importances / SHAP in plain language

TODO: `top_raw_features()` in `src/06_stack_calibrate.py` already computes LightGBM gain
importance for the meta-model's extra features — paste the top 10 and explain each in one
plain sentence. Add SHAP only if `shap` is installed (`pip show shap`); otherwise note it was
skipped.

## 10. Next ideas

- Measure `MAX_CONTEXT_SIZE` empirically (plan Task 6 Step 3) instead of using the placeholder.
- If TabPFN installs and fits in VRAM with a very small context, compare it against TabICL head-to-head.
- Revisit the Optuna trial budget (`src/05_optuna.py: trial_budget()`) once real per-fold
  timings on 700k rows are known — the current 20/50-trial defaults are a guess.
- Consider polynomial features (`src.features.add_polynomial_features`) on the top-8 LightGBM-gain
  features once section 9's importances are known; not auto-applied by `03_features.py`.

## 11. Final self-check (spec section 9)

To be completed after a real run — each line needs evidence (a file, a printed log line), not
just a checkmark:

- [ ] No target information used outside the training fold.
- [ ] Every categorical level name used in code was verified against the data (`reports/cleaning_summary.md`).
- [ ] Every feature block in the final model has a logged ablation result (`experiments.csv`).
- [ ] Stage 0 hardware check ran, `hardware.json` exists, and the GPU/CPU decision is stated above.
- [ ] OOF predictions come from the same folds (`folds.csv`) for all Layer-1 models, and
      foundation-model context never contained validation rows (enforced by an `assert` in
      `src/layer1_foundation.py::fit_predict_oof_foundation`, also covered by
      `tests/test_layer1_foundation.py`).
- [ ] Stack and blend beat the best single model on CV, or this report says they did not (section 6).
- [ ] `submission.csv` passes all sanity checks (`src/07_blend_submit.py` asserts row count,
      id order, no NaN, `[0, 1]` range, and prints the predicted vs. train fraud rate).
- [ ] Results reproduce from the saved seeds and parameters (`params/*.json`, `folds.csv`, seed=42 throughout).
