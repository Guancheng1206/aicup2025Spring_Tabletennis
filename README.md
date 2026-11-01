# AI CUP 2025 Spring — Smart Racket Table Tennis Classification

[![AI CUP 2025 Spring](https://img.shields.io/badge/AI%20CUP-2025%20Spring-orange.svg)](https://tbrain.trendmicro.com.tw/Competitions/Details/39)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](requirements.txt)
[![Private LB AUC](https://img.shields.io/badge/Private%20LB%20AUC-0.806-success.svg)](#competition-result)

Solution for the AI CUP 2025 Spring smart-racket challenge. The task is to predict four table-tennis
player attributes from six-axis IMU swing recordings:

- `gender`
- `hold racket handed`
- `play years`
- `level`

The whole pipeline lives in a single, self-contained notebook
([aicup_submission.ipynb](aicup_submission.ipynb)): feature extraction, per-target hyperparameter tuning,
training, and test-set prediction.

This is the corrected version (v7) of the notebook submitted to the competition (v6). A post-competition
review found nine problems in the submitted version, including feature-selection leakage in its
cross-validation; they are listed in [Changes from the submitted version](#changes-from-the-submitted-version).
The submitted notebook remains available in the git history at commit `8170835`.

## Competition result

The competition score comes from the **submitted version (v6)**: **Private Leaderboard ROC AUC 0.806.**

| Target | Metric | v6 Optuna best CV AUC |
| --- | --- | ---: |
| `gender` | ROC AUC | 0.7963 |
| `hold racket handed` | ROC AUC | 0.9996 |
| `play years` | micro-OvR ROC AUC | 0.6629 |
| `level` | micro-OvR ROC AUC | 0.8595 |
| **mean** | — | **0.8296** |

The headline metric is the arithmetic mean of the four per-target AUCs, matching the competition's
scoring. The private leaderboard score is the only fully out-of-sample number. The v6 CV numbers are
optimistic: `SelectKBest` was fitted on all training rows before the fold split, so it had already seen the
validation labels, and hyperparameter selection, early stopping and reporting shared the same folds.

The current notebook (v7) has been run end to end on a small subset of the data; its full-data results
have not been recorded yet.

## How it works

```mermaid
flowchart TD
    RAW["Raw IMU recording<br/>one .txt at 85 Hz, ~27 swings x 6 axes"]

    subgraph FEAT["Feature pipeline (cached)"]
        SEG["Split into 27 swings<br/>equal-width segments"]
        TS["Per-swing TSFEL features<br/>6 axes + 2 L2 magnitudes, 992 dims"]
        AGG["Aggregate to recording level<br/>7 statistics, 6,944 dims"]
        SEG --> TS --> AGG
    end

    VT["Drop near-constant features<br/>VarianceThreshold 0.01 (no labels used)"]

    subgraph TRAIN["Per-target tuning and training (x4)"]
        PRE["Fold-local steps, fitted on training folds only<br/>SelectKBest (gender / play years / level),<br/>KNNImputer, MinMaxScaler, input noise"]
        OPT["Optuna TPE search with median pruning<br/>75 trials, 5-fold GroupKFold by player_id"]
        FIT["Fit final CatBoost (GPU) on all training data<br/>trees = mean CV best iteration"]
        PRE --> OPT --> FIT
    end

    RAW --> SEG
    AGG --> VT --> PRE
    FIT -. persist / load .-> ART[("Per-target artifacts<br/>model, imputer, scaler, params")]
    FIT --> INF["Test inference<br/>predict_proba, no noise, fallback for unreadable files"]
    INF --> SUB["Submission probabilities, 4 decimals<br/>submission_catboost_tsfel_gpu_v7.csv"]
```

Intermediate TSFEL features and the four per-target artifacts are cached to disk. With
`USE_SAVED_MODELS = True` the notebook loads existing artifacts and skips training, so a re-run only
regenerates the test predictions.

## Dataset

Each recording (`unique_id`) is a raw `.txt` file holding one test session by one player, sampled at
**85 Hz**, with six-axis IMU time series:

- Accelerometer: `Ax`, `Ay`, `Az`
- Gyroscope: `Gx`, `Gy`, `Gz`

A session normally contains 27 swings, and the file may also include disturbance before and after the
test. `cut_point` gives the organizer's swing boundaries, which may contain slight errors. Datasets are
**not** committed.

The organizer defines ten swing modes:

| `mode` | Test | Task |
| --- | --- | --- |
| 1, 2 | Power | Forehand stroke / forehand loop, air swings (27) |
| 3–8 | Power | Precise forehand hits from a serving machine at 50 balls/min (27): forehand-side diagonal / straight, side-body diagonal / straight, two-point footwork diagonal / straight |
| 9, 10 | Aerobic | Increasing ball frequency until exhaustion; files may hold fewer than 27 swings |

Modes 9 and 10 make up 1,614 of the 1,955 training recordings. Class meanings: `play years` 0 / 1 / 2 =
low / medium / high (binned from all players' playing years); `level` 2 = college Division A, 3 = college
Division B, 4 = youth national team, 5 = youth players.

```text
39_Training_Dataset/
  train_info.csv
  train_data/{unique_id}.txt

39_Test_Dataset/
  test_info.csv
  sample_submission.csv
  test_data/{unique_id}.txt
```

Training metadata columns:

```text
unique_id, player_id, mode, gender, hold racket handed, play years, level, cut_point
```

`player_id` is the grouping key used for leakage-safe cross-validation. This notebook segments each
recording into 27 equal-width windows with `np.linspace`; it does **not** use the `mode` covariate or the
`cut_point` swing boundaries, so segments do not line up exactly with swings, especially when a file
includes pre- or post-test disturbance or fewer than 27 swings.

| Dataset | Recordings | Players |
| --- | ---: | ---: |
| Training | 1,955 | 42 |
| Test | 1,430 | — |

## Evaluation protocol

- Single 5-fold `GroupKFold`, grouped by `player_id`, so a player never appears in both train and
  validation within a fold.
- Every label-dependent step (`SelectKBest`) and every fitted transform (`KNNImputer`, `MinMaxScaler`) is
  fitted on the training folds only.
- Per-target hyperparameter search with Optuna TPE: 75 trials each (300 total across four targets). The
  running mean AUC is reported after each fold so `MedianPruner` can stop weak trials; each study is
  capped at a 2.5-hour wall-clock timeout.
- Competition AUC per target: ROC AUC for the binary targets, micro-OvR ROC AUC for the multiclass
  targets.
- Mean AUC across the four targets as the headline metric.

The best Optuna trial value is still selected on the same folds it is reported on, and the validation
fold drives early stopping, so it remains somewhat optimistic. It is a single split over only 42 players
with no nested loop or uncertainty estimate; an unbiased estimate needs the chosen parameters
re-evaluated on a fresh fold split.

## Model overview

A single modeling approach, trained independently for each of the four targets.

**Features.** Each recording is split into 27 swing segments. For every segment, TSFEL extracts temporal,
spectral, and statistical features at 85 Hz from the 6 raw axes plus 2 L2 magnitudes (acceleration and
gyroscope), giving 992 per-segment dimensions (124 per signal). The 27 segments are aggregated with 7
statistics (`mean / std / median / min / max / skew / kurtosis`) into a 6,944-dim recording-level vector.
TSFEL extraction runs in parallel across CPU cores.

**Selection.** `VarianceThreshold(0.01)` removes near-constant columns. For `gender`, `play years`, and
`level`, an additional `SelectKBest(f_classif)` is fitted inside each fold, with `k` tuned by Optuna
(lower bound 20, upper bound the number of available features); each target keeps its own feature subset.
`hold racket handed` uses every feature that passes the variance threshold.

**Learner.** A per-target `CatBoostClassifier` (`task_type='GPU'`) with fold-local `KNNImputer(5)` and
`MinMaxScaler`, optional Gaussian input noise on the training data (seeded), and early stopping on the
validation fold during CV. The final model is trained on all training data with the number of trees set to
the mean best iteration across the CV folds. Class imbalance is handled with `scale_pos_weight` for
`gender` and optional balanced class weights for `play years` / `level` (toggled by Optuna).

## Changes from the submitted version

Each change is tagged `[修正 N]` in the notebook.

| # | Problem in the submitted version (v6) | Effect | Fix in v7 |
| --- | --- | --- | --- |
| 1 | `SelectKBest` is fitted on all training rows before the `GroupKFold` split | Feature selection sees the validation labels; CV for `gender` / `play years` / `level` is inflated | Fit `SelectKBest` inside each fold on the training rows only |
| 2 | `MedianPruner` is configured but `trial.report()` is never called | No trial is ever pruned | Report the running mean AUC after each fold and check `should_prune()` |
| 3 | When the best trial enables `add_input_noise`, noise is also added to the **test** features | Stochastic test predictions | No noise at inference |
| 4 | The final model is fitted without an `eval_set` | Early stopping never triggers; the final model trains all `iterations` trees, unlike the CV models | Use the mean best iteration from the CV folds |
| 5 | Input noise uses unseeded `np.random` | Runs are not reproducible | `np.random.default_rng(42)` |
| 6 | TSFEL `Spectrogram mean coefficient` columns are named by frequency bins that depend on segment length; every file is aligned to the first file's names | 248 of 1,240 per-swing columns; 92.7% (train) / 92.5% (test) of their values are NaN | Disable that feature (992 per-swing / 6,944 recording-level dims) |
| 7 | On very short recordings (about 10 samples per swing; mostly aerobic modes 9 / 10 stopped early at exhaustion) `LPCC` raises, and the whole signal is discarded | 2 training and 8 test recordings have all-NaN features, so their predictions come entirely from imputation | On failure, compute features one at a time and skip only the failing ones |
| 8 | `TSFEL_SAMPLING_FREQ = 50`, but the organizer states the sensor records 85 samples per second | Frequency-domain features are computed for the wrong sampling rate | `TSFEL_SAMPLING_FREQ = 85` |
| 9 | Probabilities are written with 6 decimal places; the organizer allows at most 4 | Submission format does not follow the rules | `float_format='%.4f'` |

Features, model artifacts and the submission use `_v7` paths, so they never mix with v6 caches.

## Engineering highlights

- Per-target Optuna TPE tuning (75 trials each) with median pruning and a 2.5-hour timeout per study.
- Player-grouped 5-fold `GroupKFold` to avoid player leakage in cross-validation.
- Fold-local `SelectKBest`, `KNNImputer` and `MinMaxScaler`, and seeded input-noise regularization on the
  training folds only.
- Per-target feature selection (`VarianceThreshold` + Optuna-tuned `SelectKBest`).
- Class-imbalance handling: `scale_pos_weight` (`gender`) and optional balanced class weights
  (`play years` / `level`).
- Cached TSFEL features (extracted in parallel) and persisted per-target artifacts
  (model / imputer / scaler / params), with `USE_SAVED_MODELS` for fast re-runs.
- Robust feature extraction for very short recordings, and inference fallbacks (binary `0.5`, multiclass
  `1 / n_classes`) for test recordings whose feature file is missing, empty or unreadable.
- GPU training via CatBoost `task_type='GPU'`.

## Project structure

```text
.
aicup_submission.ipynb                      # the entire pipeline (one notebook)
requirements.txt                            # pinned environment (Python 3.12)
README.md
.gitignore

# generated at runtime, gitignored (not committed):
39_Training_Dataset/                        # official training data
39_Test_Dataset/                            # official test data
tabular_data_train_tsfel_catboost_gpu_v7/   # cached TSFEL segment features (train)
tabular_data_test_tsfel_catboost_gpu_v7/    # cached TSFEL segment features (test)
trained_models_catboost_v7/                 # per-target artifacts (.cbm / .joblib / .json)
submission_catboost_tsfel_gpu_v7.csv        # final submission
```

## Setup

```bash
pip install -r requirements.txt jupyterlab
```

Requirements:

- Python 3.12. `tsfel==0.1.9` does not run on Python 3.13 or newer.
- `PyWavelets==1.8.0`. Newer releases change TSFEL's Wavelet feature values.
- An NVIDIA GPU + CUDA (CatBoost `task_type='GPU'`).

## Running

1. Place `39_Training_Dataset/` and `39_Test_Dataset/` at the repository root.
2. Run all cells of [aicup_submission.ipynb](aicup_submission.ipynb).

Intermediate features and model artifacts are cached, so subsequent runs skip the expensive steps. To
**retrain from scratch**, set `USE_SAVED_MODELS = False` near the top of the notebook, or delete
`trained_models_catboost_v7/` and the two `tabular_data_*_v7/` cache folders.

Main tunable constants:

| Constant | Default | Meaning |
| --- | --- | --- |
| `NUM_TOTAL_SEGMENTS_PER_FILE` | 27 | Swing segments per recording |
| `N_SPLITS_GROUPKFOLD` | 5 | GroupKFold folds |
| `OPTUNA_N_TRIALS` | 75 | Optuna trials per target |
| `TSFEL_SAMPLING_FREQ` | 85 | Sensor sampling rate (Hz) used by TSFEL |
| `N_JOBS_TSFEL` | -1 | Parallel processes for TSFEL extraction (-1 = all cores) |
| `USE_SAVED_MODELS` | `True` | Load saved artifacts and skip training |

## Honest interpretation

What carried the result was not a single clever trick but a few solid choices:

1. Player-grouped cross-validation, so no player appears on both sides of a split.
2. Fold-local selection, imputation and scaling fitted only on training folds.
3. A rich TSFEL representation aggregated to the recording level.
4. Per-target feature selection and per-target Optuna tuning rather than one shared configuration.

Known limitations:

- The reported CV is the best Optuna trial value, so model selection and evaluation share folds; it is
  optimistic rather than fully held-out.
- A single split over only 42 players, with no nested loop and no uncertainty estimate.
- Equal-width segmentation ignores the provided `cut_point` swing boundaries and the `mode` covariate.
- Everything lives in one notebook, which makes testing and reproduction harder.

## Data & license

The AI CUP competition datasets are **not** included or redistributed here and remain subject to their
original competition terms. No separate code-license file is provided in this repository.

## References

- CatBoost: <https://catboost.ai>
- TSFEL: <https://tsfel.readthedocs.io>
- Optuna: <https://optuna.org>
- scikit-learn: <https://scikit-learn.org>

## AI Assistance Disclosure

AI tools were used to assist with English translation, language polishing, and formatting. The
model design and core technical decisions were developed, evaluated, and validated by the author. All
AI-assisted content was reviewed and revised before publication.
