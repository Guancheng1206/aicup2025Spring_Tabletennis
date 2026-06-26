# AI CUP 2025 春季賽 — 桌球智慧球拍 IMU 分類

Solo 參賽,633 隊中獲全國第 14 名(佳作獎,教育部獎狀)。
Private Leaderboard ROC AUC: **0.806**。

競賽頁面:[T-Brain Competition #39](https://tbrain.trendmicro.com.tw/Competitions/Details/39)
主辦:教育部資訊及科技教育司 · 運籌:教育部人工智慧競賽與標註資料蒐集計畫辦公室

---

## 任務

從六軸 IMU 揮拍訊號預測桌球選手的四項屬性。每筆 recording 為一個 `.txt` 檔,內含 27 次揮拍的連續六軸時序資料(三軸加速度 `Ax/Ay/Az`、三軸角速度 `Gx/Gy/Gz`)。

| 目標 | 類別數 | 評估指標 |
| --- | ---: | --- |
| `gender` | 2 | ROC AUC |
| `hold racket handed` | 2 | ROC AUC |
| `play years` | 3 | Micro-OvR ROC AUC |
| `level` | 4 | Micro-OvR ROC AUC |

最終分數為四項指標之算術平均。訓練集 1,955 筆 recording,共 42 位選手;測試集 1,430 筆 recording。

## 方法

**特徵抽取**。每筆 `.txt` 平均切成 27 個 swing segment。每個 segment 對 6 軸原始訊號 + 2 個 L2 magnitude(加速度合成、角速度合成)以 TSFEL 抽取 1,240 維時序特徵(時域 + 頻域 + 統計域)。27 個 segment 經 7 種統計量聚合為 recording-level 特徵向量:

```
8,680 維 = 1,240 (TSFEL per segment) × 7 (mean / std / median / min / max / skew / kurtosis)
```

**特徵篩選**。`VarianceThreshold(0.01)` 移除近常數欄位;對 `gender / play years / level` 三個 target 再以 `SelectKBest(f_classif)` 篩選,`k` 由 Optuna 搜尋。

**模型**。四個 target 各訓練獨立的 `CatBoost` 分類器(`task_type='GPU'`)。預處理採 `KNNImputer(n_neighbors=5)` + `MinMaxScaler`。類別不平衡處理:

- `gender`:Optuna 搜尋 `scale_pos_weight`,範圍以實際樣本比 0.83 / 0.17 為中心
- `play years` / `level`:fold-local balanced class weights(每個 fold 各自以 train labels 計算)

**評估與調參**。`player_id` 分組之 5-fold `GroupKFold`,確保同一選手不會同時出現在 train 與 validation。Optuna TPE 對每個 target 跑 75 trials,搜尋空間涵蓋:

- CatBoost 超參數:`iterations` (500–2,500)、`learning_rate` (5e-4 – 5e-2)、`depth` (3–6)、`l2_leaf_reg`、`border_count`、`random_strength`、`bagging_temperature`、`early_stopping_rounds`
- `SelectKBest` 的 `k`(僅 `gender / play years / level`)
- 訓練時是否加入輸入噪聲及噪聲強度
- 多類別 target 是否啟用 balanced class weights

## 結果

Optuna best CV scores(每個 target 的內部 5-fold CV 最佳分數):

| 目標 | Best CV |
| --- | ---: |
| `gender` | 0.7963 |
| `hold racket handed` | 0.9996 |
| `play years` | 0.6629 |
| `level` | 0.8595 |
| **mean** | **0.8296** |

Private Leaderboard ROC AUC: **0.806**(主辦方在未公開測試集上計算)。

## 專案結構

```
.
├── 39_Training_Dataset/                       # 官方訓練資料(未納入 repo)
├── 39_Test_Dataset/                           # 官方測試資料(未納入 repo)
├── aicup_submission.ipynb                     # 主執行 notebook
├── tabular_data_train_tsfel_catboost_gpu_v6/  # TSFEL segment 特徵 cache(自動生成)
├── tabular_data_test_tsfel_catboost_gpu_v6/   # TSFEL segment 特徵 cache(自動生成)
├── trained_models_catboost_v6/                # 訓練後模型與 preprocessor(自動生成)
│   ├── catboost_model_{target}.cbm
│   ├── imputer_{target}.joblib
│   ├── scaler_{target}.joblib
│   └── best_params_meta_{target}.json
└── submission_catboost_tsfel_gpu_v6.csv       # 最終提交檔(自動生成)
```

## 環境

- Python 3.8+
- NVIDIA GPU + CUDA(CatBoost `task_type='GPU'`)
- 套件:`numpy`, `pandas`, `scikit-learn`, `catboost`, `optuna`, `tsfel`, `joblib`, `jupyterlab`

```bash
pip install numpy pandas scikit-learn catboost optuna tsfel joblib jupyterlab
```

## 執行

1. 將 `39_Training_Dataset/` 與 `39_Test_Dataset/` 放在 repo 根目錄
2. 在 Jupyter 中執行 `aicup_submission.ipynb` 所有 cells

中間特徵與訓練後模型會自動 cache,重複執行時跳過耗時步驟。重新訓練需設定 notebook 開頭 `USE_SAVED_MODELS = False`,或直接刪除 `trained_models_catboost_v6/` 與兩個 `tabular_data_*/` 資料夾。

## 主要可調參數

| 參數 | 預設 | 說明 |
| --- | --- | --- |
| `NUM_TOTAL_SEGMENTS_PER_FILE` | 27 | 每筆訊號切分的 segment 數 |
| `N_SPLITS_GROUPKFOLD` | 5 | GroupKFold 折數 |
| `OPTUNA_N_TRIALS` | 75 | 每個 target 的 Optuna 試驗次數 |
| `TSFEL_SAMPLING_FREQ` | 50 | TSFEL 的取樣率參數 |
| `USE_SAVED_MODELS` | `True` | 是否載入已存模型跳過訓練 |
