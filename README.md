````markdown

## AI CUP 2025 春季賽桌球智慧球拍資料的精準分析競賽

## 專案概述

本專案旨在根據羽球選手揮拍的 IMU（慣性測量單元）感測器數據，預測其四項個人屬性：
1.  **性別** (`gender`)
2.  **持拍手** (`hold racket handed`)
3.  **球齡** (`play years`)
4.  **羽球程度** (`level`)

此解決方案利用 `tsfel` 套件進行大規模時間序列特徵提取，並針對四個目標分別訓練了四個獨立的 `CatBoost` 分類模型。為了達到最佳預測效能，模型訓練整合了 `optuna` 進行超參數搜索，並採用以 `player_id` 分組的 `GroupKFold` 交叉驗證，以防止數據洩漏並提升模型的泛化能力。

## 專案架構

```
.
├── 39_Training_Dataset/      # 官方訓練資料集
│   ├── train_data/           # 包含所有訓練用 .txt 感測器數據
│   └── train_info.csv        # 訓練數據的標籤與選手資訊
├── 39_Test_Dataset/          # 官方測試資料集
│   ├── test_data/            # 包含所有測試用 .txt 感測器數據
│   ├── test_info.csv         # 測試數據的選手資訊
│   └── sample_submission.csv # 提交格式範例
├── tabular_data_train_tsfel_catboost_gpu_v6/ # (自動生成) 訓練集的中間特徵檔
├── tabular_data_test_tsfel_catboost_gpu_v6/  # (自動生成) 測試集的中間特徵檔
├── trained_models_catboost_v6/             # (自動生成) 儲存已訓練好的模型與相關檔案
│   ├── catboost_model_{target}.cbm
│   ├── imputer_{target}.joblib
│   ├── scaler_{target}.joblib
│   └── best_params_meta_{target}.json
├── .gitignore                # Git 忽略檔案設定
├── aicup_submission.ipynb    # 主執行腳本 (Jupyter Notebook)
├── submission_catboost_tsfel_gpu_v6.csv # (自動生成) 最終提交的預測結果
└── README.md                 # 本說明文件
```

## 核心方法 (Methodology)

本專案的處理流程遵循一個標準的機器學習管線 (Pipeline)：

1.  **特徵提取 (Feature Extraction)**:
    * 遍歷 `train_data` 和 `test_data` 中的每個 `.txt` 原始數據檔。
    * 將每個檔案的時序數據平均分割成 27 個片段 (segments)。
    * 使用 `tsfel` 為每個片段的六軸感測器數據（加速度計 x/y/z, 陀螺儀 x/y/z）及其合成向量提取數百個時序特徵。
    * 將提取出的特徵暫存於 `tabular_data_*` 目錄下，每個原始 `.txt` 檔對應一個 `.csv` 特徵檔。

2.  **特徵聚合 (Feature Aggregation)**:
    * 針對單一選手，讀取其對應的 `.csv` 特徵檔（包含 27 個片段的特徵）。
    * 將這些片段的特徵進行統計聚合（計算 `mean`, `std`, `median`, `min`, `max`, `skew`, `kurtosis`），形成一個代表該選手的單一特徵向量。

3.  **模型訓練與優化 (Training & Optimization)**:
    * **獨立模型**: 為四個預測目標分別訓練獨立的 `CatBoost` 模型。
    * **數據預處理**: 訓練流程中自動進行 `KNNImputer` 缺失值填補和 `MinMaxScaler` 特徵縮放。
    * **交叉驗證**: 使用 `GroupKFold` 並以 `player_id` 分組，確保同一選手的數據不會同時出現在訓練集與驗證集，以準確評估模型泛化能力。
    * **超參數搜索**: 利用 `optuna` 自動化搜索 `CatBoost` 的最佳超參數，並對特定目標進行特徵數量篩選 (`SelectKBest`) 與類別不平衡處理。
    * **模型儲存**: 每個目標的最佳模型 (`.cbm`)、預處理元件 (`.joblib`) 及參數 (`.json`) 都會被儲存至 `trained_models_catboost_v6` 目錄。

4.  **預測與提交 (Prediction & Submission)**:
    * 在測試集上重複特徵提取與聚合流程。
    * 載入已儲存的四個模型及對應的預處理元件。
    * 對處理好的測試集特徵進行預測，產出各類別的機率值。
    * 依據 `sample_submission.csv` 的格式，生成最終的 `submission_catboost_tsfel_gpu_v6.csv` 檔案。

## 環境設定與安裝

1.  **硬體需求**:
    * 強烈建議使用配備 **NVIDIA GPU** 且已安裝 **CUDA Toolkit** 的環境，因 `CatBoost` 已設定為 `task_type: 'GPU'` 以大幅加速訓練過程。

2.  **Python 環境**:
    * 建議使用 Python 3.8 或更高版本。
    * 需要安裝 Jupyter 環境（如 Jupyter Lab, Jupyter Notebook 或 VS Code 的 Jupyter 擴充套件）。

3.  **安裝必要套件**:
    您可以透過 pip 一次性安裝所有必要的 Python 函式庫。建立一個 `requirements.txt` 檔案：

    **requirements.txt:**
    ```
    numpy
    pandas
    scikit-learn
    catboost
    optuna
    tsfel
    joblib
    jupyterlab
    ```

    使用以下指令進行安裝：
    ```bash
    pip install -r requirements.txt
    ```

## 如何執行與重現結果

1.  **準備資料** 📁:
    * 將官方提供的 `39_Training_Dataset` 和 `39_Test_Dataset` 資料夾放置在與 `aicup_submission.ipynb` 相同的專案根目錄下。

2.  **執行筆記本** ▶️:
    * 在您的 Jupyter 環境中（例如 VS Code 或 Jupyter Lab）打開 `aicup_submission.ipynb` 檔案。
    * 依照由上到下的順序，執行筆記本中的所有儲存格 (cells)，即可啟動完整的執行流程。

3.  **執行流程說明**:
    * **首次執行**: 程式會自動檢查並生成所需的中間特徵檔案 (`tabular_data_*`) 與模型檔案 (`trained_models_catboost_v6`)。此過程會耗費較長時間，尤其是特徵提取與超參數搜索階段。
    * **重複執行**: 如果相關檔案已存在，程式會跳過耗時的步驟，直接載入已有的特徵和模型來加速執行。

4.  **如何重新訓練 (Reproducibility)** 🔄:
    * 若要從頭開始重現整個訓練流程，請在筆記本的程式碼中找到以下全域常數，並將其值修改為 `False`：
        ```python
        USE_SAVED_MODELS = False 
        ```
    * 或者，直接刪除 `trained_models_catboost_v6` 和 `tabular_data_*` 這兩個自動生成的資料夾，然後重新執行筆記本，即可觸發完整的特徵工程與模型訓練。

## 可配置參數

在 `aicup_submission.ipynb` 腳本的開頭，您可以調整一些全域常數來控制腳本行為：

* `NUM_TOTAL_SEGMENTS_PER_FILE`: 每個 `.txt` 檔要切分的片段數量。
* `N_SPLITS_GROUPKFOLD`: 交叉驗證的折數。
* `OPTUNA_N_TRIALS`: `optuna` 進行超參數搜索的試驗總次數。
* `USE_SAVED_MODELS`: 是否載入已儲存的模型以跳過訓練。
* `MODELS_DIR`: 存放模型的目錄名稱。
* `tsfel_cfg`: TSFEL 特徵提取的設定，可自行增減要提取的特徵。
````