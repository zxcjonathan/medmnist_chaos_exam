# AI 驅動的醫療影像分析專案

## 專案概述

此專案旨在解決一個模擬的 AI 驅動醫療影像分析筆試任務。專案內容涵蓋：

* **模型評估**：分析一個急診室胸部 X 光片預測模型的效能。
* **影像預處理**：處理手部 X 光片 DICOM 檔案，進行去背和對齊。
* **程式碼重構**：將 MedMNIST 範例程式碼重構為模組化、可執行的專案。
* **資料探索**：對 CHAOS 資料集進行探索性分析，並處理其 DICOM 檔案。

## 專案結構

/medmnist_chaos_exam
├─ src/
│  ├─ model.py          # 定義神經網路模型架構 (MedMNIST)
│  ├─ train.py          # 包含模型訓練邏輯
│  └─ test.py           # 負責載入模型權重並進行評估
├─ notebooks/
│  ├─ eda_medmnist.ipynb  # 對 MedMNIST 資料集進行 EDA
│  └─ eda_chaos.ipynb     # 處理 CHAOS 資料集中的 DICOM 檔案並進行 EDA
└─ psudo_result.csv      # 考題 1 的原始數據


## 如何運行專案

### 1. 安裝環境
請確保你的 Python 環境已安裝所有必要的套件。你可以使用以下指令：

```bash
pip install -r requirements.txt
(註：你需要手動建立 requirements.txt 檔案，將所有用到的套件列入，如 torch, torchvision, medmnist, pydicom, matplotlib 等。)

2. 執行腳本
MedMNIST 訓練與測試：

Bash

python -m src.train
python -m src.test
資料探索 (EDA)：
開啟 notebooks/eda_medmnist.ipynb 和 notebooks/eda_chaos.ipynb，按照筆記本中的步驟執行，以重現我的分析和視覺化結果。

成果與洞見
模型評估：在 psudo_result.csv 數據集上，我們計算了模型的準確率、精確率和召回率，並透過混淆矩陣分析其表現。在簡報中，我詳細解釋了為什麼在醫療場景中，召回率比準確率更為關鍵。

CHAOS 資料集：我成功地解析了 CHAOS 資料集中的 DICOM 檔案，並提取了關鍵元數據，例如患者 ID 和成像模態，這對於處理真實世界中的醫學影像數據至關重要。