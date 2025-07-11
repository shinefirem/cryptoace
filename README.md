# CryptoAce

基於強化學習的加密貨幣交易系統，使用 PPO + Transformer 架構進行智能交易決策。

## 快速開始

### 1. 安裝依賴
```bash
pip install -r requirements.txt
```

### 2. 配置設定
編輯 `config.yaml` 檔案，設定交易所 API 和交易參數。

### 3. 基本使用

#### 訓練模型
```bash
# 基本訓練（使用增強預設參數）
python main.py train

# 自訂訓練參數
python main.py train --window-size 5000 --timesteps 50000
```

#### 策略回測
```bash
python main.py backtest --experiment ./experiments/your_experiment
```

#### 實時交易
```bash
# 乾跑模式（不執行真實交易）
python main.py live --dry-run

# 實際交易
python main.py live --experiment ./experiments/your_experiment
```

## 核心功能

- **智能訓練**: 滾動窗口訓練，支援經驗回放
- **策略回測**: 完整的歷史數據回測分析
- **實時交易**: 異步交易循環，狀態持久化
- **風險管理**: 內建倉位控制和風險檢查

## 系統架構

```
core/
├── trainer.py        # 模型訓練器
├── agent.py          # 強化學習智能體  
├── backtester.py     # 策略回測器
├── live_trader.py    # 實時交易器
├── data_harvester.py # 數據收集器
├── feature_engine.py # 特徵工程引擎
└── trading_env.py    # 交易環境模擬器
```

## 命令參數

### 訓練參數
- `--window-size`: 訓練窗口大小
- `--timesteps`: 每步訓練時間步數  
- `--walk-steps`: 滾動窗口步數
- `--epochs`: 訓練輪數

### 回測參數
- `--experiment`: 實驗目錄路徑（必需）
- `--start-date`: 回測開始日期
- `--end-date`: 回測結束日期

### 實時交易參數
- `--dry-run`: 乾跑模式
- `--experiment`: 使用的訓練模型
- `--state-file`: 狀態持久化檔案

## 注意事項

⚠️ **風險提醒**: 加密貨幣交易具有高風險，請謹慎使用實時交易功能。

建議先進行充分的回測驗證，並在小額資金上測試後再進行大規模交易。
