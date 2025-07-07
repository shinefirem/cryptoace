#!/bin/bash

# CryptoAce 專案設置腳本
# 創建完整的 Python 專案目錄結構

echo "開始創建 CryptoAce 專案結構..."

# 創建根目錄
mkdir -p cryptoace

# 創建主要目錄結構
mkdir -p cryptoace/core
mkdir -p cryptoace/tests
mkdir -p cryptoace/data/raw
mkdir -p cryptoace/data/features
mkdir -p cryptoace/logs
mkdir -p cryptoace/models

echo "目錄結構創建完成"

# 創建 core 模組檔案
echo "創建核心模組檔案..."
touch cryptoace/core/__init__.py
touch cryptoace/core/configurator.py
touch cryptoace/core/logger.py
touch cryptoace/core/utils.py
touch cryptoace/core/interfaces.py
touch cryptoace/core/data_harvester.py
touch cryptoace/core/feature_engine.py
touch cryptoace/core/trading_env.py
touch cryptoace/core/agent.py
touch cryptoace/core/trainer.py
touch cryptoace/core/backtester.py
touch cryptoace/core/live_trader.py

# 創建測試檔案
echo "創建測試檔案..."
touch cryptoace/tests/__init__.py
touch cryptoace/tests/test_configurator.py
touch cryptoace/tests/test_feature_engine.py
touch cryptoace/tests/test_trading_env.py

# 創建主執行檔案
echo "創建主執行檔案..."
cat > cryptoace/main.py << 'EOF'
#!/usr/bin/env python3
"""
CryptoAce - 加密貨幣交易系統主程式
"""

import sys
import os
from pathlib import Path

# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.logger import setup_logger
from core.configurator import load_config


def main():
    """主程式入口點"""
    logger = setup_logger()
    logger.info("CryptoAce 系統啟動")
    
    try:
        # 載入配置
        config = load_config()
        logger.info("配置載入完成")
        
        # TODO: 實現主要業務邏輯
        logger.info("系統運行中...")
        
    except Exception as e:
        logger.error(f"系統運行錯誤: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
EOF

# 創建配置檔案
echo "創建配置檔案..."
cat > cryptoace/config.yaml << 'EOF'
# CryptoAce 配置檔案

# 交易設定
trading:
  exchange: "binance"
  symbols: ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
  timeframe: "1h"
  
# 數據設定
data:
  raw_path: "data/raw"
  features_path: "data/features"
  lookback_days: 30
  
# 模型設定
model:
  type: "dqn"
  save_path: "models"
  checkpoint_interval: 100
  
# 日誌設定
logging:
  level: "INFO"
  file_path: "logs/cryptoace.log"
  max_size: "10MB"
  backup_count: 5
  
# 回測設定
backtest:
  initial_balance: 10000
  commission: 0.001
  start_date: "2023-01-01"
  end_date: "2023-12-31"
  
# 即時交易設定
live_trading:
  enabled: false
  paper_trading: true
  risk_per_trade: 0.02
EOF

# 創建需求檔案
echo "創建需求檔案..."
cat > cryptoace/requirements.txt << 'EOF'
# 核心依賴
numpy>=1.21.0
pandas>=1.3.0
pyyaml>=6.0

# 機器學習
scikit-learn>=1.0.0
tensorflow>=2.8.0
stable-baselines3>=1.6.0

# 數據處理
ccxt>=3.0.0
ta-lib>=0.4.0
plotly>=5.0.0

# 資料庫
sqlalchemy>=1.4.0
psycopg2-binary>=2.9.0

# 工具
requests>=2.28.0
python-dotenv>=0.19.0
schedule>=1.1.0

# 開發工具
pytest>=7.0.0
black>=22.0.0
flake8>=4.0.0
mypy>=0.950
EOF

# 創建 .gitignore 檔案
echo "創建 .gitignore 檔案..."
cat > cryptoace/.gitignore << 'EOF'
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# PEP 582
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# 專案特定
/data/raw/*
/data/features/*
/logs/*
/models/*
*.pkl
*.h5
*.weights

# 保留目錄結構的空檔案
!.gitkeep

# 敏感資訊
.env
config_local.yaml
api_keys.txt
EOF

# 創建環境變數範本
echo "創建環境變數範本..."
cat > cryptoace/.env.example << 'EOF'
# CryptoAce 環境變數範本
# 複製此檔案為 .env 並填入實際值

# 交易所 API 金鑰
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET_KEY=your_binance_secret_key_here

# 資料庫連線
DATABASE_URL=postgresql://username:password@localhost:5432/cryptoace

# 日誌設定
LOG_LEVEL=INFO

# 開發模式
DEBUG=False

# Telegram 通知 (可選)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id

# 其他 API 金鑰
COINGECKO_API_KEY=your_coingecko_api_key
EOF

# 創建保持目錄結構的 .gitkeep 檔案
echo "創建 .gitkeep 檔案..."
touch cryptoace/data/raw/.gitkeep
touch cryptoace/data/features/.gitkeep
touch cryptoace/logs/.gitkeep
touch cryptoace/models/.gitkeep

echo ""
echo "✅ CryptoAce 專案結構創建完成！"
echo ""
echo "專案目錄結構："
echo "cryptoace/"
echo "├── core/"
echo "│   ├── __init__.py"
echo "│   ├── configurator.py"
echo "│   ├── logger.py"
echo "│   ├── utils.py"
echo "│   ├── interfaces.py"
echo "│   ├── data_harvester.py"
echo "│   ├── feature_engine.py"
echo "│   ├── trading_env.py"
echo "│   ├── agent.py"
echo "│   ├── trainer.py"
echo "│   ├── backtester.py"
echo "│   └── live_trader.py"
echo "├── tests/"
echo "│   ├── __init__.py"
echo "│   ├── test_configurator.py"
echo "│   ├── test_feature_engine.py"
echo "│   └── test_trading_env.py"
echo "├── data/"
echo "│   ├── raw/"
echo "│   └── features/"
echo "├── logs/"
echo "├── models/"
echo "├── main.py"
echo "├── config.yaml"
echo "├── requirements.txt"
echo "├── .gitignore"
echo "└── .env.example"
echo ""
echo "下一步："
echo "1. cd cryptoace"
echo "2. cp .env.example .env"
echo "3. 編輯 .env 檔案填入您的 API 金鑰"
echo "4. pip install -r requirements.txt"
echo "5. python main.py"
echo ""
echo "專案設置完成！🚀"
