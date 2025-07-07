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
