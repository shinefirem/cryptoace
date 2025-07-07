"""
CryptoAce 日誌管理模組

此模組基於 loguru 函式庫，提供全局唯一的、可配置的日誌記錄器實例。
"""

import sys
from typing import Any
from loguru import logger

# 處理相對匯入問題
try:
    from .configurator import Configurator
except ImportError:
    # 當直接執行此檔案時，使用絕對匯入
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.configurator import Configurator


def setup_logger(config: Configurator) -> Any:
    """
    設定並配置日誌記錄器
    
    Args:
        config: 配置管理器實例
        
    Returns:
        配置好的 logger 實例
    """
    # 移除 loguru 的預設處理器
    logger.remove()
    
    # 獲取日誌配置
    log_config = config.logger
    log_level = log_config.get('level', 'INFO')
    log_path = log_config.get('log_path', './logs/cryptoace.log')
    rotation = log_config.get('rotation', '10 MB')
    retention = log_config.get('retention', '7 days')
    
    # 設定日誌格式
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    # 添加控制台處理器
    logger.add(
        sys.stdout,
        format=log_format,
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # 添加文件處理器
    logger.add(
        log_path,
        format=log_format,
        level=log_level,
        rotation=rotation,
        retention=retention,
        compression="zip",
        backtrace=True,
        diagnose=True,
        encoding="utf-8"
    )
    
    # 記錄初始化訊息
    logger.info(f"日誌系統初始化完成 - 級別: {log_level}, 路徑: {log_path}")
    
    return logger


def get_logger() -> Any:
    """
    獲取全局日誌記錄器實例
    
    Returns:
        全局 logger 實例
    """
    return logger


def setup_default_logger() -> Any:
    """
    使用預設配置設定日誌記錄器
    
    Returns:
        配置好的 logger 實例
    """
    try:
        from .configurator import load_config
    except ImportError:
        # 當直接執行此檔案時，使用絕對匯入
        from core.configurator import load_config
    
    try:
        config = load_config()
        return setup_logger(config)
    except Exception as e:
        # 如果配置載入失敗，使用基本設定
        logger.remove()
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
            level="INFO"
        )
        logger.warning(f"使用預設日誌配置，配置載入失敗: {e}")
        return logger


if __name__ == "__main__":
    """單元測試區塊"""
    
    print("=== CryptoAce 日誌系統測試 ===")
    
    try:
        # 實例化配置管理器
        try:
            from .configurator import load_config
        except ImportError:
            # 當直接執行此檔案時，使用絕對匯入
            from core.configurator import load_config
            
        config = load_config()
        
        # 設定日誌記錄器
        test_logger = setup_logger(config)
        
        print("\n測試不同級別的日誌訊息:")
        
        # 測試不同級別的日誌訊息
        test_logger.debug("這是一條 DEBUG 級別的訊息")
        test_logger.info("這是一條 INFO 級別的訊息")
        test_logger.warning("這是一條 WARNING 級別的訊息")
        test_logger.error("這是一條 ERROR 級別的訊息")
        
        # 測試帶有額外資訊的日誌
        test_logger.info("系統啟動", extra={"component": "main", "version": "1.0.0"})
        
        # 測試結構化日誌
        test_logger.bind(user_id=12345, action="login").info("用戶登入成功")
        
        print("\n✅ 日誌系統測試完成！")
        print(f"日誌檔案位置: {config.logger.get('log_path', './logs/cryptoace.log')}")
        
    except Exception as e:
        print(f"❌ 日誌系統測試失敗: {e}")
        
        # 使用預設配置進行測試
        print("\n嘗試使用預設配置...")
        try:
            default_logger = setup_default_logger()
            default_logger.info("使用預設配置的日誌測試")
            print("✅ 預設配置測試成功！")
        except Exception as default_e:
            print(f"❌ 預設配置測試也失敗: {default_e}")