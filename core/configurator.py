"""
CryptoAce 配置管理模組

此模組負責讀取和管理系統配置，包括 YAML 配置檔案和環境變數的整合。
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv


class Configurator:
    """
    配置管理類別
    
    負責讀取 config.yaml 檔案和 .env 環境變數，
    並提供統一的配置訪問接口。
    """
    
    def __init__(self, config_path: str) -> None:
        """
        初始化配置管理器
        
        Args:
            config_path: config.yaml 檔案的路徑
        """
        # 載入環境變數
        load_dotenv()
        
        # 讀取並解析 YAML 配置檔案
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"配置檔案不存在: {config_path}")
            
        with open(config_file, 'r', encoding='utf-8') as file:
            self._config: Dict[str, Any] = yaml.safe_load(file)
        
        # 合併 BITGET 環境變數到配置中
        self._merge_bitget_env_vars()
    
    def _merge_bitget_env_vars(self) -> None:
        """
        將 BITGET 相關的環境變數合併到配置中
        
        將所有 BITGET_ 開頭的環境變數添加到 exchange 配置區塊
        """
        if 'exchange' not in self._config:
            self._config['exchange'] = {}
        
        # 掃描所有環境變數，尋找 BITGET_ 開頭的變數
        for key, value in os.environ.items():
            if key.startswith('BITGET_'):
                # 將環境變數名稱轉換為小寫配置鍵
                config_key = key.lower()
                self._config['exchange'][config_key] = value
    
    @property
    def exchange(self) -> Dict[str, Any]:
        """
        獲取交易所配置
        
        Returns:
            交易所相關配置字典
        """
        return self._config.get('exchange', {})
    
    @property
    def data(self) -> Dict[str, Any]:
        """
        獲取數據配置
        
        Returns:
            數據相關配置字典
        """
        return self._config.get('data', {})
    
    @property
    def trading(self) -> Dict[str, Any]:
        """
        獲取交易配置
        
        Returns:
            交易相關配置字典
        """
        return self._config.get('trading', {})
    
    @property
    def trading_env(self) -> Dict[str, Any]:
        """
        獲取交易環境配置
        
        Returns:
            交易環境相關配置字典
        """
        return self._config.get('trading_env', {})
    
    @property
    def agent(self) -> Dict[str, Any]:
        """
        獲取代理模型配置
        
        Returns:
            代理模型相關配置字典
        """
        return self._config.get('agent', {})
    
    @property
    def livetrader(self) -> Dict[str, Any]:
        """
        獲取實時交易配置
        
        Returns:
            實時交易相關配置字典
        """
        return self._config.get('livetrader', {})
    
    @property
    def logger(self) -> Dict[str, Any]:
        """
        獲取日誌配置
        
        Returns:
            日誌相關配置字典
        """
        return self._config.get('logger', {})
    
    @property
    def training(self) -> Dict[str, Any]:
        """
        獲取訓練配置
        
        Returns:
            訓練相關配置字典
        """
        return self._config.get('training', {})
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        獲取配置值
        
        Args:
            key: 配置鍵名
            default: 預設值
            
        Returns:
            配置值或預設值
        """
        return self._config.get(key, default)
    
    def get_nested(self, *keys: str, default: Any = None) -> Any:
        """
        獲取嵌套配置值
        
        Args:
            *keys: 嵌套的配置鍵名序列
            default: 預設值
            
        Returns:
            嵌套配置值或預設值
        """
        current = self._config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current


def load_config(config_path: str = "config.yaml") -> Configurator:
    """
    載入配置的便利函數
    
    Args:
        config_path: 配置檔案路徑，預設為 "config.yaml"
        
    Returns:
        配置管理器實例
    """
    # 如果路徑不是絕對路徑，則相對於專案根目錄
    if not os.path.isabs(config_path):
        project_root = Path(__file__).parent.parent
        config_path = str(project_root / config_path)
    
    return Configurator(config_path)


if __name__ == "__main__":
    """單元測試區塊"""
    
    # 創建配置管理器實例
    try:
        config = load_config()
        
        print("=== CryptoAce 配置測試 ===")
        print()
        
        # 測試交易所配置
        print("1. 交易所配置:")
        print(f"   exchange: {config.exchange}")
        print()
        
        # 測試嵌套配置訪問
        print("2. PPO 學習率:")
        learning_rate = config.get_nested('agent', 'ppo', 'learning_rate')
        print(f"   learning_rate: {learning_rate}")
        print()
        
        # 測試環境變數整合
        print("3. API 金鑰 (從環境變數):")
        api_key = config.exchange.get('bitget_api_key')
        print(f"   api_key: {api_key if api_key else '未設定'}")
        print()
        
        # 測試其他配置區塊
        print("4. 其他配置區塊:")
        print(f"   data: {config.data}")
        print(f"   trading_env.initial_balance: {config.trading_env.get('initial_balance')}")
        print(f"   logger.level: {config.logger.get('level')}")
        print()
        
        print("✅ 配置載入成功！")
        
    except Exception as e:
        print(f"❌ 配置載入失敗: {e}")
        print("請確保:")
        print("1. config.yaml 檔案存在於專案根目錄")
        print("2. .env 檔案已正確設定 (可選)")