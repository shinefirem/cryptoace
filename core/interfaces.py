"""
CryptoAce 抽象接口定義模組

此模組使用 ABC (Abstract Base Class) 定義核心類別的抽象接口，
作為模組間的契約，確保實現類別符合預期的行為規範。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
import gymnasium as gym


class IAgent(ABC):
    """
    交易代理抽象接口
    
    定義了所有交易代理必須實現的核心方法，包括訓練、預測、
    模型保存和載入等功能。
    """
    
    @abstractmethod
    def train(self, 
              env: 'ITradingEnv', 
              total_timesteps: int,
              **kwargs: Any) -> Dict[str, Any]:
        """
        訓練代理模型
        
        Args:
            env: 交易環境實例
            total_timesteps: 總訓練步數
            **kwargs: 其他訓練參數
            
        Returns:
            訓練結果字典，包含損失值、獎勵等資訊
        """
        pass
    
    @abstractmethod
    def predict(self, 
                observation: np.ndarray, 
                deterministic: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        根據觀察值預測動作
        
        Args:
            observation: 環境觀察值
            deterministic: 是否使用確定性預測
            
        Returns:
            動作值和狀態值 (如果適用)
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        保存模型到指定路徑
        
        Args:
            path: 模型保存路徑
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        從指定路徑載入模型
        
        Args:
            path: 模型載入路徑
        """
        pass


class ITradingEnv(gym.Env, ABC):
    """
    交易環境抽象接口
    
    繼承自 gymnasium.Env 和 ABC，定義了交易環境的基本結構。
    雖然 gym.Env 的方法不是嚴格的抽象方法，但通過這種方式
    可以清晰地表明我們的設計意圖。
    """
    
    @abstractmethod
    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        重置環境到初始狀態
        
        Args:
            **kwargs: 重置參數
            
        Returns:
            初始觀察值和資訊字典
        """
        pass
    
    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        執行一個動作步驟
        
        Args:
            action: 要執行的動作
            
        Returns:
            觀察值、獎勵、是否結束、是否截斷、資訊字典
        """
        pass
    
    @abstractmethod
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        渲染環境
        
        Args:
            mode: 渲染模式
            
        Returns:
            渲染結果 (如果適用)
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """
        關閉環境，釋放資源
        """
        pass
    
    @abstractmethod
    def get_portfolio_value(self) -> float:
        """
        獲取當前投資組合價值
        
        Returns:
            投資組合總價值
        """
        pass
    
    @abstractmethod
    def get_position(self) -> float:
        """
        獲取當前持倉
        
        Returns:
            當前持倉比例 (-1 到 1 之間)
        """
        pass
    
    @abstractmethod
    def get_market_data(self) -> pd.DataFrame:
        """
        獲取市場數據
        
        Returns:
            市場數據 DataFrame
        """
        pass


class IDataProvider(ABC):
    """
    數據提供者抽象接口
    
    定義了數據獲取和處理的標準接口。
    """
    
    @abstractmethod
    def fetch_data(self, 
                   symbol: str, 
                   timeframe: str,
                   start_date: str,
                   end_date: str) -> pd.DataFrame:
        """
        獲取市場數據
        
        Args:
            symbol: 交易對符號
            timeframe: 時間週期
            start_date: 開始日期
            end_date: 結束日期
            
        Returns:
            市場數據 DataFrame
        """
        pass
    
    @abstractmethod
    def get_latest_price(self, symbol: str) -> float:
        """
        獲取最新價格
        
        Args:
            symbol: 交易對符號
            
        Returns:
            最新價格
        """
        pass


class IFeatureEngine(ABC):
    """
    特徵工程抽象接口
    
    定義了特徵計算和處理的標準接口。
    """
    
    @abstractmethod
    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        計算技術指標特徵
        
        Args:
            data: 原始市場數據
            
        Returns:
            包含特徵的 DataFrame
        """
        pass
    
    @abstractmethod
    def normalize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        正規化特徵數據
        
        Args:
            data: 特徵數據
            
        Returns:
            正規化後的 DataFrame
        """
        pass


class IBacktester(ABC):
    """
    回測器抽象接口
    
    定義了回測執行和結果分析的標準接口。
    """
    
    @abstractmethod
    def run_backtest(self, 
                     strategy: IAgent,
                     data: pd.DataFrame,
                     initial_capital: float) -> Dict[str, Any]:
        """
        執行回測
        
        Args:
            strategy: 交易策略
            data: 歷史數據
            initial_capital: 初始資本
            
        Returns:
            回測結果字典
        """
        pass
    
    @abstractmethod
    def calculate_metrics(self, 
                         returns: np.ndarray,
                         benchmark_returns: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        計算績效指標
        
        Args:
            returns: 策略收益率
            benchmark_returns: 基準收益率 (可選)
            
        Returns:
            績效指標字典
        """
        pass


if __name__ == "__main__":
    """單元測試區塊"""
    
    print("=== CryptoAce 抽象接口測試 ===")
    
    # 測試接口定義
    print("\n1. 檢查接口定義:")
    
    # 檢查 IAgent 接口
    agent_methods = [method for method in dir(IAgent) if not method.startswith('_')]
    print(f"   IAgent 接口方法: {agent_methods}")
    
    # 檢查 ITradingEnv 接口
    env_methods = [method for method in dir(ITradingEnv) if not method.startswith('_')]
    print(f"   ITradingEnv 接口方法: {env_methods}")
    
    # 檢查 IDataProvider 接口
    data_methods = [method for method in dir(IDataProvider) if not method.startswith('_')]
    print(f"   IDataProvider 接口方法: {data_methods}")
    
    # 檢查 IFeatureEngine 接口
    feature_methods = [method for method in dir(IFeatureEngine) if not method.startswith('_')]
    print(f"   IFeatureEngine 接口方法: {feature_methods}")
    
    # 檢查 IBacktester 接口
    backtest_methods = [method for method in dir(IBacktester) if not method.startswith('_')]
    print(f"   IBacktester 接口方法: {backtest_methods}")
    
    # 測試抽象性
    print("\n2. 測試抽象性:")
    
    try:
        # 嘗試直接實例化抽象類 (應該失敗)
        agent = IAgent()
    except TypeError as e:
        print(f"   ✅ IAgent 正確地拒絕直接實例化: {e}")
    
    try:
        # 嘗試直接實例化抽象類 (應該失敗)
        env = ITradingEnv()
    except TypeError as e:
        print(f"   ✅ ITradingEnv 正確地拒絕直接實例化: {e}")
    
    try:
        # 嘗試直接實例化抽象類 (應該失敗)
        provider = IDataProvider()
    except TypeError as e:
        print(f"   ✅ IDataProvider 正確地拒絕直接實例化: {e}")
    
    print("\n3. 檢查方法簽名:")
    
    # 檢查 IAgent.predict 方法簽名
    import inspect
    predict_signature = inspect.signature(IAgent.predict)
    print(f"   IAgent.predict 簽名: {predict_signature}")
    
    # 檢查 ITradingEnv.step 方法簽名
    step_signature = inspect.signature(ITradingEnv.step)
    print(f"   ITradingEnv.step 簽名: {step_signature}")
    
    print("\n✅ 抽象接口測試完成！")
    print("   所有接口都正確定義了抽象方法和型別註釋。")