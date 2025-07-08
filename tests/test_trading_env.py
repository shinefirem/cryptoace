"""
TradingEnv 測試模組

測試 TradingEnv 類別的核心功能，包括：
- 環境初始化和重置
- 動作執行和狀態變化
- 風險管理機制
- Gymnasium API 兼容性
"""

import pytest
import numpy as np
import pandas as pd
import gymnasium as gym
from typing import Dict, Any
import warnings

# 抑制警告
warnings.filterwarnings('ignore')

# 添加項目根目錄到路徑
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 導入待測試的模組
from core.trading_env import TradingEnv
from core.configurator import Configurator


@pytest.fixture
def sample_data():
    """
    創建用於測試的樣本數據
    
    Returns:
        pd.DataFrame: 包含 OHLCV 和特徵數據的 DataFrame
    """
    np.random.seed(42)
    
    # 創建時間序列
    dates = pd.date_range('2023-01-01', periods=500, freq='1h')
    
    # 生成價格數據
    base_price = 50000.0
    prices = []
    current_price = base_price
    
    for i in range(500):
        # 使用隨機遊走生成價格
        change = np.random.normal(0, 0.01)  # 較小的波動率使測試更穩定
        current_price *= (1 + change)
        prices.append(current_price)
    
    # 創建 OHLCV 數據
    ohlcv_data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        high = close * (1 + abs(np.random.normal(0, 0.005)))
        low = close * (1 - abs(np.random.normal(0, 0.005)))
        open_price = prices[i-1] if i > 0 else close
        volume = np.random.uniform(1000, 5000)
        
        ohlcv_data.append({
            'timestamp': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    data = pd.DataFrame(ohlcv_data)
    data.set_index('timestamp', inplace=True)
    
    # 添加技術指標特徵
    data['returns'] = data['close'].pct_change().fillna(0)
    data['sma_20'] = data['close'].rolling(20).mean()
    data['sma_50'] = data['close'].rolling(50).mean()
    data['rsi'] = _calculate_rsi(data['close'], 14)
    data['volatility'] = data['returns'].rolling(20).std()
    
    # 添加模擬特徵
    for i in range(10):
        data[f'feature_{i}'] = np.random.randn(len(data))
    
    # 移除 NaN 值
    data = data.dropna()
    
    return data


def _calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """計算 RSI 指標"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


@pytest.fixture
def trading_env(sample_data):
    """
    創建可重用的 TradingEnv 實例
    
    Args:
        sample_data: 樣本數據 fixture
        
    Returns:
        TradingEnv: 配置好的交易環境實例
    """
    env = TradingEnv(
        data=sample_data,
        initial_balance=100000.0,
        transaction_cost=0.001,
        max_position_change_per_step=0.1,
        max_drawdown_limit=0.2,
        lookback_window=50,
        risk_free_rate=0.0
    )
    return env


class TestTradingEnvInitialization:
    """測試 TradingEnv 初始化"""
    
    def test_initialization(self, trading_env, sample_data):
        """測試環境正確初始化"""
        assert trading_env.initial_balance == 100000.0
        assert trading_env.transaction_cost == 0.001
        assert trading_env.max_position_change_per_step == 0.1
        assert trading_env.max_drawdown_limit == 0.2
        assert trading_env.data_length == len(sample_data)
        assert trading_env.n_features > 0
        
        # 檢查動作和觀察空間
        assert isinstance(trading_env.action_space, gym.spaces.Box)
        assert isinstance(trading_env.observation_space, gym.spaces.Box)
        assert trading_env.action_space.shape == (1,)
        assert trading_env.observation_space.shape == (trading_env.n_features + 3,)
    
    def test_data_validation(self, sample_data):
        """測試數據驗證功能"""
        # 測試空數據
        with pytest.raises(ValueError, match="數據不能為空"):
            TradingEnv(data=pd.DataFrame())
        
        # 測試缺少必要列
        invalid_data = sample_data.drop(columns=['open'])
        with pytest.raises(ValueError, match="缺少必要的列"):
            TradingEnv(data=invalid_data)
        
        # 測試數據太短
        short_data = sample_data.head(50)
        with pytest.raises(ValueError, match="數據長度太短"):
            TradingEnv(data=short_data)


class TestTradingEnvReset:
    """測試環境重置功能"""
    
    def test_reset_without_seed(self, trading_env):
        """測試無種子重置"""
        obs, info = trading_env.reset()
        
        # 檢查返回值類型和形狀
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
        assert obs.shape == trading_env.observation_space.shape
        assert obs.dtype == np.float32
        
        # 檢查初始狀態
        assert trading_env.current_step == 0
        assert trading_env.balance == trading_env.initial_balance
        assert trading_env.position == 0.0
        assert trading_env.equity == trading_env.initial_balance
        assert trading_env.drawdown == 0.0
        assert len(trading_env.trade_history) == 0
    
    def test_reset_with_seed(self, trading_env):
        """測試帶種子的重置"""
        seed = 42
        obs1, info1 = trading_env.reset(seed=seed)
        obs2, info2 = trading_env.reset(seed=seed)
        
        # 相同種子應該產生相同的初始觀察
        np.testing.assert_array_equal(obs1, obs2)
        
        # 檢查隨機數生成器是否正確設置
        assert hasattr(trading_env, 'np_random')
        assert trading_env.np_random is not None


class TestTradingEnvStep:
    """測試環境步進功能"""
    
    def test_step_action(self, trading_env):
        """
        測試具體動作執行：做多動作
        驗證環境狀態是否按預期變化
        """
        # 重置環境
        obs, info = trading_env.reset(seed=42)
        initial_equity = trading_env.equity
        initial_position = trading_env.position
        
        # 執行做多動作（目標倉位 0.5）
        long_action = np.array([0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = trading_env.step(long_action)
        
        # 驗證倉位變化
        expected_position = min(0.5, initial_position + trading_env.max_position_change_per_step)
        assert abs(trading_env.position - expected_position) < 1e-6
        
        # 驗證交易記錄
        assert len(trading_env.trade_history) > 0
        last_trade = trading_env.trade_history[-1]
        assert last_trade['step'] == 0  # 第一步
        assert last_trade['new_position'] == trading_env.position
        assert last_trade['cost'] > 0  # 應該有交易成本
        
        # 驗證時間步推進
        assert trading_env.current_step == 1
        
        # 驗證觀察空間
        assert obs.shape == trading_env.observation_space.shape
        assert obs.dtype == np.float32
        
        # 驗證info字典包含必要信息
        required_keys = ['step', 'equity', 'position', 'drawdown', 'total_return', 'trade_count']
        for key in required_keys:
            assert key in info
        
        print(f"初始倉位: {initial_position:.4f}")
        print(f"執行後倉位: {trading_env.position:.4f}")
        print(f"預期倉位: {expected_position:.4f}")
        print(f"交易成本: {last_trade['cost']:.2f}")
    
    def test_step_different_actions(self, trading_env):
        """測試不同類型的動作輸入"""
        obs, info = trading_env.reset(seed=42)
        
        # 測試 numpy array 動作
        obs1, reward1, term1, trunc1, info1 = trading_env.step(np.array([0.3]))
        assert trading_env.position > 0
        
        # 測試 float 動作
        obs2, reward2, term2, trunc2, info2 = trading_env.step(0.2)
        
        # 測試 int 動作
        obs3, reward3, term3, trunc3, info3 = trading_env.step(0)
        
        # 所有動作都應該被正確處理
        assert all(isinstance(obs, np.ndarray) for obs in [obs1, obs2, obs3])
        assert all(isinstance(reward, (int, float)) for reward in [reward1, reward2, reward3])


class TestRiskManagement:
    """測試風險管理功能"""
    
    def test_risk_management(self, trading_env):
        """
        測試最大倉位變動限制
        傳入極端動作，驗證倉位變動是否被正確限制
        """
        # 重置環境
        obs, info = trading_env.reset(seed=42)
        initial_position = trading_env.position  # 應該是 0.0
        
        # 傳入極端做多動作（目標倉位 1.0，遠超過最大變動限制）
        extreme_action = np.array([1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = trading_env.step(extreme_action)
        
        # 驗證倉位變動被限制在最大變動範圍內
        actual_position_change = abs(trading_env.position - initial_position)
        max_allowed_change = trading_env.max_position_change_per_step
        
        assert actual_position_change <= max_allowed_change + 1e-6  # 允許小誤差
        
        # 預期倉位應該是初始倉位 + 最大變動量
        expected_position = initial_position + max_allowed_change
        assert abs(trading_env.position - expected_position) < 1e-6
        
        # 再次傳入極端動作，驗證持續限制
        obs, reward, terminated, truncated, info = trading_env.step(extreme_action)
        new_position_change = abs(trading_env.position - expected_position)
        assert new_position_change <= max_allowed_change + 1e-6
        
        print(f"初始倉位: {initial_position:.4f}")
        print(f"極端動作目標: 1.0")
        print(f"實際倉位變動: {actual_position_change:.4f}")
        print(f"最大允許變動: {max_allowed_change:.4f}")
        print(f"最終倉位: {trading_env.position:.4f}")
    
    def test_drawdown_limit(self, trading_env):
        """測試最大回撤限制"""
        # 重置環境
        obs, info = trading_env.reset(seed=42)
        
        # 模擬一系列虧損交易，嘗試觸發回撤限制
        terminated = False
        step_count = 0
        max_steps = 100  # 防止無限循環
        
        while not terminated and step_count < max_steps:
            # 隨機動作
            action = trading_env.action_space.sample()
            obs, reward, terminated, truncated, info = trading_env.step(action)
            step_count += 1
            
            # 如果回撤超過限制，應該終止
            if trading_env.drawdown > trading_env.max_drawdown_limit:
                assert terminated, "環境應該在回撤超限時終止"
                break
        
        print(f"執行了 {step_count} 步")
        print(f"最終回撤: {trading_env.drawdown:.4f}")
        print(f"回撤限制: {trading_env.max_drawdown_limit:.4f}")
        print(f"是否終止: {terminated}")
    
    def test_action_clipping(self, trading_env):
        """測試動作值被正確限制在 [-1, 1] 範圍內"""
        obs, info = trading_env.reset(seed=42)
        
        # 測試超出範圍的動作
        extreme_actions = [2.0, -2.0, 10.0, -10.0]
        
        for extreme_action in extreme_actions:
            old_position = trading_env.position
            obs, reward, terminated, truncated, info = trading_env.step(extreme_action)
            
            # 驗證倉位在有效範圍內
            assert -1.0 <= trading_env.position <= 1.0
            
            # 驗證倉位變動不超過限制
            position_change = abs(trading_env.position - old_position)
            assert position_change <= trading_env.max_position_change_per_step + 1e-6


class TestEnvironmentAPI:
    """測試環境 API 兼容性"""
    
    def test_env_api_compliance(self, trading_env):
        """
        測試 Gymnasium API 兼容性
        使用 gymnasium.utils.env_checker.check_env 進行驗證
        """
        try:
            import gymnasium.utils.env_checker as env_checker
            
            # 這個檢查應該不會拋出異常
            env_checker.check_env(trading_env)
            
            # 如果到達這裡，說明檢查通過
            assert True, "環境通過了 Gymnasium 兼容性檢查"
            
        except ImportError:
            pytest.skip("gymnasium.utils.env_checker 不可用")
        except Exception as e:
            pytest.fail(f"環境未通過 Gymnasium 兼容性檢查: {e}")
    
    def test_action_space_sample(self, trading_env):
        """測試動作空間採樣"""
        # 重置環境
        obs, info = trading_env.reset(seed=42)
        
        # 測試動作空間採樣
        for _ in range(10):
            action = trading_env.action_space.sample()
            assert trading_env.action_space.contains(action)
            
            # 執行動作
            obs, reward, terminated, truncated, info = trading_env.step(action)
            
            if terminated or truncated:
                break
    
    def test_observation_space_contains(self, trading_env):
        """測試觀察空間包含性"""
        obs, info = trading_env.reset(seed=42)
        
        # 初始觀察應該在觀察空間內
        assert trading_env.observation_space.contains(obs)
        
        # 執行幾步，檢查所有觀察都在空間內
        for _ in range(10):
            action = trading_env.action_space.sample()
            obs, reward, terminated, truncated, info = trading_env.step(action)
            assert trading_env.observation_space.contains(obs)
            
            if terminated or truncated:
                break


class TestTradingLogic:
    """測試交易邏輯"""
    
    def test_transaction_costs(self, trading_env):
        """測試交易成本計算"""
        obs, info = trading_env.reset(seed=42)
        initial_equity = trading_env.equity
        
        # 執行有效交易
        action = np.array([0.1], dtype=np.float32)
        obs, reward, terminated, truncated, info = trading_env.step(action)
        
        # 應該有交易記錄且包含成本
        assert len(trading_env.trade_history) > 0
        trade = trading_env.trade_history[0]
        assert trade['cost'] > 0
        
        # 成本應該是 |倉位變動| * 交易成本率 * 餘額
        expected_cost = abs(trade['position_change']) * trading_env.transaction_cost * initial_equity
        assert abs(trade['cost'] - expected_cost) < 1e-6
    
    def test_future_leakage_prevention(self, trading_env):
        """測試未來函數防護"""
        obs, info = trading_env.reset(seed=42)
        
        # 記錄執行交易時的數據索引
        initial_step = trading_env.current_step
        
        # 執行動作
        action = np.array([0.1], dtype=np.float32)
        obs, reward, terminated, truncated, info = trading_env.step(action)
        
        # 驗證交易使用的是當前步的開盤價（防止未來函數）
        if len(trading_env.trade_history) > 0:
            trade = trading_env.trade_history[0]
            current_data = trading_env.data.iloc[initial_step]
            assert abs(trade['price'] - current_data['open']) < 1e-6
    
    def test_reward_calculation(self, trading_env):
        """測試獎勵計算"""
        obs, info = trading_env.reset(seed=42)
        
        # 執行幾步並收集獎勵
        rewards = []
        for _ in range(20):  # 需要足夠的步數來計算有意義的獎勵
            action = trading_env.action_space.sample()
            obs, reward, terminated, truncated, info = trading_env.step(action)
            rewards.append(reward)
            
            if terminated or truncated:
                break
        
        # 獎勵應該是有限的數值
        for reward in rewards:
            assert np.isfinite(reward)
            assert isinstance(reward, (int, float))


class TestEdgeCases:
    """測試邊界情況"""
    
    def test_episode_termination(self, trading_env):
        """測試回合終止條件"""
        obs, info = trading_env.reset(seed=42)
        
        terminated = False
        truncated = False
        step_count = 0
        
        # 運行直到終止
        while not (terminated or truncated) and step_count < trading_env.data_length:
            action = trading_env.action_space.sample()
            obs, reward, terminated, truncated, info = trading_env.step(action)
            step_count += 1
        
        # 應該在數據末尾或觸發風險限制時終止
        assert terminated or truncated or step_count >= trading_env.data_length - 1
    
    def test_no_position_change(self, trading_env):
        """測試無倉位變動的情況"""
        obs, info = trading_env.reset(seed=42)
        initial_position = trading_env.position
        
        # 傳入與當前倉位相同的動作
        no_change_action = np.array([initial_position], dtype=np.float32)
        obs, reward, terminated, truncated, info = trading_env.step(no_change_action)
        
        # 倉位應該沒有變化
        assert abs(trading_env.position - initial_position) < 1e-6
        
        # 交易歷史可能為空（因為沒有有效交易）
        if len(trading_env.trade_history) == 0:
            # 這是預期的，因為沒有有效的倉位變動
            pass
        else:
            # 如果有記錄，變動應該很小
            trade = trading_env.trade_history[-1]
            assert abs(trade['position_change']) < 1e-6


if __name__ == "__main__":
    # 運行測試
    pytest.main([__file__, "-v"])
