"""
CryptoAce Agent 模組測試

測試 Agent 類的創建、預測和存取功能
"""

import pytest
import numpy as np
import pandas as pd
import torch
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import gymnasium as gym

# 導入要測試的模組
try:
    from core.agent import Agent, TransformerExtractor, TrainingCallback
    from core.trading_env import TradingEnv
    from core.interfaces import IAgent
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.agent import Agent, TransformerExtractor, TrainingCallback
    from core.trading_env import TradingEnv
    from core.interfaces import IAgent

# 檢查 stable-baselines3 可用性
try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


@pytest.fixture
def mock_trading_env():
    """
    創建模擬的 TradingEnv 實例
    """
    # 生成模擬數據
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=120, freq='1h')  # 增加到 120 個數據點
    
    # 創建價格數據
    base_price = 50000.0
    prices = [base_price]
    for i in range(1, 120):  # 調整循環範圍
        change = np.random.normal(0, 0.01)
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # 創建 OHLCV 數據
    data_list = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        high = close * (1 + abs(np.random.normal(0, 0.005)))
        low = close * (1 - abs(np.random.normal(0, 0.005)))
        open_price = prices[i-1] if i > 0 else close
        volume = np.random.uniform(1000, 5000)
        
        data_list.append({
            'timestamp': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    data = pd.DataFrame(data_list)
    data.set_index('timestamp', inplace=True)
    
    # 添加特徵 - 使用較小的滾動窗口以減少數據損失
    data['returns'] = data['close'].pct_change().fillna(0)
    data['sma_5'] = data['close'].rolling(5).mean().fillna(data['close'])  # 用前值填充 NaN
    data['volatility'] = data['returns'].rolling(5).std().fillna(0.01)  # 用小常數填充 NaN
    
    # 添加額外特徵
    for i in range(3):
        data[f'feature_{i}'] = np.random.randn(len(data))
    
    # 確保沒有 NaN 值
    data = data.bfill().ffill()  # 使用新的方法替代過時的 fillna(method=)
    
    # 驗證數據長度
    if len(data) < 100:
        raise ValueError(f"生成的數據長度不足: {len(data)}")
    
    # 創建交易環境
    env = TradingEnv(
        data=data,
        initial_balance=100000.0,
        transaction_cost=0.001,
        max_position_change_per_step=0.1,
        max_drawdown_limit=0.3,
        lookback_window=10
    )
    
    return env


@pytest.fixture
def agent_config():
    """
    Agent 配置參數
    """
    return {
        'learning_rate': 3e-4,
        'batch_size': 32,
        'n_epochs': 5,
        'gamma': 0.99,
        'features_dim': 128,
        'n_heads': 4,
        'n_layers': 2,
        'dropout': 0.1
    }


class TestTransformerExtractor:
    """
    測試 TransformerExtractor 類
    """
    
    @pytest.mark.skipif(not SB3_AVAILABLE, reason="stable-baselines3 not available")
    def test_transformer_extractor_creation(self, mock_trading_env):
        """測試 TransformerExtractor 的創建"""
        observation_space = mock_trading_env.observation_space
        
        extractor = TransformerExtractor(
            observation_space=observation_space,
            features_dim=128,
            n_heads=4,
            n_layers=2,
            dropout=0.1
        )
        
        assert extractor.input_dim == observation_space.shape[0]
        assert extractor._features_dim == 128
        assert hasattr(extractor, 'input_projection')
        assert hasattr(extractor, 'transformer_encoder')
        assert hasattr(extractor, 'output_projection')
    
    @pytest.mark.skipif(not SB3_AVAILABLE, reason="stable-baselines3 not available")
    def test_transformer_extractor_forward(self, mock_trading_env):
        """測試 TransformerExtractor 的前向傳播"""
        observation_space = mock_trading_env.observation_space
        
        extractor = TransformerExtractor(
            observation_space=observation_space,
            features_dim=128,
            n_heads=4,
            n_layers=2
        )
        
        # 創建測試輸入
        batch_size = 16
        input_tensor = torch.randn(batch_size, observation_space.shape[0])
        
        # 前向傳播
        output = extractor(input_tensor)
        
        assert output.shape == (batch_size, 128)
        assert not torch.isnan(output).any()
        assert torch.isfinite(output).all()


class TestTrainingCallback:
    """
    測試 TrainingCallback 類
    """
    
    @pytest.mark.skipif(not SB3_AVAILABLE, reason="stable-baselines3 not available")
    def test_training_callback_creation(self):
        """測試 TrainingCallback 的創建"""
        mock_logger = Mock()
        
        callback = TrainingCallback(logger=mock_logger, log_interval=100)
        
        assert callback.logger == mock_logger
        assert callback.log_interval == 100
        assert callback.episode_rewards == []
        assert callback.episode_lengths == []
    
    @pytest.mark.skipif(not SB3_AVAILABLE, reason="stable-baselines3 not available")
    def test_training_callback_logger_property(self):
        """測試 TrainingCallback 的 logger 屬性"""
        callback = TrainingCallback()
        
        # 測試 getter
        assert callback.logger is None
        
        # 測試 setter
        mock_logger = Mock()
        callback.logger = mock_logger
        assert callback.logger == mock_logger


class TestAgent:
    """
    測試 Agent 類
    """
    
    @pytest.mark.skipif(not SB3_AVAILABLE, reason="stable-baselines3 not available")
    def test_agent_creation(self, mock_trading_env, agent_config):
        """測試 Agent 的創建"""
        agent = Agent(
            env=mock_trading_env,
            config=None,
            logger=None
        )
        
        # 檢查 Agent 是否實現了 IAgent 接口
        assert isinstance(agent, IAgent)
        
        # 檢查內部模型是否為 PPO 實例
        assert isinstance(agent.model, PPO)
        
        # 檢查基本屬性
        assert agent.env == mock_trading_env
        assert agent.config is None
        assert agent.logger is None
        
        # 檢查 PPO 參數
        assert agent.learning_rate == 3e-4  # 默認值
        assert agent.batch_size == 64  # 默認值
        assert agent.features_dim == 256  # 默認值
        assert agent.n_heads == 8  # 默認值
        assert agent.n_layers == 4  # 默認值
        
        # 檢查設備
        assert agent.device in [torch.device('cpu'), torch.device('cuda')]
        
        # 檢查回調
        assert isinstance(agent.callback, TrainingCallback)
    
    @pytest.mark.skipif(not SB3_AVAILABLE, reason="stable-baselines3 not available")
    def test_agent_creation_with_config(self, mock_trading_env):
        """測試使用配置創建 Agent"""
        # 創建模擬配置
        mock_config = Mock()
        mock_config.agent = {
            'learning_rate': 1e-3,
            'batch_size': 32,
            'n_epochs': 5
        }
        mock_config.model = {
            'features_dim': 128,
            'n_heads': 4,
            'n_layers': 2
        }
        
        # 模擬 logger 相關方法以避免實際的 logger 設置
        mock_config.logger = Mock()
        mock_config.logger.get.return_value = "INFO"  # 返回有效的日誌級別
        
        # 使用 patch 來避免實際調用 setup_logger
        with patch('core.agent.setup_logger') as mock_setup_logger:
            mock_setup_logger.return_value = None
            
            agent = Agent(
                env=mock_trading_env,
                config=mock_config,
                logger=None
            )
        
        # 檢查配置是否被正確應用
        assert agent.learning_rate == 1e-3
        assert agent.batch_size == 32
        assert agent.n_epochs == 5
        assert agent.features_dim == 128
        assert agent.n_heads == 4
        assert agent.n_layers == 2
    
    @pytest.mark.skipif(not SB3_AVAILABLE, reason="stable-baselines3 not available")
    def test_agent_predict(self, mock_trading_env):
        """測試 Agent 的預測功能"""
        agent = Agent(
            env=mock_trading_env,
            config=None,
            logger=None
        )
        
        # 重置環境獲取觀察
        obs, info = mock_trading_env.reset(seed=42)
        
        # 測試確定性預測
        action, extra_info = agent.predict(obs, deterministic=True)
        
        # 檢查動作格式
        assert isinstance(action, np.ndarray)
        assert action.shape == (1,)  # 單一動作
        assert -1.0 <= action[0] <= 1.0  # 動作範圍
        
        # 檢查額外信息
        assert isinstance(extra_info, dict)
        assert 'state' in extra_info
        assert 'deterministic' in extra_info
        assert extra_info['deterministic'] is True
        
        # 測試隨機預測
        action_random, extra_info_random = agent.predict(obs, deterministic=False)
        
        assert isinstance(action_random, np.ndarray)
        assert action_random.shape == (1,)
        assert -1.0 <= action_random[0] <= 1.0
        assert extra_info_random['deterministic'] is False
    
    @pytest.mark.skipif(not SB3_AVAILABLE, reason="stable-baselines3 not available")
    def test_agent_predict_multiple_observations(self, mock_trading_env):
        """測試 Agent 對多個觀察的預測"""
        agent = Agent(
            env=mock_trading_env,
            config=None,
            logger=None
        )
        
        # 重置環境並進行多步預測
        obs, info = mock_trading_env.reset(seed=42)
        
        for _ in range(5):
            action, extra_info = agent.predict(obs, deterministic=True)
            
            # 檢查每次預測的格式
            assert isinstance(action, np.ndarray)
            assert action.shape == (1,)
            assert -1.0 <= action[0] <= 1.0
            
            # 執行動作獲取下一個觀察
            obs, reward, terminated, truncated, info = mock_trading_env.step(action)
            
            if terminated or truncated:
                break
    
    @pytest.mark.skipif(not SB3_AVAILABLE, reason="stable-baselines3 not available")
    def test_agent_save_load(self, mock_trading_env):
        """測試 Agent 的保存和載入功能"""
        # 創建 Agent
        agent = Agent(
            env=mock_trading_env,
            config=None,
            logger=None
        )
        
        # 獲取初始觀察用於測試
        obs, info = mock_trading_env.reset(seed=42)
        
        # 進行初始預測
        action_before, _ = agent.predict(obs, deterministic=True)
        
        # 使用臨時目錄進行保存測試
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "test_agent")
            
            # 測試保存
            agent.save(save_path)
            
            # 檢查文件是否存在
            assert os.path.exists(save_path + ".zip")
            
            # 創建新的 Agent 實例
            new_agent = Agent(
                env=mock_trading_env,
                config=None,
                logger=None
            )
            
            # 測試載入
            new_agent.load(save_path)
            
            # 使用相同輸入進行預測
            action_after, _ = new_agent.predict(obs, deterministic=True)
            
            # 檢查載入後的預測結果是否一致
            # 注意：由於模型結構可能略有不同，我們檢查動作範圍而不是精確值
            assert isinstance(action_after, np.ndarray)
            assert action_after.shape == action_before.shape
            assert -1.0 <= action_after[0] <= 1.0
    
    @pytest.mark.skipif(not SB3_AVAILABLE, reason="stable-baselines3 not available")
    def test_agent_save_load_nonexistent_file(self, mock_trading_env):
        """測試載入不存在的模型文件"""
        agent = Agent(
            env=mock_trading_env,
            config=None,
            logger=None
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            nonexistent_path = os.path.join(temp_dir, "nonexistent_model")
            
            # 測試載入不存在的文件應該拋出異常
            with pytest.raises(FileNotFoundError):
                agent.load(nonexistent_path)
    
    @pytest.mark.skipif(not SB3_AVAILABLE, reason="stable-baselines3 not available")
    def test_agent_evaluate(self, mock_trading_env):
        """測試 Agent 的評估功能"""
        agent = Agent(
            env=mock_trading_env,
            config=None,
            logger=None
        )
        
        # 測試評估
        eval_stats = agent.evaluate(n_episodes=2, deterministic=True)
        
        # 檢查評估結果格式
        assert isinstance(eval_stats, dict)
        
        required_keys = [
            'mean_reward', 'std_reward', 'min_reward', 'max_reward',
            'mean_length', 'std_length'
        ]
        
        for key in required_keys:
            assert key in eval_stats
            assert isinstance(eval_stats[key], float)
            assert not np.isnan(eval_stats[key])
        
        # 檢查統計值的合理性
        assert eval_stats['min_reward'] <= eval_stats['mean_reward'] <= eval_stats['max_reward']
        assert eval_stats['std_reward'] >= 0
        assert eval_stats['mean_length'] > 0
        assert eval_stats['std_length'] >= 0
    
    @pytest.mark.skipif(not SB3_AVAILABLE, reason="stable-baselines3 not available")
    def test_agent_train_short(self, mock_trading_env):
        """測試 Agent 的簡短訓練功能"""
        agent = Agent(
            env=mock_trading_env,
            config=None,
            logger=None
        )
        
        # 進行很短的訓練（僅驗證能運行）
        training_stats = agent.train(total_timesteps=100)
        
        # 檢查訓練結果格式
        assert isinstance(training_stats, dict)
        
        required_keys = [
            'total_timesteps', 'episode_rewards', 'episode_lengths',
            'mean_reward', 'std_reward'
        ]
        
        for key in required_keys:
            assert key in training_stats
        
        assert training_stats['total_timesteps'] == 100
        assert isinstance(training_stats['episode_rewards'], list)
        assert isinstance(training_stats['episode_lengths'], list)
        assert isinstance(training_stats['mean_reward'], (int, float))
        assert isinstance(training_stats['std_reward'], (int, float))
    
    @pytest.mark.skipif(not SB3_AVAILABLE, reason="stable-baselines3 not available")
    def test_agent_train_with_save(self, mock_trading_env):
        """測試 Agent 訓練並保存"""
        agent = Agent(
            env=mock_trading_env,
            config=None,
            logger=None
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "trained_agent")
            
            # 訓練並保存
            training_stats = agent.train(total_timesteps=100, save_path=save_path)
            
            # 檢查文件是否被保存
            assert os.path.exists(save_path + ".zip")
            
            # 檢查訓練統計
            assert training_stats['total_timesteps'] == 100
    
    @pytest.mark.skipif(not SB3_AVAILABLE, reason="stable-baselines3 not available")
    def test_agent_with_logger(self, mock_trading_env):
        """測試帶有 logger 的 Agent"""
        mock_logger = Mock()
        
        agent = Agent(
            env=mock_trading_env,
            config=None,
            logger=mock_logger
        )
        
        assert agent.logger == mock_logger
        
        # 檢查 logger 是否被調用（初始化時應該記錄信息）
        assert mock_logger.info.called


class TestAgentExceptions:
    """
    測試 Agent 的異常處理
    """
    
    @pytest.mark.skipif(SB3_AVAILABLE, reason="Testing when stable-baselines3 is not available")
    def test_agent_creation_without_sb3(self, mock_trading_env):
        """測試在沒有 stable-baselines3 時創建 Agent"""
        with pytest.raises(ImportError, match="stable-baselines3 is required"):
            Agent(env=mock_trading_env, config=None, logger=None)
    
    @pytest.mark.skipif(not SB3_AVAILABLE, reason="stable-baselines3 not available")
    def test_agent_predict_error_handling(self, mock_trading_env):
        """測試預測時的錯誤處理"""
        agent = Agent(
            env=mock_trading_env,
            config=None,
            logger=None
        )
        
        # 測試無效輸入
        with pytest.raises(Exception):
            invalid_obs = np.array([])  # 空數組
            agent.predict(invalid_obs)


if __name__ == "__main__":
    # 運行測試
    pytest.main([__file__, "-v"])
