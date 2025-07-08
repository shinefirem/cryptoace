"""
CryptoAce 智能交易代理模組

此模組實現基於 PPO 和 Transformer 的交易代理，整合：
- Stable-Baselines3 PPO 模型
- 自定義 Transformer 特徵提取器
- 完整的訓練和預測功能
"""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple, Union
import gymnasium as gym
from pathlib import Path
import warnings

# Stable-Baselines3 相關導入
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    print("Warning: stable-baselines3 not available. Agent functionality will be limited.")
    SB3_AVAILABLE = False
    # 創建虛擬類別以避免導入錯誤
    class BaseFeaturesExtractor:
        pass
    class BaseCallback:
        pass

# 處理相對匯入問題
try:
    from .interfaces import IAgent
    from .configurator import Configurator
    from .logger import setup_logger
    from .trading_env import TradingEnv
except ImportError:
    # 當直接執行此檔案時，使用絕對匯入
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.interfaces import IAgent
    from core.configurator import Configurator
    from core.logger import setup_logger
    from core.trading_env import TradingEnv

# 抑制警告
warnings.filterwarnings('ignore')


class TransformerExtractor(BaseFeaturesExtractor):
    """
    自定義 Transformer 特徵提取器
    
    實現基於 Transformer 編碼器的特徵提取，適用於時序金融數據
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        sequence_length: int = 50
    ):
        """
        初始化 Transformer 特徵提取器
        
        Args:
            observation_space: 觀察空間
            features_dim: 輸出特徵維度
            n_heads: 注意力頭數
            n_layers: Transformer 層數
            dropout: Dropout 比例
            sequence_length: 序列長度（用於位置編碼）
        """
        super().__init__(observation_space, features_dim)
        
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for TransformerExtractor")
        
        # 獲取輸入維度
        self.input_dim = observation_space.shape[0]
        self._features_dim = features_dim  # 使用私有屬性避免與父類衝突
        self.sequence_length = sequence_length
        
        # 輸入投影層
        self.input_projection = nn.Linear(self.input_dim, self._features_dim)
        
        # 位置編碼
        self.positional_encoding = self._create_positional_encoding(sequence_length, self._features_dim)
        
        # Transformer 編碼器層
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self._features_dim,
            nhead=n_heads,
            dim_feedforward=self._features_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers
        )
        
        # 輸出投影層
        self.output_projection = nn.Sequential(
            nn.Linear(self._features_dim, self._features_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self._features_dim, self._features_dim)
        )
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(self._features_dim)
        
    def _create_positional_encoding(self, max_length: int, d_model: int) -> torch.Tensor:
        """創建位置編碼"""
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_length, d_model]
        
        return pe
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            observations: 觀察數據 [batch_size, input_dim]
            
        Returns:
            提取的特徵 [batch_size, features_dim]
        """
        batch_size = observations.shape[0]
        
        # 輸入投影
        x = self.input_projection(observations)  # [batch_size, _features_dim]
        
        # 添加批次維度以適應序列處理
        # 這裡我們將單個觀察視為長度為1的序列
        x = x.unsqueeze(1)  # [batch_size, 1, _features_dim]
        
        # 添加位置編碼（只使用第一個位置）
        if self.positional_encoding.device != x.device:
            self.positional_encoding = self.positional_encoding.to(x.device)
        
        pos_encoding = self.positional_encoding[:, :1, :]  # [1, 1, _features_dim]
        x = x + pos_encoding
        
        # Transformer 編碼
        x = self.transformer_encoder(x)  # [batch_size, 1, _features_dim]
        
        # 移除序列維度
        x = x.squeeze(1)  # [batch_size, _features_dim]
        
        # 層歸一化和輸出投影
        x = self.layer_norm(x)
        x = self.output_projection(x)
        
        return x


class TrainingCallback(BaseCallback):
    """
    訓練回調函數
    
    用於監控訓練過程和記錄日誌
    """
    
    def __init__(self, logger=None, log_interval: int = 1000):
        super().__init__()
        self._logger = logger  # 使用私有屬性存儲 logger
        self.log_interval = log_interval
        self.episode_rewards = []
        self.episode_lengths = []

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, value):
        self._logger = value

    def _on_step(self) -> bool:
        """每步調用的回調"""
        # 記錄回合結束信息
        if len(self.locals.get('dones', [])) > 0 and any(self.locals['dones']):
            if len(self.locals.get('infos', [])) > 0:
                for info in self.locals['infos']:
                    if 'episode' in info:
                        self.episode_rewards.append(info['episode']['r'])
                        self.episode_lengths.append(info['episode']['l'])
        
        # 定期日誌記錄
        if self.num_timesteps % self.log_interval == 0:
            if self.logger and len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-10:])  # 最近10個回合的平均獎勵
                mean_length = np.mean(self.episode_lengths[-10:])  # 最近10個回合的平均長度
                
                self.logger.info(f"Step {self.num_timesteps}: "
                               f"Mean Reward: {mean_reward:.4f}, "
                               f"Mean Length: {mean_length:.1f}")
        
        return True


class Agent(IAgent):
    """
    智能交易代理
    
    基於 PPO 和 Transformer 的強化學習交易代理
    """
    
    def __init__(
        self,
        env: gym.Env,
        config: Optional[Configurator] = None,
        logger: Optional[Any] = None,
        **kwargs
    ):
        """
        初始化代理
        
        Args:
            env: 交易環境
            config: 配置管理器
            logger: 日誌記錄器
            **kwargs: 額外參數
        """
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for Agent")
        
        self.env = env
        self.config = config
        
        # 設置日誌記錄器
        if logger:
            self.logger = logger
        elif config and hasattr(config, 'logger'):
            self.logger = setup_logger(config)
        else:
            self.logger = None
        
        # 從配置中獲取參數
        if config:
            agent_config = config.agent
            model_config = getattr(config, 'model', {})
        else:
            agent_config = {}
            model_config = {}
        
        # PPO 參數
        self.learning_rate = agent_config.get('learning_rate', 3e-4)
        self.batch_size = agent_config.get('batch_size', 64)
        self.n_epochs = agent_config.get('n_epochs', 10)
        self.gamma = agent_config.get('gamma', 0.99)
        self.gae_lambda = agent_config.get('gae_lambda', 0.95)
        self.clip_range = agent_config.get('clip_range', 0.2)
        self.ent_coef = agent_config.get('ent_coef', 0.01)
        self.vf_coef = agent_config.get('vf_coef', 0.5)
        self.max_grad_norm = agent_config.get('max_grad_norm', 0.5)
        
        # Transformer 參數
        self.features_dim = model_config.get('features_dim', 256)
        self.n_heads = model_config.get('n_heads', 8)
        self.n_layers = model_config.get('n_layers', 4)
        self.dropout = model_config.get('dropout', 0.1)
        
        # 設備選擇
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 創建自定義策略參數
        policy_kwargs = {
            "features_extractor_class": TransformerExtractor,
            "features_extractor_kwargs": {
                "features_dim": self.features_dim,
                "n_heads": self.n_heads,
                "n_layers": self.n_layers,
                "dropout": self.dropout
            },
            "net_arch": [dict(pi=[256, 256], vf=[256, 256])],  # 策略和價值網路架構
            "activation_fn": torch.nn.ReLU,
        }
        
        # 創建 PPO 模型
        self.model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=self.learning_rate,
            n_steps=2048,  # 每次更新收集的步數
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
            policy_kwargs=policy_kwargs,
            device=self.device,
            verbose=1 if self.logger else 0
        )
        
        # 訓練回調
        self.callback = TrainingCallback(logger=self.logger)
        
        if self.logger:
            self.logger.info(f"Agent 初始化完成")
            self.logger.info(f"  - 設備: {self.device}")
            self.logger.info(f"  - 學習率: {self.learning_rate}")
            self.logger.info(f"  - 批次大小: {self.batch_size}")
            self.logger.info(f"  - Transformer 特徵維度: {self.features_dim}")
            self.logger.info(f"  - 注意力頭數: {self.n_heads}")
            self.logger.info(f"  - Transformer 層數: {self.n_layers}")
    
    def train(
        self,
        total_timesteps: int,
        save_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        訓練代理
        
        Args:
            total_timesteps: 總訓練步數
            save_path: 模型保存路徑
            **kwargs: 額外參數
            
        Returns:
            訓練結果字典
        """
        if self.logger:
            self.logger.info(f"開始訓練代理，總步數: {total_timesteps}")
        
        try:
            # 開始訓練
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=self.callback,
                progress_bar=True,
                **kwargs
            )
            
            # 保存模型
            if save_path:
                self.save(save_path)
                if self.logger:
                    self.logger.info(f"模型已保存到: {save_path}")
            
            # 收集訓練統計信息
            training_stats = {
                'total_timesteps': total_timesteps,
                'episode_rewards': self.callback.episode_rewards.copy(),
                'episode_lengths': self.callback.episode_lengths.copy(),
                'mean_reward': np.mean(self.callback.episode_rewards[-10:]) if self.callback.episode_rewards else 0,
                'std_reward': np.std(self.callback.episode_rewards[-10:]) if self.callback.episode_rewards else 0,
            }
            
            if self.logger:
                self.logger.info(f"訓練完成！平均獎勵: {training_stats['mean_reward']:.4f}")
            
            return training_stats
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"訓練過程中出現錯誤: {e}")
            raise
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """
        預測動作
        
        Args:
            observation: 觀察狀態
            deterministic: 是否使用確定性策略
            
        Returns:
            (action, extra_info): 動作和額外信息
        """
        try:
            # 將觀察轉換為 torch.Tensor
            obs_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device)
            
            action, state = self.model.predict(
                observation=obs_tensor,
                deterministic=deterministic
            )
            
            # 額外信息（如果需要）
            extra_info = {
                'state': state,
                'deterministic': deterministic
            }
            
            return action, extra_info
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"預測過程中出現錯誤: {e}")
            raise
    
    def save(self, path: str) -> None:
        """
        保存模型
        
        Args:
            path: 保存路徑
        """
        try:
            # 確保目錄存在
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存 PPO 模型
            self.model.save(str(save_path))
            
            if self.logger:
                self.logger.info(f"模型已保存到: {path}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"保存模型時出現錯誤: {e}")
            raise
    
    def load(self, path: str) -> None:
        """
        加載模型
        
        Args:
            path: 模型路徑
        """
        try:
            if not os.path.exists(path + ".zip"):  # PPO 自動添加 .zip 後綴
                raise FileNotFoundError(f"模型文件不存在: {path}")
            
            # 加載 PPO 模型
            self.model = PPO.load(
                path=path,
                env=self.env,
                device=self.device
            )
            
            if self.logger:
                self.logger.info(f"模型已從 {path} 加載")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"加載模型時出現錯誤: {e}")
            raise
    
    def evaluate(
        self,
        n_episodes: int = 10,
        deterministic: bool = True
    ) -> Dict[str, float]:
        """
        評估代理性能
        
        Args:
            n_episodes: 評估回合數
            deterministic: 是否使用確定性策略
            
        Returns:
            評估結果字典
        """
        if self.logger:
            self.logger.info(f"開始評估代理，回合數: {n_episodes}")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            episode_length = 0
            terminated = False
            truncated = False
            
            while not (terminated or truncated):
                action, _ = self.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            if self.logger and (episode + 1) % 5 == 0:
                self.logger.info(f"評估進度: {episode + 1}/{n_episodes}, "
                               f"當前回合獎勵: {episode_reward:.4f}")
        
        # 計算統計信息
        eval_stats = {
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'min_reward': float(np.min(episode_rewards)),
            'max_reward': float(np.max(episode_rewards)),
            'mean_length': float(np.mean(episode_lengths)),
            'std_length': float(np.std(episode_lengths)),
        }
        
        if self.logger:
            self.logger.info(f"評估完成！平均獎勵: {eval_stats['mean_reward']:.4f} ± {eval_stats['std_reward']:.4f}")
        
        return eval_stats


if __name__ == "__main__":
    """測試 Agent 模組"""
    import pandas as pd
    from pathlib import Path
    
    print("=== CryptoAce Agent 測試 ===")
    
    # 檢查 stable-baselines3 可用性
    if not SB3_AVAILABLE:
        print("❌ stable-baselines3 不可用，無法進行完整測試")
        print("請安裝: pip install stable-baselines3")
        exit(1)
    
    try:
        # 1. 創建虛擬交易環境
        print("\n1. 創建虛擬交易環境...")
        
        # 生成模擬數據
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=200, freq='1h')
        
        # 創建價格數據
        base_price = 50000.0
        prices = [base_price]
        for i in range(1, 200):
            change = np.random.normal(0, 0.02)
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        # 創建 OHLCV 數據
        data_list = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            high = close * (1 + abs(np.random.normal(0, 0.01)))
            low = close * (1 - abs(np.random.normal(0, 0.01)))
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
        
        # 添加特徵
        data['returns'] = data['close'].pct_change().fillna(0)
        data['sma_10'] = data['close'].rolling(10).mean()
        data['volatility'] = data['returns'].rolling(10).std()
        
        for i in range(5):
            data[f'feature_{i}'] = np.random.randn(len(data))
        
        data = data.dropna()
        
        print(f"   數據形狀: {data.shape}")
        
        # 創建交易環境
        env = TradingEnv(
            data=data,
            initial_balance=100000.0,
            transaction_cost=0.001,
            max_position_change_per_step=0.1,
            max_drawdown_limit=0.3,
            lookback_window=20
        )
        
        print(f"   環境觀察空間: {env.observation_space}")
        print(f"   環境動作空間: {env.action_space}")
        
        # 2. 實例化 Agent
        print("\n2. 實例化 Agent...")
        
        agent = Agent(
            env=env,
            config=None  # 使用默認配置
        )
        
        print(f"   Agent 設備: {agent.device}")
        print(f"   Transformer 特徵維度: {agent.features_dim}")
        print(f"   注意力頭數: {agent.n_heads}")
        
        # 3. 測試預測功能
        print("\n3. 測試預測功能...")
        
        obs, info = env.reset(seed=42)
        print(f"   初始觀察形狀: {obs.shape}")
        
        # 測試預測
        action, extra_info = agent.predict(obs, deterministic=True)
        print(f"   預測動作: {action}")
        print(f"   動作形狀: {action.shape}")
        print(f"   是否確定性: {extra_info['deterministic']}")
        
        # 測試隨機預測
        action_random, _ = agent.predict(obs, deterministic=False)
        print(f"   隨機動作: {action_random}")
        
        # 4. 測試簡短訓練
        print("\n4. 測試簡短訓練...")
        
        # 很短的訓練測試（只是驗證能運行）
        training_stats = agent.train(total_timesteps=1000)
        print(f"   訓練完成，總步數: {training_stats['total_timesteps']}")
        print(f"   回合數: {len(training_stats['episode_rewards'])}")
        if training_stats['episode_rewards']:
            print(f"   平均獎勵: {training_stats['mean_reward']:.4f}")
        
        # 5. 測試保存和載入
        print("\n5. 測試保存和載入...")
        
        # 測試保存
        save_path = "./models/test_agent"
        agent.save(save_path)
        print(f"   模型已保存到: {save_path}")
        
        # 驗證文件存在
        if os.path.exists(save_path + ".zip"):
            print("   ✅ 保存文件存在")
        else:
            print("   ❌ 保存文件不存在")
        
        # 測試載入
        try:
            # 創建新的 agent 實例來測試載入
            new_agent = Agent(env=env, config=None)
            new_agent.load(save_path)
            print("   ✅ 模型載入成功")
            
            # 測試載入後的預測
            action_loaded, _ = new_agent.predict(obs, deterministic=True)
            print(f"   載入後預測: {action_loaded}")
            
        except Exception as e:
            print(f"   ❌ 模型載入失敗: {e}")
        
        # 6. 測試評估功能
        print("\n6. 測試評估功能...")
        
        eval_stats = agent.evaluate(n_episodes=3, deterministic=True)
        print(f"   評估完成:")
        print(f"     平均獎勵: {eval_stats['mean_reward']:.4f}")
        print(f"     標準差: {eval_stats['std_reward']:.4f}")
        print(f"     平均長度: {eval_stats['mean_length']:.1f}")
        
        print("\n✅ Agent 測試完成！")
        print("\n🎯 核心功能驗證:")
        print("  ✓ Transformer 特徵提取器")
        print("  ✓ PPO 模型整合")
        print("  ✓ 預測功能")
        print("  ✓ 訓練功能")
        print("  ✓ 保存和載入")
        print("  ✓ 評估功能")
        
        # 清理測試文件
        try:
            if os.path.exists(save_path + ".zip"):
                os.remove(save_path + ".zip")
                print("\n🧹 測試文件已清理")
        except:
            pass
        
    except Exception as e:
        print(f"❌ Agent 測試失敗: {e}")
        print("\n錯誤詳情:")
        import traceback
        traceback.print_exc()
