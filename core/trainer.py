"""
CryptoAce Trainer 模組

此模組負責編排整個模型的訓練流程，實現滾動窗口、經驗回放和標準化的實驗產物管理。
"""

import os
import shutil
import joblib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np

# 導入模組 - 支援直接執行和模組導入
try:
    from .configurator import Configurator
    from .logger import setup_logger
    from .data_harvester import DataHarvester
    from .feature_engine import FeatureEngine
    from .trading_env import TradingEnv
    from .agent import Agent
    from .utils import set_random_seed
except ImportError:
    # 當直接執行此檔案時使用絕對導入
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.configurator import Configurator
    from core.logger import setup_logger
    from core.data_harvester import DataHarvester
    from core.feature_engine import FeatureEngine
    from core.trading_env import TradingEnv
    from core.agent import Agent
    from core.utils import set_random_seed


class Trainer:
    """
    訓練器類別
    
    負責編排整個模型的訓練流程，包括：
    - 滾動窗口 (Walk-Forward) 訓練
    - 經驗回放機制
    - 實驗產物管理
    """
    
    def __init__(self, 
                 config: Configurator,
                 logger: Optional[Any] = None):
        """
        初始化訓練器
        
        Args:
            config: 配置管理器實例
            logger: 日誌記錄器實例
        """
        self.config = config
        self.logger = logger or setup_logger(config)
        
        # 訓練參數
        self.walk_forward_steps = config.training.get('walk_forward_steps', 30)  # 滾動窗口步數
        self.window_size = config.training.get('window_size', 1000)  # 訓練窗口大小
        self.retrain_frequency = config.training.get('retrain_frequency', 10)  # 重訓練頻率
        self.experience_replay_ratio = config.training.get('experience_replay_ratio', 0.3)  # 經驗回放比例
        
        # 組件實例
        self.data_harvester = None
        self.feature_engine = None
        self.env = None
        self.agent = None
        
        # 實驗管理
        self.experiment_dir = None
        self.results_history = []
        
        self.logger.info("Trainer 初始化完成")
    
    def _create_experiment_directory(self) -> str:
        """
        創建唯一的實驗目錄
        
        Returns:
            實驗目錄路徑
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"experiment_{timestamp}"
        experiment_dir = Path(self.config.agent.get('model_path', './models/')) / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"創建實驗目錄: {experiment_dir}")
        return str(experiment_dir)
    
    def _copy_config_to_experiment(self, experiment_dir: str) -> None:
        """
        將配置文件複製到實驗目錄
        
        Args:
            experiment_dir: 實驗目錄路徑
        """
        config_source = Path("config.yaml")
        config_dest = Path(experiment_dir) / "config.yaml"
        
        if config_source.exists():
            shutil.copy2(config_source, config_dest)
            self.logger.info(f"配置文件已複製到: {config_dest}")
        else:
            self.logger.warning("未找到 config.yaml 文件")
    
    def _initialize_components(self) -> None:
        """
        初始化訓練組件
        """
        # 初始化數據收集器
        self.data_harvester = DataHarvester(
            config=self.config,
            logger=self.logger
        )
        
        # 初始化特徵引擎
        self.feature_engine = FeatureEngine(
            config=self.config,
            logger=self.logger
        )
        
        self.logger.info("訓練組件初始化完成")
    
    def _prepare_training_data(self, 
                               start_idx: int, 
                               end_idx: int,
                               experience_buffer: Optional[List[pd.DataFrame]] = None) -> Tuple[pd.DataFrame, Any]:
        """
        準備訓練數據，包含經驗回放機制
        
        Args:
            start_idx: 數據開始索引
            end_idx: 數據結束索引
            experience_buffer: 經驗回放緩衝區
            
        Returns:
            處理後的特徵數據和標量器
        """
        # 獲取當前窗口的原始數據
        raw_data = self.data_harvester.get_data_slice(start_idx, end_idx)
        
        # 經驗回放：混合新舊數據
        if experience_buffer and len(experience_buffer) > 0:
            replay_size = int(len(raw_data) * self.experience_replay_ratio)
            
            # 從經驗緩衝區隨機採樣
            replay_data_list = []
            for buffer_data in experience_buffer[-3:]:  # 使用最近3個窗口的數據
                sample_size = min(replay_size // len(experience_buffer[-3:]), len(buffer_data))
                if sample_size > 0:
                    sampled_data = buffer_data.sample(n=sample_size, random_state=42)
                    replay_data_list.append(sampled_data)
            
            if replay_data_list:
                replay_data = pd.concat(replay_data_list, ignore_index=True)
                # 混合當前數據和回放數據
                mixed_data = pd.concat([raw_data, replay_data], ignore_index=True)
                mixed_data = mixed_data.sort_index().reset_index(drop=True)
                raw_data = mixed_data
                
                self.logger.info(f"經驗回放：添加了 {len(replay_data)} 條歷史數據")
        
        # 特徵工程
        features, scaler = self.feature_engine.fit_transform(raw_data)
        
        return features, scaler
    
    def _create_trading_environment(self, features: pd.DataFrame) -> TradingEnv:
        """
        創建交易環境
        
        Args:
            features: 特徵數據
            
        Returns:
            交易環境實例
        """
        env = TradingEnv(
            data=features,
            initial_balance=self.config.trading_env.get('initial_balance', 10000.0),
            transaction_cost=self.config.exchange.get('fee', {}).get('taker', 0.0006),
            max_position_change_per_step=self.config.trading_env.get('risk_management', {}).get(
                'max_position_change_per_step', 0.5),
            max_drawdown_limit=self.config.trading_env.get('risk_management', {}).get(
                'max_drawdown_limit', 0.20),
            lookback_window=self.config.training.get('lookback_window', 20)
        )
        
        return env
    
    def _train_agent(self, 
                     env: TradingEnv, 
                     timesteps: int,
                     is_retrain: bool = False) -> Dict[str, Any]:
        """
        訓練代理
        
        Args:
            env: 交易環境
            timesteps: 訓練步數
            is_retrain: 是否為重新訓練
            
        Returns:
            訓練結果統計
        """
        if self.agent is None or is_retrain:
            # 創建新的代理或重新初始化
            self.agent = Agent(
                env=env,
                config=self.config,
                logger=self.logger
            )
            self.logger.info("創建新的 Agent 實例")
        else:
            # 更新現有代理的環境
            self.agent.env = env
            self.logger.info("更新 Agent 的訓練環境")
        
        # 開始訓練
        training_stats = self.agent.train(total_timesteps=timesteps)
        
        return training_stats
    
    def _save_experiment_artifacts(self, 
                                   experiment_dir: str,
                                   scaler: Any,
                                   step: int) -> None:
        """
        保存實驗產物
        
        Args:
            experiment_dir: 實驗目錄路徑
            scaler: 標量器對象
            step: 當前步數
        """
        # 保存模型
        model_path = Path(experiment_dir) / f"model_step_{step}"
        self.agent.save(str(model_path))
        self.logger.info(f"模型已保存: {model_path}.zip")
        
        # 保存標量器
        scaler_path = Path(experiment_dir) / f"scaler_step_{step}.joblib"
        joblib.dump(scaler, scaler_path)
        self.logger.info(f"標量器已保存: {scaler_path}")
        
        # 保存最新版本（覆蓋）
        latest_model_path = Path(experiment_dir) / "model"
        latest_scaler_path = Path(experiment_dir) / "scaler.joblib"
        
        self.agent.save(str(latest_model_path))
        joblib.dump(scaler, latest_scaler_path)
        
        self.logger.info("最新版本的模型和標量器已保存")
    
    def _evaluate_step(self, 
                       env: TradingEnv,
                       step: int) -> Dict[str, Any]:
        """
        評估當前步驟的性能
        
        Args:
            env: 交易環境
            step: 當前步數
            
        Returns:
            評估結果
        """
        if self.agent is None:
            return {}
        
        # 評估代理性能
        eval_stats = self.agent.evaluate(n_episodes=5, deterministic=True)
        eval_stats['step'] = step
        eval_stats['timestamp'] = datetime.now().isoformat()
        
        self.results_history.append(eval_stats)
        
        self.logger.info(f"步驟 {step} 評估結果 - 平均獎勵: {eval_stats.get('mean_reward', 0):.4f}")
        
        return eval_stats
    
    def run_training(self) -> Dict[str, Any]:
        """
        執行完整的訓練流程
        
        Returns:
            訓練結果統計
        """
        try:
            # 設定隨機種子
            random_seed = self.config.agent.get('random_seed', 42)
            set_random_seed(random_seed)
            
            # 創建實驗目錄
            self.experiment_dir = self._create_experiment_directory()
            
            # 複製配置文件
            self._copy_config_to_experiment(self.experiment_dir)
            
            # 初始化組件
            self._initialize_components()
            
            # 獲取總數據量
            total_data_size = len(self.data_harvester.get_full_dataset())
            self.logger.info(f"總數據量: {total_data_size}")
            
            # 經驗回放緩衝區
            experience_buffer = []
            
            # 滾動窗口訓練主循環
            for step in range(self.walk_forward_steps):
                self.logger.info(f"=== 開始訓練步驟 {step + 1}/{self.walk_forward_steps} ===")
                
                # 計算當前窗口的數據範圍
                end_idx = min(
                    (step + 1) * self.window_size // self.walk_forward_steps + self.window_size,
                    total_data_size
                )
                start_idx = max(0, end_idx - self.window_size)
                
                self.logger.info(f"數據窗口: [{start_idx}:{end_idx}]")
                
                # 準備訓練數據（包含經驗回放）
                features, scaler = self._prepare_training_data(
                    start_idx, end_idx, experience_buffer
                )
                
                # 將當前窗口加入經驗緩衝區
                current_raw_data = self.data_harvester.get_data_slice(start_idx, end_idx)
                experience_buffer.append(current_raw_data)
                
                # 限制緩衝區大小
                if len(experience_buffer) > 5:
                    experience_buffer.pop(0)
                
                # 創建交易環境
                env = self._create_trading_environment(features)
                
                # 決定是否需要重新訓練
                is_retrain = (step % self.retrain_frequency == 0)
                
                # 計算訓練步數
                base_timesteps = self.config.training.get('base_timesteps', 10000)
                timesteps = base_timesteps if is_retrain else base_timesteps // 2
                
                # 訓練代理
                training_stats = self._train_agent(env, timesteps, is_retrain)
                
                # 評估性能
                eval_stats = self._evaluate_step(env, step + 1)
                
                # 保存實驗產物
                if (step + 1) % 5 == 0 or step == self.walk_forward_steps - 1:
                    self._save_experiment_artifacts(self.experiment_dir, scaler, step + 1)
                
                self.logger.info(f"步驟 {step + 1} 完成")
            
            # 保存最終結果
            results_summary = {
                'experiment_dir': self.experiment_dir,
                'total_steps': self.walk_forward_steps,
                'results_history': self.results_history,
                'final_performance': self.results_history[-1] if self.results_history else {},
                'config_snapshot': dict(self.config._config) if hasattr(self.config, '_config') else {}
            }
            
            # 保存結果摘要
            import json
            results_path = Path(self.experiment_dir) / "training_results.json"
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results_summary, f, indent=2, default=str)
            
            self.logger.info(f"訓練完成！實驗目錄: {self.experiment_dir}")
            self.logger.info(f"結果摘要已保存: {results_path}")
            
            return results_summary
            
        except Exception as e:
            self.logger.error(f"訓練過程中發生錯誤: {str(e)}")
            raise
    
    def load_experiment(self, experiment_dir: str) -> None:
        """
        載入已有的實驗
        
        Args:
            experiment_dir: 實驗目錄路徑
        """
        experiment_path = Path(experiment_dir)
        
        if not experiment_path.exists():
            raise FileNotFoundError(f"實驗目錄不存在: {experiment_dir}")
        
        # 載入模型
        model_path = experiment_path / "model"
        if model_path.with_suffix('.zip').exists():
            # 需要先創建環境和代理
            self._initialize_components()
            
            # 創建臨時環境用於初始化代理
            dummy_data = pd.DataFrame({
                'open': [1.0] * 100,
                'high': [1.1] * 100,
                'low': [0.9] * 100,
                'close': [1.0] * 100,
                'volume': [100.0] * 100,
                'feature_0': [0.0] * 100
            })
            temp_env = self._create_trading_environment(dummy_data)
            
            self.agent = Agent(env=temp_env, config=self.config, logger=self.logger)
            self.agent.load(str(model_path))
            
            self.logger.info(f"模型已載入: {model_path}")
        
        # 載入標量器
        scaler_path = experiment_path / "scaler.joblib"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            if self.feature_engine:
                self.feature_engine.scaler = scaler
            self.logger.info(f"標量器已載入: {scaler_path}")
        
        self.experiment_dir = str(experiment_path)
        self.logger.info(f"實驗已載入: {experiment_dir}")


if __name__ == "__main__":
    """
    單元測試區塊
    
    在此區塊中調用 trainer.run_training() 方法來啟動一個完整的訓練流程。
    """
    print("=== CryptoAce Trainer 單元測試 ===")
    
    try:
        # 載入配置
        config = Configurator(config_path="config.yaml")
        
        # 設定測試參數（縮短訓練時間）
        if not hasattr(config, '_config'):
            config._config = {}
        config._config['training'] = {
            'walk_forward_steps': 3,      # 減少步數用於測試
            'window_size': 500,           # 減少窗口大小
            'retrain_frequency': 2,       # 重訓練頻率
            'experience_replay_ratio': 0.2,  # 經驗回放比例
            'base_timesteps': 1000,       # 基礎訓練步數
            'lookback_window': 10         # 回看窗口
        }
        
        # 創建訓練器
        trainer = Trainer(config=config)
        
        print("開始執行訓練流程...")
        
        # 執行訓練
        results = trainer.run_training()
        
        print("\n=== 訓練完成 ===")
        print(f"實驗目錄: {results['experiment_dir']}")
        print(f"總步驟數: {results['total_steps']}")
        
        if results['results_history']:
            final_perf = results['final_performance']
            print(f"最終性能 - 平均獎勵: {final_perf.get('mean_reward', 0):.4f}")
            print(f"最終性能 - 標準差: {final_perf.get('std_reward', 0):.4f}")
        
        print("單元測試完成！")
        
    except Exception as e:
        print(f"單元測試失敗: {str(e)}")
        import traceback
        traceback.print_exc()
