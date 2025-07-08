"""
CryptoAce Trainer 模組測試

此模組測試 Trainer 類別的完整訓練流程，包括：
- 滾動窗口訓練
- 經驗回放機制
- 實驗產物管理
"""

import os
import sys
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# 添加專案根目錄到 Python 路徑
if __name__ == "__main__":
    # 當直接執行此檔案時，添加專案根目錄到路徑
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

from core.configurator import Configurator
from core.trainer import Trainer


class TestTrainer:
    """Trainer 類別測試"""
    
    @pytest.fixture
    def temp_config_dir(self):
        """創建臨時配置目錄，帶有強健的清理機制"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        
        # 強健的清理邏輯
        def safe_cleanup_directory(dir_path):
            """安全清理目錄，處理文件鎖定問題"""
            import time
            import gc
            import logging
            
            try:
                # 關閉可能的日誌處理器
                for logger_name in ['cryptoace', 'trainer']:
                    logger = logging.getLogger(logger_name)
                    for handler in logger.handlers[:]:
                        try:
                            handler.close()
                            logger.removeHandler(handler)
                        except:
                            pass
                
                # 強制垃圾回收
                gc.collect()
                
                # 多次嘗試刪除
                for attempt in range(3):
                    try:
                        if os.path.exists(dir_path):
                            # Windows 特殊處理
                            if os.name == 'nt':
                                for root, dirs, files in os.walk(dir_path):
                                    for file in files:
                                        try:
                                            file_path = os.path.join(root, file)
                                            os.chmod(file_path, 0o777)
                                        except:
                                            pass
                            
                            shutil.rmtree(dir_path)
                            return
                    except (PermissionError, OSError):
                        if attempt < 2:
                            time.sleep(0.1 * (attempt + 1))
                        # 最後一次嘗試失敗時靜默處理，避免干擾測試輸出
            except:
                pass  # 靜默處理清理錯誤
        
        safe_cleanup_directory(temp_dir)
    
    @pytest.fixture
    def mock_config_yaml(self, temp_config_dir):
        """創建測試用的配置文件"""
        config_content = """
exchange:
  name: bitget
  fee:
    taker: 0.0006
    maker: 0.0004

data:
  timeframe: 5m
  start_date: "2023-01-01T00:00:00Z"
  raw_data_path: "./data/raw/"
  feature_data_path: "./data/features/"

trading_env:
  initial_balance: 10000.0
  risk_management:
    max_position_change_per_step: 0.5
    max_drawdown_limit: 0.20

agent:
  model_path: "./models/"
  random_seed: 42
  ppo:
    learning_rate: 0.0003
    batch_size: 64
    n_steps: 100
  transformer:
    d_model: 256
    nhead: 8
    num_layers: 4

training:
  walk_forward_steps: 2
  window_size: 50
  retrain_frequency: 1
  experience_replay_ratio: 0.2
  base_timesteps: 100
  lookback_window: 5

logger:
  level: INFO
  file_path: "./logs/cryptoace.log"
  rotation: "1 day"
  retention: "7 days"
"""
        config_path = Path(temp_config_dir) / "config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        return str(config_path)
    
    @pytest.fixture
    def mock_data(self):
        """創建測試用的假數據"""
        # 創建 100 條簡單的OHLCV數據
        np.random.seed(42)
        n_samples = 100
        
        # 生成基礎價格序列
        base_price = 50000
        price_changes = np.random.normal(0, 0.02, n_samples).cumsum()
        close_prices = base_price * (1 + price_changes)
        
        # 生成其他價格數據
        noise = np.random.normal(0, 0.005, n_samples)
        open_prices = close_prices * (1 + noise)
        high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(noise))
        low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(noise))
        volumes = np.random.uniform(100, 1000, n_samples)
        
        data = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        })
        
        return data
    
    @pytest.fixture
    def trainer_with_mocks(self, mock_config_yaml, mock_data, temp_config_dir):
        """創建帶有模擬數據的 Trainer 實例"""
        # 切換到臨時目錄
        original_cwd = os.getcwd()
        os.chdir(temp_config_dir)
        
        try:
            # 載入配置
            config = Configurator(config_path=mock_config_yaml)
            
            # 創建必要的目錄
            Path("./data/raw/").mkdir(parents=True, exist_ok=True)
            Path("./models/").mkdir(parents=True, exist_ok=True)
            Path("./logs/").mkdir(parents=True, exist_ok=True)
            
            # 保存假數據到 parquet 檔案
            data_file = Path("./data/raw/test_data.parquet")
            mock_data.to_parquet(data_file)
            
            # 創建 Trainer 實例
            trainer = Trainer(config=config)
            
            yield trainer, config, temp_config_dir
            
        finally:
            # 確保切換回原目錄
            os.chdir(original_cwd)
            
            # 清理 trainer 相關的資源
            try:
                if 'trainer' in locals() and hasattr(trainer, 'logger'):
                    # 關閉 trainer 的日誌處理器
                    if trainer.logger and hasattr(trainer.logger, 'handlers'):
                        for handler in trainer.logger.handlers[:]:
                            try:
                                handler.close()
                                trainer.logger.removeHandler(handler)
                            except:
                                pass
            except:
                pass
    
    @pytest.mark.slow
    def test_trainer_full_workflow(self, trainer_with_mocks):
        """測試 Trainer 完整的訓練流程"""
        trainer, config, temp_dir = trainer_with_mocks
        
        # 模擬 DataHarvester 的方法
        with patch.object(trainer, '_initialize_components') as mock_init:
            # 創建模擬的數據收集器
            mock_harvester = MagicMock()
            mock_harvester.get_full_dataset.return_value = pd.DataFrame({
                'open': [50000] * 60,
                'high': [50100] * 60,
                'low': [49900] * 60,
                'close': [50050] * 60,
                'volume': [500] * 60
            })
            mock_harvester.get_data_slice.return_value = pd.DataFrame({
                'open': [50000] * 30,
                'high': [50100] * 30,
                'low': [49900] * 30,
                'close': [50050] * 30,
                'volume': [500] * 30
            })
            
            # 創建模擬的特徵引擎
            mock_feature_engine = MagicMock()
            mock_features = pd.DataFrame({
                'close': [50000] * 20,
                'feature_0': [0.1] * 20,
                'feature_1': [0.2] * 20
            })
            mock_scaler = MagicMock()
            mock_feature_engine.fit_transform.return_value = (mock_features, mock_scaler)
            
            trainer.data_harvester = mock_harvester
            trainer.feature_engine = mock_feature_engine
            
            # 模擬 Agent 的訓練和評估
            with patch('core.trainer.Agent') as mock_agent_class:
                mock_agent = MagicMock()
                mock_agent.train.return_value = {'mean_reward': -50.0}
                mock_agent.evaluate.return_value = {'mean_reward': -30.0, 'std_reward': 5.0}
                mock_agent.save = MagicMock()
                mock_agent_class.return_value = mock_agent
                
                # 執行訓練
                results = trainer.run_training()
                
                # 驗證返回結果
                assert 'experiment_dir' in results
                assert 'total_steps' in results
                assert 'results_history' in results
                assert results['total_steps'] == 2  # 配置中設定的步數
                
                # 驗證實驗目錄存在
                experiment_dir = Path(results['experiment_dir'])
                assert experiment_dir.exists()
                assert experiment_dir.is_dir()
                
                # 驗證必要的文件存在
                assert (experiment_dir / 'config.yaml').exists()
                assert (experiment_dir / 'training_results.json').exists()
                
                # 驗證模型和標量器保存被調用
                mock_agent.save.assert_called()
    
    @pytest.mark.slow  
    def test_trainer_experiment_artifacts(self, trainer_with_mocks):
        """測試實驗產物的正確創建"""
        trainer, config, temp_dir = trainer_with_mocks
        
        with patch.object(trainer, '_initialize_components'):
            # 設置模擬對象
            trainer.data_harvester = MagicMock()
            trainer.data_harvester.get_full_dataset.return_value = pd.DataFrame({
                'close': [1.0] * 40
            })
            trainer.data_harvester.get_data_slice.return_value = pd.DataFrame({
                'close': [1.0] * 20
            })
            
            trainer.feature_engine = MagicMock()
            trainer.feature_engine.fit_transform.return_value = (
                pd.DataFrame({'close': [1.0] * 15, 'feature_0': [0.0] * 15}),
                MagicMock()
            )
            
            with patch('core.trainer.Agent') as mock_agent_class:
                mock_agent = MagicMock()
                mock_agent.train.return_value = {'mean_reward': -20.0}
                mock_agent.evaluate.return_value = {'mean_reward': -15.0, 'std_reward': 2.0}
                mock_agent_class.return_value = mock_agent
                
                # 執行訓練
                results = trainer.run_training()
                
                # 檢查實驗目錄結構
                exp_dir = Path(results['experiment_dir'])
                
                # 驗證配置文件備份
                config_backup = exp_dir / 'config.yaml'
                assert config_backup.exists()
                
                # 驗證結果摘要文件
                results_file = exp_dir / 'training_results.json'
                assert results_file.exists()
                
                # 驗證模型保存調用
                assert mock_agent.save.call_count >= 1
    
    def test_trainer_load_experiment(self, trainer_with_mocks):
        """測試實驗載入功能"""
        trainer, config, temp_dir = trainer_with_mocks
        
        # 創建假的實驗目錄
        exp_dir = Path(temp_dir) / "test_experiment"
        exp_dir.mkdir(exist_ok=True)
        
        # 創建假的模型文件
        (exp_dir / "model.zip").touch()
        
        # 創建假的標量器文件
        import joblib
        from sklearn.preprocessing import StandardScaler
        fake_scaler = StandardScaler()
        joblib.dump(fake_scaler, exp_dir / "scaler.joblib")
        
        with patch.object(trainer, '_initialize_components'):
            trainer.feature_engine = MagicMock()
            
            with patch('core.trainer.Agent') as mock_agent_class:
                mock_agent = MagicMock()
                mock_agent_class.return_value = mock_agent
                
                # 測試載入實驗
                trainer.load_experiment(str(exp_dir))
                
                # 驗證實驗目錄設置
                assert trainer.experiment_dir == str(exp_dir)
    
    def test_trainer_initialization(self, mock_config_yaml):
        """測試 Trainer 初始化"""
        config = Configurator(config_path=mock_config_yaml)
        
        with patch('core.trainer.setup_logger') as mock_logger:
            mock_logger.return_value = MagicMock()
            
            trainer = Trainer(config=config)
            
            # 驗證訓練參數正確設置
            assert trainer.walk_forward_steps == 2
            assert trainer.window_size == 50
            assert trainer.retrain_frequency == 1
            assert trainer.experience_replay_ratio == 0.2
            
            # 驗證組件初始化為 None
            assert trainer.data_harvester is None
            assert trainer.feature_engine is None
            assert trainer.env is None
            assert trainer.agent is None
            
            # 驗證實驗管理變數
            assert trainer.experiment_dir is None
            assert trainer.results_history == []
    
    def test_create_experiment_directory(self, trainer_with_mocks):
        """測試實驗目錄創建"""
        trainer, config, temp_dir = trainer_with_mocks
        
        # 創建實驗目錄
        exp_dir = trainer._create_experiment_directory()
        
        # 驗證目錄存在且命名正確
        assert Path(exp_dir).exists()
        assert Path(exp_dir).is_dir()
        assert 'experiment_' in exp_dir
        assert str(Path(exp_dir).parent) == str(Path(temp_dir) / "models")


if __name__ == "__main__":
    """
    直接運行測試的主程式區塊
    
    這允許您直接執行此檔案來運行所有測試，
    而不需要使用 pytest 命令。
    """
    import sys
    
    print("=== CryptoAce Trainer 測試 ===")
    print("正在運行 Trainer 模組的完整測試套件...")
    print()
    
    # 創建測試實例
    test_instance = TestTrainer()
    
    try:
        # 運行所有測試方法
        print("1. 測試 Trainer 初始化...")
        
        # 創建臨時配置
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        try:
            # 創建配置文件
            config_content = """
exchange:
  name: bitget
  fee:
    taker: 0.0006
    maker: 0.0004

data:
  timeframe: 5m
  start_date: "2023-01-01T00:00:00Z"
  raw_data_path: "./data/raw/"
  feature_data_path: "./data/features/"

trading_env:
  initial_balance: 10000.0
  risk_management:
    max_position_change_per_step: 0.5
    max_drawdown_limit: 0.20

agent:
  model_path: "./models/"
  random_seed: 42
  ppo:
    learning_rate: 0.0003
    batch_size: 64
    n_steps: 100
  transformer:
    d_model: 256
    nhead: 8
    num_layers: 4

training:
  walk_forward_steps: 2
  window_size: 50
  retrain_frequency: 1
  experience_replay_ratio: 0.2
  base_timesteps: 100
  lookback_window: 5

logger:
  level: INFO
  file_path: "./logs/cryptoace.log"
  rotation: "1 day"
  retention: "7 days"
"""
            config_path = Path(temp_dir) / "config.yaml"
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(config_content)
            
            # 測試初始化
            test_instance.test_trainer_initialization(str(config_path))
            print("   ✅ Trainer 初始化測試通過")
            
            print("\n2. 測試實驗目錄創建...")
            
            # 創建假數據
            np.random.seed(42)
            mock_data = pd.DataFrame({
                'open': [50000] * 20,
                'high': [50100] * 20,
                'low': [49900] * 20,
                'close': [50050] * 20,
                'volume': [500] * 20
            })
            
            # 設置環境
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            
            try:
                # 創建目錄
                Path("./data/raw/").mkdir(parents=True, exist_ok=True)
                Path("./models/").mkdir(parents=True, exist_ok=True)
                Path("./logs/").mkdir(parents=True, exist_ok=True)
                
                # 保存數據
                mock_data.to_parquet("./data/raw/test_data.parquet")
                
                # 創建 trainer
                config = Configurator(config_path=str(config_path))
                trainer = Trainer(config=config)
                
                # 測試目錄創建
                exp_dir = trainer._create_experiment_directory()
                assert Path(exp_dir).exists()
                print("   ✅ 實驗目錄創建測試通過")
                
                print("\n3. 測試實驗載入功能...")
                
                # 創建假實驗
                test_exp_dir = Path(temp_dir) / "test_experiment"
                test_exp_dir.mkdir(exist_ok=True)
                (test_exp_dir / "model.zip").touch()
                
                # 創建一個真實的可序列化對象而不是 MagicMock
                import joblib
                from sklearn.preprocessing import StandardScaler
                fake_scaler = StandardScaler()
                joblib.dump(fake_scaler, test_exp_dir / "scaler.joblib")
                
                # 模擬必要組件
                trainer.feature_engine = MagicMock()
                with patch('core.trainer.Agent'):
                    trainer.load_experiment(str(test_exp_dir))
                    
                print("   ✅ 實驗載入測試通過")
                
            finally:
                os.chdir(original_cwd)
            
        finally:
            # 強健的臨時目錄清理
            def cleanup_temp_directory(temp_dir_path):
                """
                強健的臨時目錄清理函數，處理 Windows 文件鎖定問題
                """
                import time
                import gc
                import logging
                
                try:
                    # 1. 關閉所有可能的日誌處理器
                    for logger_name in ['cryptoace', 'trainer', 'root']:
                        logger = logging.getLogger(logger_name)
                        for handler in logger.handlers[:]:
                            try:
                                handler.close()
                                logger.removeHandler(handler)
                            except:
                                pass
                    
                    # 2. 關閉根日誌處理器
                    for handler in logging.root.handlers[:]:
                        try:
                            handler.close()
                            logging.root.removeHandler(handler)
                        except:
                            pass
                    
                    # 3. 強制垃圾回收，釋放文件句柄
                    gc.collect()
                    
                    # 4. 多次嘗試刪除，逐漸增加等待時間
                    for attempt in range(5):
                        try:
                            if os.path.exists(temp_dir_path):
                                # 在 Windows 上，先嘗試改變文件權限
                                if os.name == 'nt':  # Windows
                                    for root, dirs, files in os.walk(temp_dir_path):
                                        for file in files:
                                            try:
                                                file_path = os.path.join(root, file)
                                                os.chmod(file_path, 0o777)
                                            except:
                                                pass
                                
                                shutil.rmtree(temp_dir_path)
                                return True  # 成功刪除
                        except (PermissionError, OSError) as e:
                            if attempt < 4:  # 不是最後一次嘗試
                                wait_time = 0.1 * (2 ** attempt)  # 指數退避: 0.1, 0.2, 0.4, 0.8 秒
                                time.sleep(wait_time)
                            else:
                                print(f"   ⚠️  警告: 經過 {attempt + 1} 次嘗試後仍無法完全清理臨時目錄")
                                print(f"       目錄: {temp_dir_path}")
                                print(f"       錯誤: {e}")
                                return False
                    
                except Exception as cleanup_error:
                    print(f"   ⚠️  清理過程中出現意外錯誤: {cleanup_error}")
                    return False
                
                return False
            
            # 執行清理
            cleanup_temp_directory(temp_dir)
        
        print("\n4. 跳過完整工作流程測試 (標記為 @pytest.mark.slow)")
        print("   💡 要運行完整測試，請使用: pytest tests/test_trainer.py -m slow")
        
        print("\n🎉 所有快速測試通過！")
        print("\n測試摘要:")
        print("  ✅ Trainer 初始化")
        print("  ✅ 實驗目錄創建")
        print("  ✅ 實驗載入功能")
        print("  ⏩ 完整工作流程測試 (跳過)")
        print("  ⏩ 實驗產物測試 (跳過)")
        
        print(f"\n如要運行完整測試套件，請使用:")
        print(f"  pytest {__file__} -v")
        print(f"  pytest {__file__} -m slow -v  # 只運行慢速測試")
        
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        print("\n錯誤詳情:")
        import traceback
        traceback.print_exc()
        sys.exit(1)
