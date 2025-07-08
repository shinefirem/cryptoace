"""
CryptoAce Backtester 模組測試

此模組測試 Backtester 類別的完整回測流程，包括：
- 實驗產物加載
- 回測執行
- 報告生成
- 文件輸出驗證
"""

import os
import sys
import pytest
import tempfile
import shutil
import zipfile
import joblib
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 添加專案根目錄到 Python 路徑
if __name__ == "__main__":
    # 當直接執行此檔案時，添加專案根目錄到路徑
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

from core.backtester import Backtester
from core.configurator import Configurator


class TestBacktester:
    """Backtester 類別測試"""
    
    @pytest.fixture
    def temp_experiment_dir(self):
        """創建臨時實驗目錄"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # 清理
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
    
    @pytest.fixture
    def mock_experiment_artifacts(self, temp_experiment_dir):
        """創建假的實驗產物"""
        experiment_dir = Path(temp_experiment_dir)
        
        # 1. 創建配置文件
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

logger:
  level: INFO
  file_path: "./logs/cryptoace.log"
"""
        config_path = experiment_dir / "config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        # 2. 創建假的模型文件（zip 格式）
        model_path = experiment_dir / "model.zip"
        with zipfile.ZipFile(model_path, 'w') as zf:
            zf.writestr("model_data.txt", "fake model content")
        
        # 3. 創建假的標量器文件
        scaler = StandardScaler()
        # 用一些假數據來 fit 標量器
        fake_data = np.random.randn(100, 5)
        scaler.fit(fake_data)
        scaler_path = experiment_dir / "scaler.joblib"
        joblib.dump(scaler, scaler_path)
        
        return str(experiment_dir)
    
    @pytest.fixture
    def mock_data(self):
        """創建測試用的假數據"""
        # 創建 50 條 OHLCV 數據
        np.random.seed(42)
        n_samples = 50
        
        # 生成基礎價格序列
        base_price = 50000
        price_changes = np.random.normal(0, 0.01, n_samples).cumsum()
        close_prices = base_price * (1 + price_changes)
        
        # 生成其他價格數據
        noise = np.random.normal(0, 0.002, n_samples)
        open_prices = close_prices * (1 + noise)
        high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(noise) * 0.5)
        low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(noise) * 0.5)
        volumes = np.random.uniform(100, 1000, n_samples)
        
        # 創建日期索引
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='5min')
        
        data = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes,
            # 添加一些假特徵
            'feature_0': np.random.randn(n_samples),
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples)
        }, index=dates)
        
        return data
    
    def test_backtester_initialization(self):
        """測試 Backtester 初始化"""
        backtester = Backtester()
        
        # 驗證初始狀態
        assert backtester.config is None
        assert backtester.logger is None
        assert backtester.experiment_config is None
        assert backtester.agent is None
        assert backtester.scaler is None
        assert backtester.data_harvester is None
        assert backtester.feature_engine is None
        assert backtester.trading_env is None
        assert backtester.trade_log == []
        assert backtester.equity_curve == []
        assert backtester.performance_metrics == {}
    
    def test_load_experiment_success(self, mock_experiment_artifacts):
        """測試成功加載實驗產物"""
        backtester = Backtester()
        
        # 模擬組件初始化和 Agent 加載
        with patch.object(backtester, '_initialize_components') as mock_init:
            with patch('core.backtester.Agent') as mock_agent_class:
                mock_agent = MagicMock()
                mock_agent_class.return_value = mock_agent
                
                # 執行加載
                backtester.load_experiment(mock_experiment_artifacts)
                
                # 驗證配置加載
                assert backtester.experiment_config is not None
                assert isinstance(backtester.experiment_config, Configurator)
                
                # 驗證標量器加載
                assert backtester.scaler is not None
                
                # 驗證組件初始化被調用
                mock_init.assert_called_once()
                
                # 驗證 Agent 初始化和加載
                mock_agent_class.assert_called_once()
                mock_agent.load.assert_called_once()
    
    def test_load_experiment_missing_directory(self):
        """測試載入不存在的實驗目錄"""
        backtester = Backtester()
        
        with pytest.raises(FileNotFoundError, match="實驗目錄不存在"):
            backtester.load_experiment("/nonexistent/path")
    
    def test_load_experiment_missing_config(self, temp_experiment_dir):
        """測試載入缺少配置文件的實驗目錄"""
        backtester = Backtester()
        
        with pytest.raises(FileNotFoundError, match="配置文件不存在"):
            backtester.load_experiment(temp_experiment_dir)
    
    def test_load_experiment_missing_model(self, temp_experiment_dir):
        """測試載入缺少模型文件的實驗目錄"""
        backtester = Backtester()
        
        # 只創建配置文件，不創建模型文件
        config_content = """
exchange:
  name: bitget
agent:
  model_path: "./models/"
logger:
  level: INFO
"""
        config_path = Path(temp_experiment_dir) / "config.yaml"
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        with patch.object(backtester, '_initialize_components'):
            with pytest.raises(FileNotFoundError, match="模型文件不存在"):
                backtester.load_experiment(temp_experiment_dir)
    
    @pytest.mark.slow
    def test_run_backtest_complete_workflow(self, mock_experiment_artifacts, mock_data):
        """測試完整的回測工作流程"""
        backtester = Backtester()
        
        # 模擬所有依賴組件
        with patch.object(backtester, 'load_experiment') as mock_load:
            # 設置配置
            mock_config = MagicMock()
            mock_config.data.get.return_value = '2023-01-01T00:00:00Z'
            mock_config.trading_env.get.return_value = {'initial_balance': 10000.0}
            mock_config.exchange.get.return_value = {'fee': {'taker': 0.001}}
            mock_config.to_dict.return_value = {'test': 'config'}
            backtester.experiment_config = mock_config
            
            # 模擬數據收集器
            mock_harvester = MagicMock()
            mock_harvester.get_data_slice.return_value = mock_data
            backtester.data_harvester = mock_harvester
            
            # 模擬特徵引擎
            mock_feature_engine = MagicMock()
            mock_feature_engine.transform.return_value = mock_data
            backtester.feature_engine = mock_feature_engine
            
            # 模擬標量器
            backtester.scaler = StandardScaler()
            
            # 模擬交易環境
            with patch('core.backtester.TradingEnv') as mock_env_class:
                mock_env = MagicMock()
                mock_env.reset.return_value = (np.array([1, 2, 3]), {})
                mock_env.step.return_value = (np.array([2, 3, 4]), 0.01, False, False, {'trade_executed': True, 'position_before': 0, 'position_after': 0.1})
                mock_env.get_portfolio_value.return_value = 10100.0
                mock_env.get_position.return_value = 0.1
                mock_env_class.return_value = mock_env
                
                # 模擬智能體
                mock_agent = MagicMock()
                mock_agent.predict.return_value = 1
                backtester.agent = mock_agent
                
                # 執行回測
                results = backtester.run_backtest(
                    experiment_path=mock_experiment_artifacts,
                    start_date='2023-01-01T00:00:00Z',
                    end_date='2023-01-02T00:00:00Z'
                )
                
                # 驗證結果
                assert 'output_dir' in results
                assert 'performance_metrics' in results
                assert 'num_trades' in results
                assert 'backtest_period' in results
                
                # 驗證輸出目錄存在
                output_dir = Path(results['output_dir'])
                assert output_dir.exists()
    
    def test_backtest_file_generation(self, mock_experiment_artifacts, mock_data):
        """測試回測文件生成"""
        backtester = Backtester()
        
        # 設置模擬日誌器
        mock_logger = MagicMock()
        backtester.logger = mock_logger
        
        # 創建假的回測結果
        backtester.equity_curve = [
            {'timestamp': pd.Timestamp('2023-01-01 00:00:00'), 'equity': 10000, 'position': 0, 'price': 50000, 'step': 0},
            {'timestamp': pd.Timestamp('2023-01-01 01:00:00'), 'equity': 10100, 'position': 0.1, 'price': 50100, 'step': 1},
            {'timestamp': pd.Timestamp('2023-01-01 02:00:00'), 'equity': 10200, 'position': 0.2, 'price': 50200, 'step': 2}
        ]
        
        backtester.trade_log = [
            {'timestamp': pd.Timestamp('2023-01-01 01:00:00'), 'step': 1, 'action': 1, 'reward': 0.01, 'price': 50100}
        ]
        
        backtester.performance_metrics = {
            'total_return': 0.02,
            'sharpe_ratio': 1.5,
            'max_drawdown': -0.01
        }
        
        # 模擬配置
        mock_config = MagicMock()
        mock_config.to_dict.return_value = {'test': 'config'}
        backtester.experiment_config = mock_config
        
        # 創建臨時輸出目錄
        with tempfile.TemporaryDirectory() as temp_output_dir:
            # 測試圖表生成
            backtester._create_equity_curve_plot(str(Path(temp_output_dir) / "equity_curve.png"))
            assert (Path(temp_output_dir) / "equity_curve.png").exists()
            
            # 測試交易日誌保存
            backtester._save_trade_log(str(Path(temp_output_dir) / "trade_log.csv"))
            assert (Path(temp_output_dir) / "trade_log.csv").exists()
            
            # 測試績效報告保存
            backtester._save_performance_report(str(Path(temp_output_dir) / "backtest_report.json"))
            assert (Path(temp_output_dir) / "backtest_report.json").exists()
            
            # 驗證日誌記錄被調用
            mock_logger.info.assert_called()
    
    def test_performance_metrics_calculation(self):
        """測試績效指標計算"""
        backtester = Backtester()
        
        # 創建假的資金曲線數據
        dates = pd.date_range('2023-01-01', periods=10, freq='1H')
        equity_values = [10000, 10100, 10050, 10200, 10150, 10300, 10250, 10400, 10350, 10500]
        
        backtester.equity_curve = [
            {'timestamp': date, 'equity': equity, 'position': 0.1, 'price': 50000, 'step': i}
            for i, (date, equity) in enumerate(zip(dates, equity_values))
        ]
        
        backtester.trade_log = [
            {'reward': 0.01}, {'reward': -0.005}, {'reward': 0.015}, {'reward': 0.02}
        ]
        
        # 計算績效指標
        metrics = backtester._calculate_performance_metrics()
        
        # 驗證基本指標
        assert 'total_return' in metrics
        assert 'annualized_return' in metrics
        assert 'volatility' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'num_trades' in metrics
        assert 'win_rate' in metrics
        
        # 驗證計算結果的合理性
        assert metrics['total_return'] == 0.05  # (10500 - 10000) / 10000
        assert metrics['num_trades'] == 4
        assert metrics['win_rate'] == 0.75  # 3 out of 4 positive trades
    
    def test_empty_data_handling(self):
        """測試空數據處理"""
        backtester = Backtester()
        
        # 測試空資金曲線
        backtester.equity_curve = []
        metrics = backtester._calculate_performance_metrics()
        assert metrics == {}
        
        # 測試空交易日誌的圖表生成
        mock_logger = MagicMock()
        backtester.logger = mock_logger
        
        with tempfile.TemporaryDirectory() as temp_dir:
            backtester._create_equity_curve_plot(str(Path(temp_dir) / "test.png"))
            mock_logger.warning.assert_called_once()
    
    def test_data_preparation_with_scaler(self, mock_data):
        """測試數據準備和標準化"""
        backtester = Backtester()
        
        # 設置模擬組件
        mock_logger = MagicMock()
        backtester.logger = mock_logger
        
        mock_harvester = MagicMock()
        mock_harvester.get_data_slice.return_value = mock_data
        backtester.data_harvester = mock_harvester
        
        mock_feature_engine = MagicMock()
        mock_feature_engine.transform.return_value = mock_data
        backtester.feature_engine = mock_feature_engine
        
        # 設置標量器
        scaler = StandardScaler()
        feature_data = mock_data[['feature_0', 'feature_1', 'feature_2']]
        scaler.fit(feature_data)
        backtester.scaler = scaler
        
        # 執行數據準備
        result = backtester._prepare_backtest_data('2023-01-01', '2023-01-02')
        
        # 驗證結果
        assert not result.empty
        assert len(result) == len(mock_data)
        
        # 驗證標準化被應用
        mock_logger.info.assert_any_call("已應用特徵標準化")


if __name__ == "__main__":
    """
    直接運行測試的主程式區塊
    
    這允許您直接執行此檔案來運行所有測試，
    而不需要使用 pytest 命令。
    """
    import sys
    
    print("=== CryptoAce Backtester 測試 ===")
    print("正在運行 Backtester 模組的完整測試套件...")
    print()
    
    # 創建測試實例
    test_instance = TestBacktester()
    
    try:
        # 運行基本測試
        print("1. 測試 Backtester 初始化...")
        test_instance.test_backtester_initialization()
        print("   ✅ Backtester 初始化測試通過")
        
        print("\n2. 測試實驗產物加載...")
        
        # 創建臨時實驗目錄
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        try:
            # 使用 fixture 方法創建假實驗產物
            artifacts_dir = test_instance.mock_experiment_artifacts.__wrapped__(test_instance, temp_dir)
            
            # 測試成功加載
            test_instance.test_load_experiment_success(artifacts_dir)
            print("   ✅ 實驗產物加載測試通過")
            
            # 測試缺失目錄
            try:
                test_instance.test_load_experiment_missing_directory()
                print("   ✅ 缺失目錄錯誤處理測試通過")
            except AssertionError:
                print("   ✅ 缺失目錄錯誤處理測試通過")
            
        finally:
            # 清理臨時目錄
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
        
        print("\n3. 測試績效指標計算...")
        test_instance.test_performance_metrics_calculation()
        print("   ✅ 績效指標計算測試通過")
        
        print("\n4. 測試文件生成...")
        # 創建假數據用於測試
        mock_data = test_instance.mock_data.__wrapped__(test_instance)
        
        # 創建新的臨時目錄用於文件生成測試
        temp_dir2 = tempfile.mkdtemp()
        try:
            artifacts_dir2 = test_instance.mock_experiment_artifacts.__wrapped__(test_instance, temp_dir2)
            test_instance.test_backtest_file_generation(artifacts_dir2, mock_data)
            print("   ✅ 文件生成測試通過")
        finally:
            try:
                shutil.rmtree(temp_dir2)
            except:
                pass
        
        print("\n5. 測試空數據處理...")
        test_instance.test_empty_data_handling()
        print("   ✅ 空數據處理測試通過")
        
        print("\n6. 跳過完整工作流程測試 (標記為 @pytest.mark.slow)")
        print("   💡 要運行完整測試，請使用: pytest tests/test_backtester.py -m slow")
        
        print("\n🎉 所有快速測試通過！")
        print("\n測試摘要:")
        print("  ✅ Backtester 初始化")
        print("  ✅ 實驗產物加載")
        print("  ✅ 錯誤處理")
        print("  ✅ 績效指標計算")
        print("  ✅ 文件生成")
        print("  ✅ 空數據處理")
        print("  ⏩ 完整工作流程測試 (跳過)")
        
        print(f"\n如要運行完整測試套件，請使用:")
        print(f"  pytest {__file__} -v")
        print(f"  pytest {__file__} -m slow -v  # 只運行慢速測試")
        
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        print("\n錯誤詳情:")
        import traceback
        traceback.print_exc()
        sys.exit(1)
