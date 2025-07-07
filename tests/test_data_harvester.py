"""
CryptoAce DataHarvester 測試模組

此模組使用 pytest 和 mock 來測試 DataHarvester 的核心功能，
避免真實的網路請求，確保測試的穩定性和速度。
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, Mock
from pathlib import Path
import tempfile
import shutil
from datetime import datetime, timedelta
import sys
import os

# 添加專案根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_harvester import DataHarvester
from core.configurator import Configurator


class TestDataHarvester:
    """DataHarvester 測試類別"""
    
    @pytest.fixture
    def mock_config(self):
        """創建模擬配置對象"""
        config = MagicMock(spec=Configurator)
        
        # 配置交易所參數
        config.exchange = {
            'name': 'bitget',
            'default_symbol': 'BTC/USDT',
            'bitget_api_key': 'test_key',
            'bitget_api_secret': 'test_secret',
            'bitget_passphrase': 'test_passphrase'
        }
        
        # 配置數據參數
        config.data = {
            'timeframe': '5m',
            'start_date': '2023-01-01T00:00:00Z',
            'raw_data_path': './test_data/raw/',
            'feature_data_path': './test_data/features/'
        }
        
        return config
    
    @pytest.fixture
    def mock_logger(self):
        """創建模擬日誌對象"""
        logger = MagicMock()
        return logger
    
    @pytest.fixture
    def temp_directory(self):
        """創建臨時目錄用於測試"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # 清理臨時目錄
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_ohlcv_data(self):
        """創建模擬的OHLCV數據"""
        # 只生成10條簡單的測試數據，避免複雜的隨機生成
        start_time = datetime(2023, 1, 1)
        ohlcv_data = []
        
        base_price = 45000.0
        for i in range(10):
            timestamp = start_time + timedelta(minutes=5 * i)
            timestamp_ms = int(timestamp.timestamp() * 1000)
            
            # 簡單的價格變化
            price = base_price + (i * 10)  # 每次增加10
            
            # 生成OHLCV數據 [timestamp, open, high, low, close, volume]
            ohlcv_data.append([
                timestamp_ms,
                price,           # open
                price + 50,      # high
                price - 50,      # low
                price + 25,      # close
                100.0 + i        # volume
            ])
        
        return ohlcv_data
    
    @pytest.fixture
    def harvester_with_temp_paths(self, mock_config, mock_logger, temp_directory):
        """創建使用臨時路徑的DataHarvester實例"""
        # 更新配置以使用臨時目錄
        mock_config.data['raw_data_path'] = f"{temp_directory}/raw/"
        mock_config.data['feature_data_path'] = f"{temp_directory}/features/"
        
        with patch('core.data_harvester.ccxt.bitget') as mock_exchange_class:
            # 創建模擬交易所實例
            mock_exchange = MagicMock()
            mock_exchange_class.return_value = mock_exchange
            
            harvester = DataHarvester(mock_config, mock_logger)
            harvester.exchange = mock_exchange
            
            return harvester, mock_exchange
    
    def test_init_harvester(self, mock_config, mock_logger):
        """測試DataHarvester初始化"""
        with patch('core.data_harvester.ccxt.bitget') as mock_exchange_class:
            mock_exchange = MagicMock()
            mock_exchange_class.return_value = mock_exchange
            
            harvester = DataHarvester(mock_config, mock_logger)
            
            # 驗證初始化
            assert harvester.config == mock_config
            assert harvester.logger == mock_logger
            assert harvester.exchange_config == mock_config.exchange
            assert harvester.data_config == mock_config.data
            
            # 驗證目錄創建
            assert harvester.raw_data_path.exists()
            assert harvester.feature_data_path.exists()
            
            mock_logger.info.assert_called_with("DataHarvester 初始化完成")
    
    def test_fetch_ohlcv_data_success(self, harvester_with_temp_paths, mock_ohlcv_data):
        """測試成功獲取OHLCV數據"""
        harvester, mock_exchange = harvester_with_temp_paths
        
        # 設定模擬交易所返回數據，然後返回空列表表示結束
        mock_exchange.fetch_ohlcv.side_effect = [
            mock_ohlcv_data,  # 第一次調用返回數據
            []  # 第二次調用返回空列表，結束循環
        ]
        
        # 執行數據獲取
        result = harvester.fetch_ohlcv_data('BTC/USDT', '5m', '2023-01-01T00:00:00Z')
        
        # 驗證結果
        assert not result.empty
        assert len(result) == 10
        assert list(result.columns) == ['open', 'high', 'low', 'close', 'volume']
        assert result.index.name == 'timestamp'
        
        # 驗證數據類型
        for col in ['open', 'high', 'low', 'close', 'volume']:
            assert result[col].dtype in [np.float64, np.int64]
        
        # 驗證API調用
        assert mock_exchange.fetch_ohlcv.call_count >= 1
    
    def test_fetch_ohlcv_data_failure(self, harvester_with_temp_paths):
        """測試獲取OHLCV數據失敗的情況"""
        harvester, mock_exchange = harvester_with_temp_paths
        
        # 設定模擬交易所拋出異常
        mock_exchange.fetch_ohlcv.side_effect = Exception("網路錯誤")
        
        # 執行數據獲取
        result = harvester.fetch_ohlcv_data('BTC/USDT', '5m', '2023-01-01T00:00:00Z')
        
        # 驗證結果
        assert result.empty
        harvester.logger.error.assert_called_with("數據下載失敗: 網路錯誤")
    
    def test_validate_and_clean_data(self, harvester_with_temp_paths):
        """測試數據驗證和清洗功能"""
        harvester, _ = harvester_with_temp_paths
        
        # 創建包含問題的測試數據
        dirty_data = pd.DataFrame({
            'open': [100, 0, 105, 110, np.nan, 120],  # 包含0和NaN
            'high': [105, 102, 108, 115, 125, 125],
            'low': [98, 98, 103, 105, 115, 118],
            'close': [102, 101, 107, 112, 122, 123],
            'volume': [1000, 0, 1200, 1100, 1300, 1400]  # 包含0成交量
        }, index=pd.date_range('2023-01-01', periods=6, freq='5min'))
        
        # 執行清洗
        cleaned_data = harvester._validate_and_clean_data(dirty_data)
        
        # 驗證清洗結果
        assert len(cleaned_data) < len(dirty_data)  # 應該移除了一些數據
        assert cleaned_data['volume'].min() > 0  # 沒有零成交量
        assert not cleaned_data.isnull().any().any()  # 沒有NaN值
        
        # 驗證價格邏輯性
        assert (cleaned_data['high'] >= cleaned_data[['open', 'close']].max(axis=1)).all()
        assert (cleaned_data['low'] <= cleaned_data[['open', 'close']].min(axis=1)).all()
        assert (cleaned_data['high'] >= cleaned_data['low']).all()
    
    def test_validate_and_clean_data_empty(self, harvester_with_temp_paths):
        """測試空數據的清洗"""
        harvester, _ = harvester_with_temp_paths
        
        empty_data = pd.DataFrame()
        result = harvester._validate_and_clean_data(empty_data)
        
        assert result.empty
        harvester.logger.warning.assert_called_with("輸入數據為空")
    
    def test_save_raw_data(self, harvester_with_temp_paths):
        """測試保存原始數據"""
        harvester, _ = harvester_with_temp_paths
        
        # 創建測試數據
        test_data = pd.DataFrame({
            'open': [100, 102, 105],
            'high': [105, 107, 110],
            'low': [98, 100, 103],
            'close': [102, 105, 108],
            'volume': [1000, 1200, 1100]
        }, index=pd.date_range('2023-01-01', periods=3, freq='5min', name='timestamp'))
        
        # 執行保存
        filepath = harvester.save_raw_data(test_data, 'BTC/USDT', '5m')
        
        # 驗證文件存在
        assert Path(filepath).exists()
        assert 'BTC_USDT_5m_raw.parquet' in filepath
        
        # 驗證文件內容
        loaded_data = pd.read_parquet(filepath)
        
        # 比較數據值和結構
        assert loaded_data.shape == test_data.shape
        assert list(loaded_data.columns) == list(test_data.columns)
        
        # 驗證數值內容（逐列比較，避免索引問題）
        for col in test_data.columns:
            assert loaded_data[col].equals(test_data[col]), f"列 {col} 的數據不匹配"
        
        # 驗證索引值（忽略索引名稱）
        assert loaded_data.index.equals(test_data.index), "索引數據不匹配"
    
    @patch('core.data_harvester.FeatureEngine')
    @patch('joblib.dump')
    def test_run_collection_success(self, mock_joblib_dump, mock_feature_engine_class, 
                                  harvester_with_temp_paths, mock_ohlcv_data):
        """測試完整的數據收集流程成功執行"""
        harvester, mock_exchange = harvester_with_temp_paths
        
        # 設定模擬交易所返回數據（確保測試結束）
        mock_exchange.fetch_ohlcv.side_effect = [
            mock_ohlcv_data,  # 第一次調用返回數據
            []  # 第二次調用返回空列表，結束循環
        ]
        
        # 設定模擬特徵引擎
        mock_feature_engine = MagicMock()
        mock_feature_engine_class.return_value = mock_feature_engine
        
        # 創建模擬特徵數據（對應10條原始數據）
        features_df = pd.DataFrame({
            'feature1': list(range(10)),  # 簡單的特徵數據
            'feature2': list(range(10, 20)),
            'feature3': list(range(20, 30))
        })
        mock_scaler = MagicMock()
        mock_feature_engine.fit_transform.return_value = (features_df, mock_scaler)
        
        # 執行數據收集
        harvester.run_collection()
        
        # 驗證API調用
        assert mock_exchange.fetch_ohlcv.call_count >= 1
        
        # 驗證特徵引擎調用
        mock_feature_engine_class.assert_called_once_with(harvester.config, harvester.logger)
        mock_feature_engine.fit_transform.assert_called_once()
        
        # 驗證文件保存
        raw_files = list(harvester.raw_data_path.glob("*.parquet"))
        feature_files = list(harvester.feature_data_path.glob("*.parquet"))
        
        assert len(raw_files) == 1
        assert len(feature_files) == 1
        assert 'BTC_USDT_5m_raw.parquet' in str(raw_files[0])
        assert 'BTC_USDT_5m_features.parquet' in str(feature_files[0])
        
        # 驗證Scaler保存
        mock_joblib_dump.assert_called_once()
        
        # 驗證日誌記錄
        harvester.logger.info.assert_any_call("開始數據收集流程: BTC/USDT 5m")
        harvester.logger.info.assert_any_call("開始特徵工程處理")
    
    def test_run_collection_no_data(self, harvester_with_temp_paths):
        """測試無法獲取數據的情況"""
        harvester, mock_exchange = harvester_with_temp_paths
        
        # 設定模擬交易所返回空數據
        mock_exchange.fetch_ohlcv.return_value = []
        
        # 執行數據收集
        harvester.run_collection()
        
        # 驗證錯誤處理
        harvester.logger.error.assert_called_with("未能獲取任何數據，流程終止")
        
        # 驗證沒有文件被創建
        raw_files = list(harvester.raw_data_path.glob("*.parquet"))
        assert len(raw_files) == 0
    
    @patch('core.data_harvester.FeatureEngine')
    def test_run_collection_feature_engine_failure(self, mock_feature_engine_class,
                                                  harvester_with_temp_paths, mock_ohlcv_data):
        """測試特徵引擎失敗的情況"""
        harvester, mock_exchange = harvester_with_temp_paths
        
        # 設定模擬交易所返回數據
        mock_exchange.fetch_ohlcv.side_effect = [
            mock_ohlcv_data,  # 第一次調用返回數據
            []  # 第二次調用返回空列表，結束循環
        ]
        
        # 設定特徵引擎拋出異常
        mock_feature_engine = MagicMock()
        mock_feature_engine_class.return_value = mock_feature_engine
        mock_feature_engine.fit_transform.side_effect = Exception("特徵引擎錯誤")
        
        # 執行數據收集
        harvester.run_collection()
        
        # 驗證原始數據仍然被保存
        raw_files = list(harvester.raw_data_path.glob("*.parquet"))
        assert len(raw_files) == 1
        
        # 驗證錯誤日誌
        harvester.logger.error.assert_any_call("特徵工程處理失敗: 特徵引擎錯誤")
        harvester.logger.info.assert_any_call("已保存原始數據，可稍後重新處理特徵")
    
    def test_exchange_initialization_fallback(self, mock_config, mock_logger):
        """測試交易所初始化失敗時的降級處理"""
        mock_config.exchange['bitget_api_key'] = None  # 無效的API金鑰
        
        with patch('core.data_harvester.ccxt.bitget') as mock_exchange_class:
            # 第一次調用失敗（有API金鑰但無效）
            # 第二次調用成功（無API金鑰的公開連接）
            mock_exchange_class.side_effect = [Exception("API錯誤"), MagicMock()]
            
            harvester = DataHarvester(mock_config, mock_logger)
            
            # 驗證降級處理
            assert mock_exchange_class.call_count == 2
            mock_logger.error.assert_called_with("交易所連接失敗: API錯誤")
            mock_logger.warning.assert_called_with("使用公開API連接（無需認證）")
    
    def test_run_collection_basic_success(self, harvester_with_temp_paths):
        """
        基本的數據收集成功測試 - 滿足用戶的核心要求:
        1. 使用模擬避免真實網路請求
        2. 測試 run_collection() 方法
        3. 檢查數據文件是否被創建
        """
        harvester, mock_exchange = harvester_with_temp_paths
        
        # 創建簡單的模擬OHLCV數據
        simple_ohlcv_data = [
            [1672531200000, 45000.0, 45100.0, 44900.0, 45050.0, 100.0],  # 2023-01-01 00:00
            [1672531500000, 45050.0, 45150.0, 44950.0, 45100.0, 120.0],  # 2023-01-01 00:05
            [1672531800000, 45100.0, 45200.0, 45000.0, 45150.0, 110.0],  # 2023-01-01 00:10
        ]
        
        # 模擬交易所返回預定義的數據
        mock_exchange.fetch_ohlcv.return_value = simple_ohlcv_data
        
        # 使用patch來模擬特徵引擎，避免真實的特徵處理
        with patch('core.data_harvester.FeatureEngine') as mock_feature_engine_class:
            mock_feature_engine = MagicMock()
            mock_feature_engine_class.return_value = mock_feature_engine
            
            # 模擬特徵引擎返回簡單的特徵數據
            mock_features = pd.DataFrame({
                'sma_5': [45000.0, 45050.0, 45100.0],
                'rsi': [50.0, 52.0, 55.0]
            })
            mock_scaler = MagicMock()
            mock_feature_engine.fit_transform.return_value = (mock_features, mock_scaler)
            
            # 執行數據收集
            harvester.run_collection()
        
        # 檢查預期的數據文件是否被創建
        raw_files = list(harvester.raw_data_path.glob("*.parquet"))
        feature_files = list(harvester.feature_data_path.glob("*.parquet"))
        
        # 使用assert語句檢查文件創建
        assert len(raw_files) == 1, "應該創建一個原始數據文件"
        assert len(feature_files) == 1, "應該創建一個特徵數據文件"
        assert 'BTC_USDT_5m_raw.parquet' in str(raw_files[0]), "原始數據文件名應該正確"
        assert 'BTC_USDT_5m_features.parquet' in str(feature_files[0]), "特徵數據文件名應該正確"
        
        # 驗證模擬的交易所被正確調用
        mock_exchange.fetch_ohlcv.assert_called()
        
        # 驗證日誌記錄
        harvester.logger.info.assert_any_call("開始數據收集流程: BTC/USDT 5m")


if __name__ == "__main__":
    """執行測試"""
    pytest.main([__file__, "-v", "--tb=short"])
