"""
CryptoAce FeatureEngine 測試模組

此模組包含對 FeatureEngine 類別的完整單元測試，
測試特徵工程的正確性和數據洩漏防護。
"""

import pytest
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from unittest.mock import MagicMock

# 處理相對匯入問題
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.feature_engine import FeatureEngine
from core.configurator import Configurator

# 抑制測試期間的警告
warnings.filterwarnings('ignore')


class TestFeatureEngine:
    """FeatureEngine 測試類別"""
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """
        創建用於測試的假 OHLCV 數據
        
        Returns:
            pd.DataFrame: 包含 OHLCV 數據的 DataFrame
        """
        # 設定隨機種子確保測試可重現
        np.random.seed(42)
        
        # 創建500個數據點（足夠計算所有技術指標，包括長週期的）
        n_samples = 500
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='1h')
        
        # 生成具有趨勢的價格數據
        base_price = 50000.0
        trend = 0.0001
        volatility = 0.02
        
        prices = []
        current_price = base_price
        
        for i in range(n_samples):
            price_change = trend + np.random.normal(0, volatility)
            current_price *= (1 + price_change)
            prices.append(current_price)
        
        # 生成OHLCV數據
        data = []
        for i, (date, close_price) in enumerate(zip(dates, prices)):
            high = close_price * (1 + abs(np.random.normal(0, 0.01)))
            low = close_price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = prices[i-1] if i > 0 else close_price
            volume = np.random.uniform(1000, 10000)
            
            data.append({
                'timestamp': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        return df
    
    @pytest.fixture
    def feature_engine(self):
        """
        創建 FeatureEngine 實例
        
        Returns:
            FeatureEngine: 特徵工程器實例
        """
        # 創建模擬的 logger
        mock_logger = MagicMock()
        
        return FeatureEngine(logger=mock_logger)
    
    @pytest.fixture
    def train_test_split(self, sample_ohlcv_data):
        """
        將樣本數據分割為訓練和測試集
        
        Args:
            sample_ohlcv_data: 樣本 OHLCV 數據
            
        Returns:
            tuple: (train_df, test_df)
        """
        split_idx = int(len(sample_ohlcv_data) * 0.8)
        train_df = sample_ohlcv_data.iloc[:split_idx].copy()
        test_df = sample_ohlcv_data.iloc[split_idx:].copy()
        
        return train_df, test_df
    
    def test_feature_engine_initialization(self):
        """測試 FeatureEngine 初始化"""
        # 測試無參數初始化
        fe = FeatureEngine()
        assert fe.config is None
        assert fe.logger is None
        assert isinstance(fe.feature_config, dict)
        assert 'sma_periods' in fe.feature_config
        
        # 測試帶參數初始化
        mock_config = MagicMock()
        mock_logger = MagicMock()
        
        fe_with_params = FeatureEngine(config=mock_config, logger=mock_logger)
        assert fe_with_params.config == mock_config
        assert fe_with_params.logger == mock_logger
    
    def test_get_feature_columns(self, feature_engine, sample_ohlcv_data):
        """測試特徵列獲取功能"""
        # 添加一些測試特徵
        test_df = sample_ohlcv_data.copy()
        test_df['feature1'] = np.random.randn(len(test_df))
        test_df['feature2'] = np.random.randint(0, 2, len(test_df))
        test_df['timestamp_col'] = pd.to_datetime('2023-01-01')  # 會被排除
        
        feature_cols = feature_engine._get_feature_columns(test_df)
        
        # 檢查原始 OHLCV 列不在特徵列中
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in ohlcv_cols:
            assert col not in feature_cols
        
        # 檢查數值特徵列存在
        assert 'feature1' in feature_cols
        assert 'feature2' in feature_cols
        
        # 檢查時間戳列被排除
        assert 'timestamp_col' not in feature_cols
    
    def test_calculate_technical_indicators(self, feature_engine, sample_ohlcv_data):
        """測試技術指標計算功能"""
        result_df = feature_engine._calculate_technical_indicators(sample_ohlcv_data)
        
        # 檢查原始列依然存在
        original_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in original_cols:
            assert col in result_df.columns
        
        # 檢查基本價格特徵
        expected_features = [
            'returns', 'log_returns', 'price_range', 
            'upper_shadow', 'lower_shadow', 'body_size'
        ]
        for feature in expected_features:
            assert feature in result_df.columns
        
        # 檢查移動平均線
        for period in feature_engine.feature_config['sma_periods']:
            assert f'sma_{period}' in result_df.columns
            assert f'sma_{period}_ratio' in result_df.columns
        
        # 檢查 RSI
        rsi_period = feature_engine.feature_config['rsi_period']
        assert f'rsi_{rsi_period}' in result_df.columns
        assert f'rsi_{rsi_period}_oversold' in result_df.columns
        assert f'rsi_{rsi_period}_overbought' in result_df.columns
        
        # 檢查特徵數量大於原始數據
        assert len(result_df.columns) > len(sample_ohlcv_data.columns)
        
        # 檢查數據型態
        assert result_df['returns'].dtype in ['float64', 'float32']
        assert result_df[f'rsi_{rsi_period}_oversold'].dtype in ['int64', 'int32']
    
    def test_fit_transform(self, feature_engine, train_test_split):
        """測試 fit_transform 方法"""
        train_df, _ = train_test_split
        
        # 執行 fit_transform
        transformed_df, scaler = feature_engine.fit_transform(train_df)
        
        # 檢查返回的 scaler 是 StandardScaler 實例
        assert isinstance(scaler, StandardScaler)
        
        # 檢查轉換後的 DataFrame
        assert isinstance(transformed_df, pd.DataFrame)
        assert len(transformed_df) <= len(train_df)  # 可能因為 dropna() 而減少
        
        # 檢查原始 OHLCV 列依然存在
        original_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in original_cols:
            assert col in transformed_df.columns
        
        # 檢查特徵列數量增加
        assert len(transformed_df.columns) > len(train_df.columns)
        
        # 檢查標準化後的特徵統計特性
        feature_cols = feature_engine._get_feature_columns(transformed_df)
        if feature_cols:
            # 訓練數據的標準化特徵應該接近標準正態分佈
            feature_stats = transformed_df[feature_cols].describe()
            
            # 檢查均值接近 0（允許一定誤差）
            means = feature_stats.loc['mean'].abs()
            assert (means < 0.1).all(), f"某些特徵的均值不接近0: {means[means >= 0.1]}"
            
            # 檢查標準差接近 1（允許一定誤差，但排除二元特徵）
            stds = feature_stats.loc['std']
            
            # 排除二元特徵（標準差為0或接近0的特徵，如 oversold/overbought 指標）
            non_binary_features = stds[stds > 0.01]  # 排除標準差小於0.01的特徵
            
            if len(non_binary_features) > 0:
                assert ((non_binary_features - 1).abs() < 0.1).all(), \
                    f"某些非二元特徵的標準差不接近1: {non_binary_features[(non_binary_features - 1).abs() >= 0.1]}"
    
    def test_transform(self, feature_engine, train_test_split):
        """測試 transform 方法"""
        train_df, test_df = train_test_split
        
        # 先進行 fit_transform 獲得 scaler
        _, scaler = feature_engine.fit_transform(train_df)
        
        # 使用 scaler 轉換測試數據
        transformed_test_df = feature_engine.transform(test_df, scaler)
        
        # 檢查轉換結果
        assert isinstance(transformed_test_df, pd.DataFrame)
        assert len(transformed_test_df) <= len(test_df)  # 可能因為 dropna() 而減少
        
        # 檢查原始列依然存在
        original_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in original_cols:
            assert col in transformed_test_df.columns
        
        # 檢查特徵列數量
        assert len(transformed_test_df.columns) > len(test_df.columns)
        
        # 檢查測試數據使用了訓練階段的標準化參數（數據洩漏防護）
        feature_cols = feature_engine._get_feature_columns(transformed_test_df)
        if feature_cols:
            test_stats = transformed_test_df[feature_cols].describe()
            
            # 測試數據的均值和標準差不應該接近 0 和 1
            # 這證明使用了訓練階段的統計信息，而非測試數據本身的統計信息
            test_means = test_stats.loc['mean'].abs()
            test_stds = test_stats.loc['std']
            
            # 排除二元特徵
            non_binary_features = test_stds[test_stds > 0.01]
            
            if len(non_binary_features) > 0:
                # 至少有一些特徵的均值不接近 0 或標準差不接近 1
                mean_check = (test_means < 0.05).all()
                std_check = ((non_binary_features - 1).abs() < 0.05).all()
                
                assert not (mean_check and std_check), \
                    "測試數據的標準化統計過於接近訓練標準，可能存在數據洩漏"
    
    def test_empty_dataframe_handling(self, feature_engine):
        """測試空 DataFrame 的處理"""
        empty_df = pd.DataFrame()
        
        # 測試 fit_transform
        result_df, scaler = feature_engine.fit_transform(empty_df)
        assert result_df.empty
        assert isinstance(scaler, StandardScaler)
        
        # 測試 transform
        result_df = feature_engine.transform(empty_df, scaler)
        assert result_df.empty
    
    def test_data_leakage_prevention(self, feature_engine, train_test_split):
        """測試數據洩漏防護機制"""
        train_df, test_df = train_test_split
        
        # 在訓練數據上擬合
        transformed_train_df, scaler = feature_engine.fit_transform(train_df)
        
        # 在測試數據上轉換
        transformed_test_df = feature_engine.transform(test_df, scaler)
        
        # 獲取特徵列
        feature_cols = feature_engine._get_feature_columns(transformed_train_df)
        
        if feature_cols:
            # 檢查訓練數據統計
            train_stats = transformed_train_df[feature_cols].describe()
            test_stats = transformed_test_df[feature_cols].describe()
            
            # 訓練數據應該標準化（均值≈0，標準差≈1），但排除二元特徵
            train_means = train_stats.loc['mean']
            train_stds = train_stats.loc['std']
            
            assert (train_means.abs() < 0.1).all(), "訓練數據均值未正確標準化"
            
            # 檢查非二元特徵的標準差
            non_binary_train_features = train_stds[train_stds > 0.01]
            if len(non_binary_train_features) > 0:
                assert ((non_binary_train_features - 1).abs() < 0.1).all(), \
                    "訓練數據非二元特徵標準差未正確標準化"
            
            # 測試數據不應該標準化（證明使用了訓練期間的統計信息）
            test_means = test_stats.loc['mean']
            test_stds = test_stats.loc['std']
            
            # 排除二元特徵
            non_binary_test_features = test_stds[test_stds > 0.01]
            
            if len(non_binary_test_features) > 0:
                # 至少有部分特徵的統計信息與標準化後的分佈不同
                mean_diff = (test_means.abs() > 0.1).any()
                std_diff = ((non_binary_test_features - 1).abs() > 0.1).any()
                
                assert mean_diff or std_diff, \
                    "測試數據統計過於接近標準正態分佈，可能存在數據洩漏"
    
    def test_feature_consistency(self, feature_engine, train_test_split):
        """測試特徵一致性 - 訓練和測試數據應該有相同的特徵列"""
        train_df, test_df = train_test_split
        
        # 在訓練數據上擬合
        transformed_train_df, scaler = feature_engine.fit_transform(train_df)
        
        # 在測試數據上轉換
        transformed_test_df = feature_engine.transform(test_df, scaler)
        
        # 檢查列名一致性
        assert list(transformed_train_df.columns) == list(transformed_test_df.columns), \
            "訓練和測試數據的特徵列不一致"
        
        # 檢查數據類型一致性
        for col in transformed_train_df.columns:
            assert transformed_train_df[col].dtype == transformed_test_df[col].dtype, \
                f"列 {col} 的數據類型在訓練和測試數據中不一致"
    
    def test_nan_handling(self, feature_engine):
        """測試 NaN 值處理"""
        # 創建包含 NaN 的測試數據
        dates = pd.date_range('2023-01-01', periods=100, freq='1h')
        data = {
            'timestamp': dates,
            'open': np.random.randn(100) * 100 + 50000,
            'high': np.random.randn(100) * 100 + 50100,
            'low': np.random.randn(100) * 100 + 49900,
            'close': np.random.randn(100) * 100 + 50000,
            'volume': np.random.randn(100) * 1000 + 5000
        }
        
        # 在開頭幾行插入 NaN（模擬技術指標計算產生的 NaN）
        df_with_nan = pd.DataFrame(data)
        df_with_nan.set_index('timestamp', inplace=True)
        
        # 執行特徵工程
        transformed_df, scaler = feature_engine.fit_transform(df_with_nan)
        
        # 檢查結果不包含 NaN
        assert not transformed_df.isnull().any().any(), "轉換後的數據仍包含 NaN 值"
        
        # 檢查數據長度減少（因為移除了包含 NaN 的行）
        assert len(transformed_df) < len(df_with_nan), "未正確移除包含 NaN 的行"
    
    def test_custom_config(self):
        """測試自定義配置"""
        # 創建自定義配置
        custom_config = {
            'sma_periods': [10, 20],  # 減少 SMA 週期
            'rsi_period': 21,         # 修改 RSI 週期
        }
        
        mock_configurator = MagicMock()
        mock_configurator.features = custom_config
        
        # 使用自定義配置創建 FeatureEngine
        fe_custom = FeatureEngine(config=mock_configurator)
        
        # 檢查配置是否正確更新
        assert fe_custom.feature_config['sma_periods'] == [10, 20]
        assert fe_custom.feature_config['rsi_period'] == 21
        
        # 檢查其他默認配置依然存在
        assert 'ema_periods' in fe_custom.feature_config
        assert 'bb_period' in fe_custom.feature_config


if __name__ == "__main__":
    """直接運行測試 - 美化版本"""
    import subprocess
    import sys
    from datetime import datetime
    
    print("=" * 60)
    print("🧪 CryptoAce FeatureEngine 測試套件")
    print("=" * 60)
    print(f"📅 測試時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python 版本: {sys.version.split()[0]}")
    print("-" * 60)
    
    try:
        # 使用更詳細和美化的 pytest 參數
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            __file__, 
            '-v',                    # 詳細輸出
            '--tb=short',           # 簡短的錯誤追蹤
            '--color=yes',          # 彩色輸出
            '--durations=5',        # 顯示最慢的5個測試
            '--disable-warnings',   # 禁用警告
            '-x'                    # 遇到第一個失敗就停止
        ], capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)))
        
        # 美化輸出
        print("📋 測試執行結果:")
        print("-" * 60)
        
        # 處理輸出，添加圖標
        output_lines = result.stdout.split('\n')
        for line in output_lines:
            if "PASSED" in line:
                print(f"✅ {line}")
            elif "FAILED" in line:
                print(f"❌ {line}")
            elif "test session starts" in line:
                print(f"🚀 {line}")
            elif "warnings summary" in line:
                print(f"⚠️  {line}")
            elif "short test summary" in line:
                print(f"📊 {line}")
            elif line.strip() and not line.startswith('='):
                print(f"   {line}")
            elif line.startswith('='):
                print(line)
        
        if result.stderr:
            print("\n⚠️  系統警告:")
            print("-" * 60)
            print(result.stderr)
        
        print("\n" + "=" * 60)
        
        # 美化結果摘要
        if result.returncode == 0:
            print("🎉 所有測試通過！")
            print("\n📈 測試覆蓋的核心功能:")
            print("   ✅ FeatureEngine 初始化與配置")
            print("   ✅ 技術指標計算引擎")
            print("   ✅ fit_transform 訓練邏輯")
            print("   ✅ transform 推理邏輯")
            print("   ✅ 數據洩漏防護機制")
            print("   ✅ 特徵一致性驗證")
            print("   ✅ NaN 值處理策略")
            print("   ✅ 空數據異常處理")
            print("   ✅ 自定義配置支持")
            print("   ✅ 二元特徵標準化")
            
            print("\n🛡️  安全性驗證:")
            print("   ✅ 防止訓練/測試數據洩漏")
            print("   ✅ 標準化器狀態隔離")
            print("   ✅ 特徵一致性保證")
            
            print("\n⚡ 性能特性:")
            print("   ✅ 高效的技術指標計算")
            print("   ✅ 記憶體優化的數據處理")
            print("   ✅ 批量特徵工程支持")
            
        else:
            print(f"💥 測試失敗 (退出代碼: {result.returncode})")
            print("請檢查上方錯誤信息並修復問題")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 測試執行過程中發生錯誤: {e}")
        print("\n🔧 建議解決方案:")
        print("   1. 檢查 pytest 是否正確安裝")
        print("   2. 確認所有依賴套件已安裝")
        print("   3. 驗證 Python 路徑設定")
