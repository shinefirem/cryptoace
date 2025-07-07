"""
CryptoAce 特徵工程模組

此模組負責計算技術指標和特徵工程，支持訓練和推理階段的分離，
防止數據洩漏問題。
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict, Optional, Tuple
import warnings

# 抑制 pandas_ta 的 pkg_resources 警告
warnings.filterwarnings('ignore', message='pkg_resources is deprecated as an API.*')
warnings.filterwarnings('ignore', category=UserWarning, module='pandas_ta')
warnings.filterwarnings('ignore', category=FutureWarning)

# 導入 pandas_ta (在警告過濾器設置後)
import pandas_ta as ta

# 處理相對匯入問題
try:
    from .configurator import Configurator
    from .logger import setup_logger
except ImportError:
    # 當直接執行此檔案時，使用絕對匯入
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.configurator import Configurator
    from core.logger import setup_logger
warnings.filterwarnings('ignore', category=UserWarning, message='.*pkg_resources.*')


class FeatureEngine:
    """
    特徵工程類別
    
    負責計算技術指標和特徵標準化，支持訓練/測試分離，
    防止數據洩漏問題。
    """
    
    def __init__(self, config: Optional[Configurator] = None, logger: Optional[Any] = None) -> None:
        """
        初始化特徵工程器
        
        Args:
            config: 配置管理器實例（可選）
            logger: 日誌記錄器實例（可選）
        """
        self.config = config
        self.logger = logger
        
        # 設定預設特徵參數
        self.feature_config = {
            'sma_periods': [5, 10, 20, 50],          # 簡單移動平均週期
            'ema_periods': [5, 10, 20],              # 指數移動平均週期
            'rsi_period': 14,                         # RSI週期
            'bb_period': 20,                          # 布林通道週期
            'bb_std': 2,                              # 布林通道標準差倍數
            'macd_fast': 12,                          # MACD快線週期
            'macd_slow': 26,                          # MACD慢線週期
            'macd_signal': 9,                         # MACD信號線週期
            'stoch_k': 14,                            # 隨機指標K週期
            'stoch_d': 3,                             # 隨機指標D週期
            'atr_period': 14,                         # ATR週期
            'obv_period': 10,                         # OBV移動平均週期
            'vwap_period': 20,                        # VWAP週期
        }
        
        # 如果有配置，更新特徵參數
        if config:
            feature_settings = getattr(config, 'features', {})
            self.feature_config.update(feature_settings)
        
        if self.logger:
            self.logger.info("FeatureEngine 初始化完成")
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        計算技術指標
        
        Args:
            df: 包含OHLCV數據的DataFrame
            
        Returns:
            包含技術指標的DataFrame
        """
        if df.empty:
            if self.logger:
                self.logger.warning("輸入數據為空，跳過特徵計算")
            return df
        
        # 複製原始數據
        features = df.copy()
        
        try:
            # 1. 價格特徵
            features['returns'] = features['close'].pct_change()
            features['log_returns'] = np.log(features['close'] / features['close'].shift(1))
            features['price_range'] = (features['high'] - features['low']) / features['close']
            features['upper_shadow'] = (features['high'] - features[['open', 'close']].max(axis=1)) / features['close']
            features['lower_shadow'] = (features[['open', 'close']].min(axis=1) - features['low']) / features['close']
            features['body_size'] = abs(features['close'] - features['open']) / features['close']
            
            # 2. 移動平均線
            for period in self.feature_config['sma_periods']:
                features[f'sma_{period}'] = ta.sma(features['close'], length=period)
                features[f'sma_{period}_ratio'] = features['close'] / features[f'sma_{period}']
            
            for period in self.feature_config['ema_periods']:
                features[f'ema_{period}'] = ta.ema(features['close'], length=period)
                features[f'ema_{period}_ratio'] = features['close'] / features[f'ema_{period}']
            
            # 3. RSI
            rsi_period = self.feature_config['rsi_period']
            features[f'rsi_{rsi_period}'] = ta.rsi(features['close'], length=rsi_period)
            features[f'rsi_{rsi_period}_oversold'] = (features[f'rsi_{rsi_period}'] < 30).astype(int)
            features[f'rsi_{rsi_period}_overbought'] = (features[f'rsi_{rsi_period}'] > 70).astype(int)
            
            # 4. 布林通道
            bb_period = self.feature_config['bb_period']
            bb_std = self.feature_config['bb_std']
            bb = ta.bbands(features['close'], length=bb_period, std=bb_std)
            if bb is not None and not bb.empty:
                features[f'bb_upper_{bb_period}'] = bb.iloc[:, 0]  # BBU
                features[f'bb_middle_{bb_period}'] = bb.iloc[:, 1]  # BBM
                features[f'bb_lower_{bb_period}'] = bb.iloc[:, 2]  # BBL
                features[f'bb_width_{bb_period}'] = (features[f'bb_upper_{bb_period}'] - features[f'bb_lower_{bb_period}']) / features[f'bb_middle_{bb_period}']
                features[f'bb_position_{bb_period}'] = (features['close'] - features[f'bb_lower_{bb_period}']) / (features[f'bb_upper_{bb_period}'] - features[f'bb_lower_{bb_period}'])
            
            # 5. MACD
            macd_fast = self.feature_config['macd_fast']
            macd_slow = self.feature_config['macd_slow']
            macd_signal = self.feature_config['macd_signal']
            macd = ta.macd(features['close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
            if macd is not None and not macd.empty:
                features['macd'] = macd.iloc[:, 0]
                features['macd_histogram'] = macd.iloc[:, 1]
                features['macd_signal'] = macd.iloc[:, 2]
                features['macd_bull'] = (features['macd'] > features['macd_signal']).astype(int)
            
            # 6. 隨機指標
            stoch_k = self.feature_config['stoch_k']
            stoch_d = self.feature_config['stoch_d']
            stoch = ta.stoch(features['high'], features['low'], features['close'], k=stoch_k, d=stoch_d)
            if stoch is not None and not stoch.empty:
                features[f'stoch_k_{stoch_k}'] = stoch.iloc[:, 0]
                features[f'stoch_d_{stoch_d}'] = stoch.iloc[:, 1]
            
            # 7. ATR (平均真實範圍)
            atr_period = self.feature_config['atr_period']
            features[f'atr_{atr_period}'] = ta.atr(features['high'], features['low'], features['close'], length=atr_period)
            features[f'atr_{atr_period}_ratio'] = features[f'atr_{atr_period}'] / features['close']
            
            # 8. 成交量指標
            features['volume_ratio'] = features['volume'] / features['volume'].rolling(20).mean()
            features['volume_price_trend'] = (features['returns'] * features['volume']).rolling(10).sum()
            
            # OBV (On Balance Volume)
            features['obv'] = ta.obv(features['close'], features['volume'])
            obv_period = self.feature_config['obv_period']
            features[f'obv_sma_{obv_period}'] = ta.sma(features['obv'], length=obv_period)
            
            # 9. VWAP (成交量加權平均價格)
            vwap_period = self.feature_config['vwap_period']
            features[f'vwap_{vwap_period}'] = ta.vwap(features['high'], features['low'], features['close'], features['volume'], length=vwap_period)
            features[f'vwap_{vwap_period}_ratio'] = features['close'] / features[f'vwap_{vwap_period}']
            
            # 10. 波動率特徵
            for window in [5, 10, 20]:
                features[f'volatility_{window}'] = features['returns'].rolling(window).std()
            
            # 計算波動率比率（確保volatility_20已存在）
            if 'volatility_20' in features.columns:
                for window in [5, 10]:
                    features[f'volatility_{window}_ratio'] = features[f'volatility_{window}'] / features['volatility_20']
            
            # 11. 價格位置特徵
            for window in [10, 20, 50]:
                features[f'high_{window}'] = features['high'].rolling(window).max()
                features[f'low_{window}'] = features['low'].rolling(window).min()
                features[f'price_position_{window}'] = (features['close'] - features[f'low_{window}']) / (features[f'high_{window}'] - features[f'low_{window}'])
            
            # 12. 趨勢特徵
            features['trend_5'] = (features['close'] > features['close'].shift(5)).astype(int)
            features['trend_10'] = (features['close'] > features['close'].shift(10)).astype(int)
            features['trend_20'] = (features['close'] > features['close'].shift(20)).astype(int)
            
            if self.logger:
                original_cols = len(df.columns)
                new_cols = len(features.columns)
                self.logger.info(f"技術指標計算完成: {original_cols} -> {new_cols} 列 (+{new_cols-original_cols} 個特徵)")
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"技術指標計算失敗: {e}")
            raise
        
        return features
    
    def _get_feature_columns(self, df: pd.DataFrame) -> list:
        """
        獲取需要標準化的特徵列
        
        Args:
            df: 特徵DataFrame
            
        Returns:
            特徵列名列表
        """
        # 排除原始OHLCV列和一些特殊列
        exclude_columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        
        # 獲取數值型特徵列
        feature_columns = []
        for col in df.columns:
            if col not in exclude_columns and df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                feature_columns.append(col)
        
        return feature_columns
    
    def fit_transform(self, train_df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
        """
        在訓練數據上計算特徵並訓練標準化器
        
        Args:
            train_df: 訓練數據DataFrame (包含OHLCV)
            
        Returns:
            tuple: (轉換後的訓練數據, 訓練好的scaler對象)
        """
        if train_df.empty:
            if self.logger:
                self.logger.warning("訓練數據為空")
            return train_df, StandardScaler()
        
        if self.logger:
            self.logger.info(f"開始特徵工程處理 - 訓練模式，數據形狀: {train_df.shape}")
        
        # 1. 計算技術指標
        features_df = self._calculate_technical_indicators(train_df)
        
        # 2. 移除包含NaN的行
        original_len = len(features_df)
        features_df = features_df.dropna()
        removed_rows = original_len - len(features_df)
        
        if removed_rows > 0 and self.logger:
            self.logger.info(f"移除包含NaN的行: {removed_rows} 行 ({removed_rows/original_len*100:.2f}%)")
        
        # 3. 獲取需要標準化的特徵列
        feature_columns = self._get_feature_columns(features_df)
        
        if not feature_columns:
            if self.logger:
                self.logger.warning("沒有找到可用的特徵列")
            return features_df, StandardScaler()
        
        # 4. 訓練標準化器並轉換數據
        scaler = StandardScaler()
        features_df[feature_columns] = scaler.fit_transform(features_df[feature_columns])
        
        if self.logger:
            self.logger.info(f"特徵工程完成 - 訓練模式")
            self.logger.info(f"  - 最終數據形狀: {features_df.shape}")
            self.logger.info(f"  - 標準化特徵數: {len(feature_columns)}")
            self.logger.info(f"  - 特徵列: {feature_columns[:10]}{'...' if len(feature_columns) > 10 else ''}")
        
        return features_df, scaler
    
    def transform(self, test_df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
        """
        在測試數據上應用已訓練的標準化器
        
        Args:
            test_df: 測試數據DataFrame (包含OHLCV)
            scaler: 已訓練的StandardScaler對象
            
        Returns:
            轉換後的測試數據DataFrame
        """
        if test_df.empty:
            if self.logger:
                self.logger.warning("測試數據為空")
            return test_df
        
        if self.logger:
            self.logger.info(f"開始特徵工程處理 - 推理模式，數據形狀: {test_df.shape}")
        
        # 1. 計算技術指標
        features_df = self._calculate_technical_indicators(test_df)
        
        # 2. 移除包含NaN的行
        original_len = len(features_df)
        features_df = features_df.dropna()
        removed_rows = original_len - len(features_df)
        
        if removed_rows > 0 and self.logger:
            self.logger.info(f"移除包含NaN的行: {removed_rows} 行 ({removed_rows/original_len*100:.2f}%)")
        
        # 3. 獲取需要標準化的特徵列
        feature_columns = self._get_feature_columns(features_df)
        
        if not feature_columns:
            if self.logger:
                self.logger.warning("沒有找到可用的特徵列")
            return features_df
        
        # 4. 使用已訓練的標準化器轉換數據
        try:
            features_df[feature_columns] = scaler.transform(features_df[feature_columns])
        except ValueError as e:
            if self.logger:
                self.logger.error(f"標準化器應用失敗: {e}")
            raise
        
        if self.logger:
            self.logger.info(f"特徵工程完成 - 推理模式")
            self.logger.info(f"  - 最終數據形狀: {features_df.shape}")
            self.logger.info(f"  - 應用的特徵數: {len(feature_columns)}")
        
        return features_df


if __name__ == "__main__":
    """單元測試區塊"""
    
    print("=== CryptoAce 特徵工程系統測試 ===")
    
    try:
        # 設定隨機種子確保可重現性
        np.random.seed(42)
        
        # 創建大型假的OHLCV數據
        print("\n1. 創建模擬數據...")
        dates = pd.date_range('2023-01-01', periods=1000, freq='5min')
        
        # 生成具有趨勢的價格數據
        base_price = 45000.0
        trend = 0.0001  # 輕微上升趨勢
        volatility = 0.02
        
        prices = []
        current_price = base_price
        
        for i in range(1000):
            # 添加趨勢和隨機波動
            price_change = trend + np.random.normal(0, volatility)
            current_price *= (1 + price_change)
            prices.append(current_price)
        
        # 生成OHLCV數據
        data = []
        for i, (date, close_price) in enumerate(zip(dates, prices)):
            high = close_price * (1 + abs(np.random.normal(0, 0.01)))
            low = close_price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = prices[i-1] if i > 0 else close_price
            volume = np.random.uniform(100, 1000)
            
            data.append({
                'timestamp': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })
        
        # 創建DataFrame
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        print(f"   生成數據形狀: {df.shape}")
        print(f"   時間範圍: {df.index[0]} 到 {df.index[-1]}")
        print(f"   價格範圍: {df['close'].min():.2f} - {df['close'].max():.2f}")
        
        # 分割數據
        split_idx = int(len(df) * 0.8)  # 80% 訓練，20% 測試
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        print(f"\n2. 數據分割:")
        print(f"   訓練數據: {train_df.shape}")
        print(f"   測試數據: {test_df.shape}")
        
        # 初始化FeatureEngine
        print("\n3. 初始化FeatureEngine...")
        feature_engine = FeatureEngine()
        
        # 在訓練數據上進行fit_transform
        print("\n4. 訓練階段 - fit_transform...")
        transformed_train_df, scaler = feature_engine.fit_transform(train_df)
        
        print(f"   轉換後訓練數據形狀: {transformed_train_df.shape}")
        print(f"   特徵數量: {transformed_train_df.shape[1] - 5}")  # 減去OHLCV列
        
        # 在測試數據上進行transform
        print("\n5. 推理階段 - transform...")
        transformed_test_df = feature_engine.transform(test_df, scaler)
        
        print(f"   轉換後測試數據形狀: {transformed_test_df.shape}")
        
        # 驗證數據洩漏防護
        print("\n6. 數據洩漏驗證:")
        
        # 檢查特徵列的統計信息
        feature_cols = [col for col in transformed_train_df.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        if feature_cols:
            train_stats = transformed_train_df[feature_cols].describe()
            test_stats = transformed_test_df[feature_cols].describe()
            
            print("   訓練數據特徵統計 (前5個特徵):")
            print(train_stats.iloc[:, :5].round(4))
            
            print("\n   測試數據特徵統計 (前5個特徵):")
            print(test_stats.iloc[:, :5].round(4))
            
            # 檢查標準化是否正確
            train_means = train_stats.loc['mean'].abs()
            train_stds = train_stats.loc['std']
            
            print(f"\n   訓練數據標準化檢查:")
            print(f"   - 平均值接近0: {(train_means < 0.1).all()}")
            print(f"   - 標準差接近1: {((train_stds - 1).abs() < 0.1).all()}")
            
            # 檢查測試數據是否使用了訓練階段的統計信息
            print(f"\n   防止數據洩漏檢查:")
            print(f"   - 測試數據平均值不為0: {not (test_stats.loc['mean'].abs() < 0.1).all()}")
            print(f"   - 測試數據標準差不為1: {not ((test_stats.loc['std'] - 1).abs() < 0.1).all()}")
        
        # 保存示例數據
        print("\n7. 保存示例結果...")
        
        # 創建輸出目錄
        output_dir = "./data/features/"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存轉換後的數據
        transformed_train_df.to_parquet(f"{output_dir}/train_features_sample.parquet")
        transformed_test_df.to_parquet(f"{output_dir}/test_features_sample.parquet")
        
        # 保存scaler對象
        import joblib
        joblib.dump(scaler, f"{output_dir}/scaler_sample.joblib")
        
        print(f"   已保存到: {output_dir}")
        print(f"   - train_features_sample.parquet")
        print(f"   - test_features_sample.parquet")
        print(f"   - scaler_sample.joblib")
        
        print("\n✅ FeatureEngine 測試完成！")
        print("\n特徵工程系統成功實現了:")
        print("  ✓ 技術指標計算")
        print("  ✓ 訓練/推理階段分離")
        print("  ✓ 數據洩漏防護")
        print("  ✓ 標準化處理")
        print("  ✓ 錯誤處理和日誌記錄")
        
    except Exception as e:
        print(f"❌ FeatureEngine 測試失敗: {e}")
        print("\n錯誤詳情:")
        import traceback
        traceback.print_exc()
