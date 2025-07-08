"""
CryptoAce 數據收集模組

此模組負責下載、驗證、清洗原始K線數據，並調用 FeatureEngine 生成特徵數據文件。
"""

import os
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional
import joblib

# 處理相對匯入問題
try:
    from .configurator import Configurator
    from .feature_engine import FeatureEngine
    from .utils import set_random_seed
except ImportError:
    # 當直接執行此檔案時，使用絕對匯入
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.configurator import Configurator
    from core.utils import set_random_seed
    
    # 暫時創建一個簡單的 FeatureEngine 類別，避免匯入錯誤
    class FeatureEngine:
        def __init__(self, config, logger):
            self.config = config
            self.logger = logger
            self.logger.warning("使用臨時 FeatureEngine 實現")
        
        def fit_transform(self, data):
            """臨時實現：返回原始數據和空的 scaler"""
            self.logger.info("執行基本特徵處理（臨時實現）")
            # 添加基本的技術指標
            import pandas as pd
            from sklearn.preprocessing import StandardScaler
            
            features = data.copy()
            
            # 添加一些基本特徵
            features['returns'] = features['close'].pct_change()
            features['volatility'] = features['returns'].rolling(20).std()
            features['sma_20'] = features['close'].rolling(20).mean()
            features['price_position'] = features['close'] / features['sma_20']
            
            # 移除 NaN 值
            features = features.dropna()
            
            # 簡單的標準化
            scaler = StandardScaler()
            numeric_cols = ['returns', 'volatility', 'price_position']
            features[numeric_cols] = scaler.fit_transform(features[numeric_cols])
            
            return features, scaler


class DataHarvester:
    """
    數據收集器類別
    
    負責從交易所下載原始K線數據，進行數據驗證和清洗，
    然後調用特徵工程模組生成最終的特徵數據。
    """
    
    def __init__(self, config: Configurator, logger: Any) -> None:
        """
        初始化數據收集器
        
        Args:
            config: 配置管理器實例
            logger: 日誌記錄器實例
        """
        self.config = config
        self.logger = logger
        
        # 獲取配置參數
        self.exchange_config = config.exchange
        self.data_config = config.data
        
        # 初始化交易所連接
        self.exchange = self._init_exchange()
        
        # 設定數據路徑
        self.raw_data_path = Path(self.data_config.get('raw_data_path', './data/raw/'))
        self.feature_data_path = Path(self.data_config.get('feature_data_path', './data/features/'))
        
        # 確保目錄存在
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.feature_data_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("DataHarvester 初始化完成")
    
    def _init_exchange(self) -> ccxt.Exchange:
        """
        初始化交易所連接
        
        Returns:
            交易所實例
        """
        exchange_name = self.exchange_config.get('name', 'bitget').lower()
        
        try:
            if exchange_name == 'bitget':
                exchange = ccxt.bitget({
                    'apiKey': self.exchange_config.get('bitget_api_key'),
                    'secret': self.exchange_config.get('bitget_api_secret'),
                    'password': self.exchange_config.get('bitget_passphrase'),
                    'sandbox': False,  # 設為 True 使用測試環境
                    'enableRateLimit': True,
                })
            elif exchange_name == 'binance':
                exchange = ccxt.binance({
                    'apiKey': self.exchange_config.get('binance_api_key'),
                    'secret': self.exchange_config.get('binance_secret_key'),
                    'enableRateLimit': True,
                })
            else:
                raise ValueError(f"不支援的交易所: {exchange_name}")
            
            self.logger.info(f"已連接到 {exchange_name} 交易所")
            return exchange
            
        except Exception as e:
            self.logger.error(f"交易所連接失敗: {e}")
            # 使用無API密鑰的公開連接
            if exchange_name == 'bitget':
                exchange = ccxt.bitget({'enableRateLimit': True})
            else:
                exchange = ccxt.binance({'enableRateLimit': True})
            
            self.logger.warning("使用公開API連接（無需認證）")
            return exchange
    
    def fetch_ohlcv_data(self, 
                        symbol: str, 
                        timeframe: str,
                        start_date: str,
                        limit: int = 1000) -> pd.DataFrame:
        """
        從交易所獲取OHLCV數據
        
        Args:
            symbol: 交易對符號 (如 'BTC/USDT')
            timeframe: 時間週期 (如 '5m', '1h')
            start_date: 開始日期 (ISO格式)
            limit: 每次請求的數據量限制
            
        Returns:
            OHLCV數據DataFrame
        """
        try:
            # 轉換時間格式
            since = int(datetime.fromisoformat(start_date.replace('Z', '+00:00')).timestamp() * 1000)
            
            all_data = []
            current_since = since
            
            self.logger.info(f"開始下載 {symbol} {timeframe} 數據，從 {start_date}")
            
            while True:
                # 獲取數據
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_since,
                    limit=limit
                )
                
                if not ohlcv:
                    break
                
                all_data.extend(ohlcv)
                current_since = ohlcv[-1][0] + 1  # 下一批數據的起始時間
                
                self.logger.debug(f"已下載 {len(ohlcv)} 條記錄，總計 {len(all_data)} 條")
                
                # 如果返回的數據少於限制，說明已經到最新數據
                if len(ohlcv) < limit:
                    break
            
            # 轉換為DataFrame
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # 轉換數據類型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            self.logger.info(f"成功下載 {len(df)} 條 {symbol} 數據")
            return df
            
        except Exception as e:
            self.logger.error(f"數據下載失敗: {e}")
            return pd.DataFrame()
    
    def _validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        驗證和清洗數據
        
        Args:
            df: 原始OHLCV數據
            
        Returns:
            清洗後的數據
        """
        if df.empty:
            self.logger.warning("輸入數據為空")
            return df
        
        original_len = len(df)
        self.logger.info(f"開始數據清洗，原始數據: {original_len} 條")
        
        # 1. 移除缺失值
        df = df.dropna()
        
        # 2. 檢查價格邏輯性
        # high >= max(open, close) and low <= min(open, close)
        price_logic_mask = (
            (df['high'] >= df[['open', 'close']].max(axis=1)) &
            (df['low'] <= df[['open', 'close']].min(axis=1)) &
            (df['high'] >= df['low'])
        )
        df = df[price_logic_mask]
        
        # 3. 檢查成交量
        df = df[df['volume'] > 0]
        
        # 4. 移除價格為零或負數的記錄
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            df = df[df[col] > 0]
        
        # 5. 移除異常波動（價格變化超過50%的記錄）
        df['price_change'] = df['close'].pct_change().abs()
        df = df[df['price_change'] <= 0.5]  # 移除單日變化超過50%的異常數據
        df.drop('price_change', axis=1, inplace=True)
        
        # 6. 移除重複的時間戳
        df = df[~df.index.duplicated(keep='first')]
        
        # 7. 按時間排序
        df = df.sort_index()
        
        cleaned_len = len(df)
        removed_count = original_len - cleaned_len
        removal_rate = (removed_count / original_len) * 100 if original_len > 0 else 0
        
        self.logger.info(f"數據清洗完成: 保留 {cleaned_len} 條，移除 {removed_count} 條 ({removal_rate:.2f}%)")
        
        if removal_rate > 10:
            self.logger.warning(f"數據移除率較高 ({removal_rate:.2f}%)，請檢查數據品質")
        
        return df
    
    def save_raw_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> str:
        """
        保存原始數據到文件
        
        Args:
            df: 數據DataFrame
            symbol: 交易對符號
            timeframe: 時間週期
            
        Returns:
            保存的文件路徑
        """
        # 生成文件名
        symbol_safe = symbol.replace('/', '_')
        filename = f"{symbol_safe}_{timeframe}_raw.parquet"
        filepath = self.raw_data_path / filename
        
        # 保存為parquet格式（更高效）
        df.to_parquet(filepath)
        
        self.logger.info(f"原始數據已保存: {filepath}")
        return str(filepath)
    
    def run_collection(self) -> None:
        """
        執行完整的數據收集流程
        
        包括數據下載、清洗、特徵生成和保存
        """
        try:
            # 獲取配置參數
            symbol = self.exchange_config.get('default_symbol', 'BTC/USDT')
            timeframe = self.data_config.get('timeframe', '5m')
            start_date = self.data_config.get('start_date', '2023-01-01T00:00:00Z')
            
            self.logger.info(f"開始數據收集流程: {symbol} {timeframe}")
            
            # 1. 下載原始數據
            raw_data = self.fetch_ohlcv_data(symbol, timeframe, start_date)
            
            if raw_data.empty:
                self.logger.error("未能獲取任何數據，流程終止")
                return
            
            # 2. 數據驗證和清洗
            cleaned_data = self._validate_and_clean_data(raw_data)
            
            if cleaned_data.empty:
                self.logger.error("數據清洗後無有效數據，流程終止")
                return
            
            # 3. 保存原始數據
            raw_filepath = self.save_raw_data(cleaned_data, symbol, timeframe)
            
            # 4. 特徵工程
            self.logger.info("開始特徵工程處理")
            
            try:
                feature_engine = FeatureEngine(self.config, self.logger)
                
                # 生成特徵
                features_df, scaler = feature_engine.fit_transform(cleaned_data)
                
                # 保存特徵數據
                symbol_safe = symbol.replace('/', '_')
                features_filename = f"{symbol_safe}_{timeframe}_features.parquet"
                features_filepath = self.feature_data_path / features_filename
                features_df.to_parquet(features_filepath)
                
                # 保存Scaler對象
                scaler_filename = f"{symbol_safe}_{timeframe}_scaler.joblib"
                scaler_filepath = self.feature_data_path / scaler_filename
                joblib.dump(scaler, scaler_filepath)
                
                self.logger.info(f"特徵數據已保存: {features_filepath}")
                self.logger.info(f"Scaler已保存: {scaler_filepath}")
                
                # 統計信息
                self.logger.info(f"數據收集完成統計:")
                self.logger.info(f"  - 原始數據: {len(raw_data)} 條")
                self.logger.info(f"  - 清洗後數據: {len(cleaned_data)} 條")
                self.logger.info(f"  - 特徵數據: {len(features_df)} 條")
                self.logger.info(f"  - 特徵維度: {features_df.shape[1]} 個")
                
            except Exception as e:
                self.logger.error(f"特徵工程處理失敗: {e}")
                self.logger.info("已保存原始數據，可稍後重新處理特徵")
            
        except Exception as e:
            self.logger.error(f"數據收集流程失敗: {e}")
            raise
    
    def get_full_dataset(self) -> pd.DataFrame:
        """
        獲取完整的數據集
        
        Returns:
            包含所有原始數據的 DataFrame
        """
        all_files = list(self.raw_data_path.glob("*.parquet"))
        if not all_files:
            self.logger.warning("未找到任何原始數據文件")
            return pd.DataFrame()
        
        # 合併所有文件中的數據
        data_frames = [pd.read_parquet(file) for file in all_files]
        full_dataset = pd.concat(data_frames, ignore_index=True)
        self.logger.info(f"成功加載完整數據集，共 {len(full_dataset)} 條記錄")
        return full_dataset
    
    def get_data_slice(self, start_idx: int, end_idx: int) -> pd.DataFrame:
        """
        根據索引範圍獲取數據片段
        
        Args:
            start_idx: 開始索引
            end_idx: 結束索引
            
        Returns:
            指定範圍的數據片段
        """
        full_dataset = self.get_full_dataset()
        if full_dataset.empty:
            self.logger.warning("無法獲取數據片段：完整數據集為空")
            return pd.DataFrame()
        
        # 確保索引範圍有效
        start_idx = max(0, start_idx)
        end_idx = min(len(full_dataset), end_idx)
        
        if start_idx >= end_idx:
            self.logger.warning(f"無效的索引範圍: [{start_idx}:{end_idx}]")
            return pd.DataFrame()
        
        data_slice = full_dataset.iloc[start_idx:end_idx].copy()
        self.logger.info(f"獲取數據片段 [{start_idx}:{end_idx}]，共 {len(data_slice)} 條記錄")
        return data_slice


if __name__ == "__main__":
    """單元測試區塊"""
    
    print("=== CryptoAce 數據收集系統測試 ===")
    
    try:
        # 初始化依賴項
        try:
            from .configurator import load_config
            from .logger import setup_logger
        except ImportError:
            from core.configurator import load_config
            from core.logger import setup_logger
        
        # 載入配置和日誌
        config = load_config()
        logger = setup_logger(config)
        
        # 設定隨機種子
        random_seed = config.get_nested('agent', 'random_seed') or 42
        set_random_seed(random_seed)
        
        # 創建數據收集器
        harvester = DataHarvester(config, logger)
        
        print("\n開始執行數據收集流程...")
        
        # 執行完整的數據收集
        harvester.run_collection()
        
        print("\n✅ 數據收集測試完成！")
        print(f"原始數據目錄: {harvester.raw_data_path}")
        print(f"特徵數據目錄: {harvester.feature_data_path}")
        
        # 檢查生成的文件
        raw_files = list(harvester.raw_data_path.glob("*.parquet"))
        feature_files = list(harvester.feature_data_path.glob("*.parquet"))
        scaler_files = list(harvester.feature_data_path.glob("*.joblib"))
        
        print(f"\n生成的文件:")
        print(f"  原始數據文件: {len(raw_files)} 個")
        print(f"  特徵數據文件: {len(feature_files)} 個")
        print(f"  Scaler文件: {len(scaler_files)} 個")
        
    except Exception as e:
        print(f"❌ 數據收集測試失敗: {e}")
        print("\n錯誤詳情:")
        import traceback
        traceback.print_exc()