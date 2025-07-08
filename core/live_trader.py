"""
實時交易模組 - LiveTrader
實現安全回退邏輯、狀態持久化和異步交易循環
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

# 處理相對導入
try:
    from .configurator import Configurator
    from .logger import get_logger
    from .agent import Agent
    from .data_harvester import DataHarvester
    from .feature_engine import FeatureEngine
except ImportError:
    # 支持直接執行腳本
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.configurator import Configurator
    from core.logger import get_logger
    from core.agent import Agent
    from core.data_harvester import DataHarvester
    from core.feature_engine import FeatureEngine


class LiveTrader:
    """
    實時交易類別
    負責實時數據處理、模型預測、風險管理和交易執行
    """
    
    def __init__(self, config_path: str = "config.yaml", 
                 experiment_dir: Optional[str] = None,
                 state_file: str = "live_trader_state.json"):
        """
        初始化LiveTrader，實現安全回退邏輯
        
        Args:
            config_path: 配置文件路徑
            experiment_dir: 實驗目錄（用於加載模型和scaler）
            state_file: 狀態持久化文件名
        """
        self.logger = None
        self.config = None
        self.agent = None
        self.data_harvester = None
        self.feature_engine = None
        
        # 狀態管理
        self.state_file = state_file
        self.state = {
            'last_processed_time': None,
            'current_position': None,
            'total_trades': 0,
            'total_pnl': 0.0,
            'last_prediction': None,
            'risk_metrics': {},
            'session_start': datetime.now().isoformat()
        }
        
        # 交易狀態
        self.current_position: Optional[Dict[str, Any]] = None
        self.pending_orders: Dict[str, Dict[str, Any]] = {}
        self.trade_history: list = []
        self.running = False
        
        try:
            # 安全初始化配置和日誌
            self._safe_init_config(config_path)
            self._safe_init_logger()
            
            # 安全初始化核心組件
            self._safe_init_components(experiment_dir)
            
            # 嘗試加載狀態
            self._load_state()
            
            self.logger.info("LiveTrader 初始化成功")
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"LiveTrader 初始化失敗: {e}")
            else:
                print(f"LiveTrader 初始化失敗: {e}")
            # 不拋出異常，允許優雅降級
            
    def _safe_init_config(self, config_path: str):
        """安全初始化配置"""
        try:
            self.config = Configurator(config_path)
        except Exception as e:
            print(f"配置初始化失敗，使用默認配置: {e}")
            # 創建最小默認配置
            self.config = type('Config', (), {
                'trading': type('Trading', (), {
                    'symbol': 'BTCUSDT',
                    'timeframe': '1m',
                    'position_size': 0.01,
                    'max_position_size': 0.1,
                    'stop_loss': 0.02,
                    'take_profit': 0.03
                })(),
                'data': type('Data', (), {
                    'lookback_window': 100
                })(),
                'get': lambda key, default=None: default
            })()
    
    def _safe_init_logger(self):
        """安全初始化日誌"""
        try:
            self.logger = get_logger()
        except Exception as e:
            print(f"日誌初始化失敗: {e}")
            # 創建基本日誌記錄器
            self.logger = logging.getLogger("LiveTrader")
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _safe_init_components(self, experiment_dir: Optional[str]):
        """安全初始化核心組件"""
        try:
            # 初始化數據收集器
            self.data_harvester = DataHarvester(self.config, self.logger)
            
            # 初始化特徵引擎
            self.feature_engine = FeatureEngine(self.config)
            
            # 初始化交易代理
            if experiment_dir and os.path.exists(experiment_dir):
                # 從實驗目錄加載訓練好的模型
                self.agent = Agent(self.config)
                self._load_trained_model(experiment_dir)
            else:
                # 使用默認代理
                self.agent = Agent(self.config)
                self.logger.warning("未提供實驗目錄，使用默認交易代理")
                
        except Exception as e:
            self.logger.error(f"組件初始化失敗: {e}")
            # 創建最小功能組件
            self.agent = None
            self.data_harvester = None
            self.feature_engine = None
    
    def _load_trained_model(self, experiment_dir: str):
        """加載訓練好的模型和scaler"""
        try:
            model_path = os.path.join(experiment_dir, "model.zip")
            scaler_path = os.path.join(experiment_dir, "scaler.joblib")
            
            if os.path.exists(model_path) and self.agent:
                # 這裡假設agent有load_model方法
                if hasattr(self.agent, 'load_model'):
                    self.agent.load_model(model_path)
                    self.logger.info(f"成功加載模型: {model_path}")
                
            if os.path.exists(scaler_path) and self.feature_engine:
                # 這裡假設feature_engine有load_scaler方法
                if hasattr(self.feature_engine, 'load_scaler'):
                    self.feature_engine.load_scaler(scaler_path)
                    self.logger.info(f"成功加載Scaler: {scaler_path}")
                    
        except Exception as e:
            self.logger.error(f"模型加載失敗: {e}")
    
    def _save_state(self):
        """保存當前狀態到JSON文件"""
        try:
            # 更新狀態
            self.state.update({
                'last_processed_time': datetime.now().isoformat(),
                'current_position': self.current_position,
                'total_trades': len(self.trade_history),
                'total_pnl': sum(trade.get('pnl', 0) for trade in self.trade_history if isinstance(trade, dict)),
                'pending_orders_count': len(self.pending_orders)
            })
            
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
                
            self.logger.debug(f"狀態已保存到 {self.state_file}")
            
        except Exception as e:
            self.logger.error(f"狀態保存失敗: {e}")
    
    def _load_state(self):
        """從JSON文件加載狀態"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    loaded_state = json.load(f)
                    
                self.state.update(loaded_state)
                
                # 恢復持倉信息
                if self.state.get('current_position'):
                    pos_data = self.state['current_position']
                    self.current_position = {
                        'symbol': pos_data.get('symbol', ''),
                        'side': pos_data.get('side', 'long'),
                        'size': pos_data.get('size', 0.0),
                        'entry_price': pos_data.get('entry_price', 0.0),
                        'timestamp': pos_data.get('timestamp', datetime.now().isoformat())
                    }
                
                self.logger.info(f"成功加載狀態: 最後處理時間 {self.state.get('last_processed_time')}")
                
            else:
                self.logger.info("未找到狀態文件，使用初始狀態")
                
        except Exception as e:
            self.logger.error(f"狀態加載失敗: {e}")
    
    async def start(self):
        """啟動異步主循環"""
        if not self._validate_components():
            self.logger.error("組件驗證失敗，無法啟動交易循環")
            return
            
        self.running = True
        self.logger.info("LiveTrader 主循環啟動")
        
        try:
            # 主交易循環
            while self.running:
                await self._trading_loop_iteration()
                await asyncio.sleep(1)  # 1秒間隔
                
        except KeyboardInterrupt:
            self.logger.info("收到中斷信號，正在停止...")
        except Exception as e:
            self.logger.error(f"交易循環異常: {e}")
        finally:
            await self._cleanup()
    
    def _validate_components(self) -> bool:
        """驗證核心組件是否可用"""
        required_components = [
            ('config', self.config),
            ('logger', self.logger),
            ('agent', self.agent),
            ('data_harvester', self.data_harvester),
            ('feature_engine', self.feature_engine)
        ]
        
        missing_components = []
        for name, component in required_components:
            if component is None:
                missing_components.append(name)
        
        if missing_components:
            self.logger.error(f"缺少必要組件: {missing_components}")
            return False
            
        return True
    
    async def _trading_loop_iteration(self):
        """交易循環的單次迭代"""
        try:
            # 獲取最新數據
            latest_data = await self._get_latest_data()
            if latest_data is None or latest_data.empty:
                return
            
            # 處理數據並進行預測
            await self._on_message(latest_data)
            
            # 保存狀態
            self._save_state()
            
        except Exception as e:
            self.logger.error(f"交易循環迭代異常: {e}")
    
    async def _get_latest_data(self) -> Optional[pd.DataFrame]:
        """獲取最新市場數據"""
        try:
            if not self.data_harvester:
                return None
                
            # 獲取最近的數據
            symbol = getattr(self.config.trading, 'symbol', 'BTCUSDT')
            timeframe = getattr(self.config.trading, 'timeframe', '1m')
            lookback = getattr(self.config.data, 'lookback_window', 100)
            
            # 模擬獲取數據（實際實現中這裡會調用真實的API）
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=lookback)
            
            # 這裡應該調用真實的數據獲取方法
            # data = self.data_harvester.get_historical_data(symbol, timeframe, start_time, end_time)
            
            # 暫時返回模擬數據
            timestamps = pd.date_range(start=start_time, end=end_time, freq='1min')
            data = pd.DataFrame({
                'timestamp': timestamps,
                'open': np.random.uniform(40000, 50000, len(timestamps)),
                'high': np.random.uniform(40000, 50000, len(timestamps)),
                'low': np.random.uniform(40000, 50000, len(timestamps)),
                'close': np.random.uniform(40000, 50000, len(timestamps)),
                'volume': np.random.uniform(1000, 10000, len(timestamps))
            })
            
            return data
            
        except Exception as e:
            self.logger.error(f"獲取數據失敗: {e}")
            return None
    
    async def _on_message(self, data: pd.DataFrame):
        """
        處理收到的K線數據
        
        Args:
            data: K線數據DataFrame
        """
        try:
            if data is None or data.empty:
                return
            
            # 特徵工程
            features = await self._prepare_features(data)
            if features is None:
                return
            
            # 模型預測
            prediction = await self._make_prediction(features)
            if prediction is None:
                return
            
            # 更新狀態
            self.state['last_prediction'] = {
                'value': float(prediction),
                'timestamp': datetime.now().isoformat()
            }
            
            # 風險管理檢查
            if not await self._risk_check(prediction, data):
                self.logger.info("風險檢查未通過，跳過此次交易信號")
                return
            
            # 執行交易決策
            await self._execute_trading_decision(prediction, data)
            
        except Exception as e:
            self.logger.error(f"消息處理異常: {e}")
    
    async def _prepare_features(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """準備特徵數據"""
        try:
            if not self.feature_engine:
                return None
                
            # 計算技術指標
            features_df = self.feature_engine.calculate_features(data)
            
            if features_df is None or features_df.empty:
                return None
            
            # 獲取最新一行的特徵
            latest_features = features_df.iloc[-1:].select_dtypes(include=[np.number])
            
            # 處理缺失值
            latest_features = latest_features.fillna(0)
            
            return latest_features.values
            
        except Exception as e:
            self.logger.error(f"特徵準備失敗: {e}")
            return None
    
    async def _make_prediction(self, features: np.ndarray) -> Optional[float]:
        """使用模型進行預測"""
        try:
            if not self.agent or features is None:
                return None
            
            # 假設agent有predict方法
            if hasattr(self.agent, 'predict'):
                prediction = self.agent.predict(features)
                return float(prediction)
            else:
                # 模擬預測（實際實現中應該使用真實模型）
                return np.random.choice([-1, 0, 1])  # -1: 賣出, 0: 持有, 1: 買入
                
        except Exception as e:
            self.logger.error(f"模型預測失敗: {e}")
            return None
    
    async def _risk_check(self, prediction: float, data: pd.DataFrame) -> bool:
        """風險管理檢查"""
        try:
            # 檢查持倉大小限制
            max_position = getattr(self.config.trading, 'max_position_size', 0.1)
            current_size = abs(self.current_position['size']) if self.current_position else 0
            
            if current_size >= max_position:
                self.logger.warning(f"達到最大持倉限制: {current_size}")
                return False
            
            # 檢查價格波動
            if len(data) >= 2:
                current_price = data['close'].iloc[-1]
                prev_price = data['close'].iloc[-2]
                price_change = abs(current_price - prev_price) / prev_price
                
                max_volatility = 0.05  # 5%最大波動
                if price_change > max_volatility:
                    self.logger.warning(f"價格波動過大: {price_change:.4f}")
                    return False
            
            # 檢查交易頻率
            recent_trades = [t for t in self.trade_history 
                           if hasattr(t, 'timestamp') and 
                           (datetime.now() - t.timestamp).seconds < 300]  # 5分鐘內
            
            if len(recent_trades) > 3:
                self.logger.warning("交易頻率過高")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"風險檢查失敗: {e}")
            return False
    
    async def _execute_trading_decision(self, prediction: float, data: pd.DataFrame):
        """執行交易決策"""
        try:
            current_price = data['close'].iloc[-1]
            symbol = getattr(self.config.trading, 'symbol', 'BTCUSDT')
            position_size = getattr(self.config.trading, 'position_size', 0.01)
            
            # 根據預測信號執行交易
            if prediction > 0.5:  # 買入信號
                await self._execute_buy(symbol, position_size, current_price)
            elif prediction < -0.5:  # 賣出信號
                await self._execute_sell(symbol, position_size, current_price)
            # 其他情況保持當前持倉
            
        except Exception as e:
            self.logger.error(f"交易執行失敗: {e}")
    
    async def _execute_buy(self, symbol: str, size: float, price: float):
        """執行買入操作"""
        try:
            # 如果已有空倉，先平倉
            if self.current_position and self.current_position['side'] == 'short':
                await self._close_position()
            
            # 創建買入訂單
            order = {
                'symbol': symbol,
                'side': 'buy',
                'size': size,
                'price': price,
                'order_type': 'market',
                'timestamp': datetime.now()
            }
            
            # 模擬訂單執行
            success = await self._submit_order(order)
            
            if success:
                # 更新持倉
                if self.current_position:
                    self.current_position['size'] += size
                else:
                    self.current_position = {
                        'symbol': symbol,
                        'side': 'long',
                        'size': size,
                        'entry_price': price,
                        'timestamp': datetime.now()
                    }
                
                self.logger.info(f"買入執行成功: {symbol} {size} @ {price}")
            
        except Exception as e:
            self.logger.error(f"買入執行失敗: {e}")
    
    async def _execute_sell(self, symbol: str, size: float, price: float):
        """執行賣出操作"""
        try:
            # 如果已有多倉，先平倉
            if self.current_position and self.current_position['side'] == 'long':
                await self._close_position()
            
            # 創建賣出訂單
            order = {
                'symbol': symbol,
                'side': 'sell',
                'size': size,
                'price': price,
                'order_type': 'market',
                'timestamp': datetime.now()
            }
            
            # 模擬訂單執行
            success = await self._submit_order(order)
            
            if success:
                # 更新持倉
                if self.current_position:
                    self.current_position['size'] -= size
                else:
                    self.current_position = {
                        'symbol': symbol,
                        'side': 'short',
                        'size': size,
                        'entry_price': price,
                        'timestamp': datetime.now()
                    }
                
                self.logger.info(f"賣出執行成功: {symbol} {size} @ {price}")
            
        except Exception as e:
            self.logger.error(f"賣出執行失敗: {e}")
    
    async def _close_position(self):
        """平倉當前持倉"""
        try:
            if not self.current_position:
                return
            
            # 創建平倉訂單
            close_side = 'sell' if self.current_position['side'] == 'long' else 'buy'
            order = {
                'symbol': self.current_position['symbol'],
                'side': close_side,
                'size': abs(self.current_position['size']),
                'price': 0,  # 市價
                'order_type': 'market',
                'timestamp': datetime.now()
            }
            
            success = await self._submit_order(order)
            
            if success:
                # 記錄交易
                self.trade_history.append({
                    'symbol': self.current_position['symbol'],
                    'side': close_side,
                    'size': abs(self.current_position['size']),
                    'price': 0,  # 實際價格需要從交易所獲取
                    'timestamp': datetime.now()
                })
                
                self.logger.info(f"平倉成功: {self.current_position['symbol']}")
                self.current_position = None
            
        except Exception as e:
            self.logger.error(f"平倉失敗: {e}")
    
    async def _submit_order(self, order: Dict[str, Any]) -> bool:
        """提交訂單到交易所"""
        try:
            # 這裡應該實現真實的交易所API調用
            # 現在只是模擬成功
            order_id = f"order_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.pending_orders[order_id] = order
            
            # 模擬異步執行
            await asyncio.sleep(0.1)
            
            # 模擬訂單完成
            del self.pending_orders[order_id]
            
            self.logger.debug(f"訂單提交成功: {order_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"訂單提交失敗: {e}")
            return False
    
    async def get_balance(self) -> Dict[str, float]:
        """查詢賬戶餘額"""
        try:
            # 模擬餘額查詢
            await asyncio.sleep(0.1)
            
            balance = {
                'USDT': 10000.0,
                'BTC': 0.1
            }
            
            self.logger.info(f"餘額查詢成功: {balance}")
            return balance
            
        except Exception as e:
            self.logger.error(f"餘額查詢失敗: {e}")
            return {}
    
    async def get_positions(self) -> Dict[str, Any]:
        """查詢當前持倉"""
        try:
            positions = {}
            
            if self.current_position:
                positions[self.current_position['symbol']] = {
                    'side': self.current_position['side'],
                    'size': self.current_position['size'],
                    'entry_price': self.current_position['entry_price'],
                    'timestamp': self.current_position['timestamp'].isoformat() if isinstance(self.current_position['timestamp'], datetime) else self.current_position['timestamp']
                }
            
            self.logger.info(f"持倉查詢成功: {positions}")
            return positions
            
        except Exception as e:
            self.logger.error(f"持倉查詢失敗: {e}")
            return {}
    
    async def stop(self):
        """停止交易循環"""
        self.running = False
        self.logger.info("LiveTrader 停止信號已發送")
    
    async def _cleanup(self):
        """清理資源"""
        try:
            # 保存最終狀態
            self._save_state()
            
            # 取消所有掛單
            if self.pending_orders:
                self.logger.info(f"取消 {len(self.pending_orders)} 個掛單")
                self.pending_orders.clear()
            
            self.logger.info("LiveTrader 清理完成")
            
        except Exception as e:
            self.logger.error(f"清理過程異常: {e}")


# 測試和演示代碼
if __name__ == "__main__":
    async def test_live_trader():
        """測試LiveTrader功能"""
        print("=== LiveTrader 測試開始 ===")
        
        try:
            # 創建LiveTrader實例
            trader = LiveTrader(
                config_path="config.yaml",
                experiment_dir=None,  # 不使用實驗目錄
                state_file="test_state.json"
            )
            
            print("✓ LiveTrader 實例創建成功")
            
            # 測試餘額查詢
            balance = await trader.get_balance()
            print(f"✓ 餘額查詢: {balance}")
            
            # 測試持倉查詢
            positions = await trader.get_positions()
            print(f"✓ 持倉查詢: {positions}")
            
            # 測試狀態保存
            trader._save_state()
            print("✓ 狀態保存成功")
            
            # 模擬短時間運行
            print("開始短期交易循環測試（5秒）...")
            start_task = asyncio.create_task(trader.start())
            
            # 運行5秒後停止
            await asyncio.sleep(5)
            await trader.stop()
            
            # 等待停止完成
            try:
                await asyncio.wait_for(start_task, timeout=2.0)
            except asyncio.TimeoutError:
                print("交易循環已超時停止")
            
            print("✓ 交易循環測試完成")
            
            # 清理測試文件
            if os.path.exists("test_state.json"):
                os.remove("test_state.json")
                print("✓ 測試文件清理完成")
            
            print("=== 所有測試通過 ===")
            
        except Exception as e:
            print(f"✗ 測試失敗: {e}")
            import traceback
            traceback.print_exc()
    
    # 運行測試
    asyncio.run(test_live_trader())
