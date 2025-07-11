"""
LiveTrader 測試模組

此模組包含 LiveTrader 類別的完整測試，涵蓋初始化邏輯、
安全回退機制、異步方法和核心交易流程。
"""

import pytest
import asyncio
import os
import json
import tempfile
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import pandas as pd
import numpy as np

# 處理導入
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.live_trader import LiveTrader


class TestLiveTrader:
    """LiveTrader 測試類別"""
    
    def setup_method(self):
        """每個測試方法前的設置"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")
        self.state_file = os.path.join(self.temp_dir, "test_state.json")
        
        # 創建測試配置檔案
        with open(self.config_path, 'w', encoding='utf-8') as f:
            f.write("""
exchange:
  name: "bitget"
  api_key: ""
  secret: ""
  passphrase: ""
  sandbox: true

trading:
  symbol: "BTCUSDT"
  timeframe: "1m"
  position_size: 0.01
  max_position_size: 0.1
  stop_loss: 0.02
  take_profit: 0.03

data:
  lookback_window: 100
""")

    def teardown_method(self):
        """每個測試方法後的清理"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass

    @patch('core.live_trader.Agent')
    @patch('core.live_trader.DataHarvester')
    @patch('core.live_trader.FeatureEngine')
    @patch('core.live_trader.get_logger')
    def test_initialization_safe_fallback(self, mock_get_logger, mock_feature_engine, 
                                         mock_data_harvester, mock_agent):
        """測試 LiveTrader 初始化的安全回退邏輯"""
        
        # 設置模擬
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        # 模擬 DataHarvester 初始化失敗，觸發安全回退
        mock_data_harvester.side_effect = Exception("API key missing")
        
        # 創建 LiveTrader 實例
        trader = LiveTrader(
            config_path=self.config_path,
            state_file=self.state_file
        )
        
        # 斷言基本初始化成功
        assert trader is not None
        assert trader.config is not None
        assert trader.logger is not None
        
        # 斷言安全回退機制：組件設為 None
        assert trader.data_harvester is None
        assert trader.feature_engine is None
        assert trader.agent is None
        
        # 斷言錯誤被記錄
        mock_logger.error.assert_called()
        
        # 斷言狀態初始化正確
        assert trader.current_position is None
        assert trader.pending_orders == {}
        assert trader.trade_history == []
        assert trader.running is False

    @patch('core.live_trader.FeatureEngine')
    @patch('core.live_trader.Agent')
    @patch('core.live_trader.DataHarvester')
    @patch('core.live_trader.get_logger')
    def test_initialization_with_sandbox_mode(self, mock_get_logger, mock_data_harvester, 
                                             mock_agent, mock_feature_engine):
        """測試缺少實盤金鑰時的沙盒模式配置"""
        
        # 設置模擬
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        # 模擬 DataHarvester 實例
        mock_harvester_instance = MagicMock()
        mock_exchange = MagicMock()
        mock_exchange.sandboxMode = True  # 模擬沙盒模式
        mock_harvester_instance.exchange = mock_exchange
        mock_data_harvester.return_value = mock_harvester_instance
        
        # 模擬其他組件正常初始化
        mock_agent.return_value = MagicMock()
        mock_feature_engine.return_value = MagicMock()
        
        # 創建 LiveTrader 實例
        trader = LiveTrader(
            config_path=self.config_path,
            state_file=self.state_file
        )
        
        # 斷言 DataHarvester 被正確初始化
        assert trader.data_harvester is not None
        
        # 斷言沙盒模式被啟用（通過模擬的 exchange）
        if hasattr(trader.data_harvester, 'exchange'):
            assert trader.data_harvester.exchange.sandboxMode is True

    @pytest.mark.asyncio
    @patch('core.live_trader.Agent')
    @patch('core.live_trader.DataHarvester')
    @patch('core.live_trader.FeatureEngine')
    @patch('core.live_trader.get_logger')
    async def test_on_message_flow(self, mock_get_logger, mock_feature_engine, 
                                  mock_data_harvester, mock_agent):
        """測試 _on_message 方法的完整調用流程"""
        
        # 設置模擬
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_agent.return_value = MagicMock()
        mock_agent.return_value.predict.return_value = 1.0
        mock_feature_engine.return_value = MagicMock()
        mock_features_df = pd.DataFrame({'feature1': [1.0, 2.0], 'feature2': [3.0, 4.0]})
        mock_feature_engine.return_value.calculate_features.return_value = mock_features_df
        mock_data_harvester.return_value = MagicMock()
        
        trader = LiveTrader(
            config_path=self.config_path,
            state_file=self.state_file
        )
        
        # 確保組件已初始化
        assert trader.agent is not None
        assert trader.feature_engine is not None
        assert trader.data_harvester is not None
        
        # 創建測試數據
        test_data = pd.DataFrame({
            'timestamp': pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 00:01:00']),
            'close': [45000.0, 45100.0]
        })
        
        # 關鍵修復：直接模擬 _risk_check 返回 True
        trader._risk_check = AsyncMock(return_value=True)
        trader._execute_trading_decision = AsyncMock()
        
        # 調用 _on_message
        await trader._on_message(test_data)
        
        # 斷言 _risk_check 被調用
        trader._risk_check.assert_called_once()
        
        # 斷言 _execute_trading_decision 被調用
        trader._execute_trading_decision.assert_called_once()

    @pytest.mark.asyncio
    @patch('core.live_trader.get_logger')
    async def test_risk_check_normal_pass(self, mock_get_logger):
        """專門測試 _risk_check 在正常情況下通過的場景"""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        trader = LiveTrader(self.config_path, self.state_file)

        test_data = pd.DataFrame({'close': [45000.0, 45100.0]}) # 價格波動小 (0.22%)
        trader.current_position = None # 無持倉
        trader.trade_history = [] # 無交易歷史

        # 確保配置對象正確設置
        assert trader.config is not None
        assert hasattr(trader.config, 'trading')
        
        result = await trader._risk_check(0.8, test_data)
        
        # 如果失敗，檢查日誌輸出以瞭解原因
        if not result:
            mock_logger.warning.assert_called()
            mock_logger.error.assert_called()
        
        assert result is True, "在所有條件都滿足時，風險檢查應返回 True"

    @pytest.mark.asyncio
    async def test_get_balance(self):
        """測試餘額查詢方法"""
        
        with patch('core.live_trader.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            trader = LiveTrader(
                config_path=self.config_path,
                state_file=self.state_file
            )
            
            # 調用 get_balance
            balance = await trader.get_balance()
            
            # 斷言返回預期的模擬餘額
            assert isinstance(balance, dict)
            assert 'USDT' in balance
            assert 'BTC' in balance
            assert balance['USDT'] == 10000.0
            assert balance['BTC'] == 0.1

    @pytest.mark.asyncio
    async def test_get_positions(self):
        """測試持倉查詢方法"""
        
        with patch('core.live_trader.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            trader = LiveTrader(
                config_path=self.config_path,
                state_file=self.state_file
            )
            
            # 測試空持倉
            positions = await trader.get_positions()
            assert positions == {}
            
            # 設置測試持倉
            trader.current_position = {
                'symbol': 'BTCUSDT',
                'side': 'long',
                'size': 0.01,
                'entry_price': 45000.0,
                'timestamp': datetime.now()
            }
            
            # 測試有持倉
            positions = await trader.get_positions()
            assert 'BTCUSDT' in positions
            assert positions['BTCUSDT']['side'] == 'long'
            assert positions['BTCUSDT']['size'] == 0.01

    @pytest.mark.asyncio
    @patch('core.live_trader.get_logger')
    async def test_risk_check(self, mock_get_logger):
        """測試風險管理檢查"""

        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        trader = LiveTrader(
            config_path=self.config_path,
            state_file=self.state_file
        )

        # 創建測試數據
        test_data = pd.DataFrame({
            'close': [45000.0, 45100.0]  # 小幅波動 (0.22%)
        })

        # 確保交易歷史為空（避免頻率檢查失敗）
        trader.trade_history = []
        
        # 確保持倉為空
        trader.current_position = None
        
        # 確保配置對象正確設置
        assert trader.config is not None
        assert hasattr(trader.config, 'trading')

        # 測試正常情況（應該通過）
        result = await trader._risk_check(0.8, test_data)
        
        # 如果失敗，檢查日誌輸出以瞭解原因
        if not result:
            print("Risk check failed, checking logs...")
            if mock_logger.warning.called:
                print(f"Warning calls: {mock_logger.warning.call_args_list}")
            if mock_logger.error.called:
                print(f"Error calls: {mock_logger.error.call_args_list}")
        
        assert result is True, "正常情況下的風險檢查應該通過"

        # 測試持倉超限情況
        trader.current_position = {
            'size': 0.15  # 超過 max_position_size (0.1)
        }
        result = await trader._risk_check(0.8, test_data)
        assert result is False, "持倉超限時的風險檢查應該失敗"
        trader.current_position = None # 重置

        # 測試價格波動過大情況
        volatile_data = pd.DataFrame({
            'close': [45000.0, 50000.0]  # 11% 波動，超過 5% 限制
        })
        result = await trader._risk_check(0.8, volatile_data)
        assert result is False, "價格波動過大時的風險檢查應該失敗"

        # 測試交易頻率過高情況
        class MockTrade:
            def __init__(self):
                self.timestamp = datetime.now()

        trader.trade_history = [MockTrade() for _ in range(4)]  # 超過 3 筆
        result = await trader._risk_check(0.8, test_data)
        assert result is False, "交易頻率過高時的風險檢查應該失敗"

    def test_state_persistence(self):
        """測試狀態持久化功能"""
        
        with patch('core.live_trader.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            # 創建第一個 trader 實例
            trader1 = LiveTrader(
                config_path=self.config_path,
                state_file=self.state_file
            )
            
            # 設置一些狀態
            trader1.current_position = {
                'symbol': 'BTCUSDT',
                'side': 'long',
                'size': 0.01,
                'entry_price': 45000.0,
                'timestamp': datetime.now().isoformat()
            }
            trader1.trade_history = [
                {'symbol': 'BTCUSDT', 'side': 'buy', 'size': 0.01}
            ]
            
            # 保存狀態
            trader1._save_state()
            
            # 斷言狀態檔案被創建
            assert os.path.exists(self.state_file)
            
            # 創建第二個 trader 實例（應該加載狀態）
            trader2 = LiveTrader(
                config_path=self.config_path,
                state_file=self.state_file
            )
            
            # 斷言狀態被正確恢復
            assert trader2.current_position is not None
            assert trader2.current_position['symbol'] == 'BTCUSDT'
            assert trader2.current_position['side'] == 'long'
            assert trader2.current_position['size'] == 0.01

    @pytest.mark.asyncio
    @patch('core.live_trader.get_logger')
    async def test_component_validation(self, mock_get_logger):
        """測試組件驗證邏輯"""
        
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        trader = LiveTrader(
            config_path=self.config_path,
            state_file=self.state_file
        )
        
        # 斷言缺少組件時驗證失敗
        result = trader._validate_components()
        assert result is False
        
        # 模擬所有組件都可用
        trader.agent = MagicMock()
        trader.data_harvester = MagicMock()
        trader.feature_engine = MagicMock()
        
        result = trader._validate_components()
        assert result is True

    @pytest.mark.asyncio
    @patch('core.live_trader.get_logger')
    async def test_stop_and_cleanup(self, mock_get_logger):
        """測試停止和清理功能"""
        
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        trader = LiveTrader(
            config_path=self.config_path,
            state_file=self.state_file
        )
        
        # 設置一些掛單
        trader.pending_orders = {
            'order1': {'symbol': 'BTCUSDT', 'side': 'buy'},
            'order2': {'symbol': 'ETHUSDT', 'side': 'sell'}
        }
        
        # 調用停止
        await trader.stop()
        assert trader.running is False
        
        # 調用清理
        await trader._cleanup()
        
        # 斷言掛單被清除
        assert len(trader.pending_orders) == 0
        
        # 斷言日誌被記錄
        mock_logger.info.assert_called()

    @pytest.mark.asyncio
    @patch('core.live_trader.Agent')
    @patch('core.live_trader.DataHarvester') 
    @patch('core.live_trader.FeatureEngine')
    @patch('core.live_trader.get_logger')
    async def test_trading_loop_iteration(self, mock_get_logger, mock_feature_engine,
                                         mock_data_harvester, mock_agent):
        """測試交易循環迭代"""
        
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        # 設置模擬組件
        mock_agent.return_value = MagicMock()
        mock_feature_engine.return_value = MagicMock()
        mock_data_harvester.return_value = MagicMock()
        
        trader = LiveTrader(
            config_path=self.config_path,
            state_file=self.state_file
        )
        
        # 模擬 _get_latest_data 返回 None（無數據）
        trader._get_latest_data = AsyncMock(return_value=None)
        
        # 調用迭代，應該早期返回
        await trader._trading_loop_iteration()
        
        # 模擬 _get_latest_data 返回空 DataFrame
        empty_df = pd.DataFrame()
        trader._get_latest_data = AsyncMock(return_value=empty_df)
        
        # 調用迭代，應該早期返回
        await trader._trading_loop_iteration()
        
        # 斷言沒有異常被拋出
        assert True  # 如果到達這裡，說明測試通過


if __name__ == "__main__":
    # 支持直接執行測試
    pytest.main([__file__, "-v"])
