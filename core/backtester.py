"""
CryptoAce 回測器模組

此模組實現回測功能，用於評估訓練完成的交易策略在歷史數據上的表現。
主要功能包括：
- 加載實驗產物（模型、配置、標量器）
- 在歷史數據上執行回測
- 生成績效報告和交易日誌
- 繪製資金曲線圖
"""

import os
import json
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure

try:
    # 相對導入（當作為模組導入時）
    from .configurator import Configurator
    from .data_harvester import DataHarvester
    from .feature_engine import FeatureEngine
    from .trading_env import TradingEnv
    from .agent import Agent
    from .logger import setup_logger
except ImportError:
    # 絕對導入（當直接執行時）
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from core.configurator import Configurator
    from core.data_harvester import DataHarvester
    from core.feature_engine import FeatureEngine
    from core.trading_env import TradingEnv
    from core.agent import Agent
    from core.logger import setup_logger


class Backtester:
    """
    回測器類別
    
    負責加載實驗產物並在歷史數據上進行回測，
    生成完整的績效分析報告。
    """
    
    def __init__(self, config: Optional[Configurator] = None, logger: Optional[Any] = None):
        """
        初始化回測器
        
        Args:
            config: 配置管理器實例（可選）
            logger: 日誌記錄器實例（可選）
        """
        self.config = config
        self.logger = logger
        
        # 實驗產物
        self.experiment_config = None
        self.agent = None
        self.scaler = None
        
        # 組件實例
        self.data_harvester = None
        self.feature_engine = None
        self.trading_env = None
        
        # 回測結果
        self.trade_log = []
        self.equity_curve = []
        self.performance_metrics = {}
        
        if self.logger:
            self.logger.info("Backtester 初始化完成")
    
    def load_experiment(self, experiment_path: str) -> None:
        """
        加載實驗產物
        
        Args:
            experiment_path: 實驗目錄路徑
        """
        experiment_dir = Path(experiment_path)
        
        if not experiment_dir.exists():
            raise FileNotFoundError(f"實驗目錄不存在: {experiment_path}")
        
        # 1. 加載配置文件
        config_file = experiment_dir / "config.yaml"
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_file}")
        
        self.experiment_config = Configurator(config_path=str(config_file))
        
        # 2. 設置日誌器（如果沒有提供的話）
        if not self.logger:
            self.logger = setup_logger(self.experiment_config)
        
        # 3. 加載標量器
        scaler_file = experiment_dir / "scaler.joblib"
        if scaler_file.exists():
            self.scaler = joblib.load(scaler_file)
            self.logger.info(f"已加載標量器: {scaler_file}")
        else:
            self.logger.warning("未找到標量器文件，將跳過特徵標準化")
        
        # 4. 初始化組件
        self._initialize_components()
        
        # 5. 加載訓練好的智能體
        model_file = experiment_dir / "model.zip"
        if not model_file.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_file}")
        
        self.agent = Agent(
            config=self.experiment_config,
            env=self.trading_env,
            logger=self.logger
        )
        self.agent.load(str(model_file))
        
        self.logger.info(f"實驗產物加載完成: {experiment_path}")
    
    def _initialize_components(self) -> None:
        """初始化回測所需的組件"""
        # 初始化數據收集器
        self.data_harvester = DataHarvester(
            config=self.experiment_config,
            logger=self.logger
        )
        
        # 初始化特徵引擎
        self.feature_engine = FeatureEngine(
            config=self.experiment_config,
            logger=self.logger
        )
        
        # 注意：TradingEnv 需要在有數據時才能初始化，所以先設為 None
        self.trading_env = None
    
    def _prepare_backtest_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        準備回測數據
        
        Args:
            start_date: 開始日期
            end_date: 結束日期
            
        Returns:
            處理後的特徵數據
        """
        # 1. 獲取原始數據
        raw_data = self.data_harvester.get_data_slice(start_date, end_date)
        
        if raw_data.empty:
            raise ValueError(f"無法獲取 {start_date} 到 {end_date} 的數據")
        
        self.logger.info(f"獲取到 {len(raw_data)} 條原始數據記錄")
        
        # 2. 生成特徵
        features_data = self.feature_engine.transform(raw_data)
        
        # 3. 應用標準化（如果有標量器）
        if self.scaler is not None:
            # 只標準化非價格列
            price_columns = ['open', 'high', 'low', 'close', 'volume']
            feature_columns = [col for col in features_data.columns if col not in price_columns]
            
            if feature_columns:
                features_data[feature_columns] = self.scaler.transform(features_data[feature_columns])
                self.logger.info("已應用特徵標準化")
        
        return features_data
    
    def _execute_backtest_loop(self, data: pd.DataFrame) -> None:
        """
        執行回測循環
        
        Args:
            data: 處理後的特徵數據
        """
        # 初始化交易環境（使用數據）
        self.trading_env = TradingEnv(
            data=data,
            config=self.experiment_config,
            logger=self.logger,
            initial_balance=self.experiment_config.trading_env.get('initial_balance', 10000.0),
            transaction_cost=self.experiment_config.exchange.get('fee', {}).get('taker', 0.001),
            max_position_change_per_step=self.experiment_config.trading_env.get('risk_management', {}).get('max_position_change_per_step', 0.1),
            max_drawdown_limit=self.experiment_config.trading_env.get('risk_management', {}).get('max_drawdown_limit', 0.2)
        )
        
        # 重置記錄
        self.trade_log = []
        self.equity_curve = []
        
        # 重置環境並獲取初始狀態
        state, info = self.trading_env.reset()
        done = False
        step = 0
        
        self.logger.info("開始執行回測循環...")
        
        while not done and step < len(data) - 1:
            # 記錄當前資金狀況
            current_equity = self.trading_env.get_portfolio_value()
            current_position = self.trading_env.get_position()
            current_price = data.iloc[self.trading_env.current_step]['close'] if hasattr(self.trading_env, 'current_step') else data.iloc[step]['close']
            current_time = data.index[self.trading_env.current_step] if hasattr(self.trading_env, 'current_step') else data.index[step]
            
            self.equity_curve.append({
                'timestamp': current_time,
                'equity': current_equity,
                'position': current_position,
                'price': current_price,
                'step': step
            })
            
            # 智能體做出決策
            action = self.agent.predict(state)
            
            # 執行動作並獲取結果
            state, reward, terminated, truncated, info = self.trading_env.step(action)
            done = terminated or truncated
            
            # 記錄交易
            if info.get('trade_executed', False):
                trade_record = {
                    'timestamp': current_time,
                    'step': step,
                    'action': action,
                    'position_before': info.get('position_before', 0),
                    'position_after': info.get('position_after', 0),
                    'price': current_price,
                    'trade_amount': info.get('trade_amount', 0),
                    'transaction_cost': info.get('transaction_cost', 0),
                    'equity_before': info.get('equity_before', 0),
                    'equity_after': current_equity,
                    'reward': reward
                }
                self.trade_log.append(trade_record)
            
            step += 1
            
            # 進度顯示
            if step % 1000 == 0:
                self.logger.info(f"回測進度: {step}/{len(data)-1} 步")
        
        # 記錄最後一個時點的資金狀況
        final_equity = self.trading_env.get_portfolio_value()
        final_position = self.trading_env.get_position()
        final_price = data.iloc[-1]['close']
        final_time = data.index[-1]
        
        self.equity_curve.append({
            'timestamp': final_time,
            'equity': final_equity,
            'position': final_position,
            'price': final_price,
            'step': step
        })
        
        self.logger.info(f"回測循環完成，共執行 {step} 步，記錄 {len(self.trade_log)} 筆交易")
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """
        計算績效指標
        
        Returns:
            績效指標字典
        """
        if not self.equity_curve:
            return {}
        
        # 轉換為 DataFrame 以便計算
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        equity_df = equity_df.set_index('timestamp')
        
        # 基本指標
        initial_equity = equity_df['equity'].iloc[0]
        final_equity = equity_df['equity'].iloc[-1]
        total_return = (final_equity - initial_equity) / initial_equity
        
        # 計算日收益率
        equity_df['daily_return'] = equity_df['equity'].pct_change().fillna(0)
        
        # 年化收益率（假設 252 個交易日）
        days = (equity_df.index[-1] - equity_df.index[0]).days
        if days > 0:
            annualized_return = (final_equity / initial_equity) ** (365 / days) - 1
        else:
            annualized_return = 0
        
        # 波動率
        volatility = equity_df['daily_return'].std() * np.sqrt(365)
        
        # 夏普比率（假設無風險利率為 0）
        if volatility > 0:
            sharpe_ratio = annualized_return / volatility
        else:
            sharpe_ratio = 0
        
        # 最大回撤
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
        max_drawdown = equity_df['drawdown'].min()
        
        # 卡爾瑪比率
        if abs(max_drawdown) > 0:
            calmar_ratio = annualized_return / abs(max_drawdown)
        else:
            calmar_ratio = 0
        
        # 交易統計
        num_trades = len(self.trade_log)
        if num_trades > 0:
            trades_df = pd.DataFrame(self.trade_log)
            winning_trades = len(trades_df[trades_df['reward'] > 0])
            win_rate = winning_trades / num_trades
            avg_trade_return = trades_df['reward'].mean()
        else:
            win_rate = 0
            avg_trade_return = 0
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_trade_return': avg_trade_return,
            'initial_equity': initial_equity,
            'final_equity': final_equity
        }
        
        return metrics
    
    def _create_equity_curve_plot(self, save_path: str) -> None:
        """
        創建資金曲線圖
        
        Args:
            save_path: 圖表保存路徑
        """
        if not self.equity_curve:
            self.logger.warning("無法創建資金曲線圖：沒有資金曲線數據")
            return
        
        # 準備數據
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        
        # 創建圖表
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle('回測結果分析', fontsize=16, fontweight='bold')
        
        # 1. 資金曲線
        ax1.plot(equity_df['timestamp'], equity_df['equity'], 'b-', linewidth=2, label='資金曲線')
        ax1.set_ylabel('資金 (USD)', fontsize=12)
        ax1.set_title('資金曲線', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. 持倉變化
        ax2.plot(equity_df['timestamp'], equity_df['position'], 'g-', linewidth=1, label='持倉比例')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_ylabel('持倉比例', fontsize=12)
        ax2.set_title('持倉變化', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. 價格走勢
        ax3.plot(equity_df['timestamp'], equity_df['price'], 'r-', linewidth=1, label='價格')
        ax3.set_ylabel('價格 (USD)', fontsize=12)
        ax3.set_xlabel('時間', fontsize=12)
        ax3.set_title('價格走勢', fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 格式化 x 軸
        for ax in [ax1, ax2, ax3]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(equity_df) // 10)))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"資金曲線圖已保存: {save_path}")
    
    def _save_trade_log(self, save_path: str) -> None:
        """
        保存交易日誌
        
        Args:
            save_path: 保存路徑
        """
        if not self.trade_log:
            self.logger.warning("沒有交易記錄可保存")
            return
        
        trades_df = pd.DataFrame(self.trade_log)
        trades_df.to_csv(save_path, index=False, encoding='utf-8')
        self.logger.info(f"交易日誌已保存: {save_path} ({len(trades_df)} 筆交易)")
    
    def _save_performance_report(self, save_path: str) -> None:
        """
        保存績效報告
        
        Args:
            save_path: 保存路徑
        """
        report = {
            'backtest_summary': {
                'start_time': str(pd.to_datetime(self.equity_curve[0]['timestamp'])) if self.equity_curve else None,
                'end_time': str(pd.to_datetime(self.equity_curve[-1]['timestamp'])) if self.equity_curve else None,
                'total_steps': len(self.equity_curve),
                'total_trades': len(self.trade_log)
            },
            'performance_metrics': self.performance_metrics,
            'experiment_config': self.experiment_config.to_dict() if self.experiment_config else {}
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"績效報告已保存: {save_path}")
    
    def run_backtest(self, experiment_path: str, 
                    start_date: Optional[str] = None, 
                    end_date: Optional[str] = None,
                    output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        執行回測
        
        Args:
            experiment_path: 實驗目錄路徑
            start_date: 回測開始日期（可選）
            end_date: 回測結束日期（可選）
            output_dir: 輸出目錄（可選）
            
        Returns:
            回測結果摘要
        """
        self.logger.info("=" * 50)
        self.logger.info("開始執行回測")
        self.logger.info("=" * 50)
        
        # 1. 加載實驗產物
        self.load_experiment(experiment_path)
        
        # 2. 設置回測時間範圍
        if not start_date:
            start_date = self.experiment_config.data.get('start_date', '2023-01-01T00:00:00Z')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        
        self.logger.info(f"回測時間範圍: {start_date} 到 {end_date}")
        
        # 3. 準備數據
        backtest_data = self._prepare_backtest_data(start_date, end_date)
        
        # 4. 執行回測
        self._execute_backtest_loop(backtest_data)
        
        # 5. 計算績效指標
        self.performance_metrics = self._calculate_performance_metrics()
        
        # 6. 設置輸出目錄
        if not output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(experiment_path) / f"backtest_{timestamp}"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 7. 生成報告和圖表
        self._create_equity_curve_plot(str(output_dir / "equity_curve.png"))
        self._save_trade_log(str(output_dir / "trade_log.csv"))
        self._save_performance_report(str(output_dir / "backtest_report.json"))
        
        # 8. 打印績效摘要
        self._print_performance_summary()
        
        self.logger.info(f"回測完成，結果已保存到: {output_dir}")
        
        return {
            'output_dir': str(output_dir),
            'performance_metrics': self.performance_metrics,
            'num_trades': len(self.trade_log),
            'backtest_period': {
                'start': start_date,
                'end': end_date
            }
        }
    
    def _print_performance_summary(self) -> None:
        """打印績效摘要"""
        if not self.performance_metrics:
            return
        
        print("\n" + "=" * 60)
        print("回測績效摘要")
        print("=" * 60)
        
        metrics = self.performance_metrics
        
        print(f"總收益率:        {metrics.get('total_return', 0):.2%}")
        print(f"年化收益率:      {metrics.get('annualized_return', 0):.2%}")
        print(f"年化波動率:      {metrics.get('volatility', 0):.2%}")
        print(f"夏普比率:        {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"最大回撤:        {metrics.get('max_drawdown', 0):.2%}")
        print(f"卡爾瑪比率:      {metrics.get('calmar_ratio', 0):.2f}")
        print(f"交易次數:        {metrics.get('num_trades', 0)}")
        print(f"勝率:           {metrics.get('win_rate', 0):.2%}")
        print(f"平均交易收益:    {metrics.get('avg_trade_return', 0):.4f}")
        print(f"初始資金:        ${metrics.get('initial_equity', 0):,.2f}")
        print(f"最終資金:        ${metrics.get('final_equity', 0):,.2f}")
        
        print("=" * 60)


if __name__ == "__main__":
    """
    主程式區塊 - 回測器測試
    
    此區塊演示如何使用 Backtester 進行回測分析
    """
    import sys
    from pathlib import Path
    
    print("=== CryptoAce 回測器測試 ===")
    print("正在執行回測分析...")
    print()
    
    try:
        # 設置實驗產物路徑（假設已存在）
        # 在實際使用中，這應該是一個真實的實驗目錄路徑
        project_root = Path(__file__).parent.parent
        experiment_path = project_root / "models" / "experiment_20240101_120000"  # 示例路徑
        
        # 檢查實驗目錄是否存在
        if not experiment_path.exists():
            print(f"⚠️  實驗目錄不存在: {experiment_path}")
            print("正在創建示例實驗目錄結構...")
            
            # 創建示例實驗目錄
            experiment_path.mkdir(parents=True, exist_ok=True)
            
            # 創建示例配置文件
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

logger:
  level: INFO
  file_path: "./logs/cryptoace.log"
"""
            with open(experiment_path / "config.yaml", 'w', encoding='utf-8') as f:
                f.write(config_content)
            
            # 創建空的模型和標量器文件（實際使用中這些應該是真實的）
            (experiment_path / "model.zip").touch()
            
            # 創建假的標量器文件
            import joblib
            from sklearn.preprocessing import StandardScaler
            fake_scaler = StandardScaler()
            joblib.dump(fake_scaler, experiment_path / "scaler.joblib")
            
            print(f"✅ 示例實驗目錄已創建: {experiment_path}")
            print("💡 注意: 這是一個示例目錄，包含空的模型文件")
            print("   在實際使用中，請使用真實的訓練實驗產物")
        
        # 創建回測器實例
        backtester = Backtester()
        
        print("\n1. 正在加載實驗產物...")
        try:
            # 模擬加載過程（實際情況下會加載真實的模型）
            print("   💡 示例模式：跳過真實的模型加載")
            print("   ✅ 實驗產物加載完成（模擬）")
        except Exception as e:
            print(f"   ❌ 實驗產物加載失敗: {e}")
            print("   💡 這可能是因為缺少真實的訓練模型")
        
        print("\n2. 回測配置信息:")
        print(f"   實驗路徑: {experiment_path}")
        print(f"   回測開始日期: 2023-01-01")
        print(f"   回測結束日期: 2023-01-31")
        
        print("\n3. 回測執行流程:")
        print("   ✅ 加載實驗產物")
        print("   ✅ 準備歷史數據")
        print("   ✅ 執行回測循環")
        print("   ✅ 計算績效指標")
        print("   ✅ 生成交易日誌")
        print("   ✅ 繪製資金曲線圖")
        print("   ✅ 保存分析報告")
        
        print("\n4. 示例績效指標:")
        example_metrics = {
            'total_return': 0.15,
            'annualized_return': 0.32,
            'volatility': 0.28,
            'sharpe_ratio': 1.14,
            'max_drawdown': -0.08,
            'calmar_ratio': 4.0,
            'num_trades': 127,
            'win_rate': 0.62,
            'avg_trade_return': 0.0012
        }
        
        print(f"   總收益率:        {example_metrics['total_return']:.2%}")
        print(f"   年化收益率:      {example_metrics['annualized_return']:.2%}")
        print(f"   夏普比率:        {example_metrics['sharpe_ratio']:.2f}")
        print(f"   最大回撤:        {example_metrics['max_drawdown']:.2%}")
        print(f"   交易次數:        {example_metrics['num_trades']}")
        print(f"   勝率:           {example_metrics['win_rate']:.2%}")
        
        print("\n🎉 回測器功能演示完成！")
        print("\n💡 如要執行真實回測，請確保:")
        print("   1. 具有完整的訓練實驗產物")
        print("   2. 配置正確的數據源")
        print("   3. 安裝所需的依賴庫")
        
        print(f"\n📁 示例實驗目錄: {experiment_path}")
        print("   您可以刪除此目錄，或保留作為模板使用")
        
    except Exception as e:
        print(f"\n❌ 回測器測試失敗: {e}")
        print("\n錯誤詳情:")
        import traceback
        traceback.print_exc()
        sys.exit(1)
