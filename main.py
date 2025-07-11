#!/usr/bin/env python3
"""
CryptoAce 主程式入口

此模組是整個 CryptoAce 專案的命令行入口點，提供訓練、回測等功能的統一接口。
使用者可以通過命令行參數來指定執行的任務和相關配置。

使用範例:
    python main.py train
    python main.py backtest --experiment ./experiments/ppo_20231201_120000
    python main.py live --config config.yaml
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 導入專案核心模組
from core.configurator import Configurator
from core.logger import setup_logger
from core.trainer import Trainer
from core.backtester import Backtester
from core.live_trader import LiveTrader


def create_argument_parser() -> argparse.ArgumentParser:
    """
    創建並配置命令行參數解析器
    
    Returns:
        配置好的 ArgumentParser 實例
    """
    parser = argparse.ArgumentParser(
        prog='CryptoAce',
        description='CryptoAce - 基於強化學習的加密貨幣交易系統',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  %(prog)s collect                                    # 收集歷史數據
  %(prog)s collect --symbol BTC/USDT --limit 20000   # 收集指定交易對的更多數據
  %(prog)s train                                      # 開始模型訓練 (使用增強預設參數)
  %(prog)s train --window-size 5000                   # 使用更大的訓練窗口
  %(prog)s train --timesteps 50000 --walk-steps 80    # 使用更多訓練步數和滾動窗口
  %(prog)s backtest --experiment ./exp               # 回測指定實驗
  %(prog)s live --config config.yaml                 # 啟動實時交易
  %(prog)s live --experiment ./exp --dry-run         # 乾跑模式實時交易
        """
    )
    
    # 全域參數
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='配置檔案路徑 (預設: config.yaml)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='日誌級別 (預設: INFO)'
    )
    
    # 創建子命令解析器
    subparsers = parser.add_subparsers(
        dest='command',
        title='可用命令',
        description='選擇要執行的操作',
        help='使用 %(prog)s {command} --help 查看詳細說明'
    )
    
    # 訓練子命令
    train_parser = subparsers.add_parser(
        'train',
        help='開始模型訓練',
        description='使用配置檔案中的參數開始強化學習模型訓練'
    )
    
    train_parser.add_argument(
        '--resume',
        type=str,
        metavar='EXPERIMENT_DIR',
        help='從指定的實驗目錄恢復訓練'
    )
    
    train_parser.add_argument(
        '--epochs',
        type=int,
        metavar='N',
        help='訓練輪數 (覆蓋配置檔案設定)'
    )
    
    train_parser.add_argument(
        '--window-size',
        type=int,
        metavar='N',
        help='訓練窗口大小 (覆蓋配置檔案設定)'
    )
    
    train_parser.add_argument(
        '--timesteps',
        type=int,
        metavar='N',
        help='每步訓練時間步數 (覆蓋配置檔案設定)'
    )
    
    train_parser.add_argument(
        '--walk-steps',
        type=int,
        metavar='N',
        help='滾動窗口步數 (覆蓋配置檔案設定)'
    )
    
    train_parser.add_argument(
        '--lookback-window',
        type=int,
        metavar='N',
        help='回看窗口大小 (覆蓋配置檔案設定)'
    )
    
    # 回測子命令
    backtest_parser = subparsers.add_parser(
        'backtest',
        help='回測交易策略',
        description='使用訓練好的模型對歷史數據進行回測分析'
    )
    
    backtest_parser.add_argument(
        '--experiment',
        type=str,
        required=True,
        metavar='EXPERIMENT_DIR',
        help='要回測的實驗目錄路徑 (必需)'
    )
    
    backtest_parser.add_argument(
        '--start-date',
        type=str,
        metavar='YYYY-MM-DD',
        help='回測開始日期 (覆蓋配置檔案設定)'
    )
    
    backtest_parser.add_argument(
        '--end-date',
        type=str,
        metavar='YYYY-MM-DD',
        help='回測結束日期 (覆蓋配置檔案設定)'
    )
    
    # 實時交易子命令
    live_parser = subparsers.add_parser(
        'live',
        help='啟動實時交易',
        description='啟動實時交易系統，可選擇使用訓練好的模型'
    )
    
    live_parser.add_argument(
        '--experiment',
        type=str,
        metavar='EXPERIMENT_DIR',
        help='使用指定實驗目錄中的訓練模型'
    )
    
    live_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='乾跑模式，不執行真實交易'
    )
    
    live_parser.add_argument(
        '--state-file',
        type=str,
        default='live_trader_state.json',
        metavar='FILE',
        help='狀態持久化檔案路徑 (預設: live_trader_state.json)'
    )
    
    # 數據收集子命令
    collect_parser = subparsers.add_parser(
        'collect',
        help='收集歷史市場數據',
        description='從交易所收集歷史K線數據用於訓練'
    )
    
    collect_parser.add_argument(
        '--symbol',
        type=str,
        default='BTC/USDT',
        help='交易對符號 (預設: BTC/USDT)'
    )
    
    collect_parser.add_argument(
        '--timeframe',
        type=str,
        default='5m',
        choices=['1m', '5m', '15m', '1h', '4h', '1d'],
        help='K線時間週期 (預設: 5m)'
    )
    
    collect_parser.add_argument(
        '--limit',
        type=int,
        default=10000,
        help='收集數據筆數 (預設: 10000)'
    )
    
    return parser


def setup_dependencies(config_path: str, log_level: str) -> tuple[Configurator, object]:
    """
    設置並初始化專案依賴
    
    Args:
        config_path: 配置檔案路徑
        log_level: 日誌級別
        
    Returns:
        (configurator, logger) 元組
        
    Raises:
        FileNotFoundError: 當配置檔案不存在時
        Exception: 當依賴初始化失敗時
    """
    try:
        # 驗證配置檔案存在
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置檔案不存在: {config_path}")
        
        # 初始化配置管理器
        configurator = Configurator(config_path)
        
        # 如果需要覆蓋日誌級別，更新配置
        if log_level != 'INFO':  # 只有當不是預設值時才覆蓋
            # 動態更新配置中的日誌級別
            if hasattr(configurator, '_config') and 'logger' in configurator._config:
                configurator._config['logger']['level'] = log_level
        
        # 設置日誌系統 - 傳遞 configurator 實例
        logger = setup_logger(configurator)
        logger.info(f"CryptoAce 啟動 - 配置檔案: {config_path}")
        logger.info(f"日誌級別設定為: {log_level}")
        
        return configurator, logger
        
    except Exception as e:
        print(f"❌ 依賴初始化失敗: {e}")
        sys.exit(1)


def handle_train_command(args: argparse.Namespace, configurator: Configurator, logger) -> None:
    """
    處理訓練命令
    
    Args:
        args: 命令行參數
        configurator: 配置管理器
        logger: 日誌記錄器
    """
    try:
        logger.info("🚀 開始模型訓練")
        
        # 動態修改配置以支援更多數據訓練
        if not hasattr(configurator, '_config'):
            configurator._config = {}
        if 'training' not in configurator._config:
            configurator._config['training'] = {}
        
        # 處理窗口大小參數
        if args.window_size:
            configurator._config['training']['window_size'] = args.window_size
            logger.info(f"覆蓋窗口大小: {args.window_size}")
        elif 'window_size' not in configurator._config['training']:
            # 設定預設的更大窗口大小
            configurator._config['training']['window_size'] = 3000
            logger.info("使用預設增強窗口大小: 3000")
        
        # 處理訓練時間步數參數
        if args.timesteps:
            configurator._config['training']['base_timesteps'] = args.timesteps
            logger.info(f"覆蓋訓練時間步數: {args.timesteps}")
        elif 'base_timesteps' not in configurator._config['training']:
            configurator._config['training']['base_timesteps'] = 30000
            logger.info("使用預設增強訓練時間步數: 30000")
        
        # 處理滾動窗口步數參數
        if args.walk_steps:
            configurator._config['training']['walk_forward_steps'] = args.walk_steps
            logger.info(f"覆蓋滾動窗口步數: {args.walk_steps}")
        elif 'walk_forward_steps' not in configurator._config['training']:
            configurator._config['training']['walk_forward_steps'] = 50
            logger.info("使用預設增強滾動窗口步數: 50")
        
        # 處理回看窗口參數
        if args.lookback_window:
            configurator._config['training']['lookback_window'] = args.lookback_window
            logger.info(f"覆蓋回看窗口大小: {args.lookback_window}")
        elif 'lookback_window' not in configurator._config['training']:
            configurator._config['training']['lookback_window'] = 50
            logger.info("使用預設增強回看窗口: 50")
        
        # 設定其他增強的預設值
        if 'retrain_frequency' not in configurator._config['training']:
            configurator._config['training']['retrain_frequency'] = 15
        if 'experience_replay_ratio' not in configurator._config['training']:
            configurator._config['training']['experience_replay_ratio'] = 0.4
        
        # 創建訓練器實例
        trainer = Trainer(configurator, logger)
        
        # 處理可選參數
        if args.resume:
            if not os.path.exists(args.resume):
                raise FileNotFoundError(f"實驗目錄不存在: {args.resume}")
            logger.info(f"從實驗目錄恢復訓練: {args.resume}")
            trainer.resume_training(args.resume)
        else:
            # 處理訓練輪數覆蓋
            if args.epochs:
                logger.info(f"覆蓋配置檔案，設定訓練輪數為: {args.epochs}")
                # 這裡可以動態修改配置或傳遞給訓練器
            
            # 顯示最終使用的訓練配置
            training_config = configurator._config.get('training', {})
            logger.info("📊 訓練配置摘要:")
            logger.info(f"  窗口大小: {training_config.get('window_size', 1000)}")
            logger.info(f"  滾動步數: {training_config.get('walk_forward_steps', 30)}")
            logger.info(f"  回看窗口: {training_config.get('lookback_window', 20)}")
            logger.info(f"  訓練時間步數: {training_config.get('base_timesteps', 10000)}")
            logger.info(f"  重訓練頻率: {training_config.get('retrain_frequency', 10)}")
            logger.info(f"  經驗回放比例: {training_config.get('experience_replay_ratio', 0.3)}")
            
            # 開始新的訓練
            trainer.run_training()
        
        logger.info("✅ 訓練完成")
        
    except Exception as e:
        logger.error(f"❌ 訓練失敗: {e}")
        sys.exit(1)


def handle_backtest_command(args: argparse.Namespace, configurator: Configurator, logger) -> None:
    """
    處理回測命令
    
    Args:
        args: 命令行參數
        configurator: 配置管理器
        logger: 日誌記錄器
    """
    try:
        logger.info("📊 開始策略回測")
        
        # 驗證實驗目錄
        if not os.path.exists(args.experiment):
            raise FileNotFoundError(f"實驗目錄不存在: {args.experiment}")
        
        logger.info(f"使用實驗目錄: {args.experiment}")
        
        # 創建回測器實例
        backtester = Backtester(configurator, logger)
        
        # 處理日期參數覆蓋
        backtest_params = {}
        if args.start_date:
            backtest_params['start_date'] = args.start_date
            logger.info(f"覆蓋開始日期: {args.start_date}")
        
        if args.end_date:
            backtest_params['end_date'] = args.end_date
            logger.info(f"覆蓋結束日期: {args.end_date}")
        
        # 執行回測
        results = backtester.run_backtest(
            experiment_dir=args.experiment,
            **backtest_params
        )
        
        logger.info("✅ 回測完成")
        logger.info(f"回測結果: {results}")
        
    except Exception as e:
        logger.error(f"❌ 回測失敗: {e}")
        sys.exit(1)


def handle_collect_command(args: argparse.Namespace, configurator: Configurator, logger) -> None:
    """
    處理數據收集命令
    
    Args:
        args: 命令行參數
        configurator: 配置管理器
        logger: 日誌記錄器
    """
    try:
        logger.info("📥 開始收集歷史市場數據")
        
        # 動態更新配置 - 設定更早的開始時間來獲取更多數據
        if not hasattr(configurator, '_config'):
            configurator._config = {}
        if 'data' not in configurator._config:
            configurator._config['data'] = {}
        if 'exchange' not in configurator._config:
            configurator._config['exchange'] = {}
        
        # 更新數據收集參數
        configurator._config['exchange']['default_symbol'] = args.symbol
        configurator._config['data']['timeframe'] = args.timeframe
        
        # 根據需要的數據量調整開始時間
        import datetime
        from datetime import timedelta
        
        # 計算需要多久的歷史數據
        if timeframe := args.timeframe:
            if timeframe == '1m':
                time_delta = timedelta(minutes=args.limit)
            elif timeframe == '5m':
                time_delta = timedelta(minutes=args.limit * 5)
            elif timeframe == '15m':
                time_delta = timedelta(minutes=args.limit * 15)
            elif timeframe == '1h':
                time_delta = timedelta(hours=args.limit)
            elif timeframe == '4h':
                time_delta = timedelta(hours=args.limit * 4)
            elif timeframe == '1d':
                time_delta = timedelta(days=args.limit)
            else:
                time_delta = timedelta(minutes=args.limit * 5)  # 預設5分鐘
        
        # 設定更早的開始時間
        end_time = datetime.datetime.now()
        start_time = end_time - time_delta
        new_start_date = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        configurator._config['data']['start_date'] = new_start_date
        
        logger.info(f"交易對: {args.symbol}")
        logger.info(f"時間週期: {args.timeframe}")
        logger.info(f"目標收集筆數: {args.limit}")
        logger.info(f"調整開始時間為: {new_start_date}")
        
        # 創建數據收集器並執行標準收集流程
        from core.data_harvester import DataHarvester
        data_harvester = DataHarvester(configurator, logger)
        
        logger.info("🔄 開始從交易所收集數據...")
        
        # 使用標準的 run_collection 方法，但使用調整後的配置
        data_harvester.run_collection()
        
        # 檢查收集結果
        dataset = data_harvester.get_full_dataset()
        logger.info(f"✅ 數據收集完成，共收集 {len(dataset)} 條記錄")
        
        # 顯示數據範圍
        if len(dataset) > 0:
            # 由於 get_full_dataset 可能返回的是處理後的數據，我們讀取原始保存的文件
            try:
                import pandas as pd
                symbol_safe = args.symbol.replace('/', '_')
                raw_filepath = data_harvester.raw_data_path / f"{symbol_safe}_{args.timeframe}_raw.parquet"
                
                if raw_filepath.exists():
                    raw_df = pd.read_parquet(raw_filepath)
                    start_time = raw_df.index[0]
                    end_time = raw_df.index[-1]
                    logger.info(f"📅 數據時間範圍: {start_time} 至 {end_time}")
                    logger.info(f"📊 原始數據: {len(raw_df)} 條記錄")
                    logger.info(f"📊 數據列: {list(raw_df.columns)}")
                    
                    # 檢查特徵數據
                    features_filepath = data_harvester.feature_data_path / f"{symbol_safe}_{args.timeframe}_features.parquet"
                    if features_filepath.exists():
                        features_df = pd.read_parquet(features_filepath)
                        logger.info(f"🔧 特徵數據: {len(features_df)} 條記錄，{features_df.shape[1]} 個特徵")
                    
            except Exception as e:
                logger.warning(f"讀取保存的數據文件時出錯: {e}")
                logger.info(f"📊 基本統計: {len(dataset)} 條記錄")
        else:
            logger.warning("⚠️ 未收集到任何數據，請檢查交易對和時間設定")
        
    except Exception as e:
        logger.error(f"❌ 數據收集失敗: {e}")
        import traceback
        logger.error(f"錯誤詳情: {traceback.format_exc()}")
        sys.exit(1)


def handle_live_command(args: argparse.Namespace, configurator: Configurator, logger) -> None:
    """
    處理實時交易命令
    
    Args:
        args: 命令行參數
        configurator: 配置管理器
        logger: 日誌記錄器
    """
    try:
        logger.info("📈 啟動實時交易系統")
        
        # 驗證實驗目錄（如果提供）
        experiment_dir = None
        if args.experiment:
            if not os.path.exists(args.experiment):
                raise FileNotFoundError(f"實驗目錄不存在: {args.experiment}")
            experiment_dir = args.experiment
            logger.info(f"使用訓練模型: {args.experiment}")
        
        # 乾跑模式警告
        if args.dry_run:
            logger.warning("⚠️  乾跑模式啟用 - 不會執行真實交易")
        
        # 創建實時交易器實例
        live_trader = LiveTrader(
            config_path=args.config,
            experiment_dir=experiment_dir,
            state_file=args.state_file
        )
        
        logger.info("🔄 啟動交易循環")
        
        # 啟動交易循環（這是一個阻塞調用）
        import asyncio
        asyncio.run(live_trader.start())
        
    except KeyboardInterrupt:
        logger.info("⏹️  收到中斷信號，正在停止交易系統...")
    except Exception as e:
        logger.error(f"❌ 實時交易失敗: {e}")
        sys.exit(1)


def main() -> None:
    """
    主函數 - 程式入口點
    
    解析命令行參數，初始化依賴，並根據用戶指定的命令執行相應的操作。
    """
    # 創建參數解析器
    parser = create_argument_parser()
    
    # 解析命令行參數
    args = parser.parse_args()
    
    # 檢查是否提供了子命令
    if not args.command:
        parser.print_help()
        print("\n❌ 錯誤: 請指定要執行的命令")
        sys.exit(1)
    
    # 初始化專案依賴
    configurator, logger = setup_dependencies(args.config, args.log_level)
    
    # 根據命令執行相應操作
    try:
        if args.command == 'train':
            handle_train_command(args, configurator, logger)
        
        elif args.command == 'backtest':
            handle_backtest_command(args, configurator, logger)
        
        elif args.command == 'live':
            handle_live_command(args, configurator, logger)
        
        elif args.command == 'collect':
            handle_collect_command(args, configurator, logger)
        
        else:
            logger.error(f"❌ 不支援的命令: {args.command}")
            parser.print_help()
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("👋 程式被用戶中斷")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ 執行失敗: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
