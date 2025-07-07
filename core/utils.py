"""
CryptoAce 工具函數模組

此模組提供專案中的通用工具函數，包括隨機種子設定等功能。
"""

import random
import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    """
    設定全局隨機種子，確保實驗的可重現性
    
    此函數會同時設定 Python 內建 random、NumPy 和 PyTorch 的隨機種子，
    並啟用 PyTorch 的確定性運算模式。
    
    Args:
        seed: 隨機種子值
    
    Note:
        啟用確定性模式可能會影響部分操作的性能，但能確保結果完全一致。
    """
    # 設定 Python 內建 random 模組的種子
    random.seed(seed)
    
    # 設定 NumPy 的隨機種子
    np.random.seed(seed)
    
    # 設定 PyTorch 的隨機種子
    torch.manual_seed(seed)
    
    # 如果有 CUDA 可用，也要設定 CUDA 的隨機種子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多 GPU 環境
    
    # 啟用 PyTorch 的確定性運算模式
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"已設定全局隨機種子: {seed}")


def get_device() -> torch.device:
    """
    獲取最佳的計算設備
    
    Returns:
        torch.device: 可用的最佳計算設備 (CUDA 或 CPU)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用 GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("使用 CPU")
    
    return device


def format_number(num: float, precision: int = 4) -> str:
    """
    格式化數字顯示
    
    Args:
        num: 要格式化的數字
        precision: 小數位數
        
    Returns:
        格式化後的字符串
    """
    if abs(num) >= 1e6:
        return f"{num/1e6:.{precision}f}M"
    elif abs(num) >= 1e3:
        return f"{num/1e3:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    計算夏普比率
    
    Args:
        returns: 收益率序列
        risk_free_rate: 無風險利率
        
    Returns:
        夏普比率值
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate
    if np.std(excess_returns) == 0:
        return 0.0
    
    return np.mean(excess_returns) / np.std(excess_returns)


def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    """
    計算最大回撤
    
    Args:
        equity_curve: 資產淨值曲線
        
    Returns:
        最大回撤比例
    """
    if len(equity_curve) == 0:
        return 0.0
    
    # 計算累計最高點
    peak = np.maximum.accumulate(equity_curve)
    
    # 計算回撤
    drawdown = (equity_curve - peak) / peak
    
    # 返回最大回撤
    return abs(np.min(drawdown))


if __name__ == "__main__":
    """單元測試區塊"""
    
    print("=== CryptoAce 工具函數測試 ===")
    
    # 測試隨機種子設定
    print("\n1. 測試隨機種子設定:")
    set_random_seed(42)
    
    # 生成隨機數測試
    random_val = random.random()
    numpy_val = np.random.random()
    torch_val = torch.rand(1).item()
    
    print(f"   Python random: {random_val:.6f}")
    print(f"   NumPy random: {numpy_val:.6f}")
    print(f"   PyTorch random: {torch_val:.6f}")
    
    # 測試設備檢測
    print("\n2. 測試設備檢測:")
    device = get_device()
    print(f"   選擇的設備: {device}")
    
    # 測試數字格式化
    print("\n3. 測試數字格式化:")
    test_numbers = [1234.5678, 1234567.89, 0.123456]
    for num in test_numbers:
        formatted = format_number(num)
        print(f"   {num} -> {formatted}")
    
    # 測試夏普比率計算
    print("\n4. 測試夏普比率計算:")
    test_returns = np.array([0.01, 0.02, -0.01, 0.03, 0.005])
    sharpe = calculate_sharpe_ratio(test_returns)
    print(f"   測試收益率: {test_returns}")
    print(f"   夏普比率: {sharpe:.4f}")
    
    # 測試最大回撤計算
    print("\n5. 測試最大回撤計算:")
    test_equity = np.array([1000, 1100, 1050, 1200, 1000, 1300])
    max_dd = calculate_max_drawdown(test_equity)
    print(f"   資產淨值曲線: {test_equity}")
    print(f"   最大回撤: {max_dd:.4f} ({max_dd*100:.2f}%)")
    
    print("\n✅ 工具函數測試完成！")