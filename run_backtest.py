#!/usr/bin/env python3
"""
回测框架启动脚本
快速运行回测的简单接口
"""

import sys
import os
import argparse
from datetime import datetime

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtest import BacktestFramework, DataManager, MomentumStrategy
from config import *


def run_quick_backtest(stock_codes=None, 
                      start_date='2023-01-01', 
                      end_date='2023-06-30',
                      initial_cash=500000):
    """
    快速运行回测
    
    Args:
        stock_codes: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        initial_cash: 初始资金
    """
    
    if stock_codes is None:
        stock_codes = ['000001.SZ', '000002.SZ', '000858.SZ']
    
    print(f"🚀 开始运行动量策略回测")
    print(f"📊 股票池: {stock_codes}")
    print(f"📅 回测期间: {start_date} 到 {end_date}")
    print(f"💰 初始资金: {initial_cash:,.2f}")
    print("="*60)
    
    try:
        # 1. 初始化数据管理器
        print("📥 正在获取数据...")
        data_manager = DataManager()
        
        # 2. 下载历史数据
        data_dict = data_manager.download_stock_data(
            stock_codes, 
            period=DATA_CONFIG['data_period'],
            start_date=start_date,
            end_date=end_date
        )
        
        if not data_dict:
            print("❌ 没有获取到有效数据，退出回测")
            return None
        
        print(f"✅ 成功获取 {len(data_dict)} 只股票的数据")
        
        # 3. 创建回测框架
        print("🔧 初始化回测框架...")
        framework = BacktestFramework(initial_cash=initial_cash)
        framework.setup_cerebro()
        framework.add_data(data_dict)
        
        # 4. 获取策略参数
        strategy_params = STRATEGY_CONFIG.get('momentum', {})
        
        # 5. 添加策略
        framework.add_strategy(MomentumStrategy, **strategy_params)
        framework.add_analyzers()
        
        # 6. 运行回测
        print("⚡ 开始运行回测...")
        strategy, metrics = framework.run_backtest()
        
        # 7. 保存结果
        results = {
            'strategy_name': 'MomentumStrategy',
            'backtest_period': f"{start_date} to {end_date}",
            'stock_codes': stock_codes,
            'initial_cash': initial_cash,
            'final_cash': framework.cerebro.broker.getvalue(),
            'total_return': metrics.get('total_return', 0),
            'annual_return': metrics.get('annual_return', 0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'max_drawdown': metrics.get('max_drawdown', 0),
            'win_rate': metrics.get('win_rate', 0),
            'trades_count': len(strategy.trades) if hasattr(strategy, 'trades') else 0,
            'strategy_params': strategy_params,
            'metrics': metrics
        }
        
        # 创建结果目录
        os.makedirs(PERFORMANCE_CONFIG['results_path'], exist_ok=True)
        
        # 保存结果到文件
        result_file = f"{PERFORMANCE_CONFIG['results_path']}/momentum_{start_date}_{end_date}.json"
        import json
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"💾 回测结果已保存到: {result_file}")
        
        return results
        
    except Exception as e:
        print(f"❌ 回测过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='回测框架启动脚本')
    parser.add_argument('--stocks', '-t', nargs='+', 
                       default=['000001.SZ', '000002.SZ', '000858.SZ'],
                       help='股票代码列表')
    parser.add_argument('--start', default='2023-01-01', help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', default='2023-06-30', help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--cash', '-c', type=float, default=500000, help='初始资金')
    
    args = parser.parse_args()
    
    # 运行回测
    results = run_quick_backtest(
        stock_codes=args.stocks,
        start_date=args.start,
        end_date=args.end,
        initial_cash=args.cash
    )
    
    if results:
        print("\n🎉 回测完成！")
    else:
        print("\n❌ 回测失败！")


if __name__ == "__main__":
    main() 