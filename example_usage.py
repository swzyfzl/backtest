"""
回测框架使用示例
展示如何使用框架进行不同策略的回测
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtest import BacktestFramework, DataManager, PerformanceAnalyzer
from strategies import get_strategy_class
from config import *

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_single_strategy_backtest(strategy_name: str, stock_codes: list = None, 
                               start_date: str = None, end_date: str = None,
                               initial_cash: float = None):
    """
    运行单个策略回测
    
    Args:
        strategy_name: 策略名称
        stock_codes: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        initial_cash: 初始资金
    """
    
    # 使用默认参数
    if stock_codes is None:
        stock_codes = ['000001.SZ', '000002.SZ', '000858.SZ', '600000.SH', '600036.SH']
    if start_date is None:
        start_date = DATA_CONFIG['start_date']
    if end_date is None:
        end_date = DATA_CONFIG['end_date']
    if initial_cash is None:
        initial_cash = BACKTEST_CONFIG['initial_cash']
    
    logger.info(f"开始运行 {strategy_name} 策略回测")
    logger.info(f"股票池: {stock_codes}")
    logger.info(f"回测期间: {start_date} 到 {end_date}")
    logger.info(f"初始资金: {initial_cash:,.2f}")
    
    # 1. 初始化数据管理器
    data_manager = DataManager()
    
    # 2. 下载历史数据
    data_dict = data_manager.download_stock_data(
        stock_codes, 
        period=DATA_CONFIG['data_period'],
        start_date=start_date,
        end_date=end_date
    )
    
    if not data_dict:
        logger.error("没有获取到有效数据，退出回测")
        return None
    
    # 3. 创建回测框架
    framework = BacktestFramework(initial_cash=initial_cash)
    
    # 4. 设置回测引擎
    framework.setup_cerebro()
    
    # 5. 添加数据源
    framework.add_data(data_dict)
    
    # 6. 获取策略类和参数
    strategy_class = get_strategy_class(strategy_name)
    strategy_params = STRATEGY_CONFIG.get(strategy_name, {})
    
    # 7. 添加策略
    framework.add_strategy(strategy_class, **strategy_params)
    
    # 8. 添加分析器
    framework.add_analyzers()
    
    # 9. 运行回测
    strategy, metrics = framework.run_backtest()
    
    # 10. 保存结果
    results = {
        'strategy_name': strategy_name,
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
    result_file = f"{PERFORMANCE_CONFIG['results_path']}/{strategy_name}_{start_date}_{end_date}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    logger.info(f"回测结果已保存到: {result_file}")
    
    return results


def run_multiple_strategies_comparison(strategy_names: list, stock_codes: list = None,
                                     start_date: str = None, end_date: str = None,
                                     initial_cash: float = None):
    """
    运行多个策略对比回测
    
    Args:
        strategy_names: 策略名称列表
        stock_codes: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        initial_cash: 初始资金
    """
    
    logger.info(f"开始运行多策略对比回测: {strategy_names}")
    
    results_comparison = {}
    
    for strategy_name in strategy_names:
        try:
            results = run_single_strategy_backtest(
                strategy_name=strategy_name,
                stock_codes=stock_codes,
                start_date=start_date,
                end_date=end_date,
                initial_cash=initial_cash
            )
            
            if results:
                results_comparison[strategy_name] = results
                
        except Exception as e:
            logger.error(f"运行策略 {strategy_name} 时出错: {e}")
    
    # 生成对比报告
    if results_comparison:
        generate_comparison_report(results_comparison)
    
    return results_comparison


def generate_comparison_report(results_comparison: dict):
    """生成策略对比报告"""
    
    # 创建对比数据
    comparison_data = []
    
    for strategy_name, results in results_comparison.items():
        comparison_data.append({
            '策略名称': strategy_name,
            '总收益率': f"{results['total_return']:.2f}",
            '年化收益率': f"{results['annual_return']:.2%}",
            '夏普比率': f"{results['sharpe_ratio']:.2f}",
            '最大回撤': f"{results['max_drawdown']:.2%}",
            '胜率': f"{results['win_rate']:.2%}",
            '交易次数': results['trades_count'],
            '最终资金': f"{results['final_cash']:,.2f}"
        })
    
    # 转换为DataFrame
    df = pd.DataFrame(comparison_data)
    
    # 保存对比报告
    report_file = f"{PERFORMANCE_CONFIG['results_path']}/strategy_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(report_file, index=False, encoding='utf-8-sig')
    
    # 打印对比表格
    print("\n" + "="*80)
    print("策略对比报告")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    logger.info(f"对比报告已保存到: {report_file}")
    
    return df


def run_parameter_optimization(strategy_name: str, param_ranges: dict, 
                             stock_codes: list = None, start_date: str = None, 
                             end_date: str = None, initial_cash: float = None):
    """
    运行参数优化
    
    Args:
        strategy_name: 策略名称
        param_ranges: 参数范围字典，格式: {'param_name': [value1, value2, ...]}
        stock_codes: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        initial_cash: 初始资金
    """
    
    logger.info(f"开始运行 {strategy_name} 参数优化")
    
    # 生成参数组合
    import itertools
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    param_combinations = list(itertools.product(*param_values))
    
    optimization_results = []
    
    for i, param_combo in enumerate(param_combinations):
        # 构建参数字典
        params = dict(zip(param_names, param_combo))
        
        logger.info(f"测试参数组合 {i+1}/{len(param_combinations)}: {params}")
        
        try:
            # 运行回测
            results = run_single_strategy_backtest(
                strategy_name=strategy_name,
                stock_codes=stock_codes,
                start_date=start_date,
                end_date=end_date,
                initial_cash=initial_cash,
                strategy_params=params
            )
            
            if results:
                # 添加参数信息
                results['parameters'] = params
                optimization_results.append(results)
                
        except Exception as e:
            logger.error(f"参数组合 {params} 测试失败: {e}")
    
    # 生成优化报告
    if optimization_results:
        generate_optimization_report(optimization_results, strategy_name)
    
    return optimization_results


def generate_optimization_report(optimization_results: list, strategy_name: str):
    """生成参数优化报告"""
    
    # 按年化收益率排序
    optimization_results.sort(key=lambda x: x['annual_return'], reverse=True)
    
    # 创建报告数据
    report_data = []
    
    for i, result in enumerate(optimization_results[:10]):  # 显示前10个最佳结果
        report_data.append({
            '排名': i + 1,
            '年化收益率': f"{result['annual_return']:.2%}",
            '夏普比率': f"{result['sharpe_ratio']:.2f}",
            '最大回撤': f"{result['max_drawdown']:.2%}",
            '胜率': f"{result['win_rate']:.2%}",
            '参数': str(result['parameters'])
        })
    
    # 转换为DataFrame
    df = pd.DataFrame(report_data)
    
    # 保存优化报告
    report_file = f"{PERFORMANCE_CONFIG['results_path']}/{strategy_name}_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(report_file, index=False, encoding='utf-8-sig')
    
    # 打印优化报告
    print("\n" + "="*80)
    print(f"{strategy_name} 参数优化报告 (前10名)")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    logger.info(f"优化报告已保存到: {report_file}")
    
    return df


def main():
    """主函数：演示各种回测功能"""
    
    # 示例1：运行单个策略回测
    print("="*60)
    print("示例1：运行动量策略回测")
    print("="*60)
    
    results = run_single_strategy_backtest(
        strategy_name='momentum',
        stock_codes=['000001.SZ', '000002.SZ', '000858.SZ'],
        start_date='2023-01-01',
        end_date='2023-06-30',
        initial_cash=500000
    )
    
    # 示例2：运行多策略对比
    print("\n" + "="*60)
    print("示例2：运行多策略对比回测")
    print("="*60)
    
    comparison_results = run_multiple_strategies_comparison(
        strategy_names=['momentum', 'ma_cross', 'rsi'],
        stock_codes=['000001.SZ', '000002.SZ', '000858.SZ'],
        start_date='2023-01-01',
        end_date='2023-06-30',
        initial_cash=500000
    )
    
    # 示例3：参数优化（可选）
    print("\n" + "="*60)
    print("示例3：动量策略参数优化")
    print("="*60)
    
    param_ranges = {
        'lookback_period': [10, 15, 20, 25],
        'momentum_threshold': [0.03, 0.05, 0.07],
        'position_size': [0.05, 0.1, 0.15]
    }
    
    optimization_results = run_parameter_optimization(
        strategy_name='momentum',
        param_ranges=param_ranges,
        stock_codes=['000001.SZ', '000002.SZ'],
        start_date='2023-01-01',
        end_date='2023-06-30',
        initial_cash=300000
    )
    
    print("\n回测演示完成！")


if __name__ == "__main__":
    main() 