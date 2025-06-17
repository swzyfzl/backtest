"""
基准数据获取模块
用于获取沪深300等基准数据，计算收益率曲线，并提供绘图功能
"""

import pandas as pd
import numpy as np
from xtquant import xtdata
import logging
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import List, Optional, Tuple

# 配置日志
logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def get_benchmark_data(start_date, end_date, benchmark_code='000300.SH'):
    """
    下载基准数据并计算收益率曲线
    确保时间戳与回测数据一致
    
    Args:
        start_date (str): 开始日期，格式：'YYYY-MM-DD'
        end_date (str): 结束日期，格式：'YYYY-MM-DD'
        benchmark_code (str): 基准代码，默认沪深300指数
        
    Returns:
        tuple: (dates, cumulative_returns) 日期列表和累计收益率列表
    """
    try:
        print(f"正在获取{benchmark_code}基准数据...")
        
        # 方法1：先订阅数据
        try:
            xtdata.subscribe_quote(benchmark_code, period='1d', count=-1)
        except Exception as e:
            print(f"订阅失败: {e}")
        
        # 等待订阅完成
        time.sleep(1)
        
        # 方法2：下载历史数据
        try:
            xtdata.download_history_data(
                benchmark_code,
                period="1d",
                start_time=start_date,
                end_time=end_date,
                incrementally=True
            )
        except Exception as e:
            print(f"历史数据下载失败: {e}")
        
        # 方法3：尝试多种获取方式
        market_data = None
        
        # 方式3.1：使用get_market_data_ex
        try:
            market_data = xtdata.get_market_data_ex(
                [], [benchmark_code], period="1d", count=-1
            )
        except Exception as e:
            print(f"get_market_data_ex失败: {e}")
        
        # 方式3.2：如果方式3.1失败，尝试get_market_data
        if not market_data or benchmark_code not in market_data:
            try:
                market_data = xtdata.get_market_data(
                    field_list=['time', 'open', 'high', 'low', 'close', 'volume'],
                    stock_list=[benchmark_code],
                    period='1d',
                    count=-1,
                    dividend_type='none',
                    fill_data=True
                )
            except Exception as e:
                print(f"get_market_data失败: {e}")
        
        # 方式3.3：尝试使用get_market_data_ex_ori
        if not market_data or (isinstance(market_data, dict) and benchmark_code not in market_data):
            try:
                market_data = xtdata.get_market_data_ex_ori(
                    [], [benchmark_code], period="1d", count=-1
                )
            except Exception as e:
                print(f"get_market_data_ex_ori失败: {e}")

        # 处理获取到的数据
        if market_data and benchmark_code in market_data:
            benchmark_data = market_data[benchmark_code]
            
            # 转换为DataFrame
            if isinstance(benchmark_data, dict):
                df = pd.DataFrame(benchmark_data)
            else:
                df = benchmark_data

            if df.empty:
                print(f"❌ {benchmark_code}数据为空")
                return [], []
            
            # 数据清洗
            original_len = len(df)
            df = df.drop_duplicates()
            df = df.dropna(subset=['close', 'volume'])
            
            if df.empty:
                print(f"❌ {benchmark_code}数据清洗后为空")
                return [], []

            # 处理日期列
            datetime_column = None
            for col_name in ['datetime', 'time', 'date', 'timestamp']:
                if col_name in df.columns:
                    datetime_column = col_name
                    break
            
            if datetime_column:
                if not df.empty:
                    sample_time = df[datetime_column].iloc[0]
                    
                    # 处理时间戳转换
                    if pd.api.types.is_numeric_dtype(df[datetime_column]):
                        if sample_time > 1e10:  # 毫秒时间戳（13位）
                            df['datetime'] = pd.to_datetime(df[datetime_column], unit='ms')
                        elif sample_time > 1e8:  # 秒时间戳（10位）
                            df['datetime'] = pd.to_datetime(df[datetime_column], unit='s')
                        else:
                            df['datetime'] = pd.to_datetime(df[datetime_column])
                    else:
                        df['datetime'] = pd.to_datetime(df[datetime_column])
                    
                    # 删除日期转换失败的行
                    df = df.dropna(subset=['datetime'])
                    df = df.sort_values('datetime')
                    df = df.reset_index(drop=True)
                    
                    if not df.empty:
                        # 根据指定时间范围过滤数据
                        start_dt = pd.to_datetime(start_date)
                        end_dt = pd.to_datetime(end_date)
                        df = df[(df['datetime'] >= start_dt) & (df['datetime'] <= end_dt)]
                        
                        if not df.empty:
                            # 计算累计收益率
                            if len(df) > 0:
                                prices = df['close'].values
                                dates = pd.to_datetime(df['datetime']).dt.date.values
                                # 计算累计收益率（首日为1）
                                cumulative_returns = prices / prices[0]
                                
                                print(f"✅ {benchmark_code}基准数据获取成功")
                                print(f"   数据点数量: {len(df)}")
                                print(f"   时间范围: {df['datetime'].min().date()} 到 {df['datetime'].max().date()}")
                                print(f"   累计收益率: {(prices[-1]/prices[0] - 1)*100:.2f}%")
                                
                                return dates, cumulative_returns
                            else:
                                print(f"❌ {benchmark_code}基准数据过滤后为空")
                                return [], []
                        else:
                            print(f"❌ {benchmark_code}时间过滤后数据为空")
                            return [], []
                    else:
                        print(f"❌ {benchmark_code}时间列处理失败")
                        return [], []
                else:
                    print(f"❌ {benchmark_code}数据为空")
                    return [], []
            else:
                print(f"❌ {benchmark_code}数据中没有找到时间列")
                return [], []
        else:
            print(f"❌ 未获取到{benchmark_code}数据")
            return [], []
            
    except Exception as e:
        print(f"❌ 获取{benchmark_code}数据失败: {e}")
        return [], []


def get_hs300_etf_data(start_date, end_date):
    """
    获取沪深300ETF数据作为基准
    
    Args:
        start_date (str): 开始日期
        end_date (str): 结束日期
        
    Returns:
        tuple: (dates, cumulative_returns)
    """
    return get_benchmark_data(start_date, end_date, '510300.SH')


def get_hs300_index_data(start_date, end_date):
    """
    获取沪深300指数数据作为基准
    
    Args:
        start_date (str): 开始日期
        end_date (str): 结束日期
        
    Returns:
        tuple: (dates, cumulative_returns)
    """
    return get_benchmark_data(start_date, end_date, '000300.SH')


def get_shanghai_index_data(start_date, end_date):
    """
    获取上证指数数据作为基准
    
    Args:
        start_date (str): 开始日期
        end_date (str): 结束日期
        
    Returns:
        tuple: (dates, cumulative_returns)
    """
    return get_benchmark_data(start_date, end_date, '000001.SH')


def adjust_benchmark_timing(benchmark_dates, benchmark_curve, skip_days=20):
    """
    调整基准数据的时间范围，跳过前N个交易日，使其与策略对齐
    
    Args:
        benchmark_dates: 基准日期列表
        benchmark_curve: 基准收益率曲线
        skip_days: 跳过的交易日数量（默认20个交易日）
        
    Returns:
        tuple: (adjusted_dates, adjusted_curve) 调整后的日期和收益率
    """
    try:
        if len(benchmark_dates) <= skip_days:
            print(f"⚠️ 基准数据长度({len(benchmark_dates)})小于跳过天数({skip_days})，返回原始数据")
            return benchmark_dates, benchmark_curve
        
        # 跳过前N个交易日
        adjusted_dates = benchmark_dates[skip_days:]
        adjusted_curve = benchmark_curve[skip_days:]
        
        print(f"基准数据时间调整:")
        print(f"原始数据长度: {len(benchmark_dates)}")
        print(f"跳过天数: {skip_days}")
        print(f"调整后长度: {len(adjusted_dates)}")
        print(f"原始开始日期: {benchmark_dates[0]}")
        print(f"调整后开始日期: {adjusted_dates[0]}")
        print(f"原始结束日期: {benchmark_dates[-1]}")
        print(f"调整后结束日期: {adjusted_dates[-1]}")
        
        return adjusted_dates, adjusted_curve
        
    except Exception as e:
        print(f"调整基准数据时间时出错: {e}")
        return benchmark_dates, benchmark_curve


def plot_strategy_analysis(strategy, initial_cash: float, benchmark_dates=None, benchmark_curve=None, 
                          benchmark_name='沪深300基准', save_path=None, skip_warmup_days=20):
    """
    绘制策略分析图表（资金曲线、基准对比、仓位变化）
    
    Args:
        strategy: BackTrader策略对象，包含portfolio_values, cash_values, dates属性
        initial_cash (float): 初始资金
        benchmark_dates: 基准日期列表
        benchmark_curve: 基准收益率曲线
        benchmark_name (str): 基准名称
        save_path (str, optional): 图片保存路径
        skip_warmup_days (int): 跳过的预热期天数（默认20个交易日）
    """
    try:
        print(f"\n{'='*50}")
        print("绘制策略分析图表")
        print(f"{'='*50}")

        if hasattr(strategy, 'portfolio_values') and hasattr(strategy, 'cash_values') and hasattr(strategy, 'dates'):
            portfolio_values = strategy.portfolio_values
            cash_values = strategy.cash_values
            dates = strategy.dates
            
            print(f"portfolio_values长度: {len(portfolio_values)}")
            print(f"cash_values长度: {len(cash_values)}")
            print(f"dates长度: {len(dates)}")
            
            if len(dates) > 0:
                # 创建子图 - 现在有3个子图
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), height_ratios=[3, 1, 1])
                
                # 上图：收益率曲线（百分比）
                # 计算策略收益率（百分比）
                strategy_returns_pct = (np.array(portfolio_values) / initial_cash - 1) * 100
                ax1.plot(dates, strategy_returns_pct, label='策略收益率', color='blue', linewidth=2)
                
                # 添加调试信息
                print(f"\n策略收益率计算调试:")
                print(f"初始资金: {initial_cash:,.2f}")
                print(f"数据点数量: {len(dates)}")
                print(f"资金数据长度: {len(portfolio_values)}")
                if len(portfolio_values) > 0:
                    print(f"第一个资金值: {portfolio_values[0]:,.2f}")
                    print(f"最后一个资金值: {portfolio_values[-1]:,.2f}")
                    print(f"第一个收益率: {strategy_returns_pct[0]:.2f}%")
                    print(f"最后一个收益率: {strategy_returns_pct[-1]:.2f}%")
                    print(f"日期范围: {dates[0]} 到 {dates[-1]}")
                
                # 检查是否有异常值
                if len(strategy_returns_pct) > 0:
                    min_return = np.min(strategy_returns_pct)
                    max_return = np.max(strategy_returns_pct)
                    print(f"收益率范围: {min_return:.2f}% 到 {max_return:.2f}%")
                    
                    # 检查是否有零收益率
                    zero_returns = np.sum(strategy_returns_pct == 0)
                    if zero_returns > 0:
                        print(f"⚠️ 发现 {zero_returns} 个零收益率数据点")
                
                # 添加零线
                ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, 
                           label='基准线', linewidth=1)
                
                # 绘制基准曲线
                if benchmark_dates is not None and benchmark_curve is not None and len(benchmark_dates) > 0:
                    print(f"benchmark_dates长度: {len(benchmark_dates)}")
                    print(f"benchmark_curve长度: {len(benchmark_curve)}")
                    
                    # 调整基准数据时间，跳过预热期
                    adjusted_benchmark_dates, adjusted_benchmark_curve = adjust_benchmark_timing(
                        benchmark_dates, benchmark_curve, skip_warmup_days
                    )
                    
                    # 计算基准收益率（百分比）- 确保从0开始
                    # 将基准收益率重新计算，以调整后的第一个值为基准
                    if len(adjusted_benchmark_curve) > 0:
                        # 以调整后的第一个值为基准，计算相对收益率
                        base_value = adjusted_benchmark_curve[0]
                        benchmark_returns_pct = ((np.array(adjusted_benchmark_curve) / base_value) - 1) * 100
                        
                        print(f"基准收益率计算:")
                        print(f"基准值: {base_value:.4f}")
                        print(f"第一个收益率: {benchmark_returns_pct[0]:.2f}%")
                        print(f"最后一个收益率: {benchmark_returns_pct[-1]:.2f}%")
                    else:
                        benchmark_returns_pct = np.array([])
                    
                    ax1.plot(adjusted_benchmark_dates, benchmark_returns_pct, label=f'{benchmark_name}收益率', color='red', linewidth=2)
                    
                    # 计算并绘制超额收益曲线
                    if len(portfolio_values) > 0 and len(adjusted_benchmark_curve) > 0:
                        # 数据对齐：找到共同的日期范围
                        strategy_dates_set = set(dates)
                        benchmark_dates_set = set(adjusted_benchmark_dates)
                        common_dates = sorted(strategy_dates_set.intersection(benchmark_dates_set))
                        
                        print(f"策略日期范围: {min(dates)} 到 {max(dates)}")
                        print(f"调整后基准日期范围: {min(adjusted_benchmark_dates)} 到 {max(adjusted_benchmark_dates)}")
                        print(f"共同日期数量: {len(common_dates)}")
                        
                        if len(common_dates) > 0:
                            # 创建日期到索引的映射
                            strategy_date_to_idx = {date: idx for idx, date in enumerate(dates)}
                            benchmark_date_to_idx = {date: idx for idx, date in enumerate(adjusted_benchmark_dates)}
                            
                            # 提取对齐的数据
                            aligned_strategy_returns = []
                            aligned_benchmark_returns = []
                            aligned_dates = []
                            
                            for common_date in common_dates:
                                if common_date in strategy_date_to_idx and common_date in benchmark_date_to_idx:
                                    strategy_idx = strategy_date_to_idx[common_date]
                                    benchmark_idx = benchmark_date_to_idx[common_date]
                                    
                                    if strategy_idx < len(strategy_returns_pct) and benchmark_idx < len(benchmark_returns_pct):
                                        aligned_strategy_returns.append(strategy_returns_pct[strategy_idx])
                                        aligned_benchmark_returns.append(benchmark_returns_pct[benchmark_idx])
                                        aligned_dates.append(common_date)
                            
                            if len(aligned_strategy_returns) > 0:
                                # 计算超额收益（百分比）
                                excess_returns_pct = np.array(aligned_strategy_returns) - np.array(aligned_benchmark_returns)
                                
                                # 绘制超额收益曲线
                                ax1.plot(aligned_dates, excess_returns_pct, label='超额收益', color='green', linewidth=2, alpha=0.8)
                                
                                print(f"超额收益曲线数据点: {len(aligned_dates)}")
                            else:
                                print("⚠️ 没有找到对齐的数据点，跳过超额收益曲线绘制")
                        
                        # 计算策略相对基准的表现（使用最终值）
                        strategy_return = (portfolio_values[-1] - initial_cash) / initial_cash
                        # 使用调整后的基准收益率计算
                        if len(adjusted_benchmark_curve) > 0:
                            benchmark_return = (adjusted_benchmark_curve[-1] / adjusted_benchmark_curve[0]) - 1
                        else:
                            benchmark_return = 0
                        excess_return = strategy_return - benchmark_return
                        
                        print(f"\n策略与基准对比:")
                        print(f"策略总收益率: {strategy_return:.2%}")
                        print(f"{benchmark_name}收益率: {benchmark_return:.2%}")
                        print(f"超额收益率: {excess_return:.2%}")
                
                ax1.set_title('策略收益率分析', fontsize=16, fontweight='bold')
                ax1.set_ylabel('收益率（%）', fontsize=12)
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # 格式化x轴日期
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
                
                # 中图：仓位柱状图
                # 计算仓位 = 总资产 - 现金
                positions = np.array(portfolio_values) - np.array(cash_values)
                
                # 绘制仓位柱状图
                ax2.bar(dates, positions, label='仓位', color='orange', alpha=0.7, width=0.8)
                
                # 添加零线
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.5)
                
                ax2.set_title('仓位变化', fontsize=14, fontweight='bold')
                ax2.set_ylabel('仓位（元）', fontsize=12)
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # 格式化x轴日期
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
                
                # 下图：每日盈亏柱状图
                # 计算每日盈亏
                daily_pnl = np.diff(portfolio_values)
                daily_pnl_dates = dates[1:]  # 从第二天开始，因为第一天没有盈亏
                
                # 绘制每日盈亏柱状图
                colors = ['green' if pnl >= 0 else 'red' for pnl in daily_pnl]
                ax3.bar(daily_pnl_dates, daily_pnl, color=colors, alpha=0.7, width=0.8, label='每日盈亏')
                
                # 添加零线
                ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.5)
                
                ax3.set_title('每日盈亏', fontsize=14, fontweight='bold')
                ax3.set_xlabel('日期', fontsize=12)
                ax3.set_ylabel('盈亏（元）', fontsize=12)
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                
                # 格式化x轴日期
                ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
                plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
                
                plt.tight_layout()
                
                # 保存图片
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    print(f"图片已保存到: {save_path}")
                
                plt.show()
                
                # 打印统计信息
                if portfolio_values:
                    final_value = portfolio_values[-1]
                    final_cash = cash_values[-1]
                    final_position = final_value - final_cash
                    max_position = max(positions)
                    min_position = min(positions)
                    
                    # 计算每日盈亏统计
                    if len(daily_pnl) > 0:
                        total_pnl = np.sum(daily_pnl)
                        positive_days = np.sum(daily_pnl > 0)
                        negative_days = np.sum(daily_pnl < 0)
                        zero_days = np.sum(daily_pnl == 0)
                        max_daily_gain = np.max(daily_pnl)
                        max_daily_loss = np.min(daily_pnl)
                        avg_daily_pnl = np.mean(daily_pnl)
                    
                    print(f"\n策略分析统计:")
                    print(f"初始资金: {initial_cash:,.2f}")
                    print(f"最终总资产: {final_value:,.2f}")
                    print(f"最终现金: {final_cash:,.2f}")
                    print(f"最终仓位: {final_position:,.2f}")
                    print(f"最大仓位: {max_position:,.2f}")
                    print(f"最小仓位: {min_position:,.2f}")
                    print(f"总收益率: {(final_value - initial_cash) / initial_cash:.2%}")
                    print(f"数据点数量: {len(dates)}")
                        
                    print(f"\n每日盈亏统计:")
                    print(f"总盈亏: {total_pnl:,.2f}")
                    print(f"盈利天数: {positive_days}")
                    print(f"亏损天数: {negative_days}")
                    print(f"持平天数: {zero_days}")
                    print(f"最大单日盈利: {max_daily_gain:,.2f}")
                    print(f"最大单日亏损: {max_daily_loss:,.2f}")
                    print(f"平均每日盈亏: {avg_daily_pnl:,.2f}")
                    print(f"胜率: {positive_days/(positive_days+negative_days)*100:.1f}%" if (positive_days+negative_days) > 0 else "胜率: 0%")
                
                print("✅ 策略分析图表绘制完成")
            else:
                print("❌ 没有收集到有效的资金数据")
                
        else:
            print("❌ 没有找到手动记录的资金数据")
            
    except Exception as e:
        print(f"策略分析图表绘制出错: {e}")
        import traceback
        traceback.print_exc()


def plot_comparison(strategy_returns: List[float], strategy_dates: List, 
                   benchmark_returns: List[float], benchmark_dates: List,
                   strategy_name: str = '策略', benchmark_name: str = '沪深300基准',
                   initial_cash: float = 100000, save_path: Optional[str] = None):
    """
    绘制策略与基准的收益率对比图
    
    Args:
        strategy_returns: 策略收益率序列
        strategy_dates: 策略日期序列
        benchmark_returns: 基准收益率序列
        benchmark_dates: 基准日期序列
        strategy_name: 策略名称
        benchmark_name: 基准名称
        initial_cash: 初始资金
        save_path: 图片保存路径
    """
    try:
        plt.figure(figsize=(14, 6))
        
        # 计算累计收益率（转换为百分比）
        strategy_cumulative = (np.array(strategy_returns) - 1) * 100  # 转换为百分比
        benchmark_cumulative = (np.array(benchmark_returns) - 1) * 100  # 转换为百分比
        
        # 绘制策略和基准曲线
        plt.plot(strategy_dates, strategy_cumulative, label=strategy_name, color='blue', linewidth=2)
        plt.plot(benchmark_dates, benchmark_cumulative, label=benchmark_name, color='red', linewidth=2)
        
        # 添加零线
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7, 
                   label='基准线', linewidth=1)
        
        plt.title('策略与基准收益率对比', fontsize=16, fontweight='bold')
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('收益率（%）', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 格式化x轴日期
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # 保存图片
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图片已保存到: {save_path}")
        
        plt.show()
        
        # 计算对比指标
        if len(strategy_returns) > 0 and len(benchmark_returns) > 0:
            strategy_total_return = strategy_returns[-1] - 1
            benchmark_total_return = benchmark_returns[-1] - 1
            excess_return = strategy_total_return - benchmark_total_return
            
            print(f"\n策略与基准对比:")
            print(f"{strategy_name}总收益率: {strategy_total_return:.2%}")
            print(f"{benchmark_name}总收益率: {benchmark_total_return:.2%}")
            print(f"超额收益率: {excess_return:.2%}")
        
        print("✅ 收益率对比图绘制完成")
        
    except Exception as e:
        print(f"收益率对比图绘制出错: {e}")
        import traceback
        traceback.print_exc()


def calculate_benchmark_metrics(strategy_returns: List[float], benchmark_returns: List[float],
                              strategy_name: str = '策略', benchmark_name: str = '沪深300基准'):
    """
    计算策略相对于基准的绩效指标
    
    Args:
        strategy_returns: 策略收益率序列
        benchmark_returns: 基准收益率序列
        strategy_name: 策略名称
        benchmark_name: 基准名称
        
    Returns:
        dict: 包含各项指标的字典
    """
    try:
        if len(strategy_returns) < 2 or len(benchmark_returns) < 2:
            print("❌ 数据点不足，无法计算指标")
            return {}
        
        # 转换为numpy数组
        strategy_returns = np.array(strategy_returns)
        benchmark_returns = np.array(benchmark_returns)
        
        # 计算日收益率
        strategy_daily_returns = np.diff(strategy_returns) / strategy_returns[:-1]
        benchmark_daily_returns = np.diff(benchmark_returns) / benchmark_returns[:-1]
        
        # 计算超额收益
        excess_returns = strategy_daily_returns - benchmark_daily_returns
        
        # 计算各项指标
        metrics = {
            'strategy_total_return': strategy_returns[-1] - 1,
            'benchmark_total_return': benchmark_returns[-1] - 1,
            'excess_return': strategy_returns[-1] - benchmark_returns[-1],
            'strategy_volatility': np.std(strategy_daily_returns) * np.sqrt(252),
            'benchmark_volatility': np.std(benchmark_daily_returns) * np.sqrt(252),
            'excess_volatility': np.std(excess_returns) * np.sqrt(252),
            'information_ratio': np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0,
            'tracking_error': np.std(excess_returns) * np.sqrt(252),
            'beta': np.cov(strategy_daily_returns, benchmark_daily_returns)[0, 1] / np.var(benchmark_daily_returns) if np.var(benchmark_daily_returns) > 0 else 0,
            'alpha': np.mean(excess_returns) * 252,
            'correlation': np.corrcoef(strategy_daily_returns, benchmark_daily_returns)[0, 1]
        }
        
        # 打印指标
        print(f"\n{'='*50}")
        print(f"{strategy_name} vs {benchmark_name} 绩效分析")
        print(f"{'='*50}")
        print(f"{strategy_name}总收益率: {metrics['strategy_total_return']:.2%}")
        print(f"{benchmark_name}总收益率: {metrics['benchmark_total_return']:.2%}")
        print(f"超额收益率: {metrics['excess_return']:.2%}")
        print(f"{strategy_name}年化波动率: {metrics['strategy_volatility']:.2%}")
        print(f"{benchmark_name}年化波动率: {metrics['benchmark_volatility']:.2%}")
        print(f"信息比率: {metrics['information_ratio']:.4f}")
        print(f"跟踪误差: {metrics['tracking_error']:.2%}")
        print(f"Beta系数: {metrics['beta']:.4f}")
        print(f"Alpha系数: {metrics['alpha']:.2%}")
        print(f"相关系数: {metrics['correlation']:.4f}")
        print(f"{'='*50}")
        
        return metrics
        
    except Exception as e:
        print(f"计算基准指标时出错: {e}")
        return {}


# if __name__ == "__main__":
#     # 测试函数
#     print("测试基准数据获取...")
#     dates, returns = get_hs300_index_data('2023-01-01', '2023-01-31')
#     print(f"获取到 {len(dates)} 个数据点")
#     if len(dates) > 0:
#         print(f"首日: {dates[0]}, 收益率: {returns[0]:.4f}")
#         print(f"末日: {dates[-1]}, 收益率: {returns[-1]:.4f}") 