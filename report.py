"""
简单的回测报告生成模块
包含绩效分析报告、分析图表和每日持仓买卖情况
"""

import os
import json
import base64
import io
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)


class SimpleBacktestReport:
    """
    简单回测报告生成器
    """
    
    def __init__(self, strategy, initial_cash: float, benchmark_dates=None, benchmark_curve=None, 
                 benchmark_name='沪深300基准', skip_warmup_days=20):
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.benchmark_dates = benchmark_dates
        self.benchmark_curve = benchmark_curve
        self.benchmark_name = benchmark_name
        self.skip_warmup_days = skip_warmup_days
        
        # 获取策略数据
        self.dates = strategy.dates if hasattr(strategy, 'dates') else []
        self.portfolio_values = strategy.portfolio_values if hasattr(strategy, 'portfolio_values') else []
        self.cash_values = strategy.cash_values if hasattr(strategy, 'cash_values') else []
        
        # 分析器结果
        self.analyzer_results = {}
        if hasattr(strategy, 'analyzers'):
            self._extract_analyzer_results()
    
    def _extract_analyzer_results(self):
        """提取分析器结果"""
        try:
            analyzers = self.strategy.analyzers
            for attr_name in dir(analyzers):
                if not attr_name.startswith('_'):
                    try:
                        analyzer = getattr(analyzers, attr_name)
                        if hasattr(analyzer, 'get_analysis'):
                            analysis_result = analyzer.get_analysis()
                            if analysis_result:
                                self.analyzer_results[attr_name] = analysis_result
                    except Exception as e:
                        logger.warning(f"提取分析器 {attr_name} 结果失败: {e}")
        except Exception as e:
            logger.error(f"提取分析器结果失败: {e}")
    
    def _adjust_benchmark_timing(self, benchmark_dates, benchmark_curve, skip_warmup_days):
        """调整基准数据时间，跳过预热期"""
        if benchmark_dates is None or benchmark_curve is None or len(benchmark_dates) == 0:
            return [], []
        
        # 跳过预热期的交易日
        if skip_warmup_days > 0 and len(benchmark_dates) > skip_warmup_days:
            adjusted_dates = benchmark_dates[skip_warmup_days:]
            adjusted_curve = benchmark_curve[skip_warmup_days:]
            return adjusted_dates, adjusted_curve
        
        return benchmark_dates, benchmark_curve
    
    def _create_performance_chart(self) -> str:
        """创建性能图表并返回base64编码"""
        if len(self.dates) == 0:
            return ""
        
        # 创建图表
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), height_ratios=[3, 1, 1])
        
        # 上图：收益率曲线
        strategy_returns_pct = (np.array(self.portfolio_values) / self.initial_cash - 1) * 100
        ax1.plot(self.dates, strategy_returns_pct, label='策略收益率', color='blue', linewidth=2)
        
        # 添加零线
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='基准线', linewidth=1)
        
        # 绘制基准曲线
        if self.benchmark_dates is not None and self.benchmark_curve is not None and len(self.benchmark_dates) > 0:
            adjusted_benchmark_dates, adjusted_benchmark_curve = self._adjust_benchmark_timing(
                self.benchmark_dates, self.benchmark_curve, self.skip_warmup_days
            )
            
            if len(adjusted_benchmark_curve) > 0:
                base_value = adjusted_benchmark_curve[0]
                benchmark_returns_pct = ((np.array(adjusted_benchmark_curve) / base_value) - 1) * 100
                ax1.plot(adjusted_benchmark_dates, benchmark_returns_pct, 
                        label=f'{self.benchmark_name}收益率', color='red', linewidth=2)
                
                # 计算超额收益
                if len(self.portfolio_values) > 0:
                    strategy_dates_set = set(self.dates)
                    benchmark_dates_set = set(adjusted_benchmark_dates)
                    common_dates = sorted(strategy_dates_set.intersection(benchmark_dates_set))
                    
                    if len(common_dates) > 0:
                        strategy_date_to_idx = {date: idx for idx, date in enumerate(self.dates)}
                        benchmark_date_to_idx = {date: idx for idx, date in enumerate(adjusted_benchmark_dates)}
                        
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
                            excess_returns_pct = np.array(aligned_strategy_returns) - np.array(aligned_benchmark_returns)
                            ax1.plot(aligned_dates, excess_returns_pct, label='超额收益', 
                                   color='green', linewidth=2, alpha=0.8)
        
        ax1.set_title('策略收益率分析', fontsize=16, fontweight='bold')
        ax1.set_ylabel('收益率（%）', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 中图：仓位柱状图
        positions = np.array(self.portfolio_values) - np.array(self.cash_values)
        ax2.bar(self.dates, positions, label='仓位', color='orange', alpha=0.7, width=0.8)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.5)
        ax2.set_title('仓位变化', fontsize=14, fontweight='bold')
        ax2.set_ylabel('仓位（元）', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # 下图：每日盈亏柱状图
        daily_pnl = np.diff(self.portfolio_values)
        daily_pnl_dates = self.dates[1:]
        colors = ['green' if pnl >= 0 else 'red' for pnl in daily_pnl]
        ax3.bar(daily_pnl_dates, daily_pnl, color=colors, alpha=0.7, width=0.8, label='每日盈亏')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.5)
        ax3.set_title('每日盈亏', fontsize=14, fontweight='bold')
        ax3.set_xlabel('日期', fontsize=12)
        ax3.set_ylabel('盈亏（元）', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # 转换为base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def _calculate_statistics(self) -> Dict:
        """计算统计数据"""
        stats = {}
        
        # 添加调试信息
        print(f"\n=== 报告统计计算调试 ===")
        print(f"分析器结果: {self.analyzer_results}")
        
        if len(self.portfolio_values) > 0:
            final_value = self.portfolio_values[-1]
            final_cash = self.cash_values[-1]
            final_position = final_value - final_cash
            positions = np.array(self.portfolio_values) - np.array(self.cash_values)
            
            # 基础统计
            stats['initial_cash'] = self.initial_cash
            stats['final_value'] = final_value
            stats['final_cash'] = final_cash
            stats['final_position'] = final_position
            stats['max_position'] = max(positions)
            stats['min_position'] = min(positions)
            stats['data_points'] = len(self.dates)
            
            # 每日盈亏统计
            if len(self.portfolio_values) > 1:
                daily_pnl = np.diff(self.portfolio_values)
                stats['total_pnl'] = np.sum(daily_pnl)
                stats['positive_days'] = np.sum(daily_pnl > 0)
                stats['negative_days'] = np.sum(daily_pnl < 0)
                stats['zero_days'] = np.sum(daily_pnl == 0)
                stats['max_daily_gain'] = np.max(daily_pnl)
                stats['max_daily_loss'] = np.min(daily_pnl)
                stats['avg_daily_pnl'] = np.mean(daily_pnl)
                stats['win_rate'] = (stats['positive_days'] / (stats['positive_days'] + stats['negative_days']) * 100) if (stats['positive_days'] + stats['negative_days']) > 0 else 0
        
        # 使用分析器结果中的实际值
        if 'returns' in self.analyzer_results:
            returns_data = self.analyzer_results['returns']
            # 使用分析器计算的总收益率（rtot是小数形式）
            stats['total_return'] = returns_data.get('rtot', 0)
            stats['annual_return'] = returns_data.get('rnorm100', 0)
            print(f"从分析器获取总收益率: {stats['total_return']:.4f}")
            print(f"从分析器获取年化收益率: {stats['annual_return']:.2f}%")
        
        if 'sharpe' in self.analyzer_results:
            sharpe_data = self.analyzer_results['sharpe']
            stats['sharpe_ratio'] = sharpe_data.get('sharperatio', 0)
            print(f"从分析器获取夏普比率: {stats['sharpe_ratio']:.4f}")
            print(f"夏普比率原始数据: {sharpe_data}")
            
            # 解释夏普比率
            if stats['sharpe_ratio'] > 0:
                print(f"✅ 夏普比率为正，策略风险调整后收益良好")
            elif stats['sharpe_ratio'] < 0:
                print(f"⚠️ 夏普比率为负，策略风险调整后收益较差")
            else:
                print(f"ℹ️ 夏普比率为零，策略风险调整后收益与无风险利率相当")
        
        if 'drawdown' in self.analyzer_results:
            drawdown_data = self.analyzer_results['drawdown']
            max_drawdown = drawdown_data.get('max', {}).get('drawdown', 0)
            max_drawdown_len = drawdown_data.get('max', {}).get('len', 0)
            stats['max_drawdown'] = max_drawdown
            stats['max_drawdown_len'] = max_drawdown_len
            print(f"从分析器获取最大回撤原始值: {max_drawdown}")
            print(f"从分析器获取最大回撤长度: {max_drawdown_len}")
            
            # 处理最大回撤的显示格式
            if abs(max_drawdown) > 1:  # 如果值大于1，说明已经是百分比形式
                stats['max_drawdown_display'] = f"{max_drawdown:.2f}%"
                print(f"最大回撤已经是百分比形式: {max_drawdown:.2f}%")
            else:  # 如果值小于1，说明是小数形式
                stats['max_drawdown_display'] = f"{max_drawdown:.2%}"
                print(f"最大回撤是小数形式: {max_drawdown:.2%}")
        
        if 'trades' in self.analyzer_results:
            trades_data = self.analyzer_results['trades']
            stats['total_trades'] = trades_data.get('total', {}).get('total', 0)
            stats['won_trades'] = trades_data.get('won', {}).get('total', 0)
            stats['lost_trades'] = trades_data.get('lost', {}).get('total', 0)
            
            if stats['total_trades'] > 0:
                stats['trade_win_rate'] = (stats['won_trades'] / stats['total_trades']) * 100
                
                if 'won' in trades_data and 'pnl' in trades_data['won']:
                    stats['avg_win'] = trades_data['won']['pnl'].get('average', 0)
                    stats['total_win'] = trades_data['won']['pnl'].get('total', 0)
                
                if 'lost' in trades_data and 'pnl' in trades_data['lost']:
                    stats['avg_loss'] = trades_data['lost']['pnl'].get('average', 0)
                    stats['total_loss'] = trades_data['lost']['pnl'].get('total', 0)
                
                if stats.get('total_loss', 0) != 0:
                    stats['profit_loss_ratio'] = abs(stats.get('total_win', 0) / stats['total_loss'])
            
            print(f"从分析器获取交易统计: 总交易{stats['total_trades']}, 盈利{stats['won_trades']}, 亏损{stats['lost_trades']}")
        
        print(f"=== 报告统计计算完成 ===\n")
        return stats
    
    def generate_html_report(self, save_path: str = 'Choose/backtest_report.html', auto_open: bool = True):
        """生成HTML报告"""
        # 计算统计数据
        stats = self._calculate_statistics()
        
        # 创建图表
        chart_base64 = self._create_performance_chart()
        
        # 生成HTML内容
        html_content = self._generate_html_content(stats, chart_base64)
        
        # 保存HTML文件
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML报告已保存到: {save_path}")
        
        # 自动打开HTML报告
        if auto_open:
            try:
                import webbrowser
                import os
                # 获取绝对路径
                abs_report_path = os.path.abspath(save_path)
                # 转换为文件URL格式
                file_url = f"file:///{abs_report_path.replace(os.sep, '/')}"
                print(f"\n正在打开HTML报告...")
                print(f"报告路径: {abs_report_path}")
                webbrowser.open(file_url)
                print("✅ HTML报告已在默认浏览器中打开")
            except Exception as e:
                logger.warning(f"自动打开报告失败: {e}")
                print(f"请手动打开报告文件: {save_path}")
        
        return save_path
    
    def _generate_html_content(self, stats: Dict, chart_base64: str) -> str:
        """生成HTML内容"""
        html_template = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>回测分析报告</title>
    <style>
        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header p {{
            margin: 10px 0 0 0;
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .content {{
            padding: 30px;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section h2 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            transition: transform 0.2s;
        }}
        .stat-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }}
        .stat-label {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        .positive {{
            color: #27ae60;
        }}
        .negative {{
            color: #e74c3c;
        }}
        .neutral {{
            color: #3498db;
        }}
        .chart-container {{
            text-align: center;
            margin: 30px 0;
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        .table-container {{
            overflow-x: auto;
            margin: 20px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: 500;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .footer {{
            background-color: #2c3e50;
            color: white;
            text-align: center;
            padding: 20px;
            margin-top: 40px;
        }}
        @media (max-width: 768px) {{
            .stats-grid {{
                grid-template-columns: 1fr;
            }}
            .header h1 {{
                font-size: 2em;
            }}
            .content {{
                padding: 20px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 回测分析报告</h1>
            <p>策略性能分析与可视化展示</p>
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="content">
            <!-- 基础统计 -->
            <div class="section">
                <h2>📈 基础统计</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value positive">¥{stats.get('initial_cash', 0):,.0f}</div>
                        <div class="stat-label">初始资金</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value positive">¥{stats.get('final_value', 0):,.0f}</div>
                        <div class="stat-label">最终资产</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value {'positive' if stats.get('total_return', 0) >= 0 else 'negative'}">{stats.get('total_return', 0):.2%}</div>
                        <div class="stat-label">总收益率</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value neutral">{stats.get('data_points', 0)}</div>
                        <div class="stat-label">数据点数量</div>
                    </div>
                </div>
            </div>
            
            <!-- 风险指标 -->
            <div class="section">
                <h2>⚠️ 风险指标</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value negative">{stats.get('max_drawdown_display', '0.00%')}</div>
                        <div class="stat-label">最大回撤</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value neutral">{stats.get('max_drawdown_len', 0)}</div>
                        <div class="stat-label">最大回撤天数</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value {'positive' if stats.get('sharpe_ratio', 0) > 0 else 'negative'}">{stats.get('sharpe_ratio', 0):.4f}</div>
                        <div class="stat-label">夏普比率</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value neutral">{stats.get('annual_return', 0):.2f}%</div>
                        <div class="stat-label">年化收益率</div>
                    </div>
                </div>
            </div>
            
            <!-- 交易统计 -->
            <div class="section">
                <h2>💰 交易统计</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value neutral">{stats.get('total_trades', 0)}</div>
                        <div class="stat-label">总交易次数</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value positive">{stats.get('won_trades', 0)}</div>
                        <div class="stat-label">盈利交易</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value negative">{stats.get('lost_trades', 0)}</div>
                        <div class="stat-label">亏损交易</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value {'positive' if stats.get('trade_win_rate', 0) > 50 else 'negative'}">{stats.get('trade_win_rate', 0):.1f}%</div>
                        <div class="stat-label">交易胜率</div>
                    </div>
                </div>
            </div>
            
            <!-- 每日盈亏统计 -->
            <div class="section">
                <h2>📊 每日盈亏统计</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value {'positive' if stats.get('total_pnl', 0) >= 0 else 'negative'}">¥{stats.get('total_pnl', 0):,.0f}</div>
                        <div class="stat-label">总盈亏</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value positive">{stats.get('positive_days', 0)}</div>
                        <div class="stat-label">盈利天数</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value negative">{stats.get('negative_days', 0)}</div>
                        <div class="stat-label">亏损天数</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value {'positive' if stats.get('win_rate', 0) > 50 else 'negative'}">{stats.get('win_rate', 0):.1f}%</div>
                        <div class="stat-label">日胜率</div>
                    </div>
                </div>
                
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>指标</th>
                                <th>数值</th>
                                <th>说明</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>最大单日盈利</td>
                                <td class="positive">¥{stats.get('max_daily_gain', 0):,.0f}</td>
                                <td>单日最大盈利金额</td>
                            </tr>
                            <tr>
                                <td>最大单日亏损</td>
                                <td class="negative">¥{stats.get('max_daily_loss', 0):,.0f}</td>
                                <td>单日最大亏损金额</td>
                            </tr>
                            <tr>
                                <td>平均每日盈亏</td>
                                <td class="{'positive' if stats.get('avg_daily_pnl', 0) >= 0 else 'negative'}">¥{stats.get('avg_daily_pnl', 0):,.0f}</td>
                                <td>每日平均盈亏金额</td>
                            </tr>
                            <tr>
                                <td>持平天数</td>
                                <td class="neutral">{stats.get('zero_days', 0)}</td>
                                <td>无盈亏的交易日数量</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
            
            <!-- 图表展示 -->
            <div class="section">
                <h2>📈 策略表现图表</h2>
                <div class="chart-container">
                    <img src="data:image/png;base64,{chart_base64}" alt="策略表现图表">
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>© 2024 回测分析报告 | 基于Backtrader框架生成</p>
        </div>
    </div>
</body>
</html>
        """
        return html_template


def generate_backtest_report(strategy, initial_cash: float, analyzer_results: Dict = None, 
                           benchmark_dates=None, benchmark_curve=None, 
                           benchmark_name='沪深300基准', skip_warmup_days=20, 
                           save_path='Choose/backtest_report.html', auto_open: bool = True):
    """
    生成回测报告的便捷函数
    
    Args:
        strategy: 策略对象
        initial_cash: 初始资金
        analyzer_results: 分析器结果字典
        benchmark_dates: 基准日期列表
        benchmark_curve: 基准曲线数据
        benchmark_name: 基准名称
        skip_warmup_days: 跳过的预热期天数
        save_path: 保存路径
        auto_open: 是否自动打开HTML报告
    
    Returns:
        str: 保存的文件路径
    """
    report = SimpleBacktestReport(
        strategy=strategy,
        initial_cash=initial_cash,
        benchmark_dates=benchmark_dates,
        benchmark_curve=benchmark_curve,
        benchmark_name=benchmark_name,
        skip_warmup_days=skip_warmup_days
    )
    
    # 如果提供了分析器结果，使用它们
    if analyzer_results:
        report.analyzer_results = analyzer_results
    
    return report.generate_html_report(save_path, auto_open)


if __name__ == "__main__":
    # 测试代码
    print("回测报告生成模块已加载")
    print("使用方法:")
    print("from report import generate_backtest_report")
    print("generate_backtest_report(strategy, initial_cash, benchmark_dates, benchmark_curve)") 