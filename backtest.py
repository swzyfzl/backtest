"""
使用Backtrader构建的完整回测框架
包含数据层、策略逻辑层、执行层、风险控制层和评估层
策略使用的框架有20个交易日的"预热期"来进行技术指标的计算,指定时间时应往前推20个交易日
"""

import backtrader as bt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta
from xtquant import xtdata
import time
import os
import json
from typing import Dict, List, Optional, Tuple
import logging

# 导入基准数据模块
from plt_benchmark import get_hs300_index_data, get_hs300_etf_data, get_shanghai_index_data, plot_strategy_analysis

# 导入报告生成模块
from report import generate_backtest_report

print('====================所需的包已经导入完毕')
print(f'backtrader源码路径{bt.__path__}')
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class XTDataFeed(bt.feeds.PandasData):
    """
    自定义数据源，适配xtquant数据格式
    """
    params = (
        ('datetime', None),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', None),
    )

#初始化数据管理器
class DataManager:
    """
    数据层：历史行情数据、基本面数据、事件数据的获取与清洗
    """

    def __init__(self):
        self.stock_list = None
        self.data_cache = {}

    def get_index_stock_list(self, index_code='000300.SH'):
        """获取指定指数的成分股列表"""
        try:
            # 下载指数权重数据
            xtdata.download_index_weight()
            # 获取指定指数的权重数据
            data_weight = xtdata.get_index_weight(index_code)
            # 获取成分股列表
            self.stock_list = list(data_weight.keys())
            logger.info(f"获取到 {index_code} 的成分股数量: {len(self.stock_list)}")
            return self.stock_list
        except Exception as e:
            logger.error(f"获取指数成分股失败: {e}")
            return []

    def download_stock_data(self, stock_codes: List[str], period="1d", start_date=None, end_date=None):
        """下载股票历史数据"""
        if not stock_codes:
            logger.warning("股票代码列表为空")
            return {}

        data_dict = {}

        for stock_code in stock_codes:
            try:
                # 下载历史数据
                xtdata.download_history_data(
                    stock_code,
                    period=period,
                    start_time=start_date,
                    end_time=end_date,
                    incrementally=True
                )

                # 获取数据
                market_data = xtdata.get_market_data_ex(
                    [], [stock_code], period=period, count=-1
                )

                if stock_code in market_data:
                    stock_data = market_data[stock_code]

                    # 转换为DataFrame
                    if isinstance(stock_data, dict):
                        df = pd.DataFrame(stock_data)
                    else:
                        df = stock_data

                    # 添加调试信息
                    logger.info(f"{stock_code} 原始数据列: {list(df.columns)}")
                    if 'time' in df.columns:
                        logger.info(f"{stock_code} time列前5个值: {df['time'].head().tolist()}")
                        logger.info(f"{stock_code} 原始数据时间范围: {df['time'].min()} 到 {df['time'].max()}")

                    # 数据清洗
                    df = self._clean_data(df)

                    # 根据指定时间范围过滤数据
                    if not df.empty and start_date and end_date:
                        start_dt = pd.to_datetime(start_date)
                        end_dt = pd.to_datetime(end_date)
                        df = df[(df['datetime'] >= start_dt) & (df['datetime'] <= end_dt)]
                        logger.info(f"{stock_code} 过滤后数据时间范围: {df['datetime'].min()} 到 {df['datetime'].max()}")

                    if not df.empty:
                        data_dict[stock_code] = df
                        logger.info(f"成功获取 {stock_code} 数据，共 {len(df)} 条记录")
                    else:
                        logger.warning(f"{stock_code} 数据为空")

            except Exception as e:
                logger.error(f"获取 {stock_code} 数据失败: {e}")

        return data_dict

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据清洗"""
        if df.empty:
            return df

        # 删除重复数据
        df = df.drop_duplicates()

        # 处理缺失值
        df = df.dropna(subset=['close', 'volume'])

        # 确保数据类型正确
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 删除异常值（价格或成交量为0或负数）
        df = df[
            (df['close'] > 0) &
            (df['volume'] > 0) &
            (df['high'] >= df['low']) &
            (df['high'] >= df['close']) &
            (df['high'] >= df['open']) &
            (df['low'] <= df['close']) &
            (df['low'] <= df['open'])
            ]

        # 处理日期列 - 检查多种可能的时间列名
        datetime_column = None
        for col_name in ['datetime', 'time', 'date', 'timestamp']:
            if col_name in df.columns:
                datetime_column = col_name
                break
        
        if datetime_column:
            logger.info(f"找到时间列: {datetime_column}")
            
            # 检查时间戳格式
            sample_time = df[datetime_column].iloc[0]
            logger.info(f"时间列样本值: {sample_time}, 类型: {type(sample_time)}")
            
            # 处理时间戳转换
            if pd.api.types.is_numeric_dtype(df[datetime_column]):
                # 如果是数值类型，可能是时间戳
                if sample_time > 1e10:  # 毫秒时间戳（13位）
                    logger.info("检测到毫秒时间戳，转换为datetime")
                    df['datetime'] = pd.to_datetime(df[datetime_column], unit='ms')
                elif sample_time > 1e8:  # 秒时间戳（10位）
                    logger.info("检测到秒时间戳，转换为datetime")
                    df['datetime'] = pd.to_datetime(df[datetime_column], unit='s')
                else:
                    logger.info("按默认方式转换时间戳")
                    df['datetime'] = pd.to_datetime(df[datetime_column])
            else:
                # 如果是字符串或其他类型，直接转换
                df['datetime'] = pd.to_datetime(df[datetime_column])
            
            # 删除日期转换失败的行
            df = df.dropna(subset=['datetime'])
            # 按时间排序
            df = df.sort_values('datetime')
            # 重置索引
            df = df.reset_index(drop=True)
            logger.info(f"时间列处理完成，数据范围: {df['datetime'].min()} 到 {df['datetime'].max()}")
        else:
            logger.warning(f"数据中没有找到时间列，可用列: {list(df.columns)}")
            return pd.DataFrame()  # 返回空DataFrame

        return df

    def get_fundamental_data(self, stock_codes: List[str], fields: List[str] = None):
        """获取基本面数据（示例）"""
        # 这里可以添加基本面数据获取逻辑
        # 例如：财务数据、估值指标等
        logger.info("基本面数据获取功能待实现")
        return {}

    def get_event_data(self, stock_codes: List[str], event_types: List[str] = None):
        """获取事件数据（示例）"""
        # 这里可以添加事件数据获取逻辑
        # 例如：公告、新闻、评级变化等
        logger.info("事件数据获取功能待实现")
        return {}

class RiskManager:
    """
    风险控制层：止损规则、仓位上限、风险指标监控
    """
    
    def __init__(self, max_position_pct=0.1, stop_loss_pct=0.05, max_drawdown_pct=0.2):
        self.max_position_pct = max_position_pct  # 单只股票最大仓位比例
        self.stop_loss_pct = stop_loss_pct  # 止损比例
        self.max_drawdown_pct = max_drawdown_pct  # 最大回撤限制
        self.var_confidence = 0.95  # VaR置信度
        
    def calculate_var(self, returns: pd.Series, confidence: float = None) -> float:
        """计算VaR (Value at Risk)"""
        if confidence is None:
            confidence = self.var_confidence
        return np.percentile(returns, (1 - confidence) * 100)
    
    def calculate_max_drawdown(self, equity_curve: pd.Series) -> Tuple[float, int, int]:
        """计算最大回撤"""
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_dd = drawdown.min()
        max_dd_idx = drawdown.idxmin()
        peak_idx = rolling_max.loc[:max_dd_idx].idxmax()
        return max_dd, peak_idx, max_dd_idx
    
    def check_position_limit(self, current_position: float, portfolio_value: float) -> bool:
        """检查仓位限制"""
        position_pct = current_position / portfolio_value
        return position_pct <= self.max_position_pct
    
    def check_stop_loss(self, entry_price: float, current_price: float) -> bool:
        """检查止损条件"""
        loss_pct = (entry_price - current_price) / entry_price
        return loss_pct >= self.stop_loss_pct


class TransactionCosts:
    """
    执行层：模拟交易滑点、手续费、流动性限制等交易成本
    """
    
    def __init__(self, commission_pct=0.0003, slippage_pct=0.0001, min_trade_amount=1000):
        self.commission_pct = commission_pct  # 手续费率
        self.slippage_pct = slippage_pct  # 滑点率
        self.min_trade_amount = min_trade_amount  # 最小交易金额
        
    def calculate_commission(self, trade_value: float) -> float:
        """计算手续费"""
        return trade_value * self.commission_pct
    
    def calculate_slippage(self, trade_value: float, is_buy: bool) -> float:
        """计算滑点成本"""
        slippage_cost = trade_value * self.slippage_pct
        return slippage_cost if is_buy else -slippage_cost
    
    def get_effective_price(self, price: float, is_buy: bool) -> float:
        """获取考虑滑点的有效价格"""
        if is_buy:
            return price * (1 + self.slippage_pct)
        else:
            return price * (1 - self.slippage_pct)


class MomentumStrategy(bt.Strategy):
    """
    策略逻辑层：动量策略示例
    包含信号生成、仓位计算、交易规则的算法实现
    记录资金数据用于绘图
    """
    
    params = (
        ('lookback_period', 20),  # 回看周期
        ('momentum_threshold', 0.05),  # 动量阈值
        ('position_size', 0.1),  # 仓位大小
        ('stop_loss', 0.05),  # 止损比例
        ('take_profit', 0.15),  # 止盈比例
    )
    
    def __init__(self):
        # 初始化技术指标
        self.momentum = {}
        self.sma = {}
        self.rsi = {}
        self.entry_prices = {}
        
        # 记录资金数据
        self.portfolio_values = []
        self.cash_values = []
        self.dates = []
        
        # 为每个数据源初始化指标
        for data in self.datas:
            # 动量指标
            self.momentum[data._name] = bt.indicators.MomentumOscillator(
                data, period=self.params.lookback_period
            )
            # 移动平均线
            self.sma[data._name] = bt.indicators.SimpleMovingAverage(
                data.close, period=self.params.lookback_period
            )
            # RSI指标
            self.rsi[data._name] = bt.indicators.RSI(
                data.close, period=14
            )
            
        # 风险管理和交易成本
        self.risk_manager = RiskManager()
        self.transaction_costs = TransactionCosts()
        
    def log(self, txt, dt=None):
        """日志记录"""
        dt = dt or self.datas[0].datetime.date(0)
        logger.info(f'{dt.isoformat()} {txt}')
        
    def next(self):
        """策略主逻辑"""
        # 手动记录当前资金数据
        current_date = self.datas[0].datetime.date(0)
        portfolio_value = self.broker.getvalue()
        cash_value = self.broker.getcash()
        
        self.dates.append(current_date)
        self.portfolio_values.append(portfolio_value)
        self.cash_values.append(cash_value)
        
        # 获取当前投资组合价值
        portfolio_value = self.broker.getvalue()
        
        for data in self.datas:
            symbol = data._name
            
            # 检查是否有足够的数据
            if len(data) < self.params.lookback_period:
                continue
                
            # 获取当前持仓
            position = self.getposition(data).size
            current_price = data.close[0]
            
            # 计算动量信号
            momentum_value = self.momentum[symbol][0]
            sma_value = self.sma[symbol][0]
            rsi_value = self.rsi[symbol][0]
            
            # 信号生成逻辑
            buy_signal = (
                momentum_value > self.params.momentum_threshold and
                current_price > sma_value and
                rsi_value < 70
            )
            
            sell_signal = (
                momentum_value < -self.params.momentum_threshold or
                current_price < sma_value or
                rsi_value > 80
            )
            
            # 止损止盈检查
            if position > 0 and symbol in self.entry_prices:
                entry_price = self.entry_prices[symbol]
                profit_pct = (current_price - entry_price) / entry_price
                
                # 止损
                if profit_pct <= -self.params.stop_loss:
                    sell_signal = True
                    self.log(f'止损触发: {symbol}, 亏损: {profit_pct:.2%}')
                
                # 止盈
                elif profit_pct >= self.params.take_profit:
                    sell_signal = True
                    self.log(f'止盈触发: {symbol}, 盈利: {profit_pct:.2%}')
            
            # 执行交易
            if buy_signal and position == 0:
                # 计算仓位大小
                position_value = portfolio_value * self.params.position_size
                shares = int(position_value / current_price)
                
                if shares > 0:
                    # 考虑交易成本的有效价格
                    effective_price = self.transaction_costs.get_effective_price(
                        current_price, is_buy=True
                    )
                    
                    # 检查仓位限制
                    if self.risk_manager.check_position_limit(
                        shares * effective_price, portfolio_value
                    ):
                        self.buy(data=data, size=shares)
                        self.entry_prices[symbol] = effective_price
                        self.log(f'买入 {symbol}: {shares}股, 价格: {effective_price:.2f}')
                        
            elif sell_signal and position > 0:
                # 考虑交易成本的有效价格
                effective_price = self.transaction_costs.get_effective_price(
                    current_price, is_buy=False
                )
                
                self.sell(data=data, size=position)
                if symbol in self.entry_prices:
                    del self.entry_prices[symbol]
                self.log(f'卖出 {symbol}: {position}股, 价格: {effective_price:.2f}')
    
    def notify_trade(self, trade):
        """交易通知 - 由BackTrader的Trades观测器自动处理"""
        if trade.isclosed:
            self.log(f'交易完成: {trade.data._name}, 盈亏: {trade.pnlcomm:.2f}')


class PerformanceAnalyzer:
    """
    评估层：使用BackTrader观测器数据计算绩效指标
    """
    
    def __init__(self, strategy):
        self.strategy = strategy
        self.results = {}
        
    def calculate_metrics(self, portfolio_value: float, risk_free_rate: float = 0.03):
        """使用BackTrader观测器数据计算绩效指标"""
        # 获取BackTrader分析器结果
        if not hasattr(self.strategy, 'analyzers'):
            logger.warning("没有找到BackTrader分析器数据")
            return {}
        
        analyzers = self.strategy.analyzers
        
        # 从分析器获取数据
        try:
            # 获取夏普比率
            sharpe_ratio = 0
            if hasattr(analyzers, 'sharpe'):
                sharpe_data = analyzers.sharpe.get_analysis()
                if sharpe_data and 'sharperatio' in sharpe_data:
                    sharpe_ratio = sharpe_data['sharperatio'] or 0
            
            # 获取回撤数据
            max_drawdown = 0
            max_drawdown_len = 0
            if hasattr(analyzers, 'drawdown'):
                drawdown_data = analyzers.drawdown.get_analysis()
                if drawdown_data and 'max' in drawdown_data:
                    max_drawdown = drawdown_data['max'].get('drawdown', 0) or 0
                    max_drawdown_len = drawdown_data['max'].get('len', 0) or 0
            
            # 获取收益率数据
            annual_return = 0
            total_return = 0
            if hasattr(analyzers, 'returns'):
                returns_data = analyzers.returns.get_analysis()
                if returns_data:
                    annual_return = returns_data.get('rnorm100', 0) or 0
                    total_return = returns_data.get('rtot', 0) or 0
            
            # 获取交易数据
            total_trades = 0
            won_trades = 0
            lost_trades = 0
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_loss_ratio = 0
            
            if hasattr(analyzers, 'trades'):
                trades_data = analyzers.trades.get_analysis()
                if trades_data:
                    total_trades = trades_data.get('total', {}).get('total', 0) or 0
                    won_trades = trades_data.get('won', {}).get('total', 0) or 0
                    lost_trades = trades_data.get('lost', {}).get('total', 0) or 0
                    
                    if total_trades > 0:
                        win_rate = (won_trades / total_trades) * 100
                        
                        # 获取平均盈亏
                        if 'won' in trades_data and 'pnl' in trades_data['won']:
                            avg_win = trades_data['won']['pnl'].get('average', 0) or 0
                        if 'lost' in trades_data and 'pnl' in trades_data['lost']:
                            avg_loss = trades_data['lost']['pnl'].get('average', 0) or 0
                        
                        # 计算盈亏比
                        if avg_loss != 0:
                            profit_loss_ratio = abs(avg_win / avg_loss)
            
            # 计算卡玛比率
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # 计算索提诺比率（简化版本）
            sortino_ratio = sharpe_ratio  # 简化处理，实际应该计算下行波动率
            
            # 计算VaR（简化版本）
            var_95 = max_drawdown * 0.5  # 简化处理
            
            self.results = {
                'total_return': total_return,  # rtot已经是小数形式，直接使用
                'annual_return': annual_return,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_loss_ratio': profit_loss_ratio,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'max_drawdown': max_drawdown,
                'max_drawdown_len': max_drawdown_len,
                'var_95': var_95,
            }
            
        except Exception as e:
            logger.error(f"计算绩效指标时出错: {e}")
            self.results = {}
        
        return self.results
    
    def print_metrics(self):
        """打印绩效指标"""
        if not self.results:
            logger.warning("请先调用 calculate_metrics 方法")
            return
            
        print("\n" + "="*50)
        print("回测绩效分析报告 (基于BackTrader观测器)")
        print("="*50)
        print(f"总收益率: {self.results['total_return']:.2%}")
        print(f"年化收益率: {self.results['annual_return']:.2f}%")
        print(f"总交易次数: {self.results['total_trades']}")
        print(f"胜率: {self.results['win_rate']:.2f}%")
        print(f"平均盈利: {self.results['avg_win']:.2f}")
        print(f"平均亏损: {self.results['avg_loss']:.2f}")
        print(f"盈亏比: {self.results['profit_loss_ratio']:.2f}")
        print(f"夏普比率: {self.results['sharpe_ratio']:.4f}")
        print(f"索提诺比率: {self.results['sortino_ratio']:.4f}")
        print(f"卡玛比率: {self.results['calmar_ratio']:.4f}")
        print(f"最大回撤: {self.results['max_drawdown']:.2f}%")
        print(f"最大回撤长度: {self.results['max_drawdown_len']} 天")
        print(f"VaR(95%): {self.results['var_95']:.2f}%")
        print("="*50)

#回测引擎初始化
class BacktestFramework:
    """
    主回测框架类
    """
    
    def __init__(self, initial_cash=100000):
        self.initial_cash = initial_cash
        self.cerebro = bt.Cerebro()
        self.strategy = None
        self.analyzer = None
        
    def setup_cerebro(self):
        """设置回测引擎"""
        # 设置初始资金
        self.cerebro.broker.setcash(self.initial_cash)
        
        # 设置手续费
        self.cerebro.broker.setcommission(commission=0.0003)
        
        # 设置滑点
        self.cerebro.broker.set_slippage_perc(0.0001)
        
        # 设置数据回放
        self.cerebro.runstrategy = True
        
        logger.info(f"回测引擎初始化完成，初始资金: {self.initial_cash:,.2f}")
    
    def add_data(self, data_dict: Dict[str, pd.DataFrame]):
        """添加数据源"""
        print(f"\n{'='*50}")
        print("数据源时间范围检查")
        print(f"{'='*50}")
        
        for symbol, df in data_dict.items():
            if not df.empty:
                # 确保数据格式正确
                df = df.copy()
                
                # 确保datetime列存在且格式正确
                if 'datetime' in df.columns:
                    # 确保datetime列是datetime格式
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    # 设置为索引
                    df.set_index('datetime', inplace=True)
                    # 确保索引是datetime格式
                    df.index = pd.to_datetime(df.index)
                    
                    # 打印数据时间范围
                    print(f"{symbol}: {df.index.min()} 到 {df.index.max()} (共{len(df)}条记录)")
                else:
                    logger.error(f"{symbol} 数据中没有datetime列，跳过")
                    continue
                
                # 创建数据源 - 不需要指定fromdate和todate，因为数据已经在下载时过滤
                data_feed = XTDataFeed(
                    dataname=df,
                    name=symbol
                )
                self.cerebro.adddata(data_feed)
                logger.info(f"添加数据源: {symbol}, 数据范围: {df.index.min()} 到 {df.index.max()}")
        
        print(f"{'='*50}")
    
    def add_strategy(self, strategy_class, **kwargs):
        """添加策略"""
        self.strategy = self.cerebro.addstrategy(strategy_class, **kwargs)
        logger.info(f"添加策略: {strategy_class.__name__}")
    
    def add_analyzers(self):
        """添加分析器"""
        # 添加各种分析器
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe',
        riskfreerate=0.03/252,  # 假设无风险利率为3%
        timeframe=bt.TimeFrame.Days,
        compression=1,
        annualize=True)
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        logger.info("添加分析器完成")
    
    def add_observers(self):
        """添加BackTrader标准观测器 - 记录所有数据但绘图时只显示投资组合表现"""
        # 添加现金和价值观测器
        self.cerebro.addobserver(bt.observers.Broker)
        
        # 添加交易观测器 - 记录每次交易的盈亏（用于数据记录）
        self.cerebro.addobserver(bt.observers.Trades)
        
        # 添加买卖下单观测器 - 记录买卖信号（用于数据记录）
        self.cerebro.addobserver(bt.observers.BuySell)
        
        # 添加时间收益率观测器 - 记录收益序列
        self.cerebro.addobserver(bt.observers.TimeReturn)
        
        # 添加回撤观测器 - 记录回撤序列
        self.cerebro.addobserver(bt.observers.DrawDown)
        
        # 添加价值观测器 - 记录投资组合价值变化
        self.cerebro.addobserver(bt.observers.Value)
        
        logger.info("添加观测器完成 - 记录所有数据，绘图时只显示投资组合表现")
    
    def run_backtest(self):
        """运行回测"""
        logger.info("开始运行回测...")
        
        # 运行回测
        results = self.cerebro.run()
        
        # 获取策略实例
        strategy = results[0]
        
        # 获取分析器结果
        analyzer_results = {}
        try:
            if hasattr(strategy, 'analyzers'):
                analyzers = strategy.analyzers
                # 动态遍历所有分析器
                logger.info("动态遍历所有分析器:")
                for attr_name in dir(analyzers):
                    if not attr_name.startswith('_'):  # 跳过私有属性
                        try:
                            analyzer = getattr(analyzers, attr_name)
                            if hasattr(analyzer, 'get_analysis'):
                                analysis_result = analyzer.get_analysis()
                                if analysis_result:  # 只记录有结果的分析器
                                    analyzer_results[attr_name] = analysis_result
                                    logger.info(f"发现分析器: {attr_name} = {analysis_result}")
                        except Exception as e:
                            # 忽略无法获取结果的分析器
                            pass
                            
        except Exception as e:
            logger.warning(f"获取分析器结果时出错: {e}")
        
        # 创建绩效分析器
        self.analyzer = PerformanceAnalyzer(strategy)
        
        # 计算绩效指标
        final_value = self.cerebro.broker.getvalue()
        metrics = self.analyzer.calculate_metrics(final_value)
        
        # 打印结果
        print(f"\n回测完成!")
        print(f"初始资金: {self.initial_cash:,.2f}")
        print(f"最终资金: {final_value:,.2f}")
        print(f"总收益: {final_value - self.initial_cash:,.2f}")
        print(f"收益率: {(final_value - self.initial_cash) / self.initial_cash:.2%}")
        
        # 打印统一的绩效分析报告
        print(f"\n{'='*50}")
        print("回测绩效分析报告")
        print(f"{'='*50}")
        
        # 基础指标
        print(f"总收益率: {(final_value - self.initial_cash) / self.initial_cash:.2%}")
        
        # 从BackTrader分析器获取详细指标
        if 'returns' in analyzer_results:
            returns_data = analyzer_results['returns']
            print(f"年化收益率: {returns_data.get('rnorm100', 0):.2f}%")
        
        if 'sharpe' in analyzer_results:
            sharpe_data = analyzer_results['sharpe']
            print(f"夏普比率: {sharpe_data.get('sharperatio', 0):.4f}")
        
        if 'drawdown' in analyzer_results:
            drawdown_data = analyzer_results['drawdown']
            max_drawdown = drawdown_data.get('max', {}).get('drawdown', 0)
            max_drawdown_len = drawdown_data.get('max', {}).get('len', 0)
            
            # 根据值的大小决定显示格式
            if abs(max_drawdown) > 1:  # 如果值大于1，说明已经是百分比形式
                print(f"最大回撤: {max_drawdown:.2f}%")
            else:  # 如果值小于1，说明是小数形式
                print(f"最大回撤: {max_drawdown:.2%}")
            
            print(f"最大回撤长度: {max_drawdown_len} 天")
        
        if 'trades' in analyzer_results:
            trades_data = analyzer_results['trades']
            total_trades = trades_data.get('total', {}).get('total', 0)
            won_trades = trades_data.get('won', {}).get('total', 0)
            lost_trades = trades_data.get('lost', {}).get('total', 0)
            win_rate = won_trades / total_trades if total_trades > 0 else 0
            print(f"总交易次数: {total_trades}")
            print(f"盈利交易: {won_trades}")
            print(f"亏损交易: {lost_trades}")
            print(f"胜率: {win_rate:.2%}")
            
            # 显示平均盈亏
            if 'won' in trades_data and 'pnl' in trades_data['won']:
                avg_win = trades_data['won']['pnl'].get('average', 0) or 0
                total_win = trades_data['won']['pnl'].get('total', 0) or 0
                print(f"平均盈利: {avg_win:.2f}")
                print(f"总盈利: {total_win:.2f}")
            if 'lost' in trades_data and 'pnl' in trades_data['lost']:
                avg_loss = trades_data['lost']['pnl'].get('average', 0) or 0
                total_loss = trades_data['lost']['pnl'].get('total', 0) or 0
                print(f"平均亏损: {avg_loss:.2f}")
                print(f"总亏损: {total_loss:.2f}")
                
                # 计算盈亏比：总盈利/总亏损
                if total_loss != 0:
                    profit_loss_ratio = abs(total_win / total_loss)
                    print(f"盈亏比: {profit_loss_ratio:.2f}")
        
        print("="*50)
        
        return strategy, metrics, analyzer_results
    

def main():
    """主函数：演示完整的回测流程"""
    
    # 1. 初始化数据管理器
    data_manager = DataManager()
    
    # 2. 获取股票列表（这里使用少量股票作为示例）
    stock_codes = ['000001.SZ', '000002.SZ', '000858.SZ', '600000.SH', '600036.SH']
    logger.info(f"使用股票池: {stock_codes}")
    
    # 3. 下载历史数据
    start_date = '2022-01-01'
    end_date = '2024-12-31'
    data_dict = data_manager.download_stock_data(
        stock_codes, 
        period="1d",
        start_date=start_date,
        end_date=end_date
    )
    
    # 4. 获取沪深300基准数据
    benchmark_dates, benchmark_curve = get_hs300_index_data(start_date, end_date)
    
    if not data_dict:
        logger.error("没有获取到有效数据，退出回测")
        return

    # 5. 创建回测框架
    framework = BacktestFramework(initial_cash=100000)
    
    # 6. 设置回测引擎
    framework.setup_cerebro()
    
    # 7. 添加数据源
    framework.add_data(data_dict)
    
    # 8. 添加策略
    framework.add_strategy(
        MomentumStrategy,
        lookback_period=20,
        momentum_threshold=0.05,
        position_size=0.1,
        stop_loss=0.05,
        take_profit=0.15
    )
    
    # 9. 添加分析器
    framework.add_analyzers()
    
    # 10. 添加观测器
    framework.add_observers()
    
    # 11. 运行回测
    strategy, metrics, analyzer_results = framework.run_backtest()
    
    # 12. 使用benchmark模块绘制资金曲线（包含基准对比）
    plot_strategy_analysis(
        strategy=strategy, 
        initial_cash=framework.initial_cash,
        benchmark_dates=benchmark_dates, 
        benchmark_curve=benchmark_curve,
        benchmark_name='沪深300基准',
        save_path='backtest_returns_curve.png',
        skip_warmup_days=20  # 跳过20个交易日的预热期
    )
    
    # 13. 保存结果
    results = {
        'strategy_name': 'MomentumStrategy',
        'backtest_period': f"{start_date} to {end_date}",
        'initial_cash': framework.initial_cash,
        'final_cash': framework.cerebro.broker.getvalue(),
        'custom_metrics': metrics,
        'backtrader_analyzers': analyzer_results,
        'trades_count': len(strategy.trades) if hasattr(strategy, 'trades') else 0,
        'benchmark_return': benchmark_curve[-1] - 1 if len(benchmark_curve) > 0 else 0
    }
    
    # 保存到文件
    with open('backtest_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    logger.info("回测结果已保存到 backtest_results.json")
    
    # 14. 生成HTML报告
    try:
        report_path = generate_backtest_report(
            strategy=strategy,
            initial_cash=framework.initial_cash,
            analyzer_results=analyzer_results,
            benchmark_dates=benchmark_dates,
            benchmark_curve=benchmark_curve,
            benchmark_name='沪深300基准',
            skip_warmup_days=20,
            save_path='backtest_report.html'
        )
        logger.info(f"HTML报告已生成: {report_path}")
        
    except Exception as e:
        logger.error(f"生成HTML报告失败: {e}")


if __name__ == "__main__":
    main() 