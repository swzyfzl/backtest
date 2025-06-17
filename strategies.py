"""
策略模块：包含多种交易策略实现
"""

import backtrader as bt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class MACrossStrategy(bt.Strategy):
    """
    移动平均线交叉策略
    """
    
    params = (
        ('fast_period', 10),  # 快速均线周期
        ('slow_period', 30),  # 慢速均线周期
        ('position_size', 0.1),  # 仓位大小
        ('stop_loss', 0.05),  # 止损比例
        ('take_profit', 0.15),  # 止盈比例
    )
    
    def __init__(self):
        self.fast_ma = {}
        self.slow_ma = {}
        self.crossover = {}
        self.entry_prices = {}
        
        for data in self.datas:
            # 快速移动平均线
            self.fast_ma[data._name] = bt.indicators.SimpleMovingAverage(
                data.close, period=self.params.fast_period
            )
            # 慢速移动平均线
            self.slow_ma[data._name] = bt.indicators.SimpleMovingAverage(
                data.close, period=self.params.slow_period
            )
            # 交叉信号
            self.crossover[data._name] = bt.indicators.CrossOver(
                self.fast_ma[data._name], self.slow_ma[data._name]
            )
    
    def next(self):
        for data in self.datas:
            symbol = data._name
            position = self.getposition(data).size
            current_price = data.close[0]
            
            # 买入信号：快线上穿慢线
            if self.crossover[symbol] > 0 and position == 0:
                portfolio_value = self.broker.getvalue()
                position_value = portfolio_value * self.params.position_size
                shares = int(position_value / current_price)
                
                if shares > 0:
                    self.buy(data=data, size=shares)
                    self.entry_prices[symbol] = current_price
                    self.log(f'MA交叉买入 {symbol}: {shares}股, 价格: {current_price:.2f}')
            
            # 卖出信号：快线下穿慢线
            elif self.crossover[symbol] < 0 and position > 0:
                self.sell(data=data, size=position)
                if symbol in self.entry_prices:
                    del self.entry_prices[symbol]
                self.log(f'MA交叉卖出 {symbol}: {position}股, 价格: {current_price:.2f}')
            
            # 止损止盈
            elif position > 0 and symbol in self.entry_prices:
                entry_price = self.entry_prices[symbol]
                profit_pct = (current_price - entry_price) / entry_price
                
                if profit_pct <= -self.params.stop_loss:
                    self.sell(data=data, size=position)
                    del self.entry_prices[symbol]
                    self.log(f'止损: {symbol}, 亏损: {profit_pct:.2%}')
                elif profit_pct >= self.params.take_profit:
                    self.sell(data=data, size=position)
                    del self.entry_prices[symbol]
                    self.log(f'止盈: {symbol}, 盈利: {profit_pct:.2%}')
    
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        logger.info(f'{dt.isoformat()} {txt}')


class RSIStrategy(bt.Strategy):
    """
    RSI策略：超买超卖反转
    """
    
    params = (
        ('rsi_period', 14),  # RSI周期
        ('oversold', 30),  # 超卖阈值
        ('overbought', 70),  # 超买阈值
        ('position_size', 0.1),  # 仓位大小
        ('stop_loss', 0.05),  # 止损比例
        ('take_profit', 0.15),  # 止盈比例
    )
    
    def __init__(self):
        self.rsi = {}
        self.entry_prices = {}
        
        for data in self.datas:
            self.rsi[data._name] = bt.indicators.RSI(
                data.close, period=self.params.rsi_period
            )
    
    def next(self):
        for data in self.datas:
            symbol = data._name
            position = self.getposition(data).size
            current_price = data.close[0]
            rsi_value = self.rsi[symbol][0]
            
            # 买入信号：RSI超卖
            if rsi_value < self.params.oversold and position == 0:
                portfolio_value = self.broker.getvalue()
                position_value = portfolio_value * self.params.position_size
                shares = int(position_value / current_price)
                
                if shares > 0:
                    self.buy(data=data, size=shares)
                    self.entry_prices[symbol] = current_price
                    self.log(f'RSI超买买入 {symbol}: {shares}股, RSI: {rsi_value:.2f}')
            
            # 卖出信号：RSI超买
            elif rsi_value > self.params.overbought and position > 0:
                self.sell(data=data, size=position)
                if symbol in self.entry_prices:
                    del self.entry_prices[symbol]
                self.log(f'RSI超卖卖出 {symbol}: {position}股, RSI: {rsi_value:.2f}')
            
            # 止损止盈
            elif position > 0 and symbol in self.entry_prices:
                entry_price = self.entry_prices[symbol]
                profit_pct = (current_price - entry_price) / entry_price
                
                if profit_pct <= -self.params.stop_loss:
                    self.sell(data=data, size=position)
                    del self.entry_prices[symbol]
                    self.log(f'止损: {symbol}, 亏损: {profit_pct:.2%}')
                elif profit_pct >= self.params.take_profit:
                    self.sell(data=data, size=position)
                    del self.entry_prices[symbol]
                    self.log(f'止盈: {symbol}, 盈利: {profit_pct:.2%}')
    
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        logger.info(f'{dt.isoformat()} {txt}')


class BollingerBandsStrategy(bt.Strategy):
    """
    布林带策略：价格突破布林带边界
    """
    
    params = (
        ('bb_period', 20),  # 布林带周期
        ('bb_dev', 2),  # 布林带标准差倍数
        ('position_size', 0.1),  # 仓位大小
        ('stop_loss', 0.05),  # 止损比例
        ('take_profit', 0.15),  # 止盈比例
    )
    
    def __init__(self):
        self.bb = {}
        self.entry_prices = {}
        
        for data in self.datas:
            self.bb[data._name] = bt.indicators.BollingerBands(
                data.close, 
                period=self.params.bb_period,
                devfactor=self.params.bb_dev
            )
    
    def next(self):
        for data in self.datas:
            symbol = data._name
            position = self.getposition(data).size
            current_price = data.close[0]
            
            bb = self.bb[symbol]
            bb_top = bb.lines.top[0]
            bb_bottom = bb.lines.bot[0]
            bb_mid = bb.lines.mid[0]
            
            # 买入信号：价格突破下轨
            if current_price < bb_bottom and position == 0:
                portfolio_value = self.broker.getvalue()
                position_value = portfolio_value * self.params.position_size
                shares = int(position_value / current_price)
                
                if shares > 0:
                    self.buy(data=data, size=shares)
                    self.entry_prices[symbol] = current_price
                    self.log(f'布林带下轨买入 {symbol}: {shares}股, 价格: {current_price:.2f}')
            
            # 卖出信号：价格突破上轨
            elif current_price > bb_top and position > 0:
                self.sell(data=data, size=position)
                if symbol in self.entry_prices:
                    del self.entry_prices[symbol]
                self.log(f'布林带上轨卖出 {symbol}: {position}股, 价格: {current_price:.2f}')
            
            # 止损止盈
            elif position > 0 and symbol in self.entry_prices:
                entry_price = self.entry_prices[symbol]
                profit_pct = (current_price - entry_price) / entry_price
                
                if profit_pct <= -self.params.stop_loss:
                    self.sell(data=data, size=position)
                    del self.entry_prices[symbol]
                    self.log(f'止损: {symbol}, 亏损: {profit_pct:.2%}')
                elif profit_pct >= self.params.take_profit:
                    self.sell(data=data, size=position)
                    del self.entry_prices[symbol]
                    self.log(f'止盈: {symbol}, 盈利: {profit_pct:.2%}')
    
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        logger.info(f'{dt.isoformat()} {txt}')


class DualThrustStrategy(bt.Strategy):
    """
    双轨突破策略
    """
    
    params = (
        ('lookback_period', 20),  # 回看周期
        ('k1', 0.7),  # 上轨系数
        ('k2', 0.7),  # 下轨系数
        ('position_size', 0.1),  # 仓位大小
        ('stop_loss', 0.05),  # 止损比例
        ('take_profit', 0.15),  # 止盈比例
    )
    
    def __init__(self):
        self.upper_band = {}
        self.lower_band = {}
        self.entry_prices = {}
        
        for data in self.datas:
            # 计算HH和LL
            self.upper_band[data._name] = bt.indicators.Highest(
                data.high, period=self.params.lookback_period
            )
            self.lower_band[data._name] = bt.indicators.Lowest(
                data.low, period=self.params.lookback_period
            )
    
    def next(self):
        for data in self.datas:
            symbol = data._name
            position = self.getposition(data).size
            current_price = data.close[0]
            
            # 计算突破轨道
            hh = self.upper_band[symbol][0]
            ll = self.lower_band[symbol][0]
            range_val = hh - ll
            
            upper_breakout = hh + self.params.k1 * range_val
            lower_breakout = ll - self.params.k2 * range_val
            
            # 买入信号：突破上轨
            if current_price > upper_breakout and position == 0:
                portfolio_value = self.broker.getvalue()
                position_value = portfolio_value * self.params.position_size
                shares = int(position_value / current_price)
                
                if shares > 0:
                    self.buy(data=data, size=shares)
                    self.entry_prices[symbol] = current_price
                    self.log(f'双轨上轨买入 {symbol}: {shares}股, 价格: {current_price:.2f}')
            
            # 卖出信号：突破下轨
            elif current_price < lower_breakout and position > 0:
                self.sell(data=data, size=position)
                if symbol in self.entry_prices:
                    del self.entry_prices[symbol]
                self.log(f'双轨下轨卖出 {symbol}: {position}股, 价格: {current_price:.2f}')
            
            # 止损止盈
            elif position > 0 and symbol in self.entry_prices:
                entry_price = self.entry_prices[symbol]
                profit_pct = (current_price - entry_price) / entry_price
                
                if profit_pct <= -self.params.stop_loss:
                    self.sell(data=data, size=position)
                    del self.entry_prices[symbol]
                    self.log(f'止损: {symbol}, 亏损: {profit_pct:.2%}')
                elif profit_pct >= self.params.take_profit:
                    self.sell(data=data, size=position)
                    del self.entry_prices[symbol]
                    self.log(f'止盈: {symbol}, 盈利: {profit_pct:.2%}')
    
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        logger.info(f'{dt.isoformat()} {txt}')


class PortfolioStrategy(bt.Strategy):
    """
    组合策略：结合多个技术指标
    """
    
    params = (
        ('ma_fast', 10),  # 快速均线
        ('ma_slow', 30),  # 慢速均线
        ('rsi_period', 14),  # RSI周期
        ('rsi_oversold', 30),  # RSI超卖
        ('rsi_overbought', 70),  # RSI超买
        ('bb_period', 20),  # 布林带周期
        ('bb_dev', 2),  # 布林带标准差
        ('position_size', 0.1),  # 仓位大小
        ('stop_loss', 0.05),  # 止损比例
        ('take_profit', 0.15),  # 止盈比例
    )
    
    def __init__(self):
        self.indicators = {}
        self.entry_prices = {}
        
        for data in self.datas:
            symbol = data._name
            self.indicators[symbol] = {
                'ma_fast': bt.indicators.SimpleMovingAverage(
                    data.close, period=self.params.ma_fast
                ),
                'ma_slow': bt.indicators.SimpleMovingAverage(
                    data.close, period=self.params.ma_slow
                ),
                'rsi': bt.indicators.RSI(
                    data.close, period=self.params.rsi_period
                ),
                'bb': bt.indicators.BollingerBands(
                    data.close, 
                    period=self.params.bb_period,
                    devfactor=self.params.bb_dev
                ),
                'volume_ma': bt.indicators.SimpleMovingAverage(
                    data.volume, period=20
                )
            }
    
    def next(self):
        for data in self.datas:
            symbol = data._name
            position = self.getposition(data).size
            current_price = data.close[0]
            current_volume = data.volume[0]
            
            ind = self.indicators[symbol]
            
            # 综合信号计算
            ma_signal = ind['ma_fast'][0] > ind['ma_slow'][0]  # 均线多头
            rsi_signal = self.params.rsi_oversold < ind['rsi'][0] < self.params.rsi_overbought  # RSI中性
            bb_signal = ind['bb'].lines.bot[0] < current_price < ind['bb'].lines.top[0]  # 价格在布林带内
            volume_signal = current_volume > ind['volume_ma'][0]  # 放量
            
            # 买入条件：均线多头 + RSI中性 + 价格在布林带内 + 放量
            buy_signal = ma_signal and rsi_signal and bb_signal and volume_signal
            
            # 卖出条件：均线空头 或 RSI超买 或 价格突破布林带
            sell_signal = (
                not ma_signal or 
                ind['rsi'][0] > self.params.rsi_overbought or
                current_price > ind['bb'].lines.top[0]
            )
            
            # 执行交易
            if buy_signal and position == 0:
                portfolio_value = self.broker.getvalue()
                position_value = portfolio_value * self.params.position_size
                shares = int(position_value / current_price)
                
                if shares > 0:
                    self.buy(data=data, size=shares)
                    self.entry_prices[symbol] = current_price
                    self.log(f'组合策略买入 {symbol}: {shares}股, 价格: {current_price:.2f}')
            
            elif sell_signal and position > 0:
                self.sell(data=data, size=position)
                if symbol in self.entry_prices:
                    del self.entry_prices[symbol]
                self.log(f'组合策略卖出 {symbol}: {position}股, 价格: {current_price:.2f}')
            
            # 止损止盈
            elif position > 0 and symbol in self.entry_prices:
                entry_price = self.entry_prices[symbol]
                profit_pct = (current_price - entry_price) / entry_price
                
                if profit_pct <= -self.params.stop_loss:
                    self.sell(data=data, size=position)
                    del self.entry_prices[symbol]
                    self.log(f'止损: {symbol}, 亏损: {profit_pct:.2%}')
                elif profit_pct >= self.params.take_profit:
                    self.sell(data=data, size=position)
                    del self.entry_prices[symbol]
                    self.log(f'止盈: {symbol}, 盈利: {profit_pct:.2%}')
    
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        logger.info(f'{dt.isoformat()} {txt}')


# 策略注册表
STRATEGY_REGISTRY = {
    'momentum': 'MomentumStrategy',
    'ma_cross': 'MACrossStrategy', 
    'rsi': 'RSIStrategy',
    'bollinger': 'BollingerBandsStrategy',
    'dual_thrust': 'DualThrustStrategy',
    'portfolio': 'PortfolioStrategy',
}


def get_strategy_class(strategy_name: str):
    """获取策略类"""
    if strategy_name == 'momentum':
        from backtest import MomentumStrategy
        return MomentumStrategy
    elif strategy_name == 'ma_cross':
        return MACrossStrategy
    elif strategy_name == 'rsi':
        return RSIStrategy
    elif strategy_name == 'bollinger':
        return BollingerBandsStrategy
    elif strategy_name == 'dual_thrust':
        return DualThrustStrategy
    elif strategy_name == 'portfolio':
        return PortfolioStrategy
    else:
        raise ValueError(f"未知的策略: {strategy_name}") 