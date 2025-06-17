"""
回测框架配置文件
"""

# 数据配置
DATA_CONFIG = {
    'default_index': '000852.SH',  # 默认指数（中证2000）
    'data_period': '1d',  # 数据周期
    'start_date': '2023-01-01',  # 回测开始日期
    'end_date': '2023-12-31',  # 回测结束日期
    'max_stocks': 50,  # 最大股票数量（用于测试）
}

# 策略配置
STRATEGY_CONFIG = {
    'momentum': {
        'lookback_period': 20,  # 回看周期
        'momentum_threshold': 0.05,  # 动量阈值
        'position_size': 0.1,  # 仓位大小
        'stop_loss': 0.05,  # 止损比例
        'take_profit': 0.15,  # 止盈比例
    },
    'ma_cross': {
        'fast_period': 10,  # 快速均线周期
        'slow_period': 30,  # 慢速均线周期
        'position_size': 0.1,
        'stop_loss': 0.05,
        'take_profit': 0.15,
    },
    'rsi': {
        'rsi_period': 14,  # RSI周期
        'oversold': 30,  # 超卖阈值
        'overbought': 70,  # 超买阈值
        'position_size': 0.1,
        'stop_loss': 0.05,
        'take_profit': 0.15,
    }
}

# 风险控制配置
RISK_CONFIG = {
    'max_position_pct': 0.1,  # 单只股票最大仓位比例
    'stop_loss_pct': 0.05,  # 止损比例
    'max_drawdown_pct': 0.2,  # 最大回撤限制
    'var_confidence': 0.95,  # VaR置信度
    'max_leverage': 1.0,  # 最大杠杆
}

# 交易成本配置
TRANSACTION_CONFIG = {
    'commission_pct': 0.0003,  # 手续费率
    'slippage_pct': 0.0001,  # 滑点率
    'min_trade_amount': 1000,  # 最小交易金额
    'stamp_duty': 0.001,  # 印花税（卖出时收取）
}

# 回测配置
BACKTEST_CONFIG = {
    'initial_cash': 1000000,  # 初始资金
    'risk_free_rate': 0.03,  # 无风险利率
    'benchmark': '000300.SH',  # 基准指数
    'rebalance_frequency': 'daily',  # 再平衡频率
}

# 绩效评估配置
PERFORMANCE_CONFIG = {
    'metrics': [
        'total_return',
        'annual_return', 
        'sharpe_ratio',
        'sortino_ratio',
        'calmar_ratio',
        'max_drawdown',
        'var_95',
        'win_rate',
        'profit_loss_ratio'
    ],
    'plot_style': 'seaborn',  # 绘图样式
    'save_results': True,  # 是否保存结果
    'results_path': './results/',  # 结果保存路径
}

# 日志配置
LOG_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': './log/backtest.log',
    'max_bytes': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5,
} 