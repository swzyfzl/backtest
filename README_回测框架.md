# Backtrader 回测框架

这是一个基于 Backtrader 包构建的完整回测框架，包含数据层、策略逻辑层、执行层、风险控制层和评估层。

## 功能特性

### 📊 数据层
- **历史行情数据获取**：通过 xtquant 接口获取股票历史数据
- **数据清洗**：自动处理缺失值、异常值、重复数据
- **基本面数据**：预留接口，可扩展获取财务数据
- **事件数据**：预留接口，可扩展获取公告、新闻等事件数据

### 🧠 策略逻辑层
- **信号生成**：多种技术指标组合生成交易信号
- **仓位计算**：基于风险管理的仓位分配算法
- **交易规则**：完整的买入、卖出、止损、止盈逻辑

### ⚡ 执行层
- **交易成本模拟**：手续费、滑点、印花税等真实交易成本
- **流动性限制**：最小交易金额、交易量限制
- **订单执行**：模拟真实市场环境下的订单执行

### 🛡️ 风险控制层
- **止损规则**：多种止损策略（固定比例、移动止损等）
- **仓位上限**：单只股票最大仓位限制
- **风险指标监控**：VaR、最大回撤、波动率等风险指标
- **杠杆控制**：最大杠杆限制

### 📈 评估层
- **绩效指标**：夏普比率、年化收益、卡玛比率、索提诺比率
- **风险指标**：最大回撤、VaR、波动率
- **交易统计**：胜率、盈亏比、交易次数
- **可视化**：投资组合价值曲线、回撤曲线、收益分布图

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 基本使用

```python
from Choose.backtest import BacktestFramework, DataManager
from Choose.strategies import get_strategy_class

# 初始化数据管理器
data_manager = DataManager()

# 获取股票数据
stock_codes = ['000001.SZ', '000002.SZ', '000858.SZ']
data_dict = data_manager.download_stock_data(
  stock_codes,
  start_date='2023-01-01',
  end_date='2023-12-31'
)

# 创建回测框架
framework = BacktestFramework(initial_cash=1000000)
framework.setup_cerebro()
framework.add_data(data_dict)

# 添加策略
strategy_class = get_strategy_class('momentum')
framework.add_strategy(strategy_class)

# 运行回测
strategy, metrics = framework.run_backtest()
```

### 2. 使用示例脚本

```bash
python example_usage.py
```

## 内置策略

### 1. 动量策略 (MomentumStrategy)
- **原理**：基于价格动量指标进行交易
- **参数**：
  - `lookback_period`: 回看周期 (默认: 20)
  - `momentum_threshold`: 动量阈值 (默认: 0.05)
  - `position_size`: 仓位大小 (默认: 0.1)

### 2. 移动平均线交叉策略 (MACrossStrategy)
- **原理**：快线上穿慢线买入，下穿卖出
- **参数**：
  - `fast_period`: 快速均线周期 (默认: 10)
  - `slow_period`: 慢速均线周期 (默认: 30)

### 3. RSI策略 (RSIStrategy)
- **原理**：RSI超卖买入，超买卖出
- **参数**：
  - `rsi_period`: RSI周期 (默认: 14)
  - `oversold`: 超卖阈值 (默认: 30)
  - `overbought`: 超买阈值 (默认: 70)

### 4. 布林带策略 (BollingerBandsStrategy)
- **原理**：价格突破布林带边界进行交易
- **参数**：
  - `bb_period`: 布林带周期 (默认: 20)
  - `bb_dev`: 标准差倍数 (默认: 2)

### 5. 双轨突破策略 (DualThrustStrategy)
- **原理**：基于价格突破上下轨进行交易
- **参数**：
  - `lookback_period`: 回看周期 (默认: 20)
  - `k1`: 上轨系数 (默认: 0.7)
  - `k2`: 下轨系数 (默认: 0.7)

### 6. 组合策略 (PortfolioStrategy)
- **原理**：结合多个技术指标的综合策略
- **特点**：同时考虑均线、RSI、布林带、成交量等多个指标

## 配置说明

### 数据配置 (config.py)
```python
DATA_CONFIG = {
    'default_index': '000852.SH',  # 默认指数
    'data_period': '1d',           # 数据周期
    'start_date': '2023-01-01',    # 回测开始日期
    'end_date': '2023-12-31',      # 回测结束日期
    'max_stocks': 50,              # 最大股票数量
}
```

### 风险控制配置
```python
RISK_CONFIG = {
    'max_position_pct': 0.1,       # 单只股票最大仓位比例
    'stop_loss_pct': 0.05,         # 止损比例
    'max_drawdown_pct': 0.2,       # 最大回撤限制
    'var_confidence': 0.95,        # VaR置信度
}
```

### 交易成本配置
```python
TRANSACTION_CONFIG = {
    'commission_pct': 0.0003,      # 手续费率
    'slippage_pct': 0.0001,        # 滑点率
    'min_trade_amount': 1000,      # 最小交易金额
    'stamp_duty': 0.001,           # 印花税
}
```

## 高级功能

### 1. 多策略对比

```python
from Choose.example_usage import run_multiple_strategies_comparison

results = run_multiple_strategies_comparison(
  strategy_names=['momentum', 'ma_cross', 'rsi'],
  stock_codes=['000001.SZ', '000002.SZ', '000858.SZ'],
  start_date='2023-01-01',
  end_date='2023-12-31'
)
```

### 2. 参数优化

```python
from Choose.example_usage import run_parameter_optimization

param_ranges = {
  'lookback_period': [10, 15, 20, 25],
  'momentum_threshold': [0.03, 0.05, 0.07],
  'position_size': [0.05, 0.1, 0.15]
}

results = run_parameter_optimization(
  strategy_name='momentum',
  param_ranges=param_ranges,
  stock_codes=['000001.SZ', '000002.SZ']
)
```

### 3. 自定义策略
```python
import backtrader as bt

class MyCustomStrategy(bt.Strategy):
    params = (
        ('my_param', 10),
    )
    
    def __init__(self):
        # 初始化指标
        self.my_indicator = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.my_param
        )
    
    def next(self):
        # 实现交易逻辑
        if self.my_indicator[0] > self.data.close[0]:
            self.buy()
        elif self.my_indicator[0] < self.data.close[0]:
            self.sell()
```

## 输出结果

### 1. 控制台输出
```
回测绩效分析报告
==================================================
总收益率: 15.23%
年化收益率: 12.45%
总交易次数: 156
胜率: 58.33%
平均盈利: 2.34%
平均亏损: -1.67%
盈亏比: 1.40
夏普比率: 1.25
索提诺比率: 1.45
卡玛比率: 2.34
最大回撤: -5.32%
VaR(95%): -2.15%
==================================================
```

### 2. 图表输出
- 投资组合价值曲线
- 交易盈亏分布图
- 累计盈亏曲线
- 回撤曲线

### 3. 文件输出
- JSON格式的详细回测结果
- CSV格式的策略对比报告
- CSV格式的参数优化报告

## 注意事项

1. **数据获取**：确保 xtquant 接口正常工作，MiniQmt 已连接
2. **内存使用**：大量股票数据可能占用较多内存，建议分批处理
3. **计算时间**：参数优化可能需要较长时间，建议使用小规模测试
4. **风险控制**：实际交易中请根据市场情况调整风险参数

## 扩展开发

### 添加新策略
1. 在 `strategies.py` 中定义新的策略类
2. 继承 `bt.Strategy` 并实现 `__init__` 和 `next` 方法
3. 在 `STRATEGY_REGISTRY` 中注册新策略

### 添加新指标
1. 使用 Backtrader 内置指标或自定义指标
2. 在策略的 `__init__` 方法中初始化指标
3. 在 `next` 方法中使用指标生成交易信号

### 添加新分析器
1. 继承 `PerformanceAnalyzer` 类
2. 实现新的分析方法
3. 在 `BacktestFramework` 中集成新分析器

## 技术支持

如有问题或建议，请查看：
1. Backtrader 官方文档
2. xtquant 接口文档
3. 代码注释和示例

## 更新日志

### v1.0.0 (2024-01-01)
- 初始版本发布
- 支持6种内置策略
- 完整的风险控制和绩效评估
- 多策略对比和参数优化功能 