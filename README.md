# 基于Backtrader的完整回测框架

一个功能完整的量化交易回测框架，基于Backtrader构建，包含数据层、策略逻辑层、执行层、风险控制层和评估层。

## 🚀 主要功能

### 📊 核心功能
- **完整回测框架**：基于Backtrader的多层架构设计
- **数据管理**：支持xtquant数据源，自动数据清洗和格式化
- **策略开发**：提供动量策略示例，支持自定义策略开发
- **风险控制**：内置止损、仓位限制、风险指标监控
- **交易成本**：模拟手续费、滑点等真实交易成本
- **绩效分析**：全面的回测绩效指标计算

### 📈 可视化分析
- **多图表展示**：收益率曲线、仓位变化、每日盈亏
- **基准对比**：支持沪深300等基准指数对比
- **超额收益**：计算并显示策略相对基准的超额收益
- **HTML报告**：自动生成美观的HTML网页报告
- **自动打开**：回测完成后自动在浏览器中打开报告

### 🔧 技术特性
- **预热期机制**：20个交易日预热期，确保指标计算稳定
- **数据对齐**：自动处理策略数据与基准数据的时间对齐
- **错误处理**：完善的异常处理和日志记录
- **模块化设计**：各层独立，便于扩展和维护

## 📁 项目结构

```
Choose/
├── backtest.py              # 主回测框架
├── report.py                # HTML报告生成模块
├── plt_benchmark.py         # 基准数据和图表绘制模块
├── backtest_results.json    # 回测结果JSON文件
├── backtest_report.html     # HTML报告文件
└── backtest_returns_curve.png # 策略表现图表
```

## 🛠️ 安装依赖

```bash
pip install backtrader pandas numpy matplotlib xtquant
```

## 📖 使用指南

### 1. 基本使用

```python
# 直接运行回测
python Choose/backtest.py
```

### 2. 自定义策略

```python
from Choose.backtest import BacktestFramework, MomentumStrategy

# 创建回测框架
framework = BacktestFramework(initial_cash=100000)

# 设置回测引擎
framework.setup_cerebro()

# 添加数据源
framework.add_data(data_dict)

# 添加自定义策略
framework.add_strategy(
    MomentumStrategy,
    lookback_period=20,
    momentum_threshold=0.05,
    position_size=0.1,
    stop_loss=0.05,
    take_profit=0.15
)

# 运行回测
strategy, metrics, analyzer_results = framework.run_backtest()
```

### 3. 生成HTML报告

```python
from Choose.report import generate_backtest_report

# 生成HTML报告
report_path = generate_backtest_report(
    strategy=strategy,
    initial_cash=100000,
    analyzer_results=analyzer_results,
    benchmark_dates=benchmark_dates,
    benchmark_curve=benchmark_curve,
    benchmark_name='沪深300基准',
    save_path='my_report.html',
    auto_open=True  # 自动打开报告
)
```

## 📊 回测报告内容

### 基础统计
- 初始资金、最终资产、总收益率
- 数据点数量、回测周期

### 风险指标
- 最大回撤及回撤天数
- 夏普比率、年化收益率
- 风险调整后收益

### 交易统计
- 总交易次数、盈利/亏损交易
- 交易胜率、平均盈亏
- 盈亏比

### 每日盈亏统计
- 总盈亏、盈利/亏损天数
- 最大单日盈利/亏损
- 平均每日盈亏、日胜率

### 可视化图表
- **收益率曲线**：策略收益率、基准收益率、超额收益
- **仓位变化**：每日持仓金额变化
- **每日盈亏**：绿色表示盈利，红色表示亏损

## ⚙️ 配置参数

### 策略参数
```python
params = (
    ('lookback_period', 20),      # 回看周期
    ('momentum_threshold', 0.05), # 动量阈值
    ('position_size', 0.1),       # 仓位大小
    ('stop_loss', 0.05),          # 止损比例
    ('take_profit', 0.15),        # 止盈比例
)
```

### 风险控制参数
```python
risk_manager = RiskManager(
    max_position_pct=0.1,    # 单只股票最大仓位比例
    stop_loss_pct=0.05,      # 止损比例
    max_drawdown_pct=0.2     # 最大回撤限制
)
```

### 交易成本参数
```python
transaction_costs = TransactionCosts(
    commission_pct=0.0003,   # 手续费率
    slippage_pct=0.0001,     # 滑点率
    min_trade_amount=1000    # 最小交易金额
)
```

## 📈 绩效指标说明

### 收益率指标
- **总收益率**：整个回测期间的累计收益率
- **年化收益率**：按年化计算的收益率
- **超额收益**：策略收益率减去基准收益率

### 风险指标
- **最大回撤**：从峰值到谷值的最大跌幅
- **夏普比率**：风险调整后收益指标
- **回撤长度**：最大回撤持续的天数

### 交易指标
- **胜率**：盈利交易占总交易的比例
- **盈亏比**：平均盈利与平均亏损的比值
- **交易频率**：单位时间内的交易次数

## 🔧 扩展开发

### 添加新策略
```python
class MyStrategy(bt.Strategy):
    params = (
        ('param1', 10),
        ('param2', 0.1),
    )
    
    def __init__(self):
        # 初始化指标
        self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=20)
    
    def next(self):
        # 策略逻辑
        if self.data.close[0] > self.sma[0]:
            self.buy()
        elif self.data.close[0] < self.sma[0]:
            self.sell()
```

### 添加新分析器
```python
# 在add_analyzers方法中添加
self.cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
```

### 自定义报告
```python
# 修改report.py中的HTML模板
# 或创建新的报告生成函数
```

## 📝 注意事项

1. **数据质量**：确保输入数据的完整性和准确性
2. **参数调优**：根据市场情况调整策略参数
3. **风险控制**：合理设置止损和仓位限制
4. **回测周期**：选择合适的回测时间范围
5. **基准选择**：选择合适的基础指数作为对比

## 🐛 常见问题

### Q: 夏普比率为负值？
A: 这是正常的，表示策略的风险调整后收益为负，需要优化策略参数。

### Q: 最大回撤显示异常？
A: 已修复最大回撤的显示格式问题，现在会根据值的格式正确显示。

### Q: 如何添加新的数据源？
A: 修改DataManager类中的download_stock_data方法，支持不同的数据源。

### Q: 报告没有自动打开？
A: 检查浏览器设置，或手动打开生成的HTML文件。

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至项目维护者

---

**最后更新**：2024年12月

**版本**：v2.0.0 