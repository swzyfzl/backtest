"""
回测框架测试脚本
用于验证框架的基本功能是否正常工作
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_mock_data():
    """创建模拟数据用于测试"""
    logger.info("创建模拟数据...")
    
    # 创建日期范围
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 6, 30)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # 过滤掉周末
    date_range = date_range[date_range.weekday < 5]
    
    data_dict = {}
    
    # 为每个股票创建模拟数据
    stock_codes = ['000001.SZ', '000002.SZ', '000858.SZ']
    
    for stock_code in stock_codes:
        # 生成随机价格数据
        np.random.seed(hash(stock_code) % 1000)  # 确保可重复性
        
        # 初始价格
        initial_price = 10.0 + np.random.random() * 20
        
        # 生成价格序列
        returns = np.random.normal(0.001, 0.02, len(date_range))  # 日收益率
        prices = [initial_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, 0.1))  # 确保价格为正
        
        # 生成OHLC数据
        data = []
        for i, (date, price) in enumerate(zip(date_range, prices)):
            # 生成当日价格波动
            daily_volatility = 0.02
            high = price * (1 + abs(np.random.normal(0, daily_volatility)))
            low = price * (1 - abs(np.random.normal(0, daily_volatility)))
            open_price = price * (1 + np.random.normal(0, daily_volatility * 0.5))
            
            # 确保价格逻辑正确
            high = max(high, open_price, price)
            low = min(low, open_price, price)
            
            # 生成成交量
            volume = int(np.random.exponential(1000000))
            
            data.append({
                'datetime': date,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(price, 2),
                'volume': volume
            })
        
        # 创建DataFrame
        df = pd.DataFrame(data)
        df.set_index('datetime', inplace=True)
        
        data_dict[stock_code] = df
        
        logger.info(f"创建 {stock_code} 模拟数据: {len(df)} 条记录")
    
    return data_dict


def test_data_manager():
    """测试数据管理器"""
    logger.info("测试数据管理器...")
    
    try:
        from backtest import DataManager
        
        data_manager = DataManager()
        
        # 测试数据清洗功能
        mock_data = create_mock_data()
        
        for stock_code, df in mock_data.items():
            cleaned_df = data_manager._clean_data(df.copy())
            logger.info(f"{stock_code} 数据清洗完成: {len(cleaned_df)} 条记录")
        
        logger.info("数据管理器测试通过")
        return True
        
    except Exception as e:
        logger.error(f"数据管理器测试失败: {e}")
        return False


def test_risk_manager():
    """测试风险管理器"""
    logger.info("测试风险管理器...")
    
    try:
        from backtest import RiskManager
        
        risk_manager = RiskManager()
        
        # 测试VaR计算
        returns = pd.Series(np.random.normal(0.001, 0.02, 1000))
        var_95 = risk_manager.calculate_var(returns, 0.95)
        logger.info(f"VaR(95%) 计算结果: {var_95:.4f}")
        
        # 测试最大回撤计算
        equity_curve = pd.Series(np.cumprod(1 + returns))
        max_dd, peak_idx, dd_idx = risk_manager.calculate_max_drawdown(equity_curve)
        logger.info(f"最大回撤: {max_dd:.4f}")
        
        # 测试仓位限制
        portfolio_value = 1000000
        position_value = 50000
        is_within_limit = risk_manager.check_position_limit(position_value, portfolio_value)
        logger.info(f"仓位限制检查: {is_within_limit}")
        
        # 测试止损检查
        entry_price = 10.0
        current_price = 9.0
        should_stop_loss = risk_manager.check_stop_loss(entry_price, current_price)
        logger.info(f"止损检查: {should_stop_loss}")
        
        logger.info("风险管理器测试通过")
        return True
        
    except Exception as e:
        logger.error(f"风险管理器测试失败: {e}")
        return False


def test_transaction_costs():
    """测试交易成本计算"""
    logger.info("测试交易成本计算...")
    
    try:
        from backtest import TransactionCosts
        
        transaction_costs = TransactionCosts()
        
        # 测试手续费计算
        trade_value = 100000
        commission = transaction_costs.calculate_commission(trade_value)
        logger.info(f"手续费: {commission:.2f}")
        
        # 测试滑点计算
        slippage_buy = transaction_costs.calculate_slippage(trade_value, True)
        slippage_sell = transaction_costs.calculate_slippage(trade_value, False)
        logger.info(f"买入滑点: {slippage_buy:.2f}, 卖出滑点: {slippage_sell:.2f}")
        
        # 测试有效价格计算
        price = 10.0
        effective_buy_price = transaction_costs.get_effective_price(price, True)
        effective_sell_price = transaction_costs.get_effective_price(price, False)
        logger.info(f"买入有效价格: {effective_buy_price:.4f}, 卖出有效价格: {effective_sell_price:.4f}")
        
        logger.info("交易成本计算测试通过")
        return True
        
    except Exception as e:
        logger.error(f"交易成本计算测试失败: {e}")
        return False


def test_performance_analyzer():
    """测试绩效分析器"""
    logger.info("测试绩效分析器...")
    
    try:
        from backtest import PerformanceAnalyzer
        
        # 创建模拟策略对象
        class MockStrategy:
            def __init__(self):
                self.trades = [
                    {
                        'symbol': '000001.SZ',
                        'open_date': datetime(2023, 1, 1),
                        'close_date': datetime(2023, 1, 10),
                        'size': 1000,
                        'price_open': 10.0,
                        'price_close': 10.5,
                        'pnl': 500,
                        'pnlcomm': 495,
                    },
                    {
                        'symbol': '000002.SZ',
                        'open_date': datetime(2023, 1, 15),
                        'close_date': datetime(2023, 1, 25),
                        'size': 1000,
                        'price_open': 20.0,
                        'price_close': 19.5,
                        'pnl': -500,
                        'pnlcomm': -505,
                    }
                ]
        
        mock_strategy = MockStrategy()
        analyzer = PerformanceAnalyzer(mock_strategy)
        
        # 测试指标计算
        portfolio_value = 1000000
        metrics = analyzer.calculate_metrics(portfolio_value)
        
        logger.info("计算得到的指标:")
        for key, value in metrics.items():
            logger.info(f"{key}: {value}")
        
        logger.info("绩效分析器测试通过")
        return True
        
    except Exception as e:
        logger.error(f"绩效分析器测试失败: {e}")
        return False


def test_backtest_framework():
    """测试回测框架"""
    logger.info("测试回测框架...")
    
    try:
        from backtest import BacktestFramework, MomentumStrategy
        
        # 创建模拟数据
        mock_data = create_mock_data()
        
        # 创建回测框架
        framework = BacktestFramework(initial_cash=100000)
        framework.setup_cerebro()
        framework.add_data(mock_data)
        
        # 添加策略
        framework.add_strategy(MomentumStrategy, lookback_period=10, position_size=0.1)
        framework.add_analyzers()
        
        # 运行回测
        strategy, metrics = framework.run_backtest()
        
        logger.info(f"回测完成，最终资金: {framework.cerebro.broker.getvalue():.2f}")
        logger.info(f"交易次数: {len(strategy.trades) if hasattr(strategy, 'trades') else 0}")
        
        logger.info("回测框架测试通过")
        return True
        
    except Exception as e:
        logger.error(f"回测框架测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    logger.info("开始运行回测框架测试...")
    
    tests = [
        ("数据管理器", test_data_manager),
        ("风险管理器", test_risk_manager),
        ("交易成本计算", test_transaction_costs),
        ("绩效分析器", test_performance_analyzer),
        ("回测框架", test_backtest_framework),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"运行测试: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            if test_func():
                passed_tests += 1
                logger.info(f"✅ {test_name} 测试通过")
            else:
                logger.error(f"❌ {test_name} 测试失败")
        except Exception as e:
            logger.error(f"❌ {test_name} 测试异常: {e}")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"测试总结: {passed_tests}/{total_tests} 通过")
    logger.info(f"{'='*50}")
    
    if passed_tests == total_tests:
        logger.info("🎉 所有测试通过！回测框架工作正常。")
    else:
        logger.warning("⚠️ 部分测试失败，请检查相关功能。")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 