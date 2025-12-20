# src/analysis/__init__.py

"""
Analysis package for cryptocurrency quantitative analysis.

This package provides modules for:
- Financial metrics calculation (risk, performance)
- Market-wide analysis and correlation
- Portfolio construction and backtesting
- Factor-based analysis
"""

from .financial_metrics import (
    calculate_volatility,
    calculate_drawdown,
    calculate_var_cvar,
    calculate_returns,
    calculate_cagr,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    get_all_metrics
)

from .market_analyzer import (
    load_all_coins_data,
    calculate_market_breadth,
    create_returns_heatmap,
    rank_by_metric,
    calculate_correlation_matrix,
    detect_volume_spike,
    identify_market_regime
)

from .portfolio_engine import (
    equal_weight_portfolio,
    risk_parity_portfolio,
    volatility_targeting_portfolio,
    backtest_portfolio,
    calculate_portfolio_metrics,
    calculate_risk_contribution
)

from .factor_analyzer import (
    calculate_momentum,
    calculate_size_factor,
    calculate_liquidity_factor,
    calculate_volatility_factor,
    create_factor_dataframe,
    factor_scatter_plot_data,
    cluster_by_factors
)

__all__ = [
    # Financial metrics
    'calculate_volatility',
    'calculate_drawdown',
    'calculate_var_cvar',
    'calculate_returns',
    'calculate_cagr',
    'calculate_sharpe_ratio',
    'calculate_sortino_ratio',
    'calculate_calmar_ratio',
    'get_all_metrics',
    
    # Market analyzer
    'load_all_coins_data',
    'calculate_market_breadth',
    'create_returns_heatmap',
    'rank_by_metric',
    'calculate_correlation_matrix',
    'detect_volume_spike',
    'identify_market_regime',
    
    # Portfolio engine
    'equal_weight_portfolio',
    'risk_parity_portfolio',
    'volatility_targeting_portfolio',
    'backtest_portfolio',
    'calculate_portfolio_metrics',
    'calculate_risk_contribution',
    
    # Factor analyzer
    'calculate_momentum',
    'calculate_size_factor',
    'calculate_liquidity_factor',
    'calculate_volatility_factor',
    'create_factor_dataframe',
    'factor_scatter_plot_data',
    'cluster_by_factors',
]
