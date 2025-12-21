"""Financial metrics: volatility, drawdown, VaR, Sharpe, Sortino, Calmar ratios."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


def calculate_returns(prices: pd.Series, method: str = 'simple') -> pd.Series:
    """
    Calculate returns from price series.
    
    Args:
        prices: Price series (usually 'close' prices)
        method: 'simple' or 'log' returns
    
    Returns:
        Returns series
    """
    if method == 'log':
        return np.log(prices / prices.shift(1))
    else:  # simple
        return prices.pct_change()


def calculate_volatility(
    prices: pd.Series, 
    window: Optional[int] = None,
    annualize: bool = True,
    periods_per_year: int = 365
) -> Union[float, pd.Series]:
    """
    Calculate volatility (standard deviation of returns).
    
    Args:
        prices: Price series
        window: Rolling window size. If None, calculate for entire series
        annualize: Whether to annualize the volatility
        periods_per_year: Number of periods per year (365 for daily data)
    
    Returns:
        Volatility (float if window=None, Series if rolling)
    """
    returns = calculate_returns(prices)
    
    if window is None:
        vol = returns.std()
    else:
        vol = returns.rolling(window=window).std()
    
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    
    return vol


def calculate_drawdown(prices: pd.Series) -> Tuple[pd.Series, float, int]:
    """
    Calculate drawdown series and maximum drawdown.
    
    Args:
        prices: Price series
    
    Returns:
        Tuple of (drawdown_series, max_drawdown, max_drawdown_duration_days)
    """
    cumulative = (1 + calculate_returns(prices)).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    
    max_dd = drawdown.min()
    
    # Calculate drawdown duration
    in_drawdown = drawdown < 0
    drawdown_periods = []
    current_period = 0
    
    for is_dd in in_drawdown:
        if is_dd:
            current_period += 1
        else:
            if current_period > 0:
                drawdown_periods.append(current_period)
            current_period = 0
    
    max_dd_duration = max(drawdown_periods) if drawdown_periods else 0
    
    return drawdown, max_dd, max_dd_duration


def calculate_var_cvar(
    prices: pd.Series, 
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate Value at Risk (VaR) and Conditional VaR (CVaR).
    
    Args:
        prices: Price series
        confidence_level: Confidence level (e.g., 0.95 for 95%)
    
    Returns:
        Tuple of (VaR, CVaR) as percentages
    """
    returns = calculate_returns(prices).dropna()
    
    # VaR is the quantile at (1 - confidence_level)
    var = returns.quantile(1 - confidence_level)
    
    # CVaR is the average of returns below VaR
    cvar = returns[returns <= var].mean()
    
    return var * 100, cvar * 100  # Return as percentages


def calculate_cagr(prices: pd.Series, periods_per_year: int = 365) -> float:
    """
    Calculate Compound Annual Growth Rate.
    
    Args:
        prices: Price series
        periods_per_year: Number of periods per year
    
    Returns:
        CAGR as a percentage
    """
    if len(prices) < 2:
        return 0.0
    
    total_return = prices.iloc[-1] / prices.iloc[0]
    n_periods = len(prices)
    n_years = n_periods / periods_per_year
    
    if n_years <= 0 or total_return <= 0:
        return 0.0
    
    cagr = (total_return ** (1 / n_years)) - 1
    return cagr * 100  # Return as percentage


def calculate_sharpe_ratio(
    prices: pd.Series, 
    risk_free_rate: float = 0.0,
    periods_per_year: int = 365
) -> float:
    """
    Calculate Sharpe Ratio (risk-adjusted return).
    
    Args:
        prices: Price series
        risk_free_rate: Annual risk-free rate (default 0)
        periods_per_year: Number of periods per year
    
    Returns:
        Sharpe ratio
    """
    returns = calculate_returns(prices).dropna()
    
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    
    # Convert annual risk-free rate to period rate
    period_rf = risk_free_rate / periods_per_year
    
    excess_returns = returns - period_rf
    sharpe = excess_returns.mean() / returns.std() * np.sqrt(periods_per_year)
    
    return sharpe


def calculate_sortino_ratio(
    prices: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 365
) -> float:
    """
    Calculate Sortino Ratio (downside risk-adjusted return).
    
    Args:
        prices: Price series
        risk_free_rate: Annual risk-free rate (default 0)
        periods_per_year: Number of periods per year
    
    Returns:
        Sortino ratio
    """
    returns = calculate_returns(prices).dropna()
    
    if len(returns) == 0:
        return 0.0
    
    # Convert annual risk-free rate to period rate
    period_rf = risk_free_rate / periods_per_year
    
    excess_returns = returns - period_rf
    
    # Downside deviation (only negative returns)
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0
    
    downside_std = downside_returns.std()
    sortino = excess_returns.mean() / downside_std * np.sqrt(periods_per_year)
    
    return sortino


def calculate_calmar_ratio(prices: pd.Series, periods_per_year: int = 365) -> float:
    """
    Calculate Calmar Ratio (CAGR / Max Drawdown).
    
    Args:
        prices: Price series
        periods_per_year: Number of periods per year
    
    Returns:
        Calmar ratio
    """
    cagr = calculate_cagr(prices, periods_per_year) / 100  # Convert back to decimal
    _, max_dd, _ = calculate_drawdown(prices)
    
    if max_dd == 0:
        return 0.0
    
    calmar = cagr / abs(max_dd)
    return calmar


def get_all_metrics(
    prices: pd.Series,
    coin_name: str = "Unknown",
    risk_free_rate: float = 0.0,
    periods_per_year: int = 365
) -> Dict[str, float]:
    """
    Calculate all financial metrics for a price series.
    
    Args:
        prices: Price series
        coin_name: Name of the cryptocurrency
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
    
    Returns:
        Dictionary containing all metrics
    """
    try:
        # Returns
        total_return = ((prices.iloc[-1] / prices.iloc[0]) - 1) * 100
        
        # Volatility
        annualized_vol = calculate_volatility(prices, annualize=True, periods_per_year=periods_per_year)
        
        # Drawdown
        _, max_dd, max_dd_duration = calculate_drawdown(prices)
        
        # VaR and CVaR
        var_95, cvar_95 = calculate_var_cvar(prices, confidence_level=0.95)
        
        # Performance metrics
        cagr = calculate_cagr(prices, periods_per_year)
        sharpe = calculate_sharpe_ratio(prices, risk_free_rate, periods_per_year)
        sortino = calculate_sortino_ratio(prices, risk_free_rate, periods_per_year)
        calmar = calculate_calmar_ratio(prices, periods_per_year)
        
        return {
            'coin': coin_name,
            'total_return': total_return,
            'cagr': cagr,
            'annualized_volatility': annualized_vol if isinstance(annualized_vol, float) else annualized_vol.iloc[-1],
            'max_drawdown': max_dd * 100,  # As percentage
            'max_drawdown_duration': max_dd_duration,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'current_price': prices.iloc[-1],
            'min_price': prices.min(),
            'max_price': prices.max(),
        }
    
    except Exception as e:
        logger.error(f"Error calculating metrics for {coin_name}: {e}")
        return {
            'coin': coin_name,
            'error': str(e)
        }


def calculate_rolling_metrics(
    prices: pd.Series,
    window: int = 30,
    metrics: list = ['volatility', 'sharpe']
) -> pd.DataFrame:
    """
    Calculate rolling metrics over time.
    
    Args:
        prices: Price series
        window: Rolling window size
        metrics: List of metrics to calculate ('volatility', 'sharpe', 'sortino')
    
    Returns:
        DataFrame with rolling metrics
    """
    result = pd.DataFrame(index=prices.index)
    
    if 'volatility' in metrics:
        result['volatility'] = calculate_volatility(prices, window=window, annualize=False)
    
    if 'sharpe' in metrics:
        returns = calculate_returns(prices)
        result['sharpe'] = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(365)
    
    if 'sortino' in metrics:
        returns = calculate_returns(prices)
        downside_std = returns.rolling(window).apply(lambda x: x[x < 0].std(), raw=False)
        result['sortino'] = returns.rolling(window).mean() / downside_std * np.sqrt(365)
    
    return result
