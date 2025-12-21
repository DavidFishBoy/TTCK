
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

def calculate_returns(prices: pd.Series, method: str = 'simple') -> pd.Series:
    if method == 'log':
        return np.log(prices / prices.shift(1))
    else:
        return prices.pct_change()

def calculate_volatility(
    prices: pd.Series, 
    window: Optional[int] = None,
    annualize: bool = True,
    periods_per_year: int = 365
) -> Union[float, pd.Series]:
    returns = calculate_returns(prices)
    
    if window is None:
        vol = returns.std()
    else:
        vol = returns.rolling(window=window).std()
    
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    
    return vol

def calculate_drawdown(prices: pd.Series) -> Tuple[pd.Series, float, int]:
    cumulative = (1 + calculate_returns(prices)).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    
    max_dd = drawdown.min()
    
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
    returns = calculate_returns(prices).dropna()
    
    var = returns.quantile(1 - confidence_level)
    
    cvar = returns[returns <= var].mean()
    
    return var * 100, cvar * 100

def calculate_cagr(prices: pd.Series, periods_per_year: int = 365) -> float:
    if len(prices) < 2:
        return 0.0
    
    total_return = prices.iloc[-1] / prices.iloc[0]
    n_periods = len(prices)
    n_years = n_periods / periods_per_year
    
    if n_years <= 0 or total_return <= 0:
        return 0.0
    
    cagr = (total_return ** (1 / n_years)) - 1
    return cagr * 100

def calculate_sharpe_ratio(
    prices: pd.Series, 
    risk_free_rate: float = 0.0,
    periods_per_year: int = 365
) -> float:
    returns = calculate_returns(prices).dropna()
    
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    
    period_rf = risk_free_rate / periods_per_year
    
    excess_returns = returns - period_rf
    sharpe = excess_returns.mean() / returns.std() * np.sqrt(periods_per_year)
    
    return sharpe

def calculate_sortino_ratio(
    prices: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 365
) -> float:
    returns = calculate_returns(prices).dropna()
    
    if len(returns) == 0:
        return 0.0
    
    period_rf = risk_free_rate / periods_per_year
    
    excess_returns = returns - period_rf
    
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0
    
    downside_std = downside_returns.std()
    sortino = excess_returns.mean() / downside_std * np.sqrt(periods_per_year)
    
    return sortino

def calculate_calmar_ratio(prices: pd.Series, periods_per_year: int = 365) -> float:
    cagr = calculate_cagr(prices, periods_per_year) / 100
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
    try:
        total_return = ((prices.iloc[-1] / prices.iloc[0]) - 1) * 100
        
        annualized_vol = calculate_volatility(prices, annualize=True, periods_per_year=periods_per_year)
        
        _, max_dd, max_dd_duration = calculate_drawdown(prices)
        
        var_95, cvar_95 = calculate_var_cvar(prices, confidence_level=0.95)
        
        cagr = calculate_cagr(prices, periods_per_year)
        sharpe = calculate_sharpe_ratio(prices, risk_free_rate, periods_per_year)
        sortino = calculate_sortino_ratio(prices, risk_free_rate, periods_per_year)
        calmar = calculate_calmar_ratio(prices, periods_per_year)
        
        return {
            'coin': coin_name,
            'total_return': total_return,
            'cagr': cagr,
            'annualized_volatility': annualized_vol if isinstance(annualized_vol, float) else annualized_vol.iloc[-1],
            'max_drawdown': max_dd * 100,
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
