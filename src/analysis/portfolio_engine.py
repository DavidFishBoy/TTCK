
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

from .financial_metrics import (
    calculate_returns,
    calculate_cagr,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_drawdown,
    calculate_volatility
)

logger = logging.getLogger(__name__)

def equal_weight_portfolio(n_coins: int) -> np.ndarray:
    return np.ones(n_coins) / n_coins

def risk_parity_portfolio(
    data_dict: Dict[str, pd.DataFrame],
    lookback: int = 90
) -> Dict[str, float]:
    volatilities = {}
    
    for coin, df in data_dict.items():
        if len(df) < lookback:
            volatilities[coin] = 0
            continue
        
        returns = calculate_returns(df['close'])
        vol = returns.tail(lookback).std()
        volatilities[coin] = vol if vol > 0 else 1e-10
    
    inv_vols = {coin: 1.0 / vol for coin, vol in volatilities.items()}
    total_inv_vol = sum(inv_vols.values())
    
    weights = {coin: inv_vol / total_inv_vol for coin, inv_vol in inv_vols.items()}
    
    return weights

def volatility_targeting_portfolio(
    data_dict: Dict[str, pd.DataFrame],
    target_vol: float = 0.15,
    lookback: int = 90
) -> Dict[str, float]:
    weights = equal_weight_portfolio(len(data_dict))
    coin_names = list(data_dict.keys())
    
    returns_dict = {}
    for coin, df in data_dict.items():
        returns = calculate_returns(df['close']).tail(lookback)
        returns_dict[coin] = returns
    
    returns_df = pd.DataFrame(returns_dict)
    cov_matrix = returns_df.cov() * 252
    
    portfolio_var = np.dot(weights, np.dot(cov_matrix.values, weights))
    portfolio_vol = np.sqrt(portfolio_var)
    
    scaling_factor = target_vol / portfolio_vol if portfolio_vol > 0 else 1.0
    scaled_weights = weights * scaling_factor
    
    scaled_weights = scaled_weights / scaled_weights.sum()
    
    return dict(zip(coin_names, scaled_weights))

def backtest_portfolio(
    data_dict: Dict[str, pd.DataFrame],
    weights: Dict[str, float],
    initial_capital: float = 10000.0,
    rebalance_freq: Optional[str] = None
) -> pd.DataFrame:
    aligned_data = {}
    for coin in weights.keys():
        if coin in data_dict:
            aligned_data[coin] = data_dict[coin]
    
    if not aligned_data:
        logger.error("No valid coins in weights dictionary")
        return pd.DataFrame()
    
    all_dates = None
    for coin, df in aligned_data.items():
        if all_dates is None:
            all_dates = df.index
        else:
            all_dates = all_dates.intersection(df.index)
    
    if len(all_dates) == 0:
        logger.error("No common dates found")
        return pd.DataFrame()
    
    price_matrix = pd.DataFrame(index=all_dates)
    for coin, df in aligned_data.items():
        price_matrix[coin] = df.loc[all_dates, 'close']
    
    n_periods = len(all_dates)
    portfolio_value = np.zeros(n_periods)
    portfolio_value[0] = initial_capital
    
    shares = {}
    for coin, weight in weights.items():
        if coin in price_matrix.columns:
            allocation = initial_capital * weight
            shares[coin] = allocation / price_matrix[coin].iloc[0]
        else:
            shares[coin] = 0
    
    for i in range(n_periods):
        total_value = 0
        for coin, n_shares in shares.items():
            if coin in price_matrix.columns:
                total_value += n_shares * price_matrix[coin].iloc[i]
        portfolio_value[i] = total_value
        
        if rebalance_freq and i > 0:
            if should_rebalance(all_dates[i], all_dates[i-1], rebalance_freq):
                for coin, weight in weights.items():
                    if coin in price_matrix.columns:
                        target_value = total_value * weight
                        shares[coin] = target_value / price_matrix[coin].iloc[i]
    
    result = pd.DataFrame({
        'portfolio_value': portfolio_value,
        'portfolio_return': (portfolio_value / initial_capital - 1) * 100
    }, index=all_dates)
    
    for coin in shares.keys():
        if coin in price_matrix.columns:
            result[f'{coin}_value'] = shares[coin] * price_matrix[coin]
    
    return result

def should_rebalance(current_date: pd.Timestamp, previous_date: pd.Timestamp, freq: str) -> bool:
    if freq == 'D':
        return True
    elif freq == 'W':
        return current_date.week != previous_date.week
    elif freq == 'M':
        return current_date.month != previous_date.month
    return False

def calculate_portfolio_metrics(
    portfolio_df: pd.DataFrame,
    risk_free_rate: float = 0.0
) -> Dict[str, float]:
    if 'portfolio_value' not in portfolio_df.columns:
        logger.error("Portfolio DataFrame must have 'portfolio_value' column")
        return {}
    
    portfolio_prices = portfolio_df['portfolio_value']
    
    total_return = ((portfolio_prices.iloc[-1] / portfolio_prices.iloc[0]) - 1) * 100
    cagr = calculate_cagr(portfolio_prices)
    sharpe = calculate_sharpe_ratio(portfolio_prices, risk_free_rate)
    sortino = calculate_sortino_ratio(portfolio_prices, risk_free_rate)
    
    _, max_dd, max_dd_duration = calculate_drawdown(portfolio_prices)
    
    annualized_vol = calculate_volatility(portfolio_prices, annualize=True)
    
    return {
        'total_return': total_return,
        'cagr': cagr,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_dd * 100,
        'max_drawdown_duration': max_dd_duration,
        'annualized_volatility': annualized_vol if isinstance(annualized_vol, float) else annualized_vol.iloc[-1],
    }

def calculate_risk_contribution(
    data_dict: Dict[str, pd.DataFrame],
    weights: Dict[str, float],
    lookback: int = 252
) -> Dict[str, float]:
    returns_dict = {}
    for coin in weights.keys():
        if coin in data_dict:
            returns = calculate_returns(data_dict[coin]['close']).tail(lookback)
            returns_dict[coin] = returns
    
    if not returns_dict:
        return {}
    
    returns_df = pd.DataFrame(returns_dict)
    cov_matrix = returns_df.cov() * 252
    
    weight_vector = np.array([weights.get(coin, 0) for coin in returns_df.columns])
    
    portfolio_var = np.dot(weight_vector, np.dot(cov_matrix.values, weight_vector))
    portfolio_vol = np.sqrt(portfolio_var)
    
    if portfolio_vol == 0:
        return {coin: 0.0 for coin in weights.keys()}
    
    mcr = np.dot(cov_matrix.values, weight_vector) / portfolio_vol
    
    cr = weight_vector * mcr
    
    pct_contribution = (cr / portfolio_vol) * 100
    
    risk_contrib = {}
    for i, coin in enumerate(returns_df.columns):
        risk_contrib[coin] = pct_contribution[i]
    
    return risk_contrib

def rebalance_simulation(
    data_dict: Dict[str, pd.DataFrame],
    weights: Dict[str, float],
    rebalance_freq: str = 'M',
    initial_capital: float = 10000.0
) -> Tuple[pd.DataFrame, List[pd.Timestamp]]:
    portfolio_df = backtest_portfolio(
        data_dict=data_dict,
        weights=weights,
        initial_capital=initial_capital,
        rebalance_freq=rebalance_freq
    )
    
    rebalance_dates = []
    prev_date = None
    
    for date in portfolio_df.index:
        if prev_date is not None:
            if should_rebalance(date, prev_date, rebalance_freq):
                rebalance_dates.append(date)
        prev_date = date
    
    return portfolio_df, rebalance_dates

def compare_portfolio_strategies(
    data_dict: Dict[str, pd.DataFrame],
    initial_capital: float = 10000.0
) -> pd.DataFrame:
    strategies = {}
    
    eq_weights = {coin: 1.0 / len(data_dict) for coin in data_dict.keys()}
    eq_portfolio = backtest_portfolio(data_dict, eq_weights, initial_capital)
    strategies['Equal Weight'] = calculate_portfolio_metrics(eq_portfolio)
    
    rp_weights = risk_parity_portfolio(data_dict)
    rp_portfolio = backtest_portfolio(data_dict, rp_weights, initial_capital)
    strategies['Risk Parity'] = calculate_portfolio_metrics(rp_portfolio)
    
    vt_weights = volatility_targeting_portfolio(data_dict, target_vol=0.20)
    vt_portfolio = backtest_portfolio(data_dict, vt_weights, initial_capital)
    strategies['Vol Targeting'] = calculate_portfolio_metrics(vt_portfolio)
    
    comparison_df = pd.DataFrame(strategies).T
    
    return comparison_df
