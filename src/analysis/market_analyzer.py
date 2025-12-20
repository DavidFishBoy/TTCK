# src/analysis/market_analyzer.py

"""
Market analyzer module for multi-coin analysis.

Provides functions for:
- Loading and aligning data from multiple coins
- Market breadth analysis
- Correlation analysis
- Volume spike detection
- Market regime identification
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

from .financial_metrics import calculate_returns

logger = logging.getLogger(__name__)


def load_all_coins_data(
    data_dir: str = "data/raw/train",
    coins: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load historical data for all coins and align to common timeframe.
    
    Args:
        data_dir: Directory containing CSV files
        coins: List of coin names. If None, loads all available coins
    
    Returns:
        Dictionary mapping coin name to DataFrame
    """
    data_dir = Path(data_dir)
    
    if coins is None:
        coins = [
            "bitcoin", "ethereum", "litecoin", "binancecoin",
            "cardano", "solana", "pancakeswap", "axieinfinity", "thesandbox"
        ]
    
    all_data = {}
    
    for coin in coins:
        # Find the most recent file for this coin
        pattern = f"{coin}_binance_*.csv"
        csv_files = list(data_dir.glob(pattern))
        
        if not csv_files:
            logger.warning(f"No data found for {coin}")
            continue
        
        latest_file = sorted(csv_files)[-1]
        
        try:
            df = pd.read_csv(latest_file)
            
            # Parse timestamp and set as index
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            elif df.index.name != 'timestamp':
                # Try to parse the index
                df.index = pd.to_datetime(df.index)
            
            # Keep only essential columns
            essential_cols = ['open', 'high', 'low', 'close', 'volume']
            if 'market_cap' in df.columns:
                essential_cols.append('market_cap')
            
            df = df[essential_cols]
            all_data[coin] = df
            
            logger.info(f"Loaded {len(df)} rows for {coin}")
            
        except Exception as e:
            logger.error(f"Error loading {coin}: {e}")
            continue
    
    # Align all dataframes to common date range
    if all_data:
        all_data = align_dataframes(all_data)
    
    return all_data


def align_dataframes(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Align all dataframes to the common date range.
    
    Args:
        data_dict: Dictionary of coin name to DataFrame
    
    Returns:
        Aligned dictionary
    """
    if not data_dict:
        return {}
    
    # Find common date range
    start_dates = [df.index.min() for df in data_dict.values()]
    end_dates = [df.index.max() for df in data_dict.values()]
    
    common_start = max(start_dates)
    common_end = min(end_dates)
    
    logger.info(f"Aligning data from {common_start} to {common_end}")
    
    aligned = {}
    for coin, df in data_dict.items():
        aligned[coin] = df.loc[common_start:common_end].copy()
    
    return aligned


def calculate_market_breadth(
    data_dict: Dict[str, pd.DataFrame],
    periods: List[int] = [1, 7, 30]
) -> pd.DataFrame:
    """
    Calculate market breadth (% of coins up/down over various periods).
    
    Args:
        data_dict: Dictionary of coin name to DataFrame
        periods: List of periods to calculate (in days)
    
    Returns:
        DataFrame with breadth metrics
    """
    results = []
    
    for period in periods:
        up_count = 0
        down_count = 0
        total_count = 0
        
        for coin, df in data_dict.items():
            if len(df) < period:
                continue
            
            period_return = (df['close'].iloc[-1] / df['close'].iloc[-period] - 1) * 100
            
            if period_return > 0:
                up_count += 1
            elif period_return < 0:
                down_count += 1
            
            total_count += 1
        
        if total_count > 0:
            results.append({
                'period': f'{period}D',
                'coins_up': up_count,
                'coins_down': down_count,
                'pct_up': (up_count / total_count) * 100,
                'pct_down': (down_count / total_count) * 100,
                'total_coins': total_count
            })
    
    return pd.DataFrame(results)


def create_returns_heatmap(
    data_dict: Dict[str, pd.DataFrame],
    periods: List[int] = [1, 7, 30]
) -> pd.DataFrame:
    """
    Create a returns heatmap for all coins across different periods.
    
    Args:
        data_dict: Dictionary of coin name to DataFrame
        periods: List of periods (in days)
    
    Returns:
        DataFrame with coins as rows and periods as columns
    """
    heatmap_data = []
    
    for coin, df in data_dict.items():
        row = {'coin': coin}
        
        for period in periods:
            if len(df) >= period:
                period_return = (df['close'].iloc[-1] / df['close'].iloc[-period] - 1) * 100
                row[f'{period}D'] = period_return
            else:
                row[f'{period}D'] = np.nan
        
        heatmap_data.append(row)
    
    return pd.DataFrame(heatmap_data)


def rank_by_metric(
    data_dict: Dict[str, pd.DataFrame],
    metric: str = 'market_cap',
    ascending: bool = False
) -> pd.DataFrame:
    """
    Rank coins by a specific metric.
    
    Args:
        data_dict: Dictionary of coin name to DataFrame
        metric: Metric to rank by ('market_cap', 'volume', 'close', 'volatility')
        ascending: Sort order
    
    Returns:
        DataFrame with rankings
    """
    rankings = []
    
    for coin, df in data_dict.items():
        row = {'coin': coin}
        
        if metric == 'volatility':
            # Calculate 14-day volatility
            returns = calculate_returns(df['close'])
            vol = returns.tail(14).std() * np.sqrt(365) * 100
            row['value'] = vol
        elif metric == 'volume':
            row['value'] = df['volume'].tail(7).mean()  # 7-day avg volume
        elif metric == 'market_cap':
            if 'market_cap' in df.columns:
                row['value'] = df['market_cap'].iloc[-1]
            else:
                row['value'] = 0
        elif metric == 'close':
            row['value'] = df['close'].iloc[-1]
        else:
            row['value'] = 0
        
        rankings.append(row)
    
    df_rank = pd.DataFrame(rankings)
    df_rank = df_rank.sort_values('value', ascending=ascending).reset_index(drop=True)
    df_rank['rank'] = range(1, len(df_rank) + 1)
    
    return df_rank


def calculate_correlation_matrix(
    data_dict: Dict[str, pd.DataFrame],
    window: Optional[int] = None
) -> pd.DataFrame:
    """
    Calculate correlation matrix of returns across all coins.
    
    Args:
        data_dict: Dictionary of coin name to DataFrame
        window: Rolling window size. If None, use entire period
    
    Returns:
        Correlation matrix as DataFrame
    """
    # Build a DataFrame of returns for all coins
    returns_dict = {}
    
    for coin, df in data_dict.items():
        returns = calculate_returns(df['close'])
        returns_dict[coin] = returns
    
    returns_df = pd.DataFrame(returns_dict)
    
    if window is None:
        corr_matrix = returns_df.corr()
    else:
        # Return the most recent rolling correlation
        corr_matrix = returns_df.tail(window).corr()
    
    return corr_matrix


def detect_volume_spike(
    df: pd.DataFrame,
    window: int = 20,
    threshold: float = 2.0
) -> pd.Series:
    """
    Detect volume spikes using z-score.
    
    Args:
        df: DataFrame with 'volume' column
        window: Rolling window for calculating mean and std
        threshold: Z-score threshold for spike detection
    
    Returns:
        Series of z-scores
    """
    volume_ma = df['volume'].rolling(window=window).mean()
    volume_std = df['volume'].rolling(window=window).std()
    
    z_scores = (df['volume'] - volume_ma) / (volume_std + 1e-10)
    
    return z_scores


def identify_market_regime(
    data_dict: Dict[str, pd.DataFrame],
    ma_period: int = 200
) -> Dict[str, str]:
    """
    Identify market regime (Bull/Bear/Sideway) for the overall market.
    
    Args:
        data_dict: Dictionary of coin name to DataFrame
        ma_period: Moving average period for trend identification
    
    Returns:
        Dictionary with regime information
    """
    # Calculate breadth
    breadth = calculate_market_breadth(data_dict, periods=[7, 30])
    
    # Count coins above/below MA
    above_ma = 0
    below_ma = 0
    
    for coin, df in data_dict.items():
        if len(df) < ma_period:
            continue
        
        ma = df['close'].rolling(window=ma_period).mean()
        current_price = df['close'].iloc[-1]
        current_ma = ma.iloc[-1]
        
        if current_price > current_ma:
            above_ma += 1
        else:
            below_ma += 1
    
    total = above_ma + below_ma
    pct_above = (above_ma / total * 100) if total > 0 else 0
    
    # Determine regime
    if pct_above > 70:
        regime = "Bull"
        description = "Market is in a strong uptrend"
    elif pct_above < 30:
        regime = "Bear"
        description = "Market is in a downtrend"
    else:
        regime = "Sideway"
        description = "Market is consolidating"
    
    # Calculate average volatility
    total_vol = 0
    vol_count = 0
    for coin, df in data_dict.items():
        returns = calculate_returns(df['close'])
        vol = returns.tail(14).std() * np.sqrt(365)
        total_vol += vol
        vol_count += 1
    
    avg_vol = (total_vol / vol_count * 100) if vol_count > 0 else 0
    
    # Volatility regime
    if avg_vol > 80:
        vol_regime = "High"
    elif avg_vol > 40:
        vol_regime = "Medium"
    else:
        vol_regime = "Low"
    
    return {
        'regime': regime,
        'description': description,
        'pct_coins_above_ma': pct_above,
        'avg_volatility': avg_vol,
        'volatility_regime': vol_regime,
        'breadth_7d_up': breadth[breadth['period'] == '7D']['pct_up'].values[0] if len(breadth) > 0 else 0,
        'breadth_30d_up': breadth[breadth['period'] == '30D']['pct_up'].values[0] if len(breadth) > 1 else 0,
    }


def calculate_rolling_correlation_with_btc(
    data_dict: Dict[str, pd.DataFrame],
    window: int = 30
) -> Dict[str, pd.Series]:
    """
    Calculate rolling correlation of each coin with Bitcoin.
    
    Args:
        data_dict: Dictionary of coin name to DataFrame
        window: Rolling window size
    
    Returns:
        Dictionary mapping coin name to rolling correlation series
    """
    if 'bitcoin' not in data_dict:
        logger.warning("Bitcoin data not found, cannot calculate correlations")
        return {}
    
    btc_returns = calculate_returns(data_dict['bitcoin']['close'])
    
    correlations = {}
    
    for coin, df in data_dict.items():
        if coin == 'bitcoin':
            continue
        
        coin_returns = calculate_returns(df['close'])
        
        # Align the two series
        aligned = pd.concat([btc_returns, coin_returns], axis=1, keys=['btc', coin]).dropna()
        
        if len(aligned) < window:
            continue
        
        rolling_corr = aligned['btc'].rolling(window=window).corr(aligned[coin])
        correlations[coin] = rolling_corr
    
    return correlations
