
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging

from .financial_metrics import calculate_returns, calculate_volatility

logger = logging.getLogger(__name__)

def calculate_momentum(
    df: pd.DataFrame,
    periods: List[int] = [30, 90]
) -> Dict[str, float]:
    momentum = {}
    
    for period in periods:
        if len(df) >= period:
            mom = (df['close'].iloc[-1] / df['close'].iloc[-period] - 1) * 100
            momentum[f'momentum_{period}d'] = mom
        else:
            momentum[f'momentum_{period}d'] = 0.0
    
    return momentum

def calculate_size_factor(df: pd.DataFrame) -> float:
    if 'market_cap' in df.columns and not df['market_cap'].isna().all():
        market_cap = df['market_cap'].iloc[-1]
        return np.log10(market_cap + 1) if market_cap > 0 else 0
    else:
        return np.log10(df['close'].iloc[-1] + 1)

def calculate_liquidity_factor(df: pd.DataFrame, window: int = 7) -> float:
    avg_volume = df['volume'].tail(window).mean()
    
    if 'market_cap' in df.columns and not df['market_cap'].isna().all():
        market_cap = df['market_cap'].iloc[-1]
        if market_cap > 0:
            return avg_volume / market_cap
    
    return avg_volume / (df['volume'].mean() + 1e-10)

def calculate_volatility_factor(df: pd.DataFrame, window: int = 30) -> float:
    returns = calculate_returns(df['close'])
    vol = returns.tail(window).std() * np.sqrt(365) * 100
    return vol

def create_factor_dataframe(data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    factor_data = []
    
    for coin, df in data_dict.items():
        row = {'coin': coin}
        
        momentum = calculate_momentum(df, periods=[30, 90])
        row.update(momentum)
        
        row['size'] = calculate_size_factor(df)
        
        row['liquidity'] = calculate_liquidity_factor(df)
        
        row['volatility'] = calculate_volatility_factor(df)
        
        row['current_price'] = df['close'].iloc[-1]
        
        if len(df) >= 7:
            row['return_7d'] = (df['close'].iloc[-1] / df['close'].iloc[-7] - 1) * 100
        else:
            row['return_7d'] = 0.0
        
        factor_data.append(row)
    
    return pd.DataFrame(factor_data)

def factor_scatter_plot_data(
    factor_df: pd.DataFrame,
    x_factor: str = 'momentum_30d',
    y_factor: str = 'volatility'
) -> pd.DataFrame:
    if x_factor not in factor_df.columns or y_factor not in factor_df.columns:
        logger.error(f"Factors {x_factor} or {y_factor} not found in DataFrame")
        return pd.DataFrame()
    
    plot_data = factor_df[['coin', x_factor, y_factor]].copy()
    
    x_median = plot_data[x_factor].median()
    y_median = plot_data[y_factor].median()
    
    def get_quadrant(row):
        if row[x_factor] >= x_median and row[y_factor] >= y_median:
            return "High Momentum, High Volatility"
        elif row[x_factor] >= x_median and row[y_factor] < y_median:
            return "High Momentum, Low Volatility"
        elif row[x_factor] < x_median and row[y_factor] >= y_median:
            return "Low Momentum, High Volatility"
        else:
            return "Low Momentum, Low Volatility"
    
    plot_data['quadrant'] = plot_data.apply(get_quadrant, axis=1)
    
    return plot_data

def cluster_by_factors(
    factor_df: pd.DataFrame,
    n_clusters: int = 3,
    features: Optional[List[str]] = None
) -> pd.DataFrame:
    if features is None:
        features = [col for col in factor_df.columns 
                   if col not in ['coin', 'current_price'] and pd.api.types.is_numeric_dtype(factor_df[col])]
    
    if not features:
        logger.error("No numeric features found for clustering")
        return factor_df
    
    X = factor_df[features].fillna(0).values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    result = factor_df.copy()
    result['cluster'] = cluster_labels
    
    cluster_descriptions = []
    for i in range(n_clusters):
        cluster_mask = result['cluster'] == i
        cluster_coins = result.loc[cluster_mask, 'coin'].tolist()
        
        avg_mom = result.loc[cluster_mask, 'momentum_30d'].mean() if 'momentum_30d' in result.columns else 0
        avg_vol = result.loc[cluster_mask, 'volatility'].mean() if 'volatility' in result.columns else 0
        
        desc = f"Cluster {i}: "
        if avg_mom > 0:
            desc += "Positive momentum, " if avg_mom > 5 else "Slight positive momentum, "
        else:
            desc += "Negative momentum, " if avg_mom < -5 else "Neutral momentum, "
        
        if avg_vol > 80:
            desc += "High volatility"
        elif avg_vol > 40:
            desc += "Medium volatility"
        else:
            desc += "Low volatility"
        
        cluster_descriptions.append({
            'cluster': i,
            'description': desc,
            'coins': ', '.join(cluster_coins),
            'avg_momentum': avg_mom,
            'avg_volatility': avg_vol
        })
    
    desc_map = {cd['cluster']: cd['description'] for cd in cluster_descriptions}
    result['cluster_description'] = result['cluster'].map(desc_map)
    
    return result

def pca_analysis(
    factor_df: pd.DataFrame,
    n_components: int = 2,
    features: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if features is None:
        features = [col for col in factor_df.columns 
                   if col not in ['coin', 'current_price'] and pd.api.types.is_numeric_dtype(factor_df[col])]
    
    if not features:
        logger.error("No numeric features found for PCA")
        return pd.DataFrame(), pd.DataFrame()
    
    X = factor_df[features].fillna(0).values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    pca_columns = [f'PC{i+1}' for i in range(n_components)]
    transformed_df = pd.DataFrame(X_pca, columns=pca_columns)
    transformed_df['coin'] = factor_df['coin'].values
    
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=pca_columns,
        index=features
    )
    
    explained_var = pd.Series(
        pca.explained_variance_ratio_ * 100,
        index=pca_columns,
        name='explained_variance_pct'
    )
    
    logger.info(f"PCA explained variance: {explained_var.to_dict()}")
    
    return transformed_df, loadings

def rank_coins_by_factor(
    factor_df: pd.DataFrame,
    factor_name: str,
    ascending: bool = False
) -> pd.DataFrame:
    if factor_name not in factor_df.columns:
        logger.error(f"Factor {factor_name} not found in DataFrame")
        return pd.DataFrame()
    
    ranked = factor_df.sort_values(by=factor_name, ascending=ascending).reset_index(drop=True)
    ranked['rank'] = range(1, len(ranked) + 1)
    
    return ranked[['rank', 'coin', factor_name]]

def identify_factor_extremes(
    factor_df: pd.DataFrame,
    top_n: int = 3
) -> Dict[str, List[str]]:
    extremes = {}
    
    numeric_factors = [col for col in factor_df.columns 
                      if col not in ['coin', 'current_price', 'cluster', 'cluster_description'] 
                      and pd.api.types.is_numeric_dtype(factor_df[col])]
    
    for factor in numeric_factors:
        sorted_df = factor_df.sort_values(by=factor, ascending=False)
        
        top_coins = sorted_df.head(top_n)['coin'].tolist()
        bottom_coins = sorted_df.tail(top_n)['coin'].tolist()
        
        extremes[factor] = {
            'top': top_coins,
            'bottom': bottom_coins,
            'top_values': sorted_df.head(top_n)[factor].tolist(),
            'bottom_values': sorted_df.tail(top_n)[factor].tolist()
        }
    
    return extremes
