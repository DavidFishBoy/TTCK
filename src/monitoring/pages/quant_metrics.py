# src/monitoring/pages/quant_metrics.py

"""
Quant Metrics Page - Risk-adjusted performance comparison
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.analysis.market_analyzer import load_all_coins_data
from src.analysis.financial_metrics import get_all_metrics


def render_quant_metrics_page():
    """Render quantitative metrics comparison page."""
    st.title("📐 Quant Metrics - Risk-Adjusted Performance")
    
    st.markdown("""
        Compare coins based on quantitative metrics used by professional investors.
    """)
    
    # Load data
    with st.spinner("Calculating metrics..."):
        data_dict = load_all_coins_data(data_dir="data/raw/train")
    
    if not data_dict:
        st.error("No data available")
        return
    
    # Calculate metrics for all coins
    all_metrics = []
    for coin, df in data_dict.items():
        metrics = get_all_metrics(df['close'], coin_name=coin)
        if 'error' not in metrics:
            all_metrics.append(metrics)
    
    if not all_metrics:
        st.error("Failed to calculate metrics")
        return
    
    metrics_df = pd.DataFrame(all_metrics)
    
    # Display ranking table
    st.subheader("🏆 Coin Rankings by Metrics")
    
    # Let user choose metric to sort by
    sort_by = st.selectbox(
        "Sort by",
        ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'cagr', 'max_drawdown'],
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    ascending = (sort_by == 'max_drawdown')
    sorted_df = metrics_df.sort_values(sort_by, ascending=ascending)
    
    # Display table with formatting
    display_df = sorted_df[[
        'coin', 'current_price', 'cagr', 'annualized_volatility',
        'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'max_drawdown'
    ]].copy()
    
    display_df.columns = [
        'Coin', 'Price', 'CAGR', 'Vol',
        'Sharpe', 'Sortino', 'Calmar', 'Max DD'
    ]
    
    st.dataframe(
        display_df.style.format({
            'Price': '${:.2f}',
            'CAGR': '{:.2f}%',
            'Vol': '{:.2f}%',
            'Sharpe': '{:.2f}',
            'Sortino': '{:.2f}',
            'Calmar': '{:.2f}',
            'Max DD': '{:.2f}%'
        }),
        use_container_width=True,
        height=400
    )
    
    # Key Metrics Explanation
    st.markdown("---")
    st.subheader("📚 Metrics Explanation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **CAGR**: Compound Annual Growth Rate  
        **Volatility**: Annualized price volatility  
        **Sharpe Ratio**: Return per unit of risk  
        """)
    
    with col2:
        st.markdown("""
        **Sortino Ratio**: Return per unit of downside risk  
        **Calmar Ratio**: Return per unit of maximum drawdown  
        **Max DD**: Largest peak-to-trough decline  
        """)
    
    # Top Performers
    st.markdown("---")
    st.subheader("🎯 Top Performers")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Best Sharpe Ratio**")
        top_sharpe = sorted_df.nlargest(3, 'sharpe_ratio')
        for _, row in top_sharpe.iterrows():
            st.write(f"- **{row['coin'].upper()}**: {row['sharpe_ratio']:.2f}")
    
    with col2:
        st.markdown("**Best Sortino Ratio**")
        top_sortino = sorted_df.nlargest(3, 'sortino_ratio')
        for _, row in top_sortino.iterrows():
            st.write(f"- **{row['coin'].upper()}**: {row['sortino_ratio']:.2f}")
    
    with col3:
        st.markdown("**Lowest Drawdown**")
        top_dd = sorted_df.nsmallest(3, 'max_drawdown')
        for _, row in top_dd.iterrows():
            st.write(f"- **{row['coin'].upper()}**: {row['max_drawdown']:.2f}%")
