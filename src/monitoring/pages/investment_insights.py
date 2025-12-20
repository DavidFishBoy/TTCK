# src/monitoring/pages/investment_insights.py

"""
Investment Insights Page - Summary and recommendations
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.analysis.market_analyzer import (
    load_all_coins_data,
    identify_market_regime,
    calculate_correlation_matrix
)
from src.analysis.financial_metrics import get_all_metrics


def render_investment_insights_page():
    """Render investment insights summary page."""
    st.title("🧠 Investment Insights")
    
    st.markdown("""
        Comprehensive summary and actionable investment recommendations.
    """)
    
    # Load data
    with st.spinner("Analyzing market..."):
        data_dict = load_all_coins_data(data_dir="data/raw/train")
    
    if not data_dict:
        st.error("No data available")
        return
    
    # Market Regime
    st.subheader("🌍 Current Market Regime")
    
    regime_info = identify_market_regime(data_dict)
    
    regime_colors = {
        "Bull": "green",
        "Bear": "red",
        "Sideway": "orange"
    }
    
    st.markdown(f"""
        <div style='padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 10px; color: white; margin-bottom: 1rem;'>
            <h2 style='margin: 0; color: white;'>{regime_info['regime']} Market</h2>
            <p style='margin: 0.5rem 0 0 0; font-size: 1.1rem;'>{regime_info['description']}</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Coins Above MA200", f"{regime_info['pct_coins_above_ma']:.0f}%")
    
    with col2:
        st.metric("Avg Volatility", f"{regime_info['avg_volatility']:.1f}%")
    
    with col3:
        st.metric("Vol Regime", regime_info['volatility_regime'])
    
    # Top 3 Watchlist
    st.markdown("---")
    st.subheader("🎯 Top 3 Coins to Watch")
    
    # Calculate all metrics
    all_metrics = []
    for coin, df in data_dict.items():
        metrics = get_all_metrics(df['close'], coin_name=coin)
        if 'error' not in metrics:
            all_metrics.append(metrics)
    
    metrics_df = pd.DataFrame(all_metrics)
    
    # Rank by Sharpe ratio
    top_3 = metrics_df.nlargest(3, 'sharpe_ratio')
    
    for idx, row in top_3.iterrows():
        with st.expander(f"#{idx+1}: {row['coin'].upper()} - Sharpe: {row['sharpe_ratio']:.2f}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Price", f"${row['current_price']:.2f}")
                st.metric("CAGR", f"{row['cagr']:.2f}%")
            
            with col2:
                st.metric("Volatility", f"{row['annualized_volatility']:.2f}%")
                st.metric("Sharpe", f"{row['sharpe_ratio']:.2f}")
            
            with col3:
                st.metric("Max Drawdown", f"{row['max_drawdown']:.2f}%")
                st.metric("Sortino", f"{row['sortino_ratio']:.2f}")
            
            st.markdown(f"""
                **Why Watch**: Strong risk-adjusted returns with Sharpe ratio of {row['sharpe_ratio']:.2f}. 
                Suitable for investors seeking balance between returns and risk.
            """)
    
    # Risk Warnings
    st.markdown("---")
    st.subheader("⚠️ Risk Warnings")
    
    # Check correlation
    corr_matrix = calculate_correlation_matrix(data_dict)
    avg_corr = corr_matrix.mean().mean()
    
    warnings = []
    
    if avg_corr > 0.7:
        warnings.append("🔴 High correlation between assets - Limited diversification benefits")
    
    if regime_info['volatility_regime'] == "High":
        warnings.append("🔴 High volatility environment - Increased risk of sharp price movements")
    
    if regime_info['regime'] == "Bear":
        warnings.append("🔴 Bear market conditions - Capital preservation should be priority")
    
    if warnings:
        for warning in warnings:
            st.warning(warning)
    else:
        st.success("✅ No major risk warnings at this time")
    
    # Action Scenarios
    st.markdown("---")
    st.subheader("📋 Recommended Actions")
    
    if regime_info['regime'] == "Bull" and regime_info['volatility_regime'] == "Low":
        st.success("""
            ### 🟢 Aggressive Growth Strategy
            
            **Market Conditions**: Bullish with low volatility
            
            **Recommended Actions**:
            - ✅ Increase exposure to high momentum stocks
            - ✅ Consider trend-following strategies
            - ✅ Can take larger position sizes
            - ⚠️ Keep stop-losses to protect gains
        """)
    
    elif regime_info['regime'] == "Bear":
        st.error("""
            ### 🔴 Defensive Strategy
            
            **Market Conditions**: Bearish trend
            
            **Recommended Actions**:
            - 🛑 Reduce overall exposure
            - 💰 Preserve capital - wait for better entry
            - 📉 Consider short positions or hedging
            - ⏰ Be patient for trend reversal signals
        """)
    
    else:
        st.info("""
            ### 🟡 Balanced Strategy
            
            **Market Conditions**: Mixed/Sideways
            
            **Recommended Actions**:
            - 🎯 Be selective with entries
            - ⚖️ Maintain balanced portfolio
            - 📊 Focus on individual coin analysis
            - 🔄 Consider range-trading strategies
        """)
