# src/monitoring/pages/portfolio_analysis.py

"""
Portfolio Analysis Page
"""

import streamlit as st
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.analysis.market_analyzer import load_all_coins_data
from src.analysis.portfolio_engine import (
    equal_weight_portfolio,
    risk_parity_portfolio,
    backtest_portfolio,
    calculate_portfolio_metrics,
    compare_portfolio_strategies
)


def render_portfolio_analysis_page():
    """Render portfolio analysis page."""
    st.title("🧺 Portfolio Analysis")
    
    st.markdown("""
        Analyze different portfolio construction strategies and their performance.
    """)
    
    # Load data
    with st.spinner("Running portfolio backtests..."):
        data_dict = load_all_coins_data(data_dir="data/raw/train")
    
    if not data_dict:
        st.error("No data available")
        return
    
    # Strategy Comparison
    st.subheader("📊 Strategy Comparison")
    
    comparison_df = compare_portfolio_strategies(data_dict, initial_capital=10000)
    
    st.dataframe(
        comparison_df.style.format({
            'total_return': '{:.2f}%',
            'cagr': '{:.2f}%',
            'sharpe_ratio': '{:.2f}',
            'sortino_ratio': '{:.2f}',
            'max_drawdown': '{:.2f}%',
            'annualized_volatility': '{:.2f}%'
        }),
        use_container_width=True
    )
    
    # Individual Strategy Analysis
    st.markdown("---")
    st.subheader("🔍 Detailed Strategy Analysis")
    
    strategy = st.selectbox(
        "Select Strategy",
        ["Equal Weight", "Risk Parity"]
    )
    
    if strategy == "Equal Weight":
        weights = {coin: 1.0 / len(data_dict) for coin in data_dict.keys()}
    else:  # Risk Parity
        weights = risk_parity_portfolio(data_dict)
    
    # Backtest
    portfolio_df = backtest_portfolio(data_dict, weights, initial_capital=10000)
    
    if not portfolio_df.empty:
        # Equity Curve
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=portfolio_df.index,
            y=portfolio_df['portfolio_value'],
            name='Portfolio Value',
            line=dict(color='#667eea', width=2)
        ))
        
        fig.update_layout(
            title=f"{strategy} Portfolio Equity Curve",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics
        metrics = calculate_portfolio_metrics(portfolio_df)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Return", f"{metrics['total_return']:.2f}%")
        
        with col2:
            st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        
        with col3:
            st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
        
        with col4:
            st.metric("CAGR", f"{metrics['cagr']:.2f}%")
        
        # Weights
        st.markdown("---")
        st.subheader("⚖️ Portfolio Weights")
        
        weights_df = pd.DataFrame.from_dict(weights, orient='index', columns=['Weight'])
        weights_df['Weight'] = weights_df['Weight'] * 100
        weights_df = weights_df.sort_values('Weight', ascending=False)
        
        st.dataframe(
            weights_df.style.format({'Weight': '{:.2f}%'}),
            use_container_width=True
        )
    
    # Recommendations
    st.markdown("---")
    st.subheader("💡 Portfolio Recommendations")
    
    best_strategy = comparison_df['sharpe_ratio'].idxmax()
    
    st.success(f"""
        **Recommended Strategy**: {best_strategy}  
        Based on risk-adjusted returns (Sharpe Ratio), {best_strategy} shows the best performance.
    """)
