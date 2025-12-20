# src/monitoring/pages/eda_correlation.py

"""
EDA: Correlation Analysis Page
"""

import streamlit as st
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.analysis.market_analyzer import (
    load_all_coins_data,
    calculate_correlation_matrix,
    calculate_rolling_correlation_with_btc
)


def render_correlation_page():
    """Render correlation analysis page."""
    st.title("🔗 EDA: Correlation Analysis")
    
    st.markdown("""
        Analyze correlations between coins to understand diversification potential.
    """)
    
    # Load data
    with st.spinner("Loading data..."):
        data_dict = load_all_coins_data(data_dir="data/raw/train")
    
    if not data_dict:
        st.error("No data available")
        return
    
    # Correlation Matrix
    st.subheader("📊 Correlation Matrix (Returns)")
    
    corr_matrix = calculate_correlation_matrix(data_dict, window=None)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=[coin.upper() for coin in corr_matrix.columns],
        y=[coin.upper() for coin in corr_matrix.index],
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Correlation Heatmap (Full Period)",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Rolling Correlation with Bitcoin
    st.markdown("---")
    st.subheader("📈 Rolling Correlation with Bitcoin (30-Day)")
    
    rolling_corrs = calculate_rolling_correlation_with_btc(data_dict, window=30)
    
    if rolling_corrs:
        fig = go.Figure()
        
        for coin, corr_series in rolling_corrs.items():
            fig.add_trace(go.Scatter(
                x=corr_series.index,
                y=corr_series,
                name=coin.upper(),
                mode='lines'
            ))
        
        fig.update_layout(
            title="Rolling 30-Day Correlation with Bitcoin",
            xaxis_title="Date",
            yaxis_title="Correlation",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Insights
    st.markdown("---")
    st.subheader("💡 Correlation Insights")
    
    avg_corr = corr_matrix.mean().mean()
    
    if avg_corr > 0.7:
        st.warning(f"⚠️ High average correlation ({avg_corr:.2f}) - Limited diversification benefits")
    elif avg_corr < 0.3:
        st.success(f"✅ Low average correlation ({avg_corr:.2f}) - Good diversification potential")
    else:
        st.info(f"ℹ️ Moderate average correlation ({avg_corr:.2f}) - Some diversification benefits")
