# src/monitoring/pages/factor_analysis.py

"""
Factor Analysis Page
"""

import streamlit as st
import plotly.express as px
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.analysis.market_analyzer import load_all_coins_data
from src.analysis.factor_analyzer import (
    create_factor_dataframe,
    factor_scatter_plot_data,
    cluster_by_factors
)


def render_factor_analysis_page():
    """Render factor analysis page."""
    st.title("🧩 Factor Analysis")
    
    st.markdown("""
        Understand what drives coin performance through factor analysis.
    """)
    
    # Load data
    with st.spinner("Analyzing factors..."):
        data_dict = load_all_coins_data(data_dir="data/raw/train")
    
    if not data_dict:
        st.error("No data available")
        return
    
    # Create factor dataframe
    factor_df = create_factor_dataframe(data_dict)
    
    # Factor Scatter Plot
    st.subheader("📊 Factor Scatter Plot")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_factor = st.selectbox("X-Axis", ['momentum_30d', 'momentum_90d', 'size', 'liquidity'])
    
    with col2:
        y_factor = st.selectbox("Y-Axis", ['volatility', 'momentum_30d', 'return_7d', 'size'])
    
    scatter_data = factor_scatter_plot_data(factor_df, x_factor=x_factor, y_factor=y_factor)
    
    if not scatter_data.empty:
        fig = px.scatter(
            scatter_data,
            x=x_factor,
            y=y_factor,
            text='coin',
            color='quadrant',
            title=f"{x_factor.replace('_', ' ').title()} vs {y_factor.replace('_', ' ').title()}",
            height=500
        )
        
        fig.update_traces(textposition='top center')
        st.plotly_chart(fig, use_container_width=True)
    
    # Clustering
    st.markdown("---")
    st.subheader("🔍 Coin Clustering")
    
    n_clusters = st.slider("Number of Clusters", 2, 5, 3)
    
    clustered_df = cluster_by_factors(factor_df, n_clusters=n_clusters)
    
    # Display clusters
    for cluster_id in sorted(clustered_df['cluster'].unique()):
        cluster_data = clustered_df[clustered_df['cluster'] == cluster_id]
        
        with st.expander(f"Cluster {cluster_id}: {cluster_data['cluster_description'].iloc[0]}"):
            st.write(f"**Coins**: {', '.join(cluster_data['coin'].str.upper())}")
            st.dataframe(
                cluster_data[['coin', 'momentum_30d', 'volatility', 'size']],
                use_container_width=True
            )
    
    # Factor Summary
    st.markdown("---")
    st.subheader("📋 Factor Summary")
    
    st.dataframe(
        factor_df[['coin', 'momentum_30d', 'momentum_90d', 'volatility', 'size', 'liquidity']].style.format({
            'momentum_30d': '{:.2f}%',
            'momentum_90d': '{:.2f}%',
            'volatility': '{:.2f}%',
            'size': '{:.2f}',
            'liquidity': '{:.4f}'
        }),
        use_container_width=True
    )
