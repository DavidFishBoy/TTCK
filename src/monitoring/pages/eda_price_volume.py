# src/monitoring/pages/eda_price_volume.py

"""
EDA: Price & Volume Analysis Page
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.analysis.market_analyzer import load_all_coins_data, detect_volume_spike


def render_price_volume_page(coin: str):
    """Render price and volume analysis page for a specific coin."""
    if not coin:
        st.warning("Please select a coin from the sidebar")
        return
    
    st.title(f"📈 EDA: Price & Volume - {coin.upper()}")
    
    # Load data for selected coin
    with st.spinner(f"Loading {coin} data..."):
        data_dict = load_all_coins_data(data_dir="data/raw/train")
    
    if coin not in data_dict:
        st.error(f"No data found for {coin}")
        return
    
    df = data_dict[coin]
    
    # Price with Moving Averages
    st.subheader("📊 Price with Moving Averages")
    
    # Calculate MAs
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()
    df['MA200'] = df['close'].rolling(window=200).mean()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['close'],
        name='Close Price',
        line=dict(color='#2E86DE', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MA20'],
        name='MA20',
        line=dict(color='orange', width=1, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MA50'],
        name='MA50',
        line=dict(color='green', width=1, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MA200'],
        name='MA200',
        line=dict(color='red', width=1, dash='dot')
    ))
    
    fig.update_layout(
        title=f"{coin.upper()} Price with Moving Averages",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Returns Distribution
    st.markdown("---")
    st.subheader("📉 Daily Returns Distribution")
    
    df['returns'] = df['close'].pct_change() * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure(data=[go.Histogram(
            x=df['returns'].dropna(),
            nbinsx=50,
            marker_color='#667eea'
        )])
        
        fig.update_layout(
            title="Returns Histogram",
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency",
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Returns statistics
        st.markdown("**Returns Statistics**")
        st.metric("Mean Return", f"{df['returns'].mean():.3f}%")
        st.metric("Std Dev", f"{df['returns'].std():.3f}%")
        st.metric("Skewness", f"{df['returns'].skew():.3f}")
        st.metric("Kurtosis", f"{df['returns'].kurtosis():.3f}")
    
    # Volume Analysis
    st.markdown("---")
    st.subheader("📊 Volume Analysis")
    
    df['volume_MA'] = df['volume'].rolling(window=20).mean()
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Volume', 'Volume with MA'),
        row_heights=[0.6, 0.4]
    )
    
    # Volume bars
    colors = ['green' if df['close'].iloc[i] >= df['open'].iloc[i] else 'red' 
              for i in range(len(df))]
    
    fig.add_trace(go.Bar(
        x=df.index, y=df['volume'],
        name='Volume',
        marker_color=colors
    ), row=1, col=1)
    
    # Volume with MA
    fig.add_trace(go.Scatter(
        x=df.index, y=df['volume'],
        name='Volume',
        line=dict(color='lightblue', width=1)
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['volume_MA'],
        name='20-Day MA',
        line=dict(color='darkblue', width=2)
    ), row=2, col=1)
    
    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Volume Spikes
    st.markdown("---")
    st.subheader("🚨 Volume Spike Detection")
    
    z_scores = detect_volume_spike(df, window=20, threshold=2.0)
    spike_dates = df.index[z_scores.abs() > 2.0]
    
    if len(spike_dates) > 0:
        st.info(f"Detected {len(spike_dates)} volume spikes (Z-score > 2)")
        
        # Show recent spikes
        recent_spikes = spike_dates[-5:]
        for date in recent_spikes:
            idx = df.index.get_loc(date)
            vol = df['volume'].iloc[idx]
            z = z_scores.iloc[idx]
            st.write(f"- **{date.date()}**: Volume = {vol:,.0f}, Z-score = {z:.2f}")
    else:
        st.success("No significant volume spikes detected")
