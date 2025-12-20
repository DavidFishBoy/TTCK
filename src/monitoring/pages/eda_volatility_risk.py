# src/monitoring/pages/eda_volatility_risk.py

"""
EDA: Volatility & Risk Analysis Page
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.analysis.market_analyzer import load_all_coins_data
from src.analysis.financial_metrics import (
    calculate_volatility,
    calculate_drawdown,
    calculate_var_cvar,
    calculate_rolling_metrics
)


def render_volatility_risk_page(coin: str):
    """Render volatility and risk analysis page."""
    if not coin:
        st.warning("Please select a coin from the sidebar")
        return
    
    st.title(f"📉 EDA: Volatility & Risk - {coin.upper()}")
    
    # Load data
    with st.spinner(f"Loading {coin} data..."):
        data_dict = load_all_coins_data(data_dir="data/raw/train")
    
    if coin not in data_dict:
        st.error(f"No data found for {coin}")
        return
    
    df = data_dict[coin]
    prices = df['close']
    
    # Rolling Volatility
    st.subheader("📊 Rolling Volatility")
    
    vol_14d = calculate_volatility(prices, window=14, annualize=False)
    vol_30d = calculate_volatility(prices, window=30, annualize=False)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index, y=vol_14d * 100,
        name='14-Day Volatility',
        line=dict(color='orange', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index, y=vol_30d * 100,
        name='30-Day Volatility',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title="Rolling Volatility Over Time",
        xaxis_title="Date",
        yaxis_title="Volatility (%)",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Drawdown Analysis
    st.markdown("---")
    st.subheader("📉 Drawdown Analysis")
    
    drawdown_series, max_dd, max_dd_duration = calculate_drawdown(prices)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Max Drawdown", f"{max_dd * 100:.2f}%")
    
    with col2:
        st.metric("Max Drawdown Duration", f"{max_dd_duration} days")
    
    # Drawdown chart (underwater plot)
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=drawdown_series * 100,
        fill='tozeroy',
        name='Drawdown',
        line=dict(color='red', width=1)
    ))
    
    fig.update_layout(
        title="Underwater Plot (Drawdown Over Time)",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk Metrics
    st.markdown("---")
    st.subheader("⚠️ Risk Metrics")
    
    var_95, cvar_95 = calculate_var_cvar(prices, confidence_level=0.95)
    annualized_vol = calculate_volatility(prices, window=None, annualize=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Annualized Volatility", f"{annualized_vol:.2f}%")
    
    with col2:
        st.metric("VaR (95%)", f"{var_95:.2f}%")
        st.caption("Expected maximum daily loss at 95% confidence")
    
    with col3:
        st.metric("CVaR (95%)", f"{cvar_95:.2f}%")
        st.caption("Average loss when VaR is exceeded")
    
    # Returns Distribution
    st.markdown("---")
    st.subheader("📊 Returns Distribution & Risk Assessment")
    
    returns = prices.pct_change().dropna() * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        marker_color='#667eea',
        name='Returns'
    ))
    
    # Add VaR line
    fig.add_vline(x=var_95, line_dash="dash", line_color="red", 
                  annotation_text=f"VaR 95% = {var_95:.2f}%")
    
    fig.update_layout(
        title="Returns Distribution with VaR",
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk Assessment
    st.markdown("---")
    st.subheader("🎯 Risk Assessment Summary")
    
    if annualized_vol > 100:
        st.error(f"🔴 **Very High Risk**: Annualized volatility of {annualized_vol:.1f}% indicates extreme risk. Only suitable for very aggressive portfolios.")
    elif annualized_vol > 60:
        st.warning(f"🟡 **High Risk**: Annualized volatility of {annualized_vol:.1f}% is above average. Suitable for high-risk tolerance investors.")
    else:
        st.success(f"🟢 **Moderate Risk**: Annualized volatility of {annualized_vol:.1f}% is relatively moderate for crypto assets.")
    
    if abs(max_dd) > 0.5:
        st.warning(f"⚠️ Maximum drawdown of {abs(max_dd)*100:.1f}% indicates potential for significant losses. Strong risk management required.")
