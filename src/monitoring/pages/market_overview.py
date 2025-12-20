# src/monitoring/pages/market_overview.py

"""
Market Overview Page - Tổng quan thị trường, xếp hạng và phân tích.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.analysis.market_analyzer import (
    load_all_coins_data,
    create_returns_heatmap,
    rank_by_metric,
    calculate_market_breadth
)


def render_market_overview_page():
    """Render trang tổng quan thị trường."""
    st.title("🌍 Tổng Quan Thị Trường")
    
    st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;'>
            <p style='color: rgba(255,255,255,0.9); margin: 0;'>
                Cái nhìn toàn diện về sức khỏe thị trường crypto và xếp hạng các đồng coin theo nhiều tiêu chí.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Đang tải dữ liệu thị trường..."):
        data_dict = load_all_coins_data(data_dir="data/raw/train")
    
    if not data_dict:
        st.error("❌ Không có dữ liệu khả dụng")
        return
    
    # Returns Heatmap
    st.markdown("---")
    st.subheader("📊 Bản Đồ Nhiệt Lợi Nhuận")
    
    st.markdown("""
        <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 8px; 
                    border-left: 4px solid #667eea; margin-bottom: 1rem;'>
            <h4 style='margin: 0 0 0.5rem 0; color: #667eea;'>📊 Biểu Đồ Này Hiển Thị Gì?</h4>
            <p style='margin: 0; color: #ccc;'>
                Bản đồ nhiệt hiển thị lợi nhuận phần trăm của từng coin qua các khoảng thời gian khác nhau. 
                Màu xanh lá thể hiện lợi nhuận dương, màu đỏ thể hiện lỗ.
            </p>
            <h4 style='margin: 1rem 0 0.5rem 0; color: #667eea;'>💡 Cách Đọc</h4>
            <ul style='margin: 0; color: #ccc; padding-left: 1.5rem;'>
                <li><strong>Xanh đậm</strong>: Lợi nhuận cao, xu hướng tăng mạnh</li>
                <li><strong>Đỏ đậm</strong>: Lỗ lớn, xu hướng giảm mạnh</li>
                <li><strong>Vàng/Trung tính</strong>: Biến động thấp, đi ngang</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    heatmap_df = create_returns_heatmap(data_dict, periods=[1, 7, 30, 90])
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_df[['1D', '7D', '30D', '90D']].values,
        x=['1 Ngày', '7 Ngày', '30 Ngày', '90 Ngày'],
        y=heatmap_df['coin'].str.upper(),
        colorscale='RdYlGn',
        zmid=0,
        text=heatmap_df[['1D', '7D', '30D', '90D']].values,
        texttemplate='%{text:.1f}%',
        textfont={"size": 10},
        colorbar=dict(title="Lợi Nhuận %")
    ))
    
    fig.update_layout(
        title="Lợi Nhuận Qua Các Khoảng Thời Gian",
        xaxis_title="Khoảng Thời Gian",
        yaxis_title="Coin",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Rankings Section
    st.markdown("---")
    st.subheader("🏆 Xếp Hạng Coin")
    
    st.markdown("""
        <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 8px; 
                    border-left: 4px solid #667eea; margin-bottom: 1rem;'>
            <h4 style='margin: 0 0 0.5rem 0; color: #667eea;'>📊 Ý Nghĩa</h4>
            <p style='margin: 0; color: #ccc;'>
                Xếp hạng các coin theo tiêu chí bạn chọn. Giúp nhanh chóng xác định coin dẫn đầu 
                hoặc coin có đặc điểm nổi bật trong từng lĩnh vực.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    ranking_metric = st.selectbox(
        "Xếp hạng theo",
        ["Vốn Hóa Thị Trường", "Khối Lượng", "Giá", "Biến Động"],
        key="ranking_metric"
    )
    
    metric_map = {
        "Vốn Hóa Thị Trường": "market_cap",
        "Khối Lượng": "volume",
        "Giá": "close",
        "Biến Động": "volatility"
    }
    
    ranked_df = rank_by_metric(
        data_dict,
        metric=metric_map[ranking_metric],
        ascending=(ranking_metric == "Biến Động")
    )
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.dataframe(
            ranked_df[['rank', 'coin', 'value']].style.format({
                'value': '{:,.2f}'
            }),
            use_container_width=True,
            height=400
        )
    
    with col2:
        fig = go.Figure(go.Bar(
            x=ranked_df['value'],
            y=ranked_df['coin'].str.upper(),
            orientation='h',
            marker=dict(
                color=ranked_df['value'],
                colorscale='Viridis',
                showscale=True
            )
        ))
        
        fig.update_layout(
            title=f"Xếp Hạng Theo {ranking_metric}",
            xaxis_title=ranking_metric,
            yaxis_title="Coin",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Market Breadth Analysis
    st.markdown("---")
    st.subheader("📈 Phân Tích Độ Rộng Thị Trường")
    
    st.markdown("""
        <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 8px; 
                    border-left: 4px solid #667eea; margin-bottom: 1rem;'>
            <h4 style='margin: 0 0 0.5rem 0; color: #667eea;'>📊 Độ Rộng Thị Trường Là Gì?</h4>
            <p style='margin: 0; color: #ccc;'>
                Độ rộng thị trường đo lường có bao nhiêu coin tham gia vào xu hướng thị trường. 
                Điều này giúp xác nhận sức mạnh của xu hướng - nhiều coin cùng di chuyển cho thấy xu hướng mạnh.
            </p>
            <h4 style='margin: 1rem 0 0.5rem 0; color: #667eea;'>💡 Nhận Định</h4>
            <ul style='margin: 0; color: #ccc; padding-left: 1.5rem;'>
                <li><strong>>70% tăng</strong>: Thị trường tăng mạnh toàn diện</li>
                <li><strong><30% tăng</strong>: Áp lực bán lan rộng</li>
                <li><strong>Phân kỳ</strong>: Chỉ vài coin dẫn dắt, cảnh báo xu hướng yếu</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    breadth_df = calculate_market_breadth(data_dict, periods=[1, 7, 14, 30, 90])
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Coin Tăng',
        x=breadth_df['period'],
        y=breadth_df['pct_up'],
        marker_color='#00d4aa'
    ))
    
    fig.add_trace(go.Bar(
        name='Coin Giảm',
        x=breadth_df['period'],
        y=breadth_df['pct_down'],
        marker_color='#ff6b6b'
    ))
    
    fig.update_layout(
        title="Độ Rộng Thị Trường - % Coin Tăng vs Giảm",
        xaxis_title="Khoảng Thời Gian",
        yaxis_title="Phần Trăm (%)",
        barmode='stack',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(
        breadth_df.style.format({
            'pct_up': '{:.1f}%',
            'pct_down': '{:.1f}%'
        }),
        use_container_width=True
    )
    
    # Liquidity Analysis
    st.markdown("---")
    st.subheader("💧 Phân Tích Thanh Khoản")
    
    st.markdown("""
        <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 8px; 
                    border-left: 4px solid #667eea; margin-bottom: 1rem;'>
            <h4 style='margin: 0 0 0.5rem 0; color: #667eea;'>📊 Tỷ Lệ Thanh Khoản</h4>
            <p style='margin: 0; color: #ccc;'>
                Tỷ lệ thanh khoản = Khối lượng giao dịch / Vốn hóa thị trường. 
                Tỷ lệ cao cho thấy coin được giao dịch tích cực, dễ mua bán với trượt giá thấp.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    liquidity_data = []
    for coin, df in data_dict.items():
        if 'market_cap' in df.columns and not df['market_cap'].isna().all():
            avg_volume = df['volume'].tail(7).mean()
            market_cap = df['market_cap'].iloc[-1]
            if market_cap > 0:
                liquidity_ratio = avg_volume / market_cap
                liquidity_data.append({
                    'coin': coin.upper(),
                    'avg_volume_7d': avg_volume,
                    'market_cap': market_cap,
                    'liquidity_ratio': liquidity_ratio * 100
                })
    
    if liquidity_data:
        liq_df = pd.DataFrame(liquidity_data).sort_values('liquidity_ratio', ascending=False)
        
        fig = go.Figure(go.Bar(
            x=liq_df['coin'],
            y=liq_df['liquidity_ratio'],
            marker=dict(
                color=liq_df['liquidity_ratio'],
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title="Thanh Khoản %")
            )
        ))
        
        fig.update_layout(
            title="Tỷ Lệ Thanh Khoản (KLTB 7 Ngày / Vốn Hóa)",
            xaxis_title="Coin",
            yaxis_title="Tỷ Lệ Thanh Khoản (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(
            liq_df.style.format({
                'avg_volume_7d': '{:,.0f}',
                'market_cap': '{:,.0f}',
                'liquidity_ratio': '{:.4f}%'
            }),
            use_container_width=True
        )
    else:
        st.warning("Không có dữ liệu vốn hóa để phân tích thanh khoản")
    
    # Key Insights
    st.markdown("---")
    st.subheader("💡 Nhận Định Thị Trường")
    
    breadth_30d = breadth_df[breadth_df['period'] == '30D']['pct_up'].values[0] if len(breadth_df) > 0 else 50
    
    col1, col2 = st.columns(2)
    
    with col1:
        if breadth_30d > 70:
            st.success("🟢 **Thị Trường Mạnh**: Hơn 70% coin tăng trong 30 ngày qua")
        elif breadth_30d < 30:
            st.error("🔴 **Thị Trường Yếu**: Dưới 30% coin tăng trong 30 ngày qua")
        else:
            st.info("🟡 **Thị Trường Trung Tính**: Không có xu hướng rõ ràng")
    
    with col2:
        if liquidity_data:
            avg_liquidity = liq_df['liquidity_ratio'].mean()
            if avg_liquidity > 0.1:
                st.success(f"💧 **Thanh Khoản Cao**: Tỷ lệ TB {avg_liquidity:.2f}%")
            else:
                st.warning(f"⚠️ **Thanh Khoản Thấp**: Tỷ lệ TB {avg_liquidity:.2f}%")
