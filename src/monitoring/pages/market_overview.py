"""Market Overview Page - Tổng quan thị trường và xếp hạng."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.analysis.market_analyzer import (
    load_all_coins_data,
    create_returns_heatmap,
    rank_by_metric,
    calculate_market_breadth
)
from src.assistant.chart_analyzer import get_chart_analyzer


def render_market_overview_page():
    """Render trang tổng quan thị trường."""
    st.title("🌍 Tổng Quan Thị Trường")
    
    st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;'>
            <h3 style='color: white; margin: 0;'>🌍 Phân Tích Tổng Quan Thị Trường</h3>
            <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;'>
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
            <h4 style='margin: 0 0 0.5rem 0; color: #667eea;'>📊 Bản Đồ Nhiệt Lợi Nhuận - So Sánh Hiệu Suất Nhiều Coin</h4>
            <p style='margin: 0; color: #ccc;'>
                Bản đồ nhiệt hiển thị lợi nhuận phần trăm của từng coin qua 3 khung thời gian: 7 ngày (ngắn hạn), 30 ngày (trung hạn), 
                và 90 ngày (dài hạn). Đây là cách nhanh nhất để nhận diện coin nào đang hot và coin nào đang yếu.
            </p>
            <ul style='margin: 0.5rem 0 0 0; color: #ccc; padding-left: 1.5rem;'>
                <li><strong>Xanh lá đậm</strong>: Lợi nhuận cao, coin đang trong xu hướng tăng mạnh - có thể là cơ hội nhưng cũng có thể đã đắt</li>
                <li><strong>Đỏ đậm</strong>: Lỗ lớn, coin đang trong xu hướng giảm - có thể là cơ hội mua vào hoặc nên tránh</li>
                <li><strong>Vàng/Trắng</strong>: Biến động thấp, coin đi ngang - chờ tín hiệu rõ ràng hơn</li>
            </ul>
            <p style='margin: 0.5rem 0 0 0; color: #ccc;'>
                <strong>Mẹo:</strong> Coin xanh cả 3 cột là coin đang có momentum tốt. Coin đỏ cả 3 cột cần thận trọng hoặc chờ đáy.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    heatmap_df = create_returns_heatmap(data_dict, periods=[7, 30, 90])
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_df[['7D', '30D', '90D']].values,
        x=['7 Ngày', '30 Ngày', '90 Ngày'],
        y=heatmap_df['coin'].str.upper(),
        colorscale='RdYlGn',
        zmid=0,
        text=heatmap_df[['7D', '30D', '90D']].values,
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
    
    st.plotly_chart(fig, width='stretch')
    
    # AI Analysis Button for Returns Heatmap
    chart_analyzer = get_chart_analyzer()
    if st.button("🤖 AI Phân Tích Bản Đồ Nhiệt Lợi Nhuận", key="analyze_heatmap"):
        with st.spinner("🔄 Đang phân tích với GPT-4..."):
            # Calculate metrics from heatmap_df
            best_coin_30d = heatmap_df.loc[heatmap_df['30D'].idxmax(), 'coin'].upper()
            best_return_30d = heatmap_df['30D'].max()
            worst_coin_30d = heatmap_df.loc[heatmap_df['30D'].idxmin(), 'coin'].upper()
            worst_return_30d = heatmap_df['30D'].min()
            coins_up_30d = int((heatmap_df['30D'] > 0).sum())
            
            chart_data = {
                "coin_count": len(heatmap_df),
                "best_coin_30d": best_coin_30d,
                "best_return_30d": best_return_30d,
                "worst_coin_30d": worst_coin_30d,
                "worst_return_30d": worst_return_30d,
                "coins_up_30d": coins_up_30d
            }
            
            analysis = chart_analyzer.analyze_chart(
                coin="all",
                chart_type="returns_heatmap",
                chart_data=chart_data,
                chart_title="Bản Đồ Nhiệt Lợi Nhuận"
            )
            st.markdown(analysis)
    
    # Rankings Section
    st.markdown("---")
    st.subheader("🏆 Xếp Hạng Coin")
    
    st.markdown("""
        <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 8px; 
                    border-left: 4px solid #667eea; margin-bottom: 1rem;'>
            <h4 style='margin: 0 0 0.5rem 0; color: #667eea;'>🏆 Xếp Hạng Coin Theo Nhiều Tiêu Chí</h4>
            <p style='margin: 0; color: #ccc;'>
                Bảng xếp hạng giúp bạn nhanh chóng xác định coin dẫn đầu hoặc coin nổi bật nhất theo tiêu chí bạn chọn.
                Mỗi tiêu chí phản ánh một khía cạnh khác nhau của coin.
            </p>
            <ul style='margin: 0.5rem 0 0 0; color: #ccc; padding-left: 1.5rem;'>
                <li><strong>Vốn Hóa</strong>: Giá trị thị trường tổng cộng - coin lớn thường ổn định hơn</li>
                <li><strong>Khối Lượng</strong>: Mức độ giao dịch - khối lượng cao = thanh khoản tốt, dễ mua bán</li>
                <li><strong>Giá</strong>: Giá hiện tại của coin</li>
                <li><strong>Biến Động</strong>: Mức dao động giá - biến động cao = rủi ro cao, tiềm năng lới lớn</li>
            </ul>
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
            width='stretch',
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
        
        st.plotly_chart(fig, width='stretch')
    
    # AI Analysis Button for Coin Ranking
    if st.button("🤖 AI Phân Tích Xếp Hạng", key="analyze_ranking"):
        with st.spinner("🔄 Đang phân tích với GPT-4..."):
            top_3 = ", ".join(ranked_df['coin'].head(3).str.upper().tolist())
            bottom_3 = ", ".join(ranked_df['coin'].tail(3).str.upper().tolist())
            range_value = f"{ranked_df['value'].max():,.2f} - {ranked_df['value'].min():,.2f}"
            
            chart_data = {
                "ranking_metric": ranking_metric,
                "top_3": top_3,
                "bottom_3": bottom_3,
                "range_value": range_value
            }
            
            analysis = chart_analyzer.analyze_chart(
                coin="all",
                chart_type="coin_ranking",
                chart_data=chart_data,
                chart_title=f"Xếp Hạng Theo {ranking_metric}"
            )
            st.markdown(analysis)
    st.markdown("---")
    st.subheader("📈 Phân Tích Độ Rộng Thị Trường")
    
    st.markdown("""
        <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 8px; 
                    border-left: 4px solid #667eea; margin-bottom: 1rem;'>
            <h4 style='margin: 0 0 0.5rem 0; color: #667eea;'>📈 Độ Rộng Thị Trường - Đo Sức Khỏe Chung</h4>
            <p style='margin: 0; color: #ccc;'>
                Độ rộng thị trường đo lường có bao nhiêu coin tham gia vào xu hướng thị trường. Biểu đồ hiển thị 
                tỷ lệ coin tăng (xanh) vs giảm (đỏ) qua các khung thời gian. Đây là chỉ số xác nhận sức mạnh xu hướng.
            </p>
            <ul style='margin: 0.5rem 0 0 0; color: #ccc; padding-left: 1.5rem;'>
                <li><strong>> 70% coin tăng</strong>: Thị trường bull mạnh, xu hướng tăng toàn diện - tín hiệu tích cực</li>
                <li><strong>< 30% coin tăng</strong>: Thị trường bear, áp lực bán lan rộng - cẩn thận với vị thế mua</li>
                <li><strong>Phân kỳ (BTC tăng nhưng độ rộng thấp)</strong>: Chỉ vài coin dẫn dắt, xu hướng có thể yếu</li>
            </ul>
            <p style='margin: 0.5rem 0 0 0; color: #ccc;'>
                <strong>Ứng dụng:</strong> Khi độ rộng mạnh (> 70%), có thể tự tin vào lệnh. Khi yếu (< 30%), nên phòng thủ.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    breadth_df = calculate_market_breadth(data_dict, periods=[7, 14, 30, 90])
    
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
    
    st.plotly_chart(fig, width='stretch')
    
    st.dataframe(
        breadth_df.style.format({
            'pct_up': '{:.1f}%',
            'pct_down': '{:.1f}%'
        }),
        width='stretch'
    )
    
    # AI Analysis Button for Market Breadth
    if st.button("🤖 AI Phân Tích Độ Rộng Thị Trường", key="analyze_breadth"):
        with st.spinner("🔄 Đang phân tích với GPT-4..."):
            pct_up_7d = breadth_df[breadth_df['period'] == '7D']['pct_up'].values[0] if len(breadth_df[breadth_df['period'] == '7D']) > 0 else 0
            pct_up_30d = breadth_df[breadth_df['period'] == '30D']['pct_up'].values[0] if len(breadth_df[breadth_df['period'] == '30D']) > 0 else 0
            pct_up_90d = breadth_df[breadth_df['period'] == '90D']['pct_up'].values[0] if len(breadth_df[breadth_df['period'] == '90D']) > 0 else 0
            breadth_trend = "TĂNG" if pct_up_30d > pct_up_90d else "GIẢM"
            
            chart_data = {
                "pct_up_7d": pct_up_7d,
                "pct_up_30d": pct_up_30d,
                "pct_up_90d": pct_up_90d,
                "breadth_trend": breadth_trend
            }
            
            analysis = chart_analyzer.analyze_chart(
                coin="all",
                chart_type="market_breadth",
                chart_data=chart_data,
                chart_title="Độ Rộng Thị Trường"
            )
            st.markdown(analysis)
    
    # Liquidity Analysis
    st.markdown("---")
    st.subheader("💧 Phân Tích Thanh Khoản")
    
    st.markdown("""
        <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 8px; 
                    border-left: 4px solid #667eea; margin-bottom: 1rem;'>
            <h4 style='margin: 0 0 0.5rem 0; color: #667eea;'>💧 Tỷ Lệ Thanh Khoản - Đánh Giá Mức Độ Giao Dịch</h4>
            <p style='margin: 0; color: #ccc;'>
                Tỷ lệ thanh khoản = Khối lượng giao dịch trung bình 7 ngày ÷ Vốn hóa thị trường. 
                Chỉ số này cho biết coin được giao dịch tích cực đến mức nào so với quy mô của nó.
            </p>
            <ul style='margin: 0.5rem 0 0 0; color: #ccc; padding-left: 1.5rem;'>
                <li><strong>Tỷ lệ cao (> 0.1%)</strong>: Coin được giao dịch nhiều - dễ mua bán, trượt giá (slippage) thấp</li>
                <li><strong>Tỷ lệ thấp (< 0.05%)</strong>: Coin ít giao dịch - có thể khó mua bán số lượng lớn</li>
            </ul>
            <p style='margin: 0.5rem 0 0 0; color: #ccc;'>
                <strong>Lưu ý:</strong> Coin thanh khoản thấp có thể biến động giá bất ngờ, cẩn thận khi giao dịch số lượng lớn.
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
        
        st.plotly_chart(fig, width='stretch')
        
        # AI Analysis Button for Liquidity
        if st.button("🤖 AI Phân Tích Thanh Khoản", key="analyze_liquidity"):
            with st.spinner("🔄 Đang phân tích với GPT-4..."):
                top_liquid = liq_df.iloc[0]
                bottom_liquid = liq_df.iloc[-1]
                
                chart_data = {
                    "top_liquid_coin": top_liquid['coin'],
                    "top_liquid_ratio": top_liquid['liquidity_ratio'],
                    "bottom_liquid_coin": bottom_liquid['coin'],
                    "bottom_liquid_ratio": bottom_liquid['liquidity_ratio'],
                    "avg_liquidity": liq_df['liquidity_ratio'].mean()
                }
                
                analysis = chart_analyzer.analyze_chart(
                    coin="all",
                    chart_type="liquidity_analysis",
                    chart_data=chart_data,
                    chart_title="Phân Tích Thanh Khoản"
                )
                st.markdown(analysis)
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
