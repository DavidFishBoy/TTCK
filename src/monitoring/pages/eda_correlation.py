"""EDA: Correlation Analysis Page."""

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
from src.assistant.chart_analyzer import get_chart_analyzer


def render_correlation_page():
    """Render trang phân tích tương quan."""
    st.title("🔗 Phân Tích Tương Quan")
    
    # Page introduction
    st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;'>
            <h3 style='color: white; margin: 0;'>🔗 Mối Quan Hệ Giữa Các Coin</h3>
            <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;'>
                Phân tích tương quan giữa các coin để hiểu mức độ đa dạng hóa của danh mục.
                Coin có tương quan thấp giúp giảm rủi ro tổng thể.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Đang tải dữ liệu..."):
        data_dict = load_all_coins_data(data_dir="data/raw/train")
    
    if not data_dict:
        st.error("❌ Không có dữ liệu")
        return
    
    # Initialize chart analyzer
    chart_analyzer = get_chart_analyzer()
    
    # =========================================================================
    # CHART 1: Correlation Matrix
    # =========================================================================
    st.subheader("📊 Ma Trận Tương Quan (Lợi Nhuận)")
    
    st.markdown("""
        <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 8px; 
                    border-left: 4px solid #667eea; margin-bottom: 1rem;'>
            <h4 style='margin: 0 0 0.5rem 0; color: #667eea;'>📊 Ma Trận Tương Quan Giữa Các Coin</h4>
            <p style='margin: 0; color: #ccc;'>
                Ma trận hiển thị hệ số tương quan giữa từng cặp coin, dao động từ -1 đến +1. Tương quan đo lường mức độ 
                hai coin di chuyển cùng chiều hay ngược chiều nhau - đây là cơ sở của việc đa dạng hóa danh mục.
            </p>
            <ul style='margin: 0.5rem 0 0 0; color: #ccc; padding-left: 1.5rem;'>
                <li><strong>+1.0 (Đỏ đậm)</strong>: Tương quan hoàn hảo - 2 coin luôn di chuyển cùng chiều 100%. Không có lợi ích đa dạng hóa</li>
                <li><strong>0.0 (Trắng)</strong>: Không tương quan - 2 coin di chuyển độc lập. Lý tưởng để đa dạng hóa</li>
                <li><strong>-1.0 (Xanh đậm)</strong>: Tương quan nghịch - 2 coin di chuyển ngược chiều. Tốt nhất cho hedge rủi ro</li>
                <li><strong>< 0.5</strong>: Tương quan thấp - tốt cho đa dạng hóa danh mục</li>
                <li><strong>> 0.7</strong>: Tương quan cao - 2 coin gần như giống nhau, nên chọn 1 trong 2</li>
            </ul>
            <p style='margin: 0.5rem 0 0 0; color: #ccc;'>
                <strong>Ứng dụng:</strong> Để xây dựng danh mục an toàn, hãy chọn các coin có tương quan thấp với nhau (< 0.5). 
                Khi 1 coin giảm, các coin khác có thể tăng và bù đắp.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
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
        colorbar=dict(title="Tương Quan")
    ))
    
    fig.update_layout(
        title="Ma Trận Tương Quan (Toàn Bộ Thời Gian)",
        height=600,
        template="plotly_dark"
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # Correlation Analysis Summary
    avg_corr = corr_matrix.mean().mean()
    max_corr = corr_matrix.where(corr_matrix != 1).max().max()
    min_corr = corr_matrix.min().min()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Tương Quan Trung Bình", f"{avg_corr:.2f}")
    with col2:
        st.metric("📈 Tương Quan Cao Nhất", f"{max_corr:.2f}")
    with col3:
        st.metric("📉 Tương Quan Thấp Nhất", f"{min_corr:.2f}")
    
    # Find highest and lowest correlation pairs
    pairs = []
    for i, coin1 in enumerate(corr_matrix.columns):
        for j, coin2 in enumerate(corr_matrix.columns):
            if i < j:
                pairs.append((coin1, coin2, corr_matrix.loc[coin1, coin2]))
    
    pairs_sorted = sorted(pairs, key=lambda x: x[2])
    lowest_pair = pairs_sorted[0] if pairs_sorted else ("N/A", "N/A", 0)
    highest_pair = pairs_sorted[-1] if pairs_sorted else ("N/A", "N/A", 0)
    
    high_corr_count = sum(1 for _, _, c in pairs if c > 0.7)
    low_corr_count = sum(1 for _, _, c in pairs if c < 0.3)
    
    # AI Analysis Button for Correlation Matrix
    if st.button("🤖 AI Phân Tích Ma Trận Tương Quan", key="analyze_corr_matrix"):
        with st.spinner("🔄 Đang phân tích với GPT-4..."):
            chart_data = {
                "avg_correlation": avg_corr,
                "highest_pair": f"{highest_pair[0].upper()} & {highest_pair[1].upper()}",
                "highest_corr": highest_pair[2],
                "lowest_pair": f"{lowest_pair[0].upper()} & {lowest_pair[1].upper()}",
                "lowest_corr": lowest_pair[2],
                "high_corr_count": high_corr_count,
                "low_corr_count": low_corr_count
            }
            
            analysis = chart_analyzer.analyze_chart(
                coin="all",
                chart_type="correlation_matrix",
                chart_data=chart_data,
                chart_title="Ma Trận Tương Quan"
            )
            st.markdown(analysis)
    
    # =========================================================================
    # CHART 2: Rolling Correlation with Bitcoin
    # =========================================================================
    st.markdown("---")
    st.subheader("📈 Tương Quan Lăn Với Bitcoin (30 Ngày)")
    
    st.markdown("""
        <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 8px; 
                    border-left: 4px solid #667eea; margin-bottom: 1rem;'>
            <h4 style='margin: 0 0 0.5rem 0; color: #667eea;'>📊 Tương Quan Lăn Với Bitcoin - Theo Dõi Theo Thời Gian</h4>
            <p style='margin: 0; color: #ccc;'>
                Biểu đồ hiển thị hệ số tương quan 30 ngày giữa các altcoin và Bitcoin theo thời gian. 
                Bitcoin là coin dẫn dắt thị trường - khi BTC tăng/giảm, hầu hết altcoin cũng theo.
            </p>
            <ul style='margin: 0.5rem 0 0 0; color: #ccc; padding-left: 1.5rem;'>
                <li><strong>Tương quan cao (> 0.7)</strong>: Altcoin theo sát Bitcoin - rủi ro hệ thống cao, khó đa dạng hóa</li>
                <li><strong>Tương quan thấp (< 0.3)</strong>: Altcoin hoạt động độc lập - có thể outperform hoặc underperform BTC</li>
                <li><strong>Tương quan âm</strong>: Hiếm gặp nhưng lý tưởng cho hedge trong thị trường giảm</li>
                <li><strong>Đường vàng (0.5)</strong>: Ngưỡng tương quan cao - coin trên đường này phụ thuộc nhiều vào BTC</li>
            </ul>
            <p style='margin: 0.5rem 0 0 0; color: #ccc;'>
                <strong>Ứng dụng:</strong> Trong thị trường bull, chọn coin tương quan cao với BTC sẽ hưởng lợi. 
                Trong thị trường bear, tìm coin tương quan thấp để bảo vệ danh mục.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    rolling_corrs = calculate_rolling_correlation_with_btc(data_dict, window=30)
    
    if rolling_corrs:
        fig = go.Figure()
        
        colors = ['#667eea', '#00d4aa', '#ffc107', '#ff6b6b', '#17a2b8', '#28a745', '#fd7e14', '#6f42c1']
        
        for i, (coin, corr_series) in enumerate(rolling_corrs.items()):
            fig.add_trace(go.Scatter(
                x=corr_series.index,
                y=corr_series,
                name=coin.upper(),
                mode='lines',
                line=dict(color=colors[i % len(colors)])
            ))
        
        fig.add_hline(y=0.5, line_dash="dash", line_color="yellow", 
                      annotation_text="Ngưỡng tương quan cao")
        
        fig.update_layout(
            title="Tương Quan Lăn 30 Ngày Với Bitcoin",
            xaxis_title="Ngày",
            yaxis_title="Hệ Số Tương Quan",
            height=500,
            hovermode='x unified',
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # AI Analysis Button for Rolling Correlation
        if st.button("🤖 AI Phân Tích Tương Quan Lăn Với BTC", key="analyze_rolling_corr"):
            with st.spinner("🔄 Đang phân tích với GPT-4..."):
                # Calculate summary stats
                correlation_summary = ""
                most_stable = None
                least_stable = None
                min_std = float('inf')
                max_std = 0
                
                for coin, corr_series in rolling_corrs.items():
                    avg = corr_series.mean()
                    std = corr_series.std()
                    correlation_summary += f"- {coin.upper()}: TB = {avg:.2f}, Std = {std:.2f}\n"
                    
                    if std < min_std:
                        min_std = std
                        most_stable = coin
                    if std > max_std:
                        max_std = std
                        least_stable = coin
                
                chart_data = {
                    "window": 30,
                    "correlation_summary": correlation_summary,
                    "most_stable_coin": most_stable.upper() if most_stable else "N/A",
                    "most_volatile_coin": least_stable.upper() if least_stable else "N/A"
                }
                
                analysis = chart_analyzer.analyze_chart(
                    coin="all",
                    chart_type="rolling_correlation",
                    chart_data=chart_data,
                    chart_title="Tương Quan Lăn Với Bitcoin"
                )
                st.markdown(analysis)
    
    # =========================================================================
    # Insights & Best Pairs
    # =========================================================================
    st.markdown("---")
    st.subheader("💡 Nhận Định Tương Quan")
    
    if avg_corr > 0.7:
        st.warning(f"""
            ⚠️ **Tương Quan Cao** ({avg_corr:.2f})
            
            Các coin có xu hướng di chuyển cùng chiều mạnh. Điều này có nghĩa:
            - Đa dạng hóa trong danh mục này có lợi ích hạn chế
            - Khi thị trường giảm, phần lớn coin sẽ giảm cùng lúc
            - Cân nhắc thêm tài sản ngoài crypto để đa dạng hóa
        """)
    elif avg_corr < 0.3:
        st.success(f"""
            ✅ **Tương Quan Thấp** ({avg_corr:.2f})
            
            Các coin hoạt động khá độc lập. Điều này có nghĩa:
            - Tiềm năng đa dạng hóa tốt trong danh mục
            - Rủi ro tổng thể có thể được giảm thiểu
            - Các coin khác nhau có thể bù đắp lẫn nhau
        """)
    else:
        st.info(f"""
            ℹ️ **Tương Quan Vừa Phải** ({avg_corr:.2f})
            
            Các coin có mức độ liên kết trung bình. Điều này có nghĩa:
            - Có một số lợi ích đa dạng hóa
            - Khi thị trường biến động mạnh, các coin vẫn có xu hướng đi cùng chiều
            - Nên chọn lọc coin có tương quan thấp để tối ưu danh mục
        """)
    
    # Best pairs for diversification
    st.markdown("---")
    st.subheader("🎯 Cặp Coin Tốt Nhất Cho Đa Dạng Hóa")
    
    pairs_sorted_low = sorted(pairs, key=lambda x: x[2])[:5]
    
    st.markdown("**5 Cặp Coin Có Tương Quan Thấp Nhất:**")
    for coin1, coin2, corr in pairs_sorted_low:
        color = "🟢" if corr < 0.3 else "🟡" if corr < 0.5 else "🟠"
        st.markdown(f"{color} **{coin1.upper()}** & **{coin2.upper()}**: Tương quan {corr:.2f}")
