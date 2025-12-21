"""EDA: Price & Volume Analysis Page."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.analysis.market_analyzer import load_all_coins_data, detect_volume_spike
from src.assistant.chart_analyzer import get_chart_analyzer


def render_price_volume_page():
    """Render trang phân tích giá và khối lượng."""
    st.title("📈 Phân Tích Giá & Khối Lượng")
    
    # Page introduction
    st.markdown("""
        <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 8px; 
                    border-left: 3px solid #667eea; margin-bottom: 1rem;'>
            <b>📖 Giới thiệu:</b> Trang này phân tích chi tiết biến động giá, đường trung bình động (MA), 
            khối lượng giao dịch và phân phối lợi nhuận cho coin bạn chọn.
        </div>
    """, unsafe_allow_html=True)
    
    # Coin selector inside page
    st.subheader("⚙️ Chọn Coin")
    
    coins = [
        "bitcoin", "ethereum", "litecoin", "binancecoin",
        "cardano", "solana", "pancakeswap", "axieinfinity", "thesandbox"
    ]
    
    coin = st.selectbox(
        "Chọn coin để phân tích:",
        coins,
        format_func=lambda x: x.upper(),
        key="price_volume_coin_selector"
    )
    
    st.markdown("---")
    
    # Page header with selected coin
    st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;'>
            <h3 style='color: white; margin: 0;'>📊 Phân Tích Kỹ Thuật {coin.upper()}</h3>
            <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;'>
                Phân tích chi tiết biến động giá, đường trung bình động (MA), 
                khối lượng giao dịch và phân phối lợi nhuận của {coin.upper()}.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load data for selected coin
    with st.spinner(f"Đang tải dữ liệu {coin}..."):
        data_dict = load_all_coins_data(data_dir="data/raw/train")
    
    if coin not in data_dict:
        st.error(f"❌ Không tìm thấy dữ liệu cho {coin}")
        return
    
    df = data_dict[coin]
    
    # Initialize chart analyzer
    chart_analyzer = get_chart_analyzer()
    
    # =========================================================================
    # CHART 1: Price with Moving Averages
    # =========================================================================
    st.subheader("📊 Giá Với Đường Trung Bình Động (MA)")
    
    st.markdown("""
        <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 8px; 
                    border-left: 4px solid #667eea; margin-bottom: 1rem;'>
            <h4 style='margin: 0 0 0.5rem 0; color: #667eea;'>📊 Biểu Đồ Giá Kết Hợp Đường Trung Bình Động</h4>
            <p style='margin: 0; color: #ccc;'>
                Biểu đồ hiển thị giá đóng cửa (đường xanh) cùng với 3 đường trung bình động (MA - Moving Average). 
                MA là công cụ phân tích kỹ thuật phổ biến nhất, giúp lọc nhiễu ngắn hạn và xác định xu hướng thị trường.
            </p>
            <ul style='margin: 0.5rem 0 0 0; color: #ccc; padding-left: 1.5rem;'>
                <li><strong>MA20 (đường cam)</strong>: Trung bình 20 ngày gần nhất - phản ánh xu hướng ngắn hạn, phù hợp cho trader</li>
                <li><strong>MA50 (đường xanh lá)</strong>: Trung bình 50 ngày - xu hướng trung hạn, cân bằng giữa độ nhạy và ổn định</li>
                <li><strong>MA200 (đường đỏ)</strong>: Trung bình 200 ngày - xu hướng dài hạn, được các nhà đầu tư tổ chức theo dõi</li>
            </ul>
            <p style='margin: 0.5rem 0 0 0; color: #ccc;'>
                <strong>Quy tắc đọc:</strong> Khi giá nằm TRÊN đường MA → thị trường đang trong xu hướng tăng. 
                Khi giá nằm DƯỚI đường MA → xu hướng giảm. <strong>Golden Cross</strong> (MA20 cắt lên MA50) là tín hiệu mua mạnh; 
                <strong>Death Cross</strong> (MA20 cắt xuống MA50) là tín hiệu bán.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Calculate MAs
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()
    df['MA200'] = df['close'].rolling(window=200).mean()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['close'],
        name='Giá Đóng Cửa',
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
        title=f"Giá {coin.upper()} Với Đường Trung Bình Động",
        xaxis_title="Ngày",
        yaxis_title="Giá (USD)",
        height=500,
        hovermode='x unified',
        template="plotly_dark"
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # Trend Analysis
    current_price = df['close'].iloc[-1]
    ma20 = df['MA20'].iloc[-1]
    ma50 = df['MA50'].iloc[-1]
    ma200 = df['MA200'].iloc[-1] if not pd.isna(df['MA200'].iloc[-1]) else current_price
    
    col1, col2, col3 = st.columns(3)
    with col1:
        trend_20 = "📈 Tăng" if current_price > ma20 else "📉 Giảm"
        st.metric("Xu Hướng Ngắn Hạn (MA20)", trend_20)
    with col2:
        trend_50 = "📈 Tăng" if current_price > ma50 else "📉 Giảm"
        st.metric("Xu Hướng Trung Hạn (MA50)", trend_50)
    with col3:
        trend_200 = "📈 Tăng" if current_price > ma200 else "📉 Giảm"
        st.metric("Xu Hướng Dài Hạn (MA200)", trend_200)
    
    # AI Analysis Button for Price/MA Chart
    if st.button("🤖 AI Phân Tích Biểu Đồ Giá & MA", key="analyze_price_ma"):
        with st.spinner("🔄 Đang phân tích với GPT-4..."):
            # Detect cross signal
            if len(df) > 50:
                ma20_prev = df['MA20'].iloc[-2]
                ma50_prev = df['MA50'].iloc[-2]
                if ma20 > ma50 and ma20_prev <= ma50_prev:
                    cross_signal = "Golden Cross (MA20 cắt lên MA50) - Tín hiệu mua"
                elif ma20 < ma50 and ma20_prev >= ma50_prev:
                    cross_signal = "Death Cross (MA20 cắt xuống MA50) - Tín hiệu bán"
                else:
                    cross_signal = "Không có tín hiệu cross gần đây"
            else:
                cross_signal = "Không đủ dữ liệu"
            
            chart_data = {
                "current_price": current_price,
                "ma20": ma20,
                "ma50": ma50,
                "ma200": ma200 if not pd.isna(ma200) else 0,
                "price_vs_ma20": "TRÊN" if current_price > ma20 else "DƯỚI",
                "price_vs_ma50": "TRÊN" if current_price > ma50 else "DƯỚI",
                "price_vs_ma200": "TRÊN" if current_price > ma200 else "DƯỚI",
                "cross_signal": cross_signal
            }
            
            analysis = chart_analyzer.analyze_chart(
                coin=coin,
                chart_type="price_ma",
                chart_data=chart_data,
                chart_title=f"Giá {coin.upper()} Với Đường Trung Bình Động"
            )
            st.markdown(analysis)
    
    # =========================================================================
    # CHART 2: Volume Analysis
    # =========================================================================
    st.markdown("---")
    st.subheader("📊 Phân Tích Khối Lượng Giao Dịch")
    
    st.markdown("""
        <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 8px; 
                    border-left: 4px solid #667eea; margin-bottom: 1rem;'>
            <h4 style='margin: 0 0 0.5rem 0; color: #667eea;'>📊 Phân Tích Mối Quan Hệ Giá - Khối Lượng</h4>
            <p style='margin: 0; color: #ccc;'>
                Biểu đồ trên hiển thị giá (trên) và khối lượng giao dịch (dưới). Khối lượng là số lượng coin được mua bán trong một ngày - 
                đây là chỉ báo quan trọng về sức mạnh của xu hướng và sự quan tâm của thị trường.
            </p>
            <ul style='margin: 0.5rem 0 0 0; color: #ccc; padding-left: 1.5rem;'>
                <li><strong>Cột xanh lá</strong>: Ngày giá tăng - khối lượng cao + giá tăng = xu hướng tăng mạnh</li>
                <li><strong>Cột đỏ</strong>: Ngày giá giảm - khối lượng cao + giá giảm = áp lực bán mạnh</li>
                <li><strong>Đột biến khối lượng</strong>: Thường xảy ra trước khi xu hướng đảo chiều hoặc breakout</li>
            </ul>
            <p style='margin: 0.5rem 0 0 0; color: #ccc;'>
                <strong>Quy tắc:</strong> Xu hướng tăng khỏe mạnh cần khối lượng tăng dần. 
                Nếu giá tăng nhưng khối lượng giảm → cảnh báo xu hướng yếu đi (divergence).
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Volume chart
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=("Giá", "Khối Lượng")
    )
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['close'],
        name='Giá',
        line=dict(color='#667eea', width=2)
    ), row=1, col=1)
    
    # Color volume bars based on price change
    colors = ['#00d4aa' if df['close'].iloc[i] >= df['close'].iloc[i-1] else '#ff6b6b' 
              for i in range(1, len(df))]
    colors = ['#00d4aa'] + colors  # First bar
    
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['volume'],
        name='Khối Lượng',
        marker_color=colors
    ), row=2, col=1)
    
    fig.update_layout(
        height=600,
        hovermode='x unified',
        showlegend=False,
        template="plotly_dark"
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # Volume Spike Detection
    st.markdown("---")
    st.subheader("🚨 Phát Hiện Đột Biến Khối Lượng")
    
    z_scores = detect_volume_spike(df, window=20, threshold=2.0)
    spikes = df[abs(z_scores) > 2.0].tail(5)
    spike_count = len(df[abs(z_scores) > 2.0])
    
    if len(spikes) > 0:
        st.warning(f"⚠️ Phát hiện {spike_count} đợt đột biến khối lượng trong toàn bộ lịch sử")
        st.markdown("**5 Đột Biến Gần Nhất:**")
        latest_spike_date = None
        latest_spike_zscore = 0
        for date, row in spikes.iterrows():
            z = z_scores.loc[date]
            spike_type = "🔥 Cao" if z > 0 else "❄️ Thấp"
            st.markdown(f"- **{date.strftime('%Y-%m-%d')}**: {spike_type} (Z-Score: {z:.2f})")
            latest_spike_date = date.strftime('%Y-%m-%d')
            latest_spike_zscore = z
    else:
        st.success("✅ Không có đột biến khối lượng đáng kể gần đây")
        latest_spike_date = "N/A"
        latest_spike_zscore = 0
    
    # Calculate volume stats
    avg_volume_20d = df['volume'].tail(20).mean()
    current_volume = df['volume'].iloc[-1]
    volume_vs_avg = current_volume / avg_volume_20d if avg_volume_20d > 0 else 1
    volume_trend = "TĂNG" if df['volume'].tail(7).mean() > df['volume'].tail(30).mean() else "GIẢM"
    
    # AI Analysis Button for Volume
    if st.button("🤖 AI Phân Tích Khối Lượng", key="analyze_volume"):
        with st.spinner("🔄 Đang phân tích với GPT-4..."):
            chart_data = {
                "avg_volume_20d": avg_volume_20d,
                "volume_vs_avg": volume_vs_avg,
                "spike_count": spike_count,
                "latest_spike_date": latest_spike_date,
                "latest_spike_zscore": latest_spike_zscore,
                "volume_trend": volume_trend
            }
            
            analysis = chart_analyzer.analyze_chart(
                coin=coin,
                chart_type="volume_analysis",
                chart_data=chart_data,
                chart_title="Phân Tích Khối Lượng Giao Dịch"
            )
            st.markdown(analysis)
    
    # =========================================================================
    # CHART 3: Returns Distribution
    # =========================================================================
    st.markdown("---")
    st.subheader("📊 Phân Phối Lợi Nhuận Hàng Ngày")
    
    st.markdown("""
        <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 8px; 
                    border-left: 4px solid #667eea; margin-bottom: 1rem;'>
            <h4 style='margin: 0 0 0.5rem 0; color: #667eea;'>📊 Histogram Phân Phối Lợi Nhuận Hàng Ngày</h4>
            <p style='margin: 0; color: #ccc;'>
                Biểu đồ histogram hiển thị tần suất xuất hiện của các mức lợi nhuận/lỗ hàng ngày (% thay đổi giá). 
                Đường thẳng đứng màu trắng là mốc 0%, đường vàng là mức lợi nhuận trung bình.
            </p>
            <ul style='margin: 0.5rem 0 0 0; color: #ccc; padding-left: 1.5rem;'>
                <li><strong>Phân phối rộng (nhiều cột xa tâm)</strong>: Coin có biến động cao, rủi ro lớn nhưng tiềm năng lợi nhuận cao</li>
                <li><strong>Phân phối hẹp (cột tập trung quanh 0%)</strong>: Coin ổn định hơn, phù hợp cho nhà đầu tư thận trọng</li>
                <li><strong>Đuôi bên trái dài</strong>: Có những ngày lỗ rất nặng (tail risk cao)</li>
                <li><strong>Đuôi bên phải dài</strong>: Có những ngày tăng đột biến (upside potential)</li>
            </ul>
            <p style='margin: 0.5rem 0 0 0; color: #ccc;'>
                <strong>Ý nghĩa:</strong> Nếu lợi nhuận TB dương và % ngày tăng > 50% → coin có xu hướng tăng nhẹ về dài hạn.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    returns = df['close'].pct_change().dropna() * 100
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        marker_color='#667eea',
        name='Lợi Nhuận'
    ))
    
    fig.add_vline(x=0, line_dash="dash", line_color="white", line_width=1)
    fig.add_vline(x=returns.mean(), line_dash="dash", line_color="yellow", 
                  annotation_text=f"TB: {returns.mean():.2f}%")
    
    fig.update_layout(
        title="Phân Phối Lợi Nhuận Hàng Ngày",
        xaxis_title="Lợi Nhuận Hàng Ngày (%)",
        yaxis_title="Tần Suất",
        height=400,
        template="plotly_dark"
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # Summary stats
    positive_days = int((returns > 0).sum())
    negative_days = int((returns < 0).sum())
    total_days = len(returns)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Lợi Nhuận TB/Ngày", f"{returns.mean():.2f}%")
    with col2:
        st.metric("Độ Lệch Chuẩn", f"{returns.std():.2f}%")
    with col3:
        st.metric("Ngày Tăng Giá", f"{positive_days} ({(returns > 0).mean()*100:.1f}%)")
    with col4:
        st.metric("Ngày Giảm Giá", f"{negative_days} ({(returns < 0).mean()*100:.1f}%)")
    
    # AI Analysis Button for Returns Histogram
    if st.button("🤖 AI Phân Tích Phân Phối Lợi Nhuận", key="analyze_returns_hist"):
        with st.spinner("🔄 Đang phân tích với GPT-4..."):
            chart_data = {
                "mean_return": returns.mean(),
                "std_return": returns.std(),
                "positive_days": positive_days,
                "negative_days": negative_days,
                "positive_pct": (positive_days / total_days) * 100 if total_days > 0 else 0,
                "negative_pct": (negative_days / total_days) * 100 if total_days > 0 else 0,
                "max_return": returns.max(),
                "min_return": returns.min()
            }
            
            analysis = chart_analyzer.analyze_chart(
                coin=coin,
                chart_type="returns_histogram",
                chart_data=chart_data,
                chart_title="Phân Phối Lợi Nhuận Hàng Ngày"
            )
            st.markdown(analysis)