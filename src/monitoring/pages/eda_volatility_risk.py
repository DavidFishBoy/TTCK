"""EDA: Volatility & Risk Analysis Page."""

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
from src.assistant.chart_analyzer import get_chart_analyzer


def render_volatility_risk_page():
    """Render trang phân tích biến động và rủi ro."""
    st.title("📉 Phân Tích Biến Động & Rủi Ro")
    
    # Coin selector inside page
    st.markdown("""
        <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 8px; 
                    border-left: 3px solid #667eea; margin-bottom: 1rem;'>
            <b>📖 Giới thiệu:</b> Trang này phân tích mức độ biến động, rủi ro sụt giảm (drawdown), 
            và các chỉ số rủi ro chuyên nghiệp như VaR và CVaR cho coin bạn chọn.
        </div>
    """, unsafe_allow_html=True)
    
    # Coin selector
    st.subheader("⚙️ Chọn Coin")
    
    coins = [
        "bitcoin", "ethereum", "litecoin", "binancecoin",
        "cardano", "solana", "pancakeswap", "axieinfinity", "thesandbox"
    ]
    
    coin = st.selectbox(
        "Chọn coin để phân tích:",
        coins,
        format_func=lambda x: x.upper(),
        key="volatility_coin_selector"
    )
    
    st.markdown("---")
    
    # Page header with selected coin
    st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;'>
            <h3 style='color: white; margin: 0;'>⚠️ Đánh Giá Rủi Ro {coin.upper()}</h3>
            <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;'>
                Phân tích mức độ biến động, rủi ro sụt giảm (drawdown), 
                và các chỉ số rủi ro chuyên nghiệp như VaR và CVaR.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner(f"Đang tải dữ liệu {coin}..."):
        data_dict = load_all_coins_data(data_dir="data/raw/train")
    
    if coin not in data_dict:
        st.error(f"❌ Không tìm thấy dữ liệu cho {coin}")
        return
    
    df = data_dict[coin]
    prices = df['close']
    
    # Initialize chart analyzer
    chart_analyzer = get_chart_analyzer()
    
    # =========================================================================
    # CHART 1: Rolling Volatility
    # =========================================================================
    st.subheader("📊 Biến Động Theo Thời Gian")
    
    st.markdown("""
        <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 8px; 
                    border-left: 4px solid #667eea; margin-bottom: 1rem;'>
            <h4 style='margin: 0 0 0.5rem 0; color: #667eea;'>📊 Biến Động (Volatility) Theo Thời Gian</h4>
            <p style='margin: 0; color: #ccc;'>
                Biến động đo lường mức độ dao động giá trong một khoảng thời gian - được tính bằng độ lệch chuẩn của lợi nhuận hàng ngày.
                Biểu đồ hiển thị 2 đường biến động lăn: 14 ngày (đường cam - ngắn hạn) và 30 ngày (đường đỏ - dài hơn).
            </p>
            <ul style='margin: 0.5rem 0 0 0; color: #ccc; padding-left: 1.5rem;'>
                <li><strong>Biến động cao (> 5%)</strong>: Rủi ro lớn nhưng cũng là cơ hội cho trader ngắn hạn</li>
                <li><strong>Biến động thấp (< 2%)</strong>: Thị trường ổn định, phù hợp cho nhà đầu tư dài hạn</li>
                <li><strong>Đường đi lên</strong>: Rủi ro đang gia tăng - cân nhắc giảm vị thế hoặc tăng cường quản lý rủi ro</li>
                <li><strong>Đường đi xuống</strong>: Thị trường bớt căng thẳng - có thể tăng vị thế</li>
            </ul>
            <p style='margin: 0.5rem 0 0 0; color: #ccc;'>
                <strong>Ứng dụng:</strong> Dùng biến động để tính position size - biến động cao thì giảm vị thế, biến động thấp thì tăng vị thế.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    vol_14d = calculate_volatility(prices, window=14, annualize=False)
    vol_30d = calculate_volatility(prices, window=30, annualize=False)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index, y=vol_14d * 100,
        name='Biến Động 14 Ngày',
        line=dict(color='orange', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index, y=vol_30d * 100,
        name='Biến Động 30 Ngày',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title="Biến Động Lăn Theo Thời Gian",
        xaxis_title="Ngày",
        yaxis_title="Biến Động (%)",
        height=400,
        hovermode='x unified',
        template="plotly_dark"
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # AI Analysis Button for Volatility Chart
    if st.button("🤖 AI Phân Tích Biểu Đồ Biến Động", key="analyze_volatility"):
        with st.spinner("🔄 Đang phân tích với GPT-4..."):
            # Prepare chart data
            vol_14d_latest = vol_14d.iloc[-1] * 100 if len(vol_14d) > 0 else 0
            vol_30d_latest = vol_30d.iloc[-1] * 100 if len(vol_30d) > 0 else 0
            vol_14d_avg = vol_14d.mean() * 100 if len(vol_14d) > 0 else 0
            vol_30d_avg = vol_30d.mean() * 100 if len(vol_30d) > 0 else 0
            
            # Determine trend
            if len(vol_14d) > 30:
                trend = "TĂNG" if vol_14d.iloc[-1] > vol_14d.iloc[-30] else "GIẢM"
            else:
                trend = "KHÔNG ĐỦ DỮ LIỆU"
            
            chart_data = {
                "vol_14d_latest": vol_14d_latest,
                "vol_30d_latest": vol_30d_latest,
                "vol_14d_avg": vol_14d_avg,
                "vol_30d_avg": vol_30d_avg,
                "volatility_trend": trend
            }
            
            analysis = chart_analyzer.analyze_chart(
                coin=coin,
                chart_type="rolling_volatility",
                chart_data=chart_data,
                chart_title="Biến Động Lăn Theo Thời Gian"
            )
            st.markdown(analysis)
    
    # =========================================================================
    # CHART 2: Drawdown Analysis
    # =========================================================================
    st.markdown("---")
    st.subheader("📉 Phân Tích Sụt Giảm (Drawdown)")
    
    st.markdown("""
        <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 8px; 
                    border-left: 4px solid #667eea; margin-bottom: 1rem;'>
            <h4 style='margin: 0 0 0.5rem 0; color: #667eea;'>📉 Drawdown - Chỉ Số Đo Rủi Ro Sụt Giảm</h4>
            <p style='margin: 0; color: #ccc;'>
                Drawdown là mức giảm từ đỉnh cao nhất trước đó so với giá hiện tại. Ví dụ: nếu coin đạt đỉnh $100 rồi giảm xuống $70, drawdown là -30%.
                Biểu đồ "Underwater" bên dưới hiển thị mức drawdown theo thời gian - vùng đỏ càng sâu thì lỗ càng lớn.
            </p>
            <ul style='margin: 0.5rem 0 0 0; color: #ccc; padding-left: 1.5rem;'>
                <li><strong>Max Drawdown</strong>: Mức lỗ tối đa từ đỉnh nếu bạn mua đúng đỉnh và bán đúng đáy - chỉ số quan trọng nhất để đánh giá rủi ro</li>
                <li><strong>Thời gian phục hồi</strong>: Bao lâu để giá quay lại đỉnh cũ - coin tốt sẽ phục hồi nhanh</li>
                <li><strong>Tần suất drawdown sâu (> 20%)</strong>: Cho biết coin thường xuyên có biến động mạnh hay không</li>
            </ul>
            <p style='margin: 0.5rem 0 0 0; color: #ccc;'>
                <strong>Lưu ý quan trọng:</strong> Max Drawdown cho biết worst-case scenario - nếu bạn không chấp nhận được mức lỗ này, hãy giảm vị thế hoặc dùng stop-loss.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    drawdown_series, max_dd, max_dd_duration = calculate_drawdown(prices)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("🔻 Max Drawdown", f"{max_dd * 100:.2f}%")
    
    with col2:
        st.metric("⏱️ Thời Gian Phục Hồi Dài Nhất", f"{max_dd_duration} ngày")
    
    # Drawdown chart (underwater plot)
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=drawdown_series * 100,
        fill='tozeroy',
        name='Drawdown',
        line=dict(color='red', width=1),
        fillcolor='rgba(255, 107, 107, 0.4)'
    ))
    
    fig.update_layout(
        title="Biểu Đồ Underwater (Drawdown Theo Thời Gian)",
        xaxis_title="Ngày",
        yaxis_title="Drawdown (%)",
        height=400,
        template="plotly_dark"
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # AI Analysis Button for Drawdown Chart
    if st.button("🤖 AI Phân Tích Biểu Đồ Drawdown", key="analyze_drawdown"):
        with st.spinner("🔄 Đang phân tích với GPT-4..."):
            current_dd = drawdown_series.iloc[-1] * 100 if len(drawdown_series) > 0 else 0
            dd_count_20 = (drawdown_series < -0.2).sum()
            
            chart_data = {
                "max_drawdown": max_dd * 100,
                "max_dd_duration": max_dd_duration,
                "current_drawdown": current_dd,
                "dd_count_20": int(dd_count_20)
            }
            
            analysis = chart_analyzer.analyze_chart(
                coin=coin,
                chart_type="drawdown",
                chart_data=chart_data,
                chart_title="Biểu Đồ Underwater (Drawdown)"
            )
            st.markdown(analysis)
    
    # =========================================================================
    # CHART 3: Risk Metrics & Returns Distribution
    # =========================================================================
    st.markdown("---")
    st.subheader("⚠️ Các Chỉ Số Rủi Ro")
    
    st.markdown("""
        <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 8px; 
                    border-left: 4px solid #667eea; margin-bottom: 1rem;'>
            <h4 style='margin: 0 0 0.5rem 0; color: #667eea;'>⚠️ Các Chỉ Số Rủi Ro Chuyên Nghiệp</h4>
            <p style='margin: 0; color: #ccc;'>
                Đây là các chỉ số được các quỹ đầu tư và nhà đầu tư chuyên nghiệp sử dụng để đánh giá và quản lý rủi ro danh mục.
            </p>
            <ul style='margin: 0.5rem 0 0 0; color: #ccc; padding-left: 1.5rem;'>
                <li><strong>VaR 95% (Value at Risk)</strong>: Mức lỗ tối đa dự kiến trong 1 ngày với độ tin cậy 95%. Ví dụ: VaR = -3% nghĩa là có 95% khả năng bạn sẽ không lỗ quá 3% trong 1 ngày</li>
                <li><strong>CVaR 95% (Conditional VaR)</strong>: Mức lỗ trung bình khi vượt quá VaR (worst 5% scenarios). CVaR luôn lớn hơn VaR và cho biết "tail risk" - rủi ro trong các sự kiện cực đoan</li>
                <li><strong>Biến động năm (Annualized Volatility)</strong>: Độ lệch chuẩn lợi nhuận đã chuẩn hóa theo năm - dùng để so sánh rủi ro giữa các tài sản</li>
            </ul>
            <p style='margin: 0.5rem 0 0 0; color: #ccc;'>
                <strong>Ứng dụng:</strong> Nếu bạn đầu tư $10,000 và VaR = -5%, thì có 95% khả năng bạn sẽ không lỗ quá $500 trong một ngày.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    var_95, cvar_95 = calculate_var_cvar(prices, confidence_level=0.95)
    annualized_vol = calculate_volatility(prices, window=None, annualize=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📊 Biến Động Năm", f"{annualized_vol:.2f}%")
        st.caption("Độ lệch chuẩn lợi nhuận hàng năm")
    
    with col2:
        st.metric("📉 VaR (95%)", f"{var_95:.2f}%")
        st.caption("Mức lỗ tối đa hàng ngày ở độ tin cậy 95%")
    
    with col3:
        st.metric("📉 CVaR (95%)", f"{cvar_95:.2f}%")
        st.caption("Mức lỗ trung bình khi vượt VaR")
    
    # Returns Distribution
    st.markdown("---")
    st.subheader("📊 Phân Phối Lợi Nhuận & Đánh Giá Rủi Ro")
    
    returns = prices.pct_change().dropna() * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        marker_color='#667eea',
        name='Lợi Nhuận'
    ))
    
    # Add VaR line
    fig.add_vline(x=var_95, line_dash="dash", line_color="red", 
                  annotation_text=f"VaR 95% = {var_95:.2f}%")
    
    fig.update_layout(
        title="Phân Phối Lợi Nhuận Với Đường VaR",
        xaxis_title="Lợi Nhuận Hàng Ngày (%)",
        yaxis_title="Tần Suất",
        height=400,
        template="plotly_dark"
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # AI Analysis Button for Returns Distribution
    if st.button("🤖 AI Phân Tích Phân Phối Lợi Nhuận", key="analyze_returns"):
        with st.spinner("🔄 Đang phân tích với GPT-4..."):
            positive_days = (returns > 0).sum()
            total_days = len(returns)
            
            chart_data = {
                "mean_return": returns.mean(),
                "std_return": returns.std(),
                "var_95": var_95,
                "cvar_95": cvar_95,
                "annualized_vol": annualized_vol,
                "positive_days_pct": (positive_days / total_days) * 100 if total_days > 0 else 0
            }
            
            analysis = chart_analyzer.analyze_chart(
                coin=coin,
                chart_type="returns_distribution",
                chart_data=chart_data,
                chart_title="Phân Phối Lợi Nhuận & VaR"
            )
            st.markdown(analysis)
    
    # =========================================================================
    # Risk Assessment
    # =========================================================================
    st.markdown("---")
    st.subheader("🎯 Tổng Kết Đánh Giá Rủi Ro")
    
    if annualized_vol > 100:
        st.error(f"""
            🔴 **Rủi Ro Rất Cao**: Biến động năm {annualized_vol:.1f}% cho thấy rủi ro cực kỳ cao. 
            Chỉ phù hợp cho các nhà đầu tư chấp nhận rủi ro rất cao.
        """)
    elif annualized_vol > 60:
        st.warning(f"""
            🟡 **Rủi Ro Cao**: Biến động năm {annualized_vol:.1f}% cao hơn trung bình. 
            Phù hợp cho nhà đầu tư có khẩu vị rủi ro cao.
        """)
    else:
        st.success(f"""
            🟢 **Rủi Ro Vừa Phải**: Biến động năm {annualized_vol:.1f}% tương đối vừa phải 
            so với các tài sản crypto khác.
        """)
    
    if abs(max_dd) > 0.5:
        st.warning(f"""
            ⚠️ **Cảnh Báo Drawdown**: Max Drawdown {abs(max_dd)*100:.1f}% cho thấy tiềm năng 
            lỗ lớn. Cần áp dụng quản lý rủi ro nghiêm ngặt như stop-loss hoặc phân bổ vốn hợp lý.
        """)
    
    # Risk recommendations
    st.markdown("---")
    st.subheader("💡 Khuyến Nghị Quản Lý Rủi Ro")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
            **📊 Đề Xuất Vị Thế**
            
            Dựa trên biến động {annualized_vol:.1f}%, nếu bạn chấp nhận rủi ro 2% tài khoản/giao dịch:
            
            - Vị thế tối đa: **{min(100, 200/annualized_vol):.1f}%** tài khoản
        """)
    
    with col2:
        st.info(f"""
            **🛡️ Stop-Loss Đề Xuất**
            
            Dựa trên VaR 95% ({var_95:.2f}%):
            
            - Stop-loss hợp lý: **{abs(var_95)*1.5:.1f}%** từ giá vào
            - Stop-loss bảo thủ: **{abs(var_95)*2:.1f}%** từ giá vào
        """)
