"""Investment Insights Page - Khuyến nghị đầu tư."""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.analysis.market_analyzer import (
    load_all_coins_data,
    identify_market_regime,
    calculate_correlation_matrix
)
from src.analysis.financial_metrics import get_all_metrics


def render_investment_insights_page():
    """Render trang khuyến nghị đầu tư."""
    st.title("🧠 Khuyến Nghị Đầu Tư")
    
    # Page introduction
    st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;'>
            <h3 style='color: white; margin: 0;'>📊 Tổng Hợp & Khuyến Nghị</h3>
            <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;'>
                Tổng hợp phân tích từ tất cả các trang và đưa ra khuyến nghị đầu tư 
                dựa trên điều kiện thị trường hiện tại.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Đang phân tích thị trường..."):
        data_dict = load_all_coins_data(data_dir="data/raw/train")
    
    if not data_dict:
        st.error("❌ Không có dữ liệu")
        return
    
    # Market Regime
    st.subheader("🌍 Tình Trạng Thị Trường Hiện Tại")
    
    regime_info = identify_market_regime(data_dict)
    
    regime_names_vi = {
        "Bull": "Tăng Giá (Bull)",
        "Bear": "Giảm Giá (Bear)",
        "Sideway": "Đi Ngang"
    }
    
    regime_desc_vi = {
        "Bull": "Thị trường đang trong xu hướng tăng. Phần lớn coin đang giao dịch trên đường MA200.",
        "Bear": "Thị trường đang trong xu hướng giảm. Cần thận trọng và ưu tiên bảo toàn vốn.",
        "Sideway": "Thị trường đang đi ngang, không có xu hướng rõ ràng."
    }
    
    regime_colors = {
        "Bull": "#00d4aa",
        "Bear": "#ff6b6b",
        "Sideway": "#ffc107"
    }
    
    st.markdown(f"""
        <div style='padding: 1.5rem; background: linear-gradient(135deg, {regime_colors[regime_info['regime']]} 0%, #667eea 100%); 
                    border-radius: 10px; color: white; margin-bottom: 1rem;'>
            <h2 style='margin: 0; color: white;'>Thị Trường {regime_names_vi[regime_info['regime']]}</h2>
            <p style='margin: 0.5rem 0 0 0; font-size: 1.1rem;'>{regime_desc_vi[regime_info['regime']]}</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📊 Coin Trên MA200", f"{regime_info['pct_coins_above_ma']:.0f}%")
    
    with col2:
        st.metric("📈 Biến Động TB", f"{regime_info['avg_volatility']:.1f}%")
    
    with col3:
        vol_vi = {"High": "Cao", "Low": "Thấp", "Normal": "Bình Thường"}
        st.metric("⚡ Mức Biến Động", vol_vi.get(regime_info['volatility_regime'], regime_info['volatility_regime']))
    
    # Top 3 Watchlist
    st.markdown("---")
    st.subheader("🎯 Top 3 Coin Đáng Theo Dõi")
    
    st.markdown("""
        <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 8px; 
                    border-left: 4px solid #667eea; margin-bottom: 1rem;'>
            <p style='margin: 0; color: #ccc;'>
                Danh sách được xếp hạng theo Sharpe Ratio - chỉ số đo lường lợi nhuận 
                điều chỉnh rủi ro. Sharpe càng cao = hiệu suất đầu tư càng tốt.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Calculate all metrics
    all_metrics = []
    for coin, df in data_dict.items():
        metrics = get_all_metrics(df['close'], coin_name=coin)
        if 'error' not in metrics:
            all_metrics.append(metrics)
    
    metrics_df = pd.DataFrame(all_metrics)
    
    # Rank by Sharpe ratio
    top_3 = metrics_df.nlargest(3, 'sharpe_ratio')
    
    medals = ["🥇", "🥈", "🥉"]
    
    for rank, (idx, row) in enumerate(top_3.iterrows()):
        with st.expander(f"{medals[rank]} #{rank+1}: {row['coin'].upper()} - Sharpe: {row['sharpe_ratio']:.2f}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("💵 Giá Hiện Tại", f"${row['current_price']:.2f}")
                st.metric("📈 CAGR", f"{row['cagr']:.2f}%")
            
            with col2:
                st.metric("📊 Biến Động", f"{row['annualized_volatility']:.2f}%")
                st.metric("⚖️ Sharpe", f"{row['sharpe_ratio']:.2f}")
            
            with col3:
                st.metric("📉 Max Drawdown", f"{row['max_drawdown']:.2f}%")
                st.metric("🎯 Sortino", f"{row['sortino_ratio']:.2f}")
            
            st.markdown(f"""
                **Tại Sao Nên Theo Dõi**: Coin này có lợi nhuận điều chỉnh rủi ro tốt với 
                Sharpe Ratio {row['sharpe_ratio']:.2f}. Phù hợp cho nhà đầu tư tìm kiếm 
                sự cân bằng giữa lợi nhuận và rủi ro.
            """)
    
    # Risk Warnings
    st.markdown("---")
    st.subheader("⚠️ Cảnh Báo Rủi Ro")
    
    # Check correlation
    corr_matrix = calculate_correlation_matrix(data_dict)
    avg_corr = corr_matrix.mean().mean()
    
    warnings = []
    
    if avg_corr > 0.7:
        warnings.append({
            "type": "error",
            "msg": "🔴 Tương quan cao giữa các coin - Lợi ích đa dạng hóa hạn chế"
        })
    
    if regime_info['volatility_regime'] == "High":
        warnings.append({
            "type": "warning",
            "msg": "🟡 Môi trường biến động cao - Rủi ro biến động giá mạnh tăng cao"
        })
    
    if regime_info['regime'] == "Bear":
        warnings.append({
            "type": "error",
            "msg": "🔴 Thị trường đang giảm - Ưu tiên bảo toàn vốn"
        })
    
    if regime_info['pct_coins_above_ma'] < 30:
        warnings.append({
            "type": "warning",
            "msg": "🟡 Ít coin trên MA200 - Thị trường suy yếu toàn diện"
        })
    
    if warnings:
        for w in warnings:
            if w["type"] == "error":
                st.error(w["msg"])
            else:
                st.warning(w["msg"])
    else:
        st.success("✅ Không có cảnh báo rủi ro lớn tại thời điểm này")
    
    # Action Scenarios
    st.markdown("---")
    st.subheader("📋 Chiến Lược Khuyến Nghị")
    
    if regime_info['regime'] == "Bull" and regime_info['volatility_regime'] == "Low":
        st.success("""
            ### 🟢 Chiến Lược Tăng Trưởng Mạnh
            
            **Điều Kiện Thị Trường**: Xu hướng tăng với biến động thấp
            
            **Hành Động Khuyến Nghị**:
            - ✅ Tăng tỷ trọng các coin có momentum cao
            - ✅ Áp dụng chiến lược theo xu hướng (trend-following)
            - ✅ Có thể mở vị thế lớn hơn
            - ⚠️ Vẫn đặt stop-loss để bảo vệ lợi nhuận
            
            **Lời Khuyên**: Đây là giai đoạn thuận lợi cho đầu tư tăng trưởng. 
            Tận dụng cơ hội nhưng không quên quản lý rủi ro.
        """)
    
    elif regime_info['regime'] == "Bear":
        st.error("""
            ### 🔴 Chiến Lược Phòng Thủ
            
            **Điều Kiện Thị Trường**: Xu hướng giảm
            
            **Hành Động Khuyến Nghị**:
            - 🛑 Giảm tổng exposure với thị trường
            - 💰 Bảo toàn vốn - chờ điểm vào tốt hơn
            - 📉 Cân nhắc các vị thế short hoặc hedging
            - ⏰ Kiên nhẫn chờ tín hiệu đảo chiều
            
            **Lời Khuyên**: Đây không phải lúc để "bắt đáy". 
            Tập trung vào bảo toàn vốn và chờ xác nhận đảo chiều.
        """)
    
    elif regime_info['volatility_regime'] == "High":
        st.warning("""
            ### 🟡 Chiến Lược Cẩn Trọng
            
            **Điều Kiện Thị Trường**: Biến động cao
            
            **Hành Động Khuyến Nghị**:
            - ⚖️ Giảm kích thước vị thế
            - 🛡️ Đặt stop-loss rộng hơn hoặc không giao dịch
            - 📊 Tập trung vào coin ít biến động
            - 💵 Giữ tỷ lệ tiền mặt cao
            
            **Lời Khuyên**: Biến động cao = Rủi ro cao. Chờ thị trường ổn định hơn 
            trước khi mở vị thế lớn.
        """)
    
    else:
        st.info("""
            ### 🟡 Chiến Lược Cân Bằng
            
            **Điều Kiện Thị Trường**: Hỗn hợp / Đi ngang
            
            **Hành Động Khuyến Nghị**:
            - 🎯 Chọn lọc kỹ điểm vào lệnh
            - ⚖️ Duy trì danh mục cân bằng
            - 📊 Tập trung phân tích từng coin riêng lẻ
            - 🔄 Cân nhắc chiến lược giao dịch trong vùng giá
            
            **Lời Khuyên**: Không có xu hướng rõ ràng = cần linh hoạt. 
            Tập trung vào cơ hội cụ thể thay vì đặt cược vào thị trường chung.
        """)
    
    # Disclaimer
    st.markdown("---")
    st.warning("""
        ⚠️ **Tuyên Bố Miễn Trừ Trách Nhiệm**: Tất cả thông tin và khuyến nghị trên đây 
        chỉ mang tính chất tham khảo và giáo dục. Không được coi là lời khuyên tài chính. 
        Thị trường tiền điện tử có tính biến động cao. Luôn tự nghiên cứu (DYOR) và 
        không bao giờ đầu tư nhiều hơn số tiền bạn có thể chấp nhận mất.
    """)
