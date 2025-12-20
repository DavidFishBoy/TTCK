# src/monitoring/pages/home.py

"""
Home Page - Trang chủ giới thiệu các mục phân tích.
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def render_home_page():
    """Render trang chủ với giới thiệu các mục phân tích."""
    st.title("🏠 Trang Chủ")
    
    # Welcome banner
    st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 16px; margin-bottom: 2rem; text-align: center;'>
            <h2 style='color: white; margin: 0; font-size: 1.8rem;'>🚀 Chào Mừng Đến Với Crypto Analytics</h2>
            <p style='color: rgba(255,255,255,0.9); margin: 1rem 0 0 0; font-size: 1.1rem;'>
                Bảng điều khiển phân tích tiền điện tử sử dụng Deep Learning & AI<br>
                Khám phá các công cụ phân tích chuyên nghiệp ở menu bên trái
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
        <div style='background: rgba(102, 126, 234, 0.1); padding: 1.5rem; border-radius: 12px; 
                    border-left: 4px solid #667eea; margin-bottom: 2rem;'>
            <h3 style='margin: 0 0 0.5rem 0; color: #667eea;'>📌 Giới Thiệu</h3>
            <p style='margin: 0; color: #ccc; line-height: 1.6;'>
                Đây là bảng điều khiển phân tích tiền điện tử toàn diện, được xây dựng để hỗ trợ 
                các nhà đầu tư và nhà giao dịch đưa ra quyết định thông minh hơn. Sử dụng thanh 
                điều hướng bên trái để truy cập vào các công cụ phân tích chi tiết.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("📋 Các Chức Năng Chính")
    
    # Features in 2 columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Market Overview
        st.markdown("""
            <div class='feature-card'>
                <h4>🌍 Tổng Quan Thị Trường</h4>
                <p>
                    <strong>Mô tả:</strong> Hiển thị bức tranh tổng thể về thị trường crypto, 
                    bao gồm bản đồ nhiệt lợi nhuận, xếp hạng coin theo các tiêu chí khác nhau, 
                    và phân tích độ rộng thị trường.<br><br>
                    <strong>Phù hợp cho:</strong> Đánh giá nhanh tình hình thị trường và so sánh 
                    hiệu suất giữa các đồng coin.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Volatility & Risk
        st.markdown("""
            <div class='feature-card'>
                <h4>📉 Phân Tích Biến Động & Rủi Ro</h4>
                <p>
                    <strong>Mô tả:</strong> Đo lường mức độ biến động giá và đánh giá rủi ro 
                    của từng đồng coin. Bao gồm các chỉ số như ATR, Bollinger Bands, và 
                    phân tích drawdown.<br><br>
                    <strong>Phù hợp cho:</strong> Quản lý rủi ro và xác định điểm vào/ra lệnh 
                    an toàn.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Quant Metrics
        st.markdown("""
            <div class='feature-card'>
                <h4>📐 Chỉ Số Định Lượng</h4>
                <p>
                    <strong>Mô tả:</strong> Các chỉ số tài chính chuyên nghiệp như Sharpe Ratio, 
                    Sortino Ratio, Calmar Ratio, CAGR, và Maximum Drawdown để đánh giá 
                    hiệu suất điều chỉnh rủi ro.<br><br>
                    <strong>Phù hợp cho:</strong> So sánh hiệu quả đầu tư giữa các coin một 
                    cách khoa học.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Portfolio Analysis
        st.markdown("""
            <div class='feature-card'>
                <h4>🧺 Phân Tích Danh Mục</h4>
                <p>
                    <strong>Mô tả:</strong> Công cụ xây dựng và kiểm thử danh mục đầu tư với 
                    các chiến lược khác nhau như Equal Weight và Risk Parity. Bao gồm 
                    backtest và đường cong vốn.<br><br>
                    <strong>Phù hợp cho:</strong> Tối ưu hóa phân bổ tài sản và đa dạng hóa 
                    danh mục.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Prediction
        st.markdown("""
            <div class='feature-card'>
                <h4>🔮 Dự Đoán Giá</h4>
                <p>
                    <strong>Mô tả:</strong> Sử dụng mô hình Deep Learning (LSTM) để dự đoán 
                    xu hướng giá trong tương lai. Hiển thị dự đoán giá với khoảng tin cậy 
                    và chỉ số độ tin cậy của mô hình.<br><br>
                    <strong>Phù hợp cho:</strong> Lập kế hoạch giao dịch và xác định cơ hội 
                    đầu tư tiềm năng.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Price & Volume
        st.markdown("""
            <div class='feature-card'>
                <h4>📈 Phân Tích Giá & Khối Lượng</h4>
                <p>
                    <strong>Mô tả:</strong> Phân tích chi tiết giá và khối lượng giao dịch 
                    của từng đồng coin. Bao gồm biểu đồ giá với đường trung bình động (MA), 
                    phân phối lợi nhuận, và phát hiện đột biến khối lượng.<br><br>
                    <strong>Phù hợp cho:</strong> Phân tích kỹ thuật và tìm kiếm điểm vào lệnh.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Correlation
        st.markdown("""
            <div class='feature-card'>
                <h4>🔗 Phân Tích Tương Quan</h4>
                <p>
                    <strong>Mô tả:</strong> Ma trận tương quan giữa các đồng coin để hiểu 
                    mối quan hệ giá cả. Giúp xác định coin nào di chuyển cùng nhau và 
                    coin nào có tính độc lập cao.<br><br>
                    <strong>Phù hợp cho:</strong> Đa dạng hóa danh mục và tìm kiếm cơ hội 
                    hedging.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Factor Analysis
        st.markdown("""
            <div class='feature-card'>
                <h4>🧩 Phân Tích Nhân Tố</h4>
                <p>
                    <strong>Mô tả:</strong> Phân tích các yếu tố ảnh hưởng đến hiệu suất 
                    coin như momentum, volatility, size, và liquidity. Bao gồm phân cụm 
                    coin theo đặc điểm tương tự.<br><br>
                    <strong>Phù hợp cho:</strong> Hiểu các động lực thúc đẩy giá và lựa 
                    chọn coin theo chiến lược.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Investment Insights
        st.markdown("""
            <div class='feature-card'>
                <h4>🧠 Khuyến Nghị Đầu Tư</h4>
                <p>
                    <strong>Mô tả:</strong> Tổng hợp phân tích và đưa ra khuyến nghị đầu tư 
                    dựa trên điều kiện thị trường hiện tại. Bao gồm danh sách coin cần 
                    theo dõi và cảnh báo rủi ro.<br><br>
                    <strong>Phù hợp cho:</strong> Đưa ra quyết định đầu tư tổng thể và 
                    quản lý danh mục.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Compare Models
        st.markdown("""
            <div class='feature-card'>
                <h4>⚖️ So Sánh Mô Hình</h4>
                <p>
                    <strong>Mô tả:</strong> So sánh hiệu suất giữa các mô hình dự đoán khác 
                    nhau: LSTM Deep Learning, Naive Baseline, Moving Average, và Exponential 
                    Moving Average.<br><br>
                    <strong>Phù hợp cho:</strong> Đánh giá độ chính xác của các mô hình và 
                    lựa chọn phương pháp phù hợp.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick stats
    st.subheader("📊 Thông Tin Nhanh")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💰 Số Coin Hỗ Trợ", "9")
    
    with col2:
        st.metric("📈 Chỉ Số Phân Tích", "20+")
    
    with col3:
        st.metric("🤖 Mô Hình AI", "LSTM")
    
    with col4:
        st.metric("📅 Cập Nhật Dữ Liệu", "Hàng Ngày")
    
    st.markdown("---")
    
    # How to use
    st.subheader("🎯 Hướng Dẫn Sử Dụng")
    
    st.markdown("""
        <div style='background: rgba(0, 212, 170, 0.1); padding: 1.5rem; border-radius: 12px; 
                    border-left: 4px solid #00d4aa;'>
            <ol style='margin: 0; color: #ccc; line-height: 1.8; padding-left: 1.5rem;'>
                <li><strong>Chọn mục phân tích</strong> từ thanh điều hướng bên trái</li>
                <li><strong>Với các trang phân tích coin</strong>, chọn coin cần xem từ dropdown</li>
                <li><strong>Đọc phần giải thích</strong> ở đầu mỗi biểu đồ để hiểu ý nghĩa</li>
                <li><strong>Tương tác với biểu đồ</strong>: zoom, hover để xem chi tiết</li>
                <li><strong>Kết hợp nhiều công cụ</strong> để có cái nhìn toàn diện</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("---")
    st.warning("""
        ⚠️ **Lưu ý quan trọng**: Tất cả thông tin và phân tích trên dashboard này chỉ mang tính 
        chất tham khảo và giáo dục. Không được coi là lời khuyên tài chính hay khuyến nghị đầu tư. 
        Thị trường tiền điện tử có tính biến động cao. Hãy tự nghiên cứu và chỉ đầu tư số tiền 
        bạn có thể chấp nhận mất.
    """)
