
"""Main dashboard entry point - Professional Crypto Analytics."""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.monitoring.pages import (
    render_home_page,
    render_market_overview_page,
    render_price_volume_page,
    render_volatility_risk_page,
    render_correlation_page,
    render_quant_metrics_page,
    render_factor_analysis_page,
    render_portfolio_analysis_page,
    render_investment_insights_page,
    render_prediction_page,
    render_compare_models_page,
    render_sentiment_analysis_page,
)


def configure_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Phân Tích Crypto - Deep Learning",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def inject_custom_css():
    """Inject professional dark theme CSS styling."""
    st.markdown("""
        <style>
        /* ============ Import Google Font ============ */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* ============ Hide default Streamlit multipage nav ============ */
        [data-testid="stSidebarNav"] {
            display: none !important;
        }
        
        /* ============ Root Variables ============ */
        :root {
            --bg-primary: #0e1117;
            --bg-secondary: #1a1d26;
            --bg-card: #21262d;
            --text-primary: #f0f6fc;
            --text-secondary: #8b949e;
            --accent-primary: #667eea;
            --accent-secondary: #764ba2;
            --accent-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --success: #00d4aa;
            --warning: #ffc107;
            --danger: #ff6b6b;
            --border-color: #30363d;
            --shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        /* ============ Global Styles ============ */
        .stApp {
            background: var(--bg-primary);
            font-family: 'Inter', sans-serif;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* ============ Gradient Banner Text Fix ============ */
        div[style*="linear-gradient"] h3,
        div[style*="linear-gradient"] p {
            color: white !important;
        }
        div[style*="linear-gradient"] p {
            color: rgba(255,255,255,0.9) !important;
        }
        
        /* ============ Sidebar Styling ============ */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1d26 0%, #0e1117 100%);
            border-right: 1px solid var(--border-color);
        }
        
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
            color: var(--text-primary);
        }
        
        /* Sidebar header */
        .sidebar-header {
            text-align: center;
            padding: 2rem 1rem;
            background: var(--accent-gradient);
            border-radius: 0 0 20px 20px;
            margin: -1rem -1rem 1.5rem -1rem;
        }
        
        .sidebar-header h1 {
            color: white !important;
            font-size: 1.2rem;
            font-weight: 700;
            margin: 0.5rem 0 0 0;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        
        .sidebar-header p {
            color: rgba(255,255,255,0.85);
            font-size: 0.8rem;
            margin: 0.3rem 0 0 0;
        }
        
        .sidebar-logo {
            font-size: 3rem;
            margin-bottom: 0.5rem;
        }
        
        /* Navigation section */
        .nav-section {
            padding: 0.5rem 0;
        }
        
        .nav-section-title {
            color: var(--accent-primary) !important;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin: 1rem 0 0.5rem 0;
            padding-left: 0.5rem;
        }
        
        /* Radio buttons styling */
        .stRadio > label {
            color: var(--text-secondary) !important;
            font-weight: 500;
        }
        
        .stRadio > div {
            gap: 0.25rem;
        }
        
        .stRadio > div > label {
            background: transparent;
            border: none;
            padding: 0.6rem 1rem;
            border-radius: 8px;
            transition: all 0.2s ease;
            color: var(--text-secondary);
        }
        
        .stRadio > div > label:hover {
            background: rgba(102, 126, 234, 0.1);
            color: var(--text-primary);
        }
        
        /* ============ Main Content Styles ============ */
        .main > div {
            padding: 1rem 2rem 2rem 2rem;
        }
        
        /* Headers */
        h1 {
            color: var(--text-primary) !important;
            font-weight: 700;
            font-size: 2rem;
            margin-bottom: 1.5rem;
        }
        
        h2, h3 {
            color: var(--text-primary) !important;
            font-weight: 600;
        }
        
        h2 {
            font-size: 1.4rem;
        }
        
        h3 {
            font-size: 1.2rem;
            color: var(--accent-primary) !important;
        }
        
        /* ============ Metric Cards ============ */
        [data-testid="stMetric"] {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1rem 1.25rem;
            box-shadow: var(--shadow);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        [data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
        }
        
        [data-testid="stMetric"] label {
            color: var(--text-secondary) !important;
            font-size: 0.85rem;
            font-weight: 500;
        }
        
        [data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: var(--text-primary) !important;
            font-weight: 700;
            font-size: 1.5rem;
        }
        
        /* ============ DataFrame Styling ============ */
        [data-testid="stDataFrame"] {
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
        }
        
        /* ============ Charts Container ============ */
        [data-testid="stPlotlyChart"] {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1rem;
            box-shadow: var(--shadow);
        }
        
        /* ============ Custom Info Boxes ============ */
        .chart-intro {
            background: rgba(102, 126, 234, 0.1);
            border-left: 4px solid var(--accent-primary);
            border-radius: 0 8px 8px 0;
            padding: 1rem 1.25rem;
            margin-bottom: 1rem;
        }
        
        .chart-intro h4 {
            color: var(--accent-primary) !important;
            margin: 0 0 0.5rem 0;
            font-size: 0.95rem;
        }
        
        .chart-intro p, .chart-intro li {
            color: var(--text-secondary);
            font-size: 0.9rem;
            margin: 0;
        }
        
        /* ============ Feature Cards ============ */
        .feature-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .feature-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
            border-color: var(--accent-primary);
        }
        
        .feature-card h4 {
            color: var(--accent-primary) !important;
            margin: 0 0 0.5rem 0;
        }
        
        .feature-card p {
            color: var(--text-secondary);
            margin: 0;
            font-size: 0.9rem;
        }
        
        /* ============ Buttons ============ */
        .stButton > button {
            background: var(--accent-gradient);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
            font-weight: 600;
            transition: all 0.2s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
        }
        
        /* ============ Select Boxes ============ */
        .stSelectbox > div > div {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-primary);
        }
        
        /* ============ Expanders ============ */
        .streamlit-expanderHeader {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-primary) !important;
        }
        
        /* ============ Alerts ============ */
        .stAlert {
            border-radius: 8px;
        }
        
        /* ============ Horizontal Rule ============ */
        hr {
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--border-color), transparent);
            margin: 2rem 0;
        }
        
        /* ============ Scrollbar ============ */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--bg-secondary);
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--accent-primary);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--accent-secondary);
        }
        
        /* ============ Footer ============ */
        .dashboard-footer {
            text-align: center;
            padding: 1.5rem;
            color: var(--text-secondary);
            font-size: 0.75rem;
            border-top: 1px solid var(--border-color);
            margin-top: 3rem;
        }
        </style>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render professional sidebar navigation."""
    with st.sidebar:
        # Header with logo
        st.markdown("""
            <div class='sidebar-header'>
                <div class='sidebar-logo'>🚀</div>
                <h1>Crypto Analytics</h1>
                <p>Bảng Điều Khiển Deep Learning</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Navigation sections
        st.markdown("<p class='nav-section-title'>📋 ĐIỀU HƯỚNG</p>", unsafe_allow_html=True)
        
        page = st.radio(
            "Điều hướng",
            [
                "🏠 Trang Chủ",
                "🌍 Tổng Quan Thị Trường",
                "📈 Phân Tích Giá & Khối Lượng",
                "📉 Phân Tích Biến Động & Rủi Ro",
                "🔗 Phân Tích Tương Quan",
                "📐 Chỉ Số Định Lượng",
                "🧩 Phân Tích Nhân Tố",
                "🧺 Phân Tích Danh Mục",
                "🧠 Khuyến Nghị Đầu Tư",
                "🔮 Dự Đoán Giá",
                "⚖️ So Sánh Mô Hình",
                "📊 Phân Tích Tâm Lý Thị Trường"
            ],
            label_visibility="collapsed"
        )
        
        # Footer
        st.markdown("---")
        st.markdown("""
            <div style='text-align: center; color: #666; font-size: 0.75rem; padding: 1rem 0;'>
                <p style='margin: 0;'>Xây dựng với 💜 bằng Streamlit</p>
                <p style='margin: 0.3rem 0 0 0;'>© 2024 Crypto Analytics</p>
            </div>
        """, unsafe_allow_html=True)
        
        return page


def main():
    """Main dashboard application."""
    configure_page()
    inject_custom_css()
    
    # Render sidebar and get selection
    page = render_sidebar()
    
    # Route to appropriate page
    if page == "🏠 Trang Chủ":
        render_home_page()
    
    elif page == "🌍 Tổng Quan Thị Trường":
        render_market_overview_page()
    
    elif page == "📈 Phân Tích Giá & Khối Lượng":
        render_price_volume_page()  # No coin param - selector inside page
    
    elif page == "📉 Phân Tích Biến Động & Rủi Ro":
        render_volatility_risk_page()  # No coin param - selector inside page
    
    elif page == "🔗 Phân Tích Tương Quan":
        render_correlation_page()
    
    elif page == "📐 Chỉ Số Định Lượng":
        render_quant_metrics_page()
    
    elif page == "🧩 Phân Tích Nhân Tố":
        render_factor_analysis_page()
    
    elif page == "🧺 Phân Tích Danh Mục":
        render_portfolio_analysis_page()
    
    elif page == "🧠 Khuyến Nghị Đầu Tư":
        render_investment_insights_page()
    
    elif page == "🔮 Dự Đoán Giá":
        render_prediction_page()
    
    elif page == "⚖️ So Sánh Mô Hình":
        render_compare_models_page()
    
    elif page == "📊 Phân Tích Tâm Lý Thị Trường":
        render_sentiment_analysis_page()
    
    # Footer
    st.markdown("""
        <div class='dashboard-footer'>
            <p>🚀 Crypto Analytics | Ứng dụng Deep Learning & Phân Tích Nâng Cao</p>
            <p>Dữ liệu chỉ mang tính chất tham khảo. Không phải lời khuyên tài chính.</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
