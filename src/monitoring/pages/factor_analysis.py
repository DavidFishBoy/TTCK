"""Factor Analysis Page - Phân tích nhân tố."""

import streamlit as st
import plotly.express as px
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.analysis.market_analyzer import load_all_coins_data
from src.analysis.factor_analyzer import (
    create_factor_dataframe,
    factor_scatter_plot_data,
    cluster_by_factors
)
from src.assistant.chart_analyzer import get_chart_analyzer


def render_factor_analysis_page():
    """Render trang phân tích nhân tố."""
    st.title("Phân Tích Nhân Tố")
    
    # Page introduction
    st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;'>
            <h3 style='color: white; margin: 0;'>Hiểu Động Lực Thúc Đẩy Giá</h3>
            <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;'>
                Phân tích các yếu tố quan trọng ảnh hưởng đến hiệu suất coin như momentum, 
                biến động, quy mô và thanh khoản. Giúp phân loại coin theo đặc điểm.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Đang phân tích các nhân tố..."):
        data_dict = load_all_coins_data(data_dir="data/raw/train")
    
    if not data_dict:
        st.error("❌ Không có dữ liệu")
        return
    
    # Create factor dataframe
    factor_df = create_factor_dataframe(data_dict)
    
    # Factor Scatter Plot
    st.subheader("Biểu Đồ Phân Tán Nhân Tố")
    
    st.markdown("""
        <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 8px; 
                    border-left: 4px solid #667eea; margin-bottom: 1rem;'>
            <h4 style='margin: 0 0 0.5rem 0; color: #667eea;'>Biểu Đồ Này Cho Biết Gì?</h4>
            <p style='margin: 0; color: #ccc;'>
                So sánh các coin theo 2 nhân tố bạn chọn. Vị trí của coin trên biểu đồ 
                cho thấy đặc điểm của nó so với các coin khác.
            </p>
            <h4 style='margin: 1rem 0 0.5rem 0; color: #667eea;'>Giải Thích Nhân Tố</h4>
            <ul style='margin: 0; color: #ccc; padding-left: 1.5rem;'>
                <li><strong>Momentum</strong>: Đà tăng/giảm giá trong 30 hoặc 90 ngày</li>
                <li><strong>Biến Động</strong>: Mức độ dao động giá</li>
                <li><strong>Quy Mô</strong>: Vốn hóa thị trường (log scale)</li>
                <li><strong>Thanh Khoản</strong>: Tỷ lệ khối lượng/vốn hóa</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    factor_names_vi = {
        'momentum_30d': 'Momentum 30 Ngày',
        'momentum_90d': 'Momentum 90 Ngày',
        'volatility': 'Biến Động',
        'size': 'Quy Mô',
        'liquidity': 'Thanh Khoản',
        'return_7d': 'Lợi Nhuận 7 Ngày'
    }
    
    with col1:
        x_factor = st.selectbox(
            "Trục X",
            ['momentum_30d', 'momentum_90d', 'size', 'liquidity'],
            format_func=lambda x: factor_names_vi.get(x, x)
        )
    
    with col2:
        y_factor = st.selectbox(
            "Trục Y",
            ['volatility', 'momentum_30d', 'return_7d', 'size'],
            format_func=lambda x: factor_names_vi.get(x, x)
        )
    
    scatter_data = factor_scatter_plot_data(factor_df, x_factor=x_factor, y_factor=y_factor)
    
    if not scatter_data.empty:
        fig = px.scatter(
            scatter_data,
            x=x_factor,
            y=y_factor,
            text='coin',
            color='quadrant',
            title=f"{factor_names_vi.get(x_factor, x_factor)} vs {factor_names_vi.get(y_factor, y_factor)}",
            height=500
        )
        
        fig.update_traces(textposition='top center', textfont_size=10)
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, width='stretch')
        
        # Quadrant Analysis
        st.markdown("**Phân Tích Theo Góc Phần Tư:**")
        quadrants = scatter_data.groupby('quadrant')['coin'].apply(list).to_dict()
        
        col1, col2 = st.columns(2)
        with col1:
            if 'High-High' in quadrants:
                st.success(f"**Cao-Cao**: {', '.join([c.upper() for c in quadrants['High-High']])}")
            if 'Low-Low' in quadrants:
                st.error(f"**Thấp-Thấp**: {', '.join([c.upper() for c in quadrants['Low-Low']])}")
        with col2:
            if 'High-Low' in quadrants:
                st.warning(f"**Cao-Thấp**: {', '.join([c.upper() for c in quadrants['High-Low']])}")
            if 'Low-High' in quadrants:
                st.info(f"**Thấp-Cao**: {', '.join([c.upper() for c in quadrants['Low-High']])}")
        
        # AI Analysis Button for Scatter Plot
        chart_analyzer = get_chart_analyzer()
        if st.button("🤖 AI Phân Tích Biểu Đồ Nhân Tố", key="analyze_factors"):
            with st.spinner("🔄 Đang phân tích với GPT-4..."):
                # Prepare scatter data summary
                scatter_summary = ""
                for _, row in scatter_data.iterrows():
                    scatter_summary += f"- {row['coin'].upper()}: {x_factor}={row[x_factor]:.2f}, {y_factor}={row[y_factor]:.2f}\n"
                
                chart_data = {
                    "x_factor": factor_names_vi.get(x_factor, x_factor),
                    "y_factor": factor_names_vi.get(y_factor, y_factor),
                    "coin_count": len(scatter_data),
                    "scatter_data": scatter_summary
                }
                
                analysis = chart_analyzer.analyze_chart(
                    coin="all",
                    chart_type="factor_scatter",
                    chart_data=chart_data,
                    chart_title=f"{factor_names_vi.get(x_factor, x_factor)} vs {factor_names_vi.get(y_factor, y_factor)}"
                )
                st.markdown(analysis)
    
    # Clustering
    st.markdown("---")
    st.subheader("Phân Cụm Coin")
    
    st.markdown("""
        <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 8px; 
                    border-left: 4px solid #667eea; margin-bottom: 1rem;'>
            <h4 style='margin: 0 0 0.5rem 0; color: #667eea;'>Phân Cụm Là Gì?</h4>
            <p style='margin: 0; color: #ccc;'>
                Thuật toán K-Means nhóm các coin có đặc điểm tương tự lại với nhau.
                Coin trong cùng cụm có xu hướng hoạt động giống nhau.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    n_clusters = st.slider("Số Lượng Cụm", 2, 5, 3)
    
    clustered_df = cluster_by_factors(factor_df, n_clusters=n_clusters)
    
    # Display clusters
    for cluster_id in sorted(clustered_df['cluster'].unique()):
        cluster_data = clustered_df[clustered_df['cluster'] == cluster_id]
        
        with st.expander(f"Cụm {cluster_id + 1}: {cluster_data['cluster_description'].iloc[0]}"):
            coins_list = ', '.join(cluster_data['coin'].str.upper())
            st.write(f"**Các Coin**: {coins_list}")
            
            # Average factors
            avg_mom = cluster_data['momentum_30d'].mean()
            avg_vol = cluster_data['volatility'].mean()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Momentum TB", f"{avg_mom:.2f}%")
            with col2:
                st.metric("Biến Động TB", f"{avg_vol:.2f}%")
            with col3:
                st.metric("Số Coin", len(cluster_data))
            
            st.dataframe(
                cluster_data[['coin', 'momentum_30d', 'volatility', 'size']].style.format({
                    'momentum_30d': '{:.2f}%',
                    'volatility': '{:.2f}%',
                    'size': '{:.2f}'
                }),
                width='stretch'
            )
    
    # AI Analysis Button for Clusters
    if st.button("🤖 AI Phân Tích Phân Cụm", key="analyze_clusters"):
        with st.spinner("🔄 Đang phân tích với GPT-4..."):
            # Prepare cluster details
            cluster_details = ""
            for cluster_id in sorted(clustered_df['cluster'].unique()):
                cluster_data = clustered_df[clustered_df['cluster'] == cluster_id]
                coins = ', '.join(cluster_data['coin'].str.upper())
                desc = cluster_data['cluster_description'].iloc[0]
                cluster_details += f"- Cluster {cluster_id + 1} ({desc}): {coins}\n"
            
            factors_used = "momentum_30d, volatility, size"
            
            chart_data = {
                "n_clusters": n_clusters,
                "factors_used": factors_used,
                "cluster_details": cluster_details
            }
            
            analysis = chart_analyzer.analyze_chart(
                coin="all",
                chart_type="factor_cluster",
                chart_data=chart_data,
                chart_title="Phân Cụm Coin"
            )
            st.markdown(analysis)
    
    # Factor Summary
    st.markdown("---")
    st.subheader("Bảng Tóm Tắt Nhân Tố")
    
    st.markdown("""
        <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 8px; 
                    border-left: 4px solid #667eea; margin-bottom: 1rem;'>
            <p style='margin: 0; color: #ccc;'>
                Bảng dưới hiển thị giá trị các nhân tố cho từng coin. 
                Có thể sử dụng để so sánh và lọc coin theo tiêu chí mong muốn.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Add ranking
    display_df = factor_df[['coin', 'momentum_30d', 'momentum_90d', 'volatility', 'size', 'liquidity']].copy()
    display_df['coin'] = display_df['coin'].str.upper()
    display_df.columns = ['Coin', 'Momentum 30N', 'Momentum 90N', 'Biến Động', 'Quy Mô', 'Thanh Khoản']
    
    st.dataframe(
        display_df.style.format({
            'Momentum 30N': '{:.2f}%',
            'Momentum 90N': '{:.2f}%',
            'Biến Động': '{:.2f}%',
            'Quy Mô': '{:.2f}',
            'Thanh Khoản': '{:.4f}'
        }),
        width='stretch'
    )
    
    # Insights
    st.markdown("---")
    st.subheader("Nhận Định Nhân Tố")
    
    # Find best momentum coin
    best_mom = factor_df.loc[factor_df['momentum_30d'].idxmax()]
    lowest_vol = factor_df.loc[factor_df['volatility'].idxmin()]
    
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"""
            **Momentum Cao Nhất**: {best_mom['coin'].upper()}
            
            Momentum 30 ngày: {best_mom['momentum_30d']:.2f}%
            
            Coin này đang có đà tăng mạnh nhất. Phù hợp cho chiến lược theo xu hướng.
        """)
    with col2:
        st.info(f"""
            **Biến Động Thấp Nhất**: {lowest_vol['coin'].upper()}
            
            Biến động: {lowest_vol['volatility']:.2f}%
            
            Coin này ổn định nhất. Phù hợp cho nhà đầu tư ưu tiên an toàn.
        """)