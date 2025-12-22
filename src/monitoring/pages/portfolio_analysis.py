"""Portfolio Analysis Page - Phân tích danh mục đầu tư."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.analysis.market_analyzer import load_all_coins_data
from src.analysis.portfolio_engine import (
    equal_weight_portfolio,
    risk_parity_portfolio,
    backtest_portfolio,
    calculate_portfolio_metrics,
    compare_portfolio_strategies
)
from src.assistant.chart_analyzer import get_chart_analyzer


def render_portfolio_analysis_page():
    """Render trang phân tích danh mục đầu tư."""
    st.title("Phân Tích Danh Mục Đầu Tư")
    
    # Page introduction
    st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;'>
            <h3 style='color: white; margin: 0;'>Xây Dựng & Kiểm Thử Danh Mục</h3>
            <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;'>
                Phân tích các chiến lược xây dựng danh mục đầu tư khác nhau và đánh giá hiệu suất lịch sử.
                So sánh giữa Equal Weight (phân bổ đều) và Risk Parity (phân bổ theo rủi ro).
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Đang chạy backtest danh mục..."):
        data_dict = load_all_coins_data(data_dir="data/raw/train")
    
    if not data_dict:
        st.error("❌ Không có dữ liệu")
        return
    
    # Strategy Comparison
    st.subheader("So Sánh Chiến Lược")
    
    st.markdown("""
        <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 8px; 
                    border-left: 4px solid #667eea; margin-bottom: 1rem;'>
            <h4 style='margin: 0 0 0.5rem 0; color: #667eea;'>Bảng So Sánh Hiệu Suất Các Chiến Lược</h4>
            <p style='margin: 0; color: #ccc;'>
                Bảng hiển thị kết quả backtest của 2 chiến lược phân bổ danh mục với vốn ban đầu $10,000 trên dữ liệu lịch sử.
                Mỗi chiến lược có cách phân bổ tỷ trọng khác nhau giữa các coin.
            </p>
            <ul style='margin: 0.5rem 0 0 0; color: #ccc; padding-left: 1.5rem;'>
                <li><strong>Total Return</strong>: Tổng lợi nhuận từ đầu đến cuối kỳ (%)</li>
                <li><strong>CAGR</strong>: Tốc độ tăng trưởng kép hàng năm - so sánh được giữa các thời kỳ khác nhau</li>
                <li><strong>Sharpe Ratio</strong>: Lợi nhuận điều chỉnh rủi ro (> 1 là tốt, > 2 là xuất sắc)</li>
                <li><strong>Max Drawdown</strong>: Mức lỗ tối đa từ đỉnh - chỉ số rủi ro quan trọng nhất</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    comparison_df = compare_portfolio_strategies(data_dict, initial_capital=10000)
    
    # Rename index to Vietnamese based on actual strategies in the dataframe
    strategy_names_vi = {
        'Equal Weight': 'Equal Weight (Phân bổ đều)',
        'Risk Parity': 'Risk Parity (Theo rủi ro)',
        'Vol Targeting': 'Vol Targeting (Mục tiêu biến động)'
    }
    comparison_df_display = comparison_df.copy()
    comparison_df_display.index = [strategy_names_vi.get(idx, idx) for idx in comparison_df.index]
    
    st.dataframe(
        comparison_df_display.style.format({
            'total_return': '{:.2f}%',
            'cagr': '{:.2f}%',
            'sharpe_ratio': '{:.2f}',
            'sortino_ratio': '{:.2f}',
            'max_drawdown': '{:.2f}%',
            'annualized_volatility': '{:.2f}%'
        }),
        width='stretch'
    )
    
    # Strategy Analysis
    best_strategy_idx = comparison_df['sharpe_ratio'].idxmax()
    best_strategy = "Equal Weight" if best_strategy_idx == "equal_weight" else "Risk Parity"
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
            **Equal Weight (Phân bổ đều)**  
            Phân bổ vốn đều cho tất cả coin (mỗi coin = 1/N).  
            Đơn giản, dễ hiểu, không cần dự đoán tương lai.
        """)
    with col2:
        st.info("""
            **Risk Parity (Theo rủi ro)**  
            Phân bổ sao cho mỗi coin đóng góp rủi ro như nhau.  
            Coin biến động ít được phân bổ nhiều hơn.
        """)
    
    # Individual Strategy Analysis
    st.markdown("---")
    st.subheader("Phân Tích Chi Tiết Chiến Lược")
    
    strategy = st.selectbox(
        "Chọn Chiến Lược",
        ["Equal Weight (Phân bổ đều)", "Risk Parity (Theo rủi ro)"]
    )
    
    strategy_key = "Equal Weight" if "Equal" in strategy else "Risk Parity"
    
    if strategy_key == "Equal Weight":
        weights = {coin: 1.0 / len(data_dict) for coin in data_dict.keys()}
    else:  # Risk Parity
        weights = risk_parity_portfolio(data_dict)
    
    # Backtest
    portfolio_df = backtest_portfolio(data_dict, weights, initial_capital=10000)
    
    if not portfolio_df.empty:
        # Equity Curve
        st.markdown("""
            <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 8px; 
                        border-left: 4px solid #667eea; margin-bottom: 1rem;'>
                <h4 style='margin: 0 0 0.5rem 0; color: #667eea;'>Đường Cong Vốn - Lịch Sử Giá Trị Danh Mục</h4>
                <p style='margin: 0; color: #ccc;'>
                    Biểu đồ cho thấy giá trị danh mục theo thời gian nếu bạn đầu tư $10,000 từ đầu kỳ.
                    Vùng tô màu bên dưới cho thấy sự tăng trưởng tổng thể.
                </p>
                <ul style='margin: 0.5rem 0 0 0; color: #ccc; padding-left: 1.5rem;'>
                    <li><strong>Đường đi lên</strong>: Danh mục đang sinh lời - chiến lược hiệu quả</li>
                    <li><strong>Đường đi xuống</strong>: Danh mục đang lỗ - cân nhắc điều chỉnh</li>
                    <li><strong>Các đợt giảm sâu</strong>: Chính là các giai đoạn drawdown - thời điểm khó khăn nhất</li>
                </ul>
                <p style='margin: 0.5rem 0 0 0; color: #ccc;'>
                    <strong>Lưu ý:</strong> Kết quả quá khứ không đảm bảo tương lai, nhưng giúp hiểu hành vi của chiến lược trong các điều kiện khác nhau.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=portfolio_df.index,
            y=portfolio_df['portfolio_value'],
            name='Giá Trị Danh Mục',
            line=dict(color='#667eea', width=2),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.2)'
        ))
        
        fig.update_layout(
            title=f"Đường Cong Vốn - {strategy}",
            xaxis_title="Ngày",
            yaxis_title="Giá Trị Danh Mục ($)",
            height=400,
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # AI Analysis Button for Portfolio Chart
        chart_analyzer = get_chart_analyzer()
        if st.button("🤖 AI Phân Tích Danh Mục", key="analyze_portfolio"):
            with st.spinner("🔄 Đang phân tích với GPT-4..."):
                metrics = calculate_portfolio_metrics(portfolio_df)
                
                # Prepare strategies summary
                strategies = "Equal Weight, Risk Parity"
                
                # Get returns and drawdowns for each strategy
                returns_summary = ""
                drawdown_summary = ""
                for idx, row in comparison_df.iterrows():
                    returns_summary += f"- {idx}: {row['total_return']:.2f}%\n"
                    drawdown_summary += f"- {idx}: {row['max_drawdown']:.2f}%\n"
                
                best_name = comparison_df['sharpe_ratio'].idxmax()
                worst_name = comparison_df['sharpe_ratio'].idxmin()
                
                chart_data = {
                    "strategies": strategies,
                    "returns_summary": returns_summary,
                    "best_strategy": best_name,
                    "best_return": comparison_df.loc[best_name, 'total_return'],
                    "worst_strategy": worst_name,
                    "worst_return": comparison_df.loc[worst_name, 'total_return'],
                    "drawdown_summary": drawdown_summary
                }
                
                analysis = chart_analyzer.analyze_chart(
                    coin="portfolio",
                    chart_type="portfolio_returns",
                    chart_data=chart_data,
                    chart_title=f"Đường Cong Vốn - {strategy}"
                )
                st.markdown(analysis)
        
        # Metrics
        metrics = calculate_portfolio_metrics(portfolio_df)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Tổng Lợi Nhuận", f"{metrics['total_return']:.2f}%")
        
        with col2:
            st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        
        with col3:
            st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
        
        with col4:
            st.metric("CAGR", f"{metrics['cagr']:.2f}%")
        
        # Weights
        st.markdown("---")
        st.subheader("Tỷ Trọng Danh Mục")
        
        st.markdown("""
            <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 8px; 
                        border-left: 4px solid #667eea; margin-bottom: 1rem;'>
                <h4 style='margin: 0 0 0.5rem 0; color: #667eea;'>Tỷ Trọng Phân Bổ Danh Mục</h4>
                <p style='margin: 0; color: #ccc;'>
                    Bảng và biểu đồ tròn hiển thị phần trăm vốn phân bổ cho mỗi coin theo chiến lược đã chọn.
                    Đây là thông tin quan trọng để bạn tái tạo danh mục trong thực tế.
                </p>
                <ul style='margin: 0.5rem 0 0 0; color: #ccc; padding-left: 1.5rem;'>
                    <li><strong>Equal Weight</strong>: Mỗi coin được phân bổ đều (VD: 9 coin = mỗi coin 11.1%)</li>
                    <li><strong>Risk Parity</strong>: Coin biến động thấp được phân bổ nhiều hơn để cân bằng rủi ro</li>
                </ul>
                <p style='margin: 0.5rem 0 0 0; color: #ccc;'>
                    <strong>Lưu ý:</strong> Tỷ trọng nên được tái cân bằng định kỳ (hàng tháng hoặc quý) để duy trì chiến lược.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        weights_df = pd.DataFrame.from_dict(weights, orient='index', columns=['Tỷ Trọng'])
        weights_df['Tỷ Trọng'] = weights_df['Tỷ Trọng'] * 100
        weights_df = weights_df.sort_values('Tỷ Trọng', ascending=False)
        weights_df.index = weights_df.index.str.upper()
        weights_df.index.name = 'Coin'
        
        # Display table and pie chart with better layout
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.markdown("**Bảng Tỷ Trọng**")
            st.dataframe(
                weights_df.style.format({'Tỷ Trọng': '{:.2f}%'}),
                width='stretch',
                height=350
            )
        
        with col2:
            # Pie chart - bigger size
            fig_pie = go.Figure(data=[go.Pie(
                labels=weights_df.index,
                values=weights_df['Tỷ Trọng'],
                hole=0.4,
                textinfo='percent+label',
                textposition='outside',
                marker=dict(colors=['#667eea', '#764ba2', '#00d4aa', '#ffc107', '#ff6b6b', '#17a2b8', '#28a745', '#fd7e14', '#6f42c1'])
            )])
            fig_pie.update_layout(
                title=dict(text="Phân Bổ Danh Mục", font=dict(size=18)),
                height=450,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5
                ),
                margin=dict(t=60, b=80, l=20, r=20)
            )
            st.plotly_chart(fig_pie, width='stretch')
        
        # AI Analysis Button for Portfolio Allocation
        if st.button("🤖 AI Phân Tích Phân Bổ Danh Mục", key="analyze_allocation"):
            with st.spinner("🔄 Đang phân tích với GPT-4..."):
                top_weight_coin = weights_df.index[0]
                top_weight = weights_df['Tỷ Trọng'].iloc[0]
                min_weight_coin = weights_df.index[-1]
                min_weight = weights_df['Tỷ Trọng'].iloc[-1]
                concentration = weights_df['Tỷ Trọng'].head(3).sum()
                
                chart_data = {
                    "strategy_name": strategy,
                    "coin_count": len(weights_df),
                    "top_weight_coin": top_weight_coin,
                    "top_weight": top_weight,
                    "min_weight_coin": min_weight_coin,
                    "min_weight": min_weight,
                    "concentration": concentration
                }
                
                analysis = chart_analyzer.analyze_chart(
                    coin="portfolio",
                    chart_type="portfolio_allocation",
                    chart_data=chart_data,
                    chart_title=f"Phân Bổ Danh Mục - {strategy}"
                )
                st.markdown(analysis)
    
    # Recommendations
    st.markdown("---")
    st.subheader("Khuyến Nghị Danh Mục")
    
    st.success(f"""
        **Chiến Lược Được Khuyến Nghị: {best_strategy}**  
        
        Dựa trên lợi nhuận điều chỉnh rủi ro (Sharpe Ratio), chiến lược **{best_strategy}** 
        cho kết quả tốt nhất trên dữ liệu lịch sử.
        
        **Lưu ý**: Kết quả quá khứ không đảm bảo kết quả tương lai. 
        Hãy đa dạng hóa và quản lý rủi ro phù hợp với khẩu vị đầu tư của bạn.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
            **Khi Nào Dùng Equal Weight?**
            - Không chắc chắn về coin nào sẽ tốt hơn
            - Muốn đơn giản, dễ tái cân bằng
            - Tin tưởng vào tất cả coin trong danh sách
        """)
    with col2:
        st.info("""
            **Khi Nào Dùng Risk Parity?**
            - Muốn kiểm soát rủi ro tốt hơn
            - Ưu tiên ổn định hơn lợi nhuận tối đa
            - Tránh coin biến động cao chiếm quá nhiều rủi ro
        """)