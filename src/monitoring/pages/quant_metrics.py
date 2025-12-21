# src/monitoring/pages/quant_metrics.py

"""
Quant Metrics Page - Trang chỉ số định lượng và so sánh hiệu suất điều chỉnh rủi ro
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.analysis.market_analyzer import load_all_coins_data
from src.analysis.financial_metrics import get_all_metrics
from src.assistant.chart_analyzer import get_chart_analyzer


def render_quant_metrics_page():
    """Render trang chỉ số định lượng."""
    st.title("📐 Chỉ Số Định Lượng")
    
    # Page introduction
    st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;'>
            <h3 style='color: white; margin: 0;'>📊 Phân Tích Hiệu Suất Điều Chỉnh Rủi Ro</h3>
            <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;'>
                So sánh các đồng coin dựa trên các chỉ số định lượng được các nhà đầu tư chuyên nghiệp sử dụng.
                Các chỉ số này giúp đánh giá hiệu suất đầu tư có tính đến yếu tố rủi ro.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Đang tính toán chỉ số..."):
        data_dict = load_all_coins_data(data_dir="data/raw/train")
    
    if not data_dict:
        st.error("❌ Không có dữ liệu")
        return
    
    # Calculate metrics for all coins
    all_metrics = []
    for coin, df in data_dict.items():
        metrics = get_all_metrics(df['close'], coin_name=coin)
        if 'error' not in metrics:
            all_metrics.append(metrics)
    
    if not all_metrics:
        st.error("❌ Không thể tính toán chỉ số")
        return
    
    metrics_df = pd.DataFrame(all_metrics)
    
    # Display ranking table
    st.subheader("🏆 Xếp Hạng Coin Theo Chỉ Số")
    
    # Chart explanation
    st.markdown("""
        <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 8px; 
                    border-left: 4px solid #667eea; margin-bottom: 1rem;'>
            <h4 style='margin: 0 0 0.5rem 0; color: #667eea;'>📊 Bảng Này Hiển Thị Gì?</h4>
            <p style='margin: 0; color: #ccc;'>
                Bảng xếp hạng các coin theo chỉ số bạn chọn. Cột "Xếp Hạng" hiển thị thứ tự từ tốt nhất đến kém nhất.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Let user choose metric to sort by
    sort_options = {
        'sharpe_ratio': 'Sharpe Ratio (Lợi nhuận/Rủi ro)',
        'sortino_ratio': 'Sortino Ratio (Lợi nhuận/Rủi ro giảm)',
        'calmar_ratio': 'Calmar Ratio (Lợi nhuận/Sụt giảm tối đa)',
        'cagr': 'CAGR (Tăng trưởng hàng năm)',
        'max_drawdown': 'Max Drawdown (Sụt giảm tối đa)'
    }
    
    sort_by = st.selectbox(
        "Sắp xếp theo",
        list(sort_options.keys()),
        format_func=lambda x: sort_options[x]
    )
    
    ascending = (sort_by == 'max_drawdown')
    sorted_df = metrics_df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
    
    # Add rank column
    sorted_df['Xếp Hạng'] = range(1, len(sorted_df) + 1)
    
    # Display table with formatting
    display_df = sorted_df[[
        'Xếp Hạng', 'coin', 'current_price', 'cagr', 'annualized_volatility',
        'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'max_drawdown'
    ]].copy()
    
    display_df.columns = [
        'Xếp Hạng', 'Coin', 'Giá', 'CAGR', 'Biến Động',
        'Sharpe', 'Sortino', 'Calmar', 'Max DD'
    ]
    
    # Convert coin names to uppercase
    display_df['Coin'] = display_df['Coin'].str.upper()
    
    st.dataframe(
        display_df.style.format({
            'Xếp Hạng': '{:.0f}',
            'Giá': '${:,.2f}',
            'CAGR': '{:.2f}%',
            'Biến Động': '{:.2f}%',
            'Sharpe': '{:.2f}',
            'Sortino': '{:.2f}',
            'Calmar': '{:.2f}',
            'Max DD': '{:.2f}%'
        }),
        use_container_width=True,
        height=400
    )
    
    # AI Analysis Button for Quant Metrics
    chart_analyzer = get_chart_analyzer()
    if st.button("🤖 AI Phân Tích Chỉ Số Định Lượng", key="analyze_quant"):
        with st.spinner("🔄 Đang phân tích với GPT-4..."):
            # Prepare metrics table
            metrics_table = ""
            for _, row in display_df.head(5).iterrows():
                metrics_table += f"| {row['Coin']} | {row['Sharpe']:.2f} | {row['Sortino']:.2f} | {row['Calmar']:.2f} | {row['Max DD']:.2f}% |\n"
            
            best_sharpe = display_df.iloc[0] if sort_by == 'sharpe_ratio' else metrics_df.nlargest(1, 'sharpe_ratio').iloc[0]
            best_sortino = metrics_df.nlargest(1, 'sortino_ratio').iloc[0]
            lowest_dd = metrics_df.nsmallest(1, 'max_drawdown').iloc[0]
            
            chart_data = {
                "metrics_table": metrics_table,
                "best_sharpe_coin": best_sharpe['coin'].upper() if 'coin' in best_sharpe else best_sharpe['Coin'],
                "best_sharpe": best_sharpe['sharpe_ratio'] if 'sharpe_ratio' in best_sharpe else best_sharpe['Sharpe'],
                "best_sortino_coin": best_sortino['coin'].upper(),
                "best_sortino": best_sortino['sortino_ratio'],
                "lowest_dd_coin": lowest_dd['coin'].upper(),
                "lowest_dd": lowest_dd['max_drawdown']
            }
            
            analysis = chart_analyzer.analyze_chart(
                coin="all",
                chart_type="quant_metrics",
                chart_data=chart_data,
                chart_title="Chỉ Số Định Lượng"
            )
            st.markdown(analysis)
    
    # Analysis based on selected sort metric
    st.markdown("---")
    st.subheader(f"📈 Phân Tích Theo {sort_options[sort_by]}")
    
    top_coin = sorted_df.iloc[0]
    bottom_coin = sorted_df.iloc[-1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        if sort_by == 'sharpe_ratio':
            st.success(f"""
                #### 🥇 Coin Tốt Nhất: {top_coin['coin'].upper()}
                **Sharpe Ratio: {top_coin['sharpe_ratio']:.2f}**
                
                Coin này có tỷ lệ lợi nhuận trên mỗi đơn vị rủi ro cao nhất. 
                Sharpe > 1 được coi là tốt, > 2 là xuất sắc.
                
                **Giải thích**: Với mỗi đơn vị rủi ro (biến động) bạn chấp nhận, 
                bạn nhận được {top_coin['sharpe_ratio']:.2f} đơn vị lợi nhuận.
            """)
        elif sort_by == 'sortino_ratio':
            st.success(f"""
                #### 🥇 Coin Tốt Nhất: {top_coin['coin'].upper()}
                **Sortino Ratio: {top_coin['sortino_ratio']:.2f}**
                
                Coin này có tỷ lệ lợi nhuận/rủi ro giảm giá tốt nhất. 
                Sortino chỉ tính rủi ro khi giá giảm, phù hợp cho nhà đầu tư 
                muốn tránh lỗ.
            """)
        elif sort_by == 'calmar_ratio':
            st.success(f"""
                #### 🥇 Coin Tốt Nhất: {top_coin['coin'].upper()}
                **Calmar Ratio: {top_coin['calmar_ratio']:.2f}**
                
                Coin này có tỷ lệ lợi nhuận/sụt giảm tối đa cao nhất. 
                Calmar Ratio cao nghĩa là coin phục hồi tốt sau các đợt giảm giá mạnh.
            """)
        elif sort_by == 'cagr':
            st.success(f"""
                #### 🥇 Coin Tốt Nhất: {top_coin['coin'].upper()}
                **CAGR: {top_coin['cagr']:.2f}%**
                
                Coin này có tốc độ tăng trưởng hàng năm cao nhất. 
                CAGR cho biết trung bình mỗi năm bạn tăng trưởng bao nhiêu phần trăm.
            """)
        else:  # max_drawdown
            st.success(f"""
                #### 🥇 Coin An Toàn Nhất: {top_coin['coin'].upper()}
                **Max Drawdown: {top_coin['max_drawdown']:.2f}%**
                
                Coin này có mức sụt giảm tối đa thấp nhất. 
                Max Drawdown thấp nghĩa là rủi ro mất vốn trong đợt downtrend thấp hơn.
            """)
    
    with col2:
        if sort_by != 'max_drawdown':
            st.warning(f"""
                #### ⚠️ Coin Cần Cân Nhắc: {bottom_coin['coin'].upper()}
                **{sort_options[sort_by].split('(')[0].strip()}: {bottom_coin[sort_by]:.2f}**
                
                Coin này xếp cuối theo chỉ số đã chọn. Tuy nhiên, 
                điều này không có nghĩa là coin xấu - hãy xem xét 
                thêm các yếu tố khác và kết hợp nhiều chỉ số.
            """)
        else:
            st.warning(f"""
                #### ⚠️ Coin Rủi Ro Cao: {bottom_coin['coin'].upper()}
                **Max Drawdown: {bottom_coin['max_drawdown']:.2f}%**
                
                Coin này có mức sụt giảm tối đa cao nhất, nghĩa là 
                trong quá khứ đã có lúc giảm rất mạnh. Cần quản lý 
                rủi ro cẩn thận nếu đầu tư vào coin này.
            """)
    
    # Key Metrics Explanation
    st.markdown("---")
    st.subheader("📚 Giải Thích Các Chỉ Số")
    
    st.markdown("""
        <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 8px; 
                    border-left: 4px solid #667eea;'>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📈 CAGR (Compound Annual Growth Rate)**  
        Tỷ lệ tăng trưởng kép hàng năm. Cho biết trung bình mỗi năm tài sản tăng bao nhiêu %.
        
        **📊 Biến Động (Volatility)**  
        Độ lệch chuẩn của lợi nhuận hàng năm. Biến động cao = rủi ro cao.
        
        **⚖️ Sharpe Ratio**  
        Lợi nhuận vượt trội trên mỗi đơn vị rủi ro. Sharpe > 1 là tốt.
        """)
    
    with col2:
        st.markdown("""
        **📉 Sortino Ratio**  
        Giống Sharpe nhưng chỉ tính rủi ro khi giá giảm. Tốt hơn Sharpe cho đánh giá downside risk.
        
        **🎯 Calmar Ratio**  
        Lợi nhuận chia cho mức sụt giảm tối đa. Đánh giá khả năng phục hồi sau downtrend.
        
        **🔻 Max Drawdown**  
        Mức giảm lớn nhất từ đỉnh xuống đáy. Cho biết rủi ro tối đa trong quá khứ.
        """)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Top Performers Summary
    st.markdown("---")
    st.subheader("🎯 Top 3 Theo Từng Chỉ Số")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🏆 Sharpe Ratio Cao Nhất**")
        top_sharpe = metrics_df.nlargest(3, 'sharpe_ratio')
        for i, (_, row) in enumerate(top_sharpe.iterrows(), 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            st.markdown(f"{medal} **{row['coin'].upper()}**: {row['sharpe_ratio']:.2f}")
    
    with col2:
        st.markdown("**🏆 Sortino Ratio Cao Nhất**")
        top_sortino = metrics_df.nlargest(3, 'sortino_ratio')
        for i, (_, row) in enumerate(top_sortino.iterrows(), 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            st.markdown(f"{medal} **{row['coin'].upper()}**: {row['sortino_ratio']:.2f}")
    
    with col3:
        st.markdown("**🏆 Max Drawdown Thấp Nhất**")
        top_dd = metrics_df.nsmallest(3, 'max_drawdown')
        for i, (_, row) in enumerate(top_dd.iterrows(), 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            st.markdown(f"{medal} **{row['coin'].upper()}**: {row['max_drawdown']:.2f}%")
