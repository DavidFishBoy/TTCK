"""Sentiment Analysis Page - Fear & Greed Index + News Sentiment."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
from scipy import stats
import asyncio
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.data_collection.sentiment_collector import (
    SentimentCollector,
    get_sentiment_data,
    merge_sentiment_with_price
)
from src.data_collection.news_collector import (
    NewsCollector,
    get_news_sentiment_data
)
from src.assistant.chart_analyzer import get_chart_analyzer


def load_coin_data(coin: str = "bitcoin") -> pd.DataFrame:
    """Load price data for a specific coin."""
    data_dir = Path(__file__).parent.parent.parent.parent / "data" / "raw" / "train"
    
    files = list(data_dir.glob(f"{coin}_binance_*.csv"))
    if not files:
        return pd.DataFrame()
    
    latest_file = sorted(files)[-1]
    df = pd.read_csv(latest_file)
    
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    elif "timestamp" in df.columns:
        sample_ts = df["timestamp"].iloc[0]
        if isinstance(sample_ts, str) and "-" in sample_ts:
            df["date"] = pd.to_datetime(df["timestamp"])
        else:
            df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    
    if "close" in df.columns and "return" not in df.columns:
        df["return"] = df["close"].pct_change() * 100
        df["log_return"] = np.log(df["close"] / df["close"].shift(1)) * 100
    
    return df


# ============ Fear & Greed Analysis Functions ============

def calculate_lag_correlations(df: pd.DataFrame, sentiment_col: str = "fng_value", lag_periods: list = [0, 1, 3, 7, 14]) -> pd.DataFrame:
    """Calculate correlations between lagged sentiment and returns."""
    if sentiment_col not in df.columns or "return" not in df.columns:
        return pd.DataFrame()
    
    correlations = []
    for lag in lag_periods:
        lagged = df[sentiment_col].shift(lag)
        valid_mask = ~(lagged.isna() | df["return"].isna())
        
        if valid_mask.sum() > 10:
            corr, pvalue = stats.pearsonr(lagged[valid_mask], df["return"][valid_mask])
            correlations.append({
                "Lag (Days)": lag,
                "Correlation": corr,
                "P-Value": pvalue,
                "Significant": pvalue < 0.05,
                "N": valid_mask.sum()
            })
    
    return pd.DataFrame(correlations)


def perform_event_study(df: pd.DataFrame, forward_periods: list = [1, 3, 7, 14], fear_threshold: int = 25, greed_threshold: int = 75) -> dict:
    """Perform event study for extreme sentiment events."""
    if "fng_value" not in df.columns or "return" not in df.columns:
        return {}
    
    df = df.copy()
    for period in forward_periods:
        df[f"return_+{period}d"] = df["return"].shift(-period).rolling(period).sum()
    
    fear_events = df[df["fng_value"] <= fear_threshold].copy()
    greed_events = df[df["fng_value"] >= greed_threshold].copy()
    
    fear_stats = {}
    for period in forward_periods:
        col = f"return_+{period}d"
        if col in fear_events.columns:
            valid = fear_events[col].dropna()
            if len(valid) > 0:
                fear_stats[f"+{period}d"] = {
                    "Median Return (%)": valid.median(),
                    "Mean Return (%)": valid.mean(),
                    "Hit Rate (%)": (valid > 0).mean() * 100,
                    "Count": len(valid)
                }
    
    greed_stats = {}
    for period in forward_periods:
        col = f"return_+{period}d"
        if col in greed_events.columns:
            valid = greed_events[col].dropna()
            if len(valid) > 0:
                greed_stats[f"+{period}d"] = {
                    "Median Return (%)": valid.median(),
                    "Mean Return (%)": valid.mean(),
                    "Hit Rate (%)": (valid > 0).mean() * 100,
                    "Count": len(valid)
                }
    
    return {
        "extreme_fear": fear_stats,
        "extreme_greed": greed_stats,
        "fear_count": len(fear_events),
        "greed_count": len(greed_events)
    }


def create_fng_timeline_chart(df: pd.DataFrame) -> go.Figure:
    """Create Fear & Greed timeline chart with color zones."""
    fig = go.Figure()
    
    fig.add_hrect(y0=0, y1=25, fillcolor="rgba(255, 0, 0, 0.1)", line_width=0)
    fig.add_hrect(y0=25, y1=45, fillcolor="rgba(255, 165, 0, 0.1)", line_width=0)
    fig.add_hrect(y0=45, y1=55, fillcolor="rgba(128, 128, 128, 0.1)", line_width=0)
    fig.add_hrect(y0=55, y1=75, fillcolor="rgba(144, 238, 144, 0.1)", line_width=0)
    fig.add_hrect(y0=75, y1=100, fillcolor="rgba(0, 128, 0, 0.1)", line_width=0)
    
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["fng_value"],
        mode="lines", name="Fear & Greed Index",
        line=dict(width=2, color="#667eea")
    ))
    
    extreme_fear = df[df["fng_value"] <= 25]
    extreme_greed = df[df["fng_value"] >= 75]
    
    fig.add_trace(go.Scatter(
        x=extreme_fear["date"], y=extreme_fear["fng_value"],
        mode="markers", name="Extreme Fear",
        marker=dict(color="red", size=8, symbol="triangle-down")
    ))
    fig.add_trace(go.Scatter(
        x=extreme_greed["date"], y=extreme_greed["fng_value"],
        mode="markers", name="Extreme Greed",
        marker=dict(color="green", size=8, symbol="triangle-up")
    ))
    
    fig.update_layout(
        title="📊 Fear & Greed Index Over Time",
        xaxis_title="Date", yaxis_title="Fear & Greed Value",
        yaxis=dict(range=[0, 100], dtick=25),
        template="plotly_dark", height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    return fig


def create_sentiment_return_overlay(df: pd.DataFrame, sentiment_col: str, coin: str) -> go.Figure:
    """Create dual-axis chart for sentiment vs return."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=df["date"], y=df[sentiment_col], name="Sentiment",
                   line=dict(color="#667eea", width=2)),
        secondary_y=False
    )
    
    colors = ["#00d4aa" if r >= 0 else "#ff6b6b" for r in df["return"].fillna(0)]
    fig.add_trace(
        go.Bar(x=df["date"], y=df["return"], name=f"{coin} Return (%)",
               marker_color=colors, opacity=0.6),
        secondary_y=True
    )
    
    fig.update_layout(
        title=f"📈 Sentiment vs {coin} Daily Return",
        template="plotly_dark", height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    fig.update_yaxes(title_text="Sentiment", secondary_y=False)
    fig.update_yaxes(title_text="Return (%)", secondary_y=True)
    return fig


def create_lag_correlation_chart(corr_df: pd.DataFrame) -> go.Figure:
    """Create bar chart for lag correlations."""
    if corr_df.empty:
        return go.Figure()
    
    colors = ["#00d4aa" if c >= 0 else "#ff6b6b" for c in corr_df["Correlation"]]
    
    fig = go.Figure(go.Bar(
        x=corr_df["Lag (Days)"].astype(str) + " days",
        y=corr_df["Correlation"],
        marker_color=colors,
        text=[f"{c:.3f}" for c in corr_df["Correlation"]],
        textposition="outside"
    ))
    
    for i, row in corr_df.iterrows():
        if row["Significant"]:
            fig.add_annotation(
                x=f"{row['Lag (Days)']} days",
                y=row["Correlation"] + (0.02 if row["Correlation"] >= 0 else -0.02),
                text="*", showarrow=False, font=dict(size=20, color="gold")
            )
    
    fig.update_layout(
        title="🔗 Lag Correlation: Sentiment(t-k) vs Return(t)",
        xaxis_title="Lag Period", yaxis_title="Pearson Correlation",
        template="plotly_dark", height=350, yaxis=dict(range=[-0.3, 0.3])
    )
    return fig


# ============ NewsAPI Analysis Functions ============

def create_news_timeline_chart(df: pd.DataFrame) -> go.Figure:
    """Create news sentiment timeline chart."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Sentiment line
    fig.add_trace(
        go.Scatter(
            x=df["date"], y=df["news_sentiment_mean"],
            mode="lines+markers", name="Sentiment (Mean)",
            line=dict(color="#667eea", width=2),
            marker=dict(size=6)
        ),
        secondary_y=False
    )
    
    # News count bars
    fig.add_trace(
        go.Bar(
            x=df["date"], y=df["news_count"],
            name="Số lượng tin", marker_color="#00d4aa", opacity=0.5
        ),
        secondary_y=True
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", secondary_y=False)
    
    fig.update_layout(
        title="📰 News Sentiment Timeline (7 ngày gần nhất)",
        template="plotly_dark", height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    fig.update_yaxes(title_text="Sentiment Score", secondary_y=False, range=[-1, 1])
    fig.update_yaxes(title_text="Số tin", secondary_y=True)
    
    return fig


def render_news_headlines(articles_df: pd.DataFrame, limit: int = 10):
    """Render top news headlines with sentiment."""
    if articles_df.empty:
        st.info("Không có tin tức")
        return
    
    st.markdown("#### 📰 Tin Tức Gần Đây")
    
    for i, row in articles_df.head(limit).iterrows():
        sentiment = row["sentiment_score"]
        label = row["sentiment_label"]
        
        # Color based on sentiment
        if label == "positive":
            color = "#00d4aa"
            icon = "🟢"
        elif label == "negative":
            color = "#ff6b6b"
            icon = "🔴"
        else:
            color = "#888"
            icon = "⚪"
        
        with st.container():
            st.markdown(f"""
                <div style='background: rgba(30,30,40,0.5); padding: 0.8rem; border-radius: 8px; 
                            margin-bottom: 0.5rem; border-left: 3px solid {color};'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <span style='font-weight: 600; color: #eee;'>{icon} {row['title'][:80]}...</span>
                        <span style='color: {color}; font-weight: bold;'>{sentiment:.2f}</span>
                    </div>
                    <div style='color: #888; font-size: 0.8rem; margin-top: 0.3rem;'>
                        {row['source']} • {row['date'].strftime('%Y-%m-%d')}
                    </div>
                </div>
            """, unsafe_allow_html=True)


# ============ Main Render Function ============

def render_sentiment_analysis_page():
    """Render the Social Sentiment Analysis page."""
    st.title("📊 Phân Tích Tâm Lý Thị Trường")
    
    # Introduction
    st.markdown("""
        <div style='background: rgba(102, 126, 234, 0.1); padding: 1.5rem; border-radius: 12px; 
                    border-left: 4px solid #667eea; margin-bottom: 2rem;'>
            <h3 style='margin: 0 0 0.5rem 0; color: #667eea;'>📌 Giới Thiệu</h3>
            <p style='margin: 0; color: #ccc; line-height: 1.6;'>
                Phân tích tâm lý thị trường từ nhiều nguồn dữ liệu để hỗ trợ quyết định đầu tư.
                Chọn nguồn dữ liệu bên dưới để xem phân tích chi tiết.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # ============ Source Selector ============
    st.subheader("⚙️ Chọn Nguồn Dữ Liệu")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        data_source = st.selectbox(
            "Nguồn Sentiment:",
            ["🎯 Alternative.me (Fear & Greed Index)", "📰 NewsAPI (Tin tức Crypto)"],
            help="Chọn nguồn dữ liệu để phân tích"
        )
    
    with col2:
        selected_coin = st.selectbox(
            "Chọn coin để so sánh:",
            ["bitcoin", "ethereum", "solana", "binancecoin", "cardano",
             "litecoin", "pancakeswap", "axieinfinity", "thesandbox"],
            format_func=lambda x: x.upper()
        )
    
    with col3:
        refresh_data = st.button("🔄 Cập nhật", width='stretch')
    
    st.markdown("---")
    
    # Load price data
    price_df = load_coin_data(selected_coin)
    
    # ============ Alternative.me (Fear & Greed) ============
    if "Alternative.me" in data_source:
        render_fear_greed_analysis(price_df, selected_coin, refresh_data)
    
    # ============ NewsAPI ============
    else:
        render_news_sentiment_analysis(price_df, selected_coin, refresh_data)
    
    # Disclaimer
    st.markdown("---")
    st.warning("""
        ⚠️ **Lưu ý quan trọng**: 
        - Sentiment là **tín hiệu rủi ro**, không phải công cụ dự đoán giá.
        - Kết hợp với các phân tích khác để đưa ra quyết định.
    """)


def render_fear_greed_analysis(price_df: pd.DataFrame, selected_coin: str, refresh: bool):
    """Render Fear & Greed Index analysis section."""
    
    # Load data
    with st.spinner("Đang tải dữ liệu Fear & Greed..."):
        try:
            sentiment_df = get_sentiment_data(refresh=refresh)
        except Exception as e:
            st.error(f"Không thể tải dữ liệu: {e}")
            try:
                collector = SentimentCollector()
                sentiment_df = asyncio.run(collector.collect_and_save())
            except Exception as e2:
                st.error(f"Lỗi thu thập: {e2}")
                return
    
    if sentiment_df.empty:
        st.warning("Không có dữ liệu. Nhấn 'Cập nhật' để thu thập.")
        return
    
    # Merge with price
    if not price_df.empty:
        merged_df = merge_sentiment_with_price(price_df, sentiment_df)
    else:
        merged_df = sentiment_df.copy()
        merged_df["return"] = np.nan
    
    # Overview metrics
    st.subheader("📈 Tổng Quan Fear & Greed")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    latest = sentiment_df.iloc[-1] if not sentiment_df.empty else {}
    
    with col1:
        st.metric("Giá trị hiện tại", f"{latest.get('fng_value', 'N/A')}")
    with col2:
        st.metric("Trạng thái", latest.get("fng_label", "N/A"))
    with col3:
        st.metric("TB 30 ngày", f"{sentiment_df['fng_value'].tail(30).mean():.1f}")
    with col4:
        st.metric("Ngày Extreme Fear", (sentiment_df["fng_value"] <= 25).sum())
    with col5:
        st.metric("Ngày Extreme Greed", (sentiment_df["fng_value"] >= 75).sum())
    
    st.markdown("---")
    
    # Timeline chart
    st.subheader("📊 Biểu đồ Fear & Greed Index")
    st.markdown("""
        <div style='background: rgba(102,126,234,0.1); padding: 1rem; border-radius: 8px; border-left: 3px solid #667eea; margin-bottom: 1rem;'>
            <b>Cách đọc:</b> 0-25 (Extreme Fear), 26-49 (Fear), 50-59 (Neutral), 60-74 (Greed), 75-100 (Extreme Greed)
        </div>
    """, unsafe_allow_html=True)
    
    fig_timeline = create_fng_timeline_chart(sentiment_df)
    st.plotly_chart(fig_timeline, width='stretch')
    
    # AI Analysis Button for Fear & Greed Chart
    chart_analyzer = get_chart_analyzer()
    if st.button("🤖 AI Phân Tích Fear & Greed", key="analyze_fng"):
        with st.spinner("🔄 Đang phân tích với GPT-4..."):
            latest = sentiment_df.iloc[-1] if not sentiment_df.empty else {}
            fng_7d_avg = sentiment_df['fng_value'].tail(7).mean()
            fng_30d_avg = sentiment_df['fng_value'].tail(30).mean()
            
            # Determine trend
            if fng_7d_avg > fng_30d_avg:
                trend = "TĂNG (chuyển từ sợ hãi sang tham lam)"
            else:
                trend = "GIẢM (chuyển từ tham lam sang sợ hãi)"
            
            # Calculate correlation if available
            if 'return' in merged_df.columns and not merged_df['return'].isna().all():
                corr = merged_df['fng_value'].corr(merged_df['return'])
            else:
                corr = 0
            
            chart_data = {
                "current_fng": latest.get('fng_value', 0),
                "fng_classification": latest.get('fng_label', 'Unknown'),
                "fng_7d_avg": fng_7d_avg,
                "fng_30d_avg": fng_30d_avg,
                "sentiment_trend": trend,
                "fng_return_correlation": corr
            }
            
            analysis = chart_analyzer.analyze_chart(
                coin=selected_coin,
                chart_type="sentiment_fng",
                chart_data=chart_data,
                chart_title="Fear & Greed Index"
            )
            st.markdown(analysis)
    
    # Overlay with return
    st.markdown("---")
    st.subheader(f"📈 Sentiment vs {selected_coin.upper()} Return")
    
    if "return" in merged_df.columns and not merged_df["return"].isna().all():
        fig_overlay = create_sentiment_return_overlay(merged_df, "fng_value", selected_coin.upper())
        st.plotly_chart(fig_overlay, width='stretch')
    else:
        st.warning(f"Không có dữ liệu giá cho {selected_coin.upper()}")
    
    # Lag correlation
    st.markdown("---")
    st.subheader("🔗 Phân Tích Tương Quan Theo Lag")
    
    if "return" in merged_df.columns and not merged_df["return"].isna().all():
        corr_df = calculate_lag_correlations(merged_df, "fng_value")
        if not corr_df.empty:
            fig_corr = create_lag_correlation_chart(corr_df)
            st.plotly_chart(fig_corr, width='stretch')
            
            with st.expander("📋 Bảng chi tiết"):
                st.dataframe(corr_df.style.format({"Correlation": "{:.4f}", "P-Value": "{:.4f}"}), width='stretch')
    
    # Event study
    st.markdown("---")
    st.subheader("📊 Event Study: Return sau Extreme Sentiment")
    
    if "return" in merged_df.columns and not merged_df["return"].isna().all():
        event_results = perform_event_study(merged_df)
        
        if event_results:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔴 Extreme Fear (≤ 25)")
                st.metric("Số sự kiện", event_results.get("fear_count", 0))
                fear_stats = event_results.get("extreme_fear", {})
                if fear_stats:
                    st.dataframe(pd.DataFrame(fear_stats).T.style.format({
                        "Median Return (%)": "{:.2f}", "Mean Return (%)": "{:.2f}", "Hit Rate (%)": "{:.1f}"
                    }), width='stretch')
            
            with col2:
                st.markdown("#### 🟢 Extreme Greed (≥ 75)")
                st.metric("Số sự kiện", event_results.get("greed_count", 0))
                greed_stats = event_results.get("extreme_greed", {})
                if greed_stats:
                    st.dataframe(pd.DataFrame(greed_stats).T.style.format({
                        "Median Return (%)": "{:.2f}", "Mean Return (%)": "{:.2f}", "Hit Rate (%)": "{:.1f}"
                    }), width='stretch')
    
    # Distribution
    st.markdown("---")
    st.subheader("📊 Phân Bố Fear & Greed Index")
    
    col1, col2 = st.columns(2)
    with col1:
        fig_hist = px.histogram(sentiment_df, x="fng_value", nbins=20, color_discrete_sequence=["#667eea"])
        fig_hist.update_layout(template="plotly_dark", height=300, title="Phân phối giá trị")
        st.plotly_chart(fig_hist, width='stretch')
    
    with col2:
        label_counts = sentiment_df["fng_label"].value_counts()
        fig_pie = px.pie(values=label_counts.values, names=label_counts.index, color_discrete_sequence=px.colors.sequential.RdBu)
        fig_pie.update_layout(template="plotly_dark", height=300, title="Tỷ lệ trạng thái")
        st.plotly_chart(fig_pie, width='stretch')


def render_news_sentiment_analysis(price_df: pd.DataFrame, selected_coin: str, refresh: bool):
    """Render News Sentiment analysis section - Professional version."""
    
    # Load data
    with st.spinner("Đang tải dữ liệu tin tức..."):
        try:
            news_data = get_news_sentiment_data(refresh=refresh)
            articles_df = news_data.get("articles", pd.DataFrame())
            daily_df = news_data.get("daily", pd.DataFrame())
        except Exception as e:
            st.error(f"Không thể tải dữ liệu: {e}")
            try:
                collector = NewsCollector()
                news_data = asyncio.run(collector.collect_and_save())
                articles_df = news_data.get("articles", pd.DataFrame())
                daily_df = news_data.get("daily", pd.DataFrame())
            except Exception as e2:
                st.error(f"Lỗi thu thập: {e2}")
                return
    
    if articles_df.empty:
        st.warning("Không có dữ liệu tin tức. Nhấn 'Cập nhật' để thu thập.")
        st.info("💡 **Lưu ý**: NewsAPI free tier giới hạn 100 requests/ngày và chỉ lấy tin trong 30 ngày gần nhất.")
        return
    
    # ============ Overview Metrics with Insights ============
    st.subheader("📰 Tổng Quan News Sentiment")
    
    st.markdown("""
        <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 8px; 
                    border-left: 3px solid #667eea; margin-bottom: 1rem;'>
            <b>📖 Về News Sentiment:</b> Phân tích cảm xúc từ tiêu đề và mô tả tin tức crypto. 
            Score từ -1 (rất tiêu cực) đến +1 (rất tích cực). Sentiment gần 0 = trung lập.
        </div>
    """, unsafe_allow_html=True)
    
    # Calculate metrics
    avg_sentiment = articles_df["sentiment_score"].mean()
    median_sentiment = articles_df["sentiment_score"].median()
    positive_count = (articles_df["sentiment_label"] == "positive").sum()
    negative_count = (articles_df["sentiment_label"] == "negative").sum()
    neutral_count = (articles_df["sentiment_label"] == "neutral").sum()
    total_count = len(articles_df)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        delta_color = "normal" if avg_sentiment >= 0 else "inverse"
        st.metric("Sentiment TB", f"{avg_sentiment:.3f}", 
                  delta=f"{'Tích cực' if avg_sentiment > 0.05 else 'Tiêu cực' if avg_sentiment < -0.05 else 'Trung lập'}")
    
    with col2:
        st.metric("Tổng số tin", f"{total_count}")
    
    with col3:
        st.metric("🟢 Tích cực", f"{positive_count} ({positive_count/total_count*100:.0f}%)")
    
    with col4:
        st.metric("🔴 Tiêu cực", f"{negative_count} ({negative_count/total_count*100:.0f}%)")
    
    with col5:
        st.metric("⚪ Trung lập", f"{neutral_count} ({neutral_count/total_count*100:.0f}%)")
    
    # Auto insight
    if avg_sentiment > 0.1:
        st.success("💡 **Insight**: Tin tức đang nghiêng về **tích cực** - thị trường có thể đang trong giai đoạn lạc quan.")
    elif avg_sentiment < -0.1:
        st.error("⚠️ **Insight**: Tin tức đang nghiêng về **tiêu cực** - có thể có sự kiện xấu hoặc FUD đang lan rộng.")
    else:
        st.info("📊 **Insight**: Tin tức đang ở trạng thái **trung lập** - thị trường chưa có xu hướng rõ ràng.")
    
    st.markdown("---")
    
    # ============ Timeline Chart with Analysis ============
    st.subheader("📈 Biểu Đồ Sentiment Theo Thời Gian")
    
    st.markdown("""
        <div style='background: rgba(0, 212, 170, 0.1); padding: 1rem; border-radius: 8px; 
                    border-left: 3px solid #00d4aa; margin-bottom: 1rem;'>
            <b>📖 Cách đọc:</b><br>
            • <b>Đường xanh (Sentiment)</b>: Giá trị trung bình sentiment mỗi ngày (-1 đến +1)<br>
            • <b>Cột xanh (Volume)</b>: Số lượng tin tức được thu thập mỗi ngày<br>
            • <b>Đường ngang (y=0)</b>: Ngưỡng trung lập - trên = tích cực, dưới = tiêu cực
        </div>
    """, unsafe_allow_html=True)
    
    if not daily_df.empty:
        # Enhanced timeline chart
        fig_timeline = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add colored area for sentiment
        colors = ["#00d4aa" if s >= 0 else "#ff6b6b" for s in daily_df["news_sentiment_mean"]]
        
        fig_timeline.add_trace(
            go.Scatter(
                x=daily_df["date"], y=daily_df["news_sentiment_mean"],
                mode="lines+markers", name="Sentiment (Mean)",
                line=dict(color="#667eea", width=3),
                marker=dict(size=10, color=colors, line=dict(width=2, color="white")),
                fill="tozeroy",
                fillcolor="rgba(102, 126, 234, 0.2)"
            ),
            secondary_y=False
        )
        
        # News count bars
        fig_timeline.add_trace(
            go.Bar(
                x=daily_df["date"], y=daily_df["news_count"],
                name="Số lượng tin", 
                marker=dict(color="#764ba2", opacity=0.5),
                width=60000000  # Adjusted bar width
            ),
            secondary_y=True
        )
        
        # Add zero line
        fig_timeline.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.5)", secondary_y=False)
        
        # Add positive/negative zones
        fig_timeline.add_hrect(y0=0, y1=1, fillcolor="rgba(0,212,170,0.05)", 
                              line_width=0, secondary_y=False)
        fig_timeline.add_hrect(y0=-1, y1=0, fillcolor="rgba(255,107,107,0.05)", 
                              line_width=0, secondary_y=False)
        
        fig_timeline.update_layout(
            title=dict(text="📰 News Sentiment Timeline", font=dict(size=18)),
            template="plotly_dark", height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            hovermode="x unified"
        )
        fig_timeline.update_yaxes(title_text="Sentiment Score", secondary_y=False, 
                                   range=[-1, 1], dtick=0.25, gridcolor="rgba(255,255,255,0.1)")
        fig_timeline.update_yaxes(title_text="Số tin", secondary_y=True)
        fig_timeline.update_xaxes(gridcolor="rgba(255,255,255,0.1)")
        
        st.plotly_chart(fig_timeline, width='stretch')
        
        # Daily analysis
        if len(daily_df) >= 2:
            latest_day = daily_df.iloc[-1]
            prev_day = daily_df.iloc[-2]
            change = latest_day["news_sentiment_mean"] - prev_day["news_sentiment_mean"]
            
            st.markdown(f"""
                <div style='background: rgba(30,30,40,0.5); padding: 1rem; border-radius: 8px; margin-top: 0.5rem;'>
                    <b>📊 Phân tích ngày gần nhất ({latest_day['date'].strftime('%Y-%m-%d')}):</b><br>
                    • Sentiment: <b style='color: {"#00d4aa" if latest_day["news_sentiment_mean"] > 0 else "#ff6b6b"}'>{latest_day["news_sentiment_mean"]:.3f}</b><br>
                    • Thay đổi so với hôm trước: <b style='color: {"#00d4aa" if change > 0 else "#ff6b6b"}'>{change:+.3f}</b><br>
                    • Số tin: <b>{int(latest_day["news_count"])}</b> bài
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ============ Headlines with Better Design ============
    st.subheader("📰 Tin Tức Gần Đây")
    
    st.markdown("""
        <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 8px; 
                    border-left: 3px solid #667eea; margin-bottom: 1rem;'>
            <b>📖 Về Sentiment Score:</b> 
            🟢 Score > 0.05 = Tích cực | 
            🔴 Score < -0.05 = Tiêu cực | 
            ⚪ Còn lại = Trung lập
        </div>
    """, unsafe_allow_html=True)
    
    # Enhanced headlines display
    for i, row in articles_df.head(8).iterrows():
        sentiment = row["sentiment_score"]
        label = row["sentiment_label"]
        
        if label == "positive":
            color, bg_color, icon = "#00d4aa", "rgba(0,212,170,0.1)", "🟢"
        elif label == "negative":
            color, bg_color, icon = "#ff6b6b", "rgba(255,107,107,0.1)", "🔴"
        else:
            color, bg_color, icon = "#888", "rgba(136,136,136,0.1)", "⚪"
        
        title = row['title'][:100] + "..." if len(row['title']) > 100 else row['title']
        
        st.markdown(f"""
            <div style='background: {bg_color}; padding: 1rem; border-radius: 10px; 
                        margin-bottom: 0.7rem; border-left: 4px solid {color};'>
                <div style='display: flex; justify-content: space-between; align-items: flex-start;'>
                    <div style='flex: 1;'>
                        <span style='font-weight: 600; color: #eee; font-size: 0.95rem;'>{icon} {title}</span>
                        <div style='color: #999; font-size: 0.8rem; margin-top: 0.4rem;'>
                            📰 {row['source']} • 📅 {row['date'].strftime('%Y-%m-%d %H:%M') if hasattr(row['date'], 'strftime') else row['date']}
                        </div>
                    </div>
                    <div style='text-align: right; min-width: 80px;'>
                        <span style='color: {color}; font-weight: bold; font-size: 1.1rem;'>{sentiment:.2f}</span>
                        <div style='color: {color}; font-size: 0.75rem; text-transform: uppercase;'>{label}</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ============ Distribution Charts with Analysis ============
    st.subheader("📊 Phân Bố Sentiment")
    
    st.markdown("""
        <div style='background: rgba(118, 75, 162, 0.1); padding: 1rem; border-radius: 8px; 
                    border-left: 3px solid #764ba2; margin-bottom: 1rem;'>
            <b>📖 Phân tích phân bố:</b> Biểu đồ histogram cho thấy sentiment của tin tức tập trung ở đâu.
            Nếu phần lớn nằm bên phải (>0) = thị trường lạc quan, bên trái (<0) = bi quan.
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Enhanced histogram
        fig_hist = go.Figure()
        
        # Separate positive and negative for coloring
        pos_scores = articles_df[articles_df["sentiment_score"] >= 0]["sentiment_score"]
        neg_scores = articles_df[articles_df["sentiment_score"] < 0]["sentiment_score"]
        
        fig_hist.add_trace(go.Histogram(x=pos_scores, nbinsx=15, name="Positive", 
                                        marker_color="#00d4aa", opacity=0.7))
        fig_hist.add_trace(go.Histogram(x=neg_scores, nbinsx=15, name="Negative", 
                                        marker_color="#ff6b6b", opacity=0.7))
        
        fig_hist.add_vline(x=0, line_dash="dash", line_color="white", line_width=2)
        fig_hist.add_vline(x=avg_sentiment, line_dash="dot", line_color="#667eea", line_width=2,
                          annotation_text=f"Mean: {avg_sentiment:.2f}", annotation_position="top")
        
        fig_hist.update_layout(
            template="plotly_dark", height=350,
            title=dict(text="Phân phối Sentiment Score", font=dict(size=14)),
            xaxis_title="Sentiment Score", yaxis_title="Số tin",
            barmode="overlay", showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig_hist, width='stretch')
    
    with col2:
        # Enhanced pie chart
        label_counts = articles_df["sentiment_label"].value_counts()
        colors_map = {"positive": "#00d4aa", "neutral": "#667eea", "negative": "#ff6b6b"}
        colors_list = [colors_map.get(l, "#888") for l in label_counts.index]
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=label_counts.index, values=label_counts.values,
            hole=0.4, marker=dict(colors=colors_list, line=dict(color='#1a1a2e', width=2)),
            textinfo="percent+label", textfont=dict(size=12),
            hovertemplate="<b>%{label}</b><br>Số tin: %{value}<br>Tỷ lệ: %{percent}<extra></extra>"
        )])
        
        fig_pie.update_layout(
            template="plotly_dark", height=350,
            title=dict(text="Tỷ lệ Sentiment", font=dict(size=14)),
            showlegend=False,
            annotations=[dict(text=f"{total_count}<br>tin", x=0.5, y=0.5, font_size=16, showarrow=False)]
        )
        st.plotly_chart(fig_pie, width='stretch')
    
    # Distribution insight
    dominant = label_counts.idxmax()
    dominant_pct = label_counts.max() / total_count * 100
    
    if dominant == "positive" and dominant_pct > 50:
        st.success(f"📈 **Phân tích**: {dominant_pct:.0f}% tin tức có sentiment tích cực - narrative thị trường đang lạc quan.")
    elif dominant == "negative" and dominant_pct > 50:
        st.error(f"📉 **Phân tích**: {dominant_pct:.0f}% tin tức có sentiment tiêu cực - có thể có FUD hoặc tin xấu.")
    else:
        st.info(f"📊 **Phân tích**: Tin tức phân bố khá đều - sentiment {dominant} chiếm {dominant_pct:.0f}%.")
    
    st.markdown("---")
    
    # ============ Source Analysis ============
    st.subheader("📰 Phân Tích Theo Nguồn Tin")
    
    st.markdown("""
        <div style='background: rgba(0, 212, 170, 0.1); padding: 1rem; border-radius: 8px; 
                    border-left: 3px solid #00d4aa; margin-bottom: 1rem;'>
            <b>📖 Ý nghĩa:</b> So sánh sentiment trung bình từ các nguồn tin khác nhau.
            Nguồn có sentiment cao = đưa tin tích cực, thấp = đưa tin tiêu cực.
        </div>
    """, unsafe_allow_html=True)
    
    source_stats = articles_df.groupby("source").agg({
        "sentiment_score": ["mean", "count"]
    }).reset_index()
    source_stats.columns = ["Source", "Avg Sentiment", "Count"]
    source_stats = source_stats.sort_values("Count", ascending=False).head(10)
    
    # Enhanced bar chart
    colors = ["#00d4aa" if s >= 0 else "#ff6b6b" for s in source_stats["Avg Sentiment"]]
    
    fig_source = go.Figure()
    fig_source.add_trace(go.Bar(
        x=source_stats["Source"], y=source_stats["Avg Sentiment"],
        marker=dict(color=colors, line=dict(width=1, color="white")),
        text=[f"{s:.2f}" for s in source_stats["Avg Sentiment"]],
        textposition="outside", textfont=dict(size=11)
    ))
    
    fig_source.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.5)")
    
    fig_source.update_layout(
        template="plotly_dark", height=400,
        title=dict(text="Sentiment Trung Bình Theo Nguồn (Top 10)", font=dict(size=16)),
        xaxis_title="Nguồn tin", yaxis_title="Avg Sentiment",
        yaxis=dict(range=[-0.5, 0.5], dtick=0.1),
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig_source, width='stretch')
    
    # Source insight
    most_positive = source_stats.loc[source_stats["Avg Sentiment"].idxmax()]
    most_negative = source_stats.loc[source_stats["Avg Sentiment"].idxmin()]
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
            <div style='background: rgba(0,212,170,0.2); padding: 1rem; border-radius: 8px;'>
                <b>🟢 Nguồn tích cực nhất:</b><br>
                <span style='color: #00d4aa; font-size: 1.1rem;'>{most_positive['Source']}</span><br>
                Sentiment: <b>{most_positive['Avg Sentiment']:.3f}</b> ({int(most_positive['Count'])} tin)
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div style='background: rgba(255,107,107,0.2); padding: 1rem; border-radius: 8px;'>
                <b>🔴 Nguồn tiêu cực nhất:</b><br>
                <span style='color: #ff6b6b; font-size: 1.1rem;'>{most_negative['Source']}</span><br>
                Sentiment: <b>{most_negative['Avg Sentiment']:.3f}</b> ({int(most_negative['Count'])} tin)
            </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    render_sentiment_analysis_page()