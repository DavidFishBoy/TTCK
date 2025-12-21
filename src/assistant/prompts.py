# src/assistant/prompts.py

"""
Prompt templates cho ChartAnalyzer - Phân tích biểu đồ bằng GPT-4.
Mỗi prompt được thiết kế riêng cho từng loại biểu đồ trong dashboard crypto.
"""

# ============================================================================
# SYSTEM PROMPT - Prompt chung cho tất cả các phân tích
# ============================================================================

SYSTEM_PROMPT = """Bạn là chuyên gia phân tích đầu tư cryptocurrency với hơn 10 năm kinh nghiệm.

## VAI TRÒ CỦA BẠN:
- Phân tích kỹ thuật và định lượng các biểu đồ crypto
- Giải thích các chỉ số tài chính một cách dễ hiểu
- Đưa ra khuyến nghị đầu tư dựa trên dữ liệu

## NGUYÊN TẮC PHÂN TÍCH:
1. **Dựa trên dữ liệu**: Chỉ phân tích dựa trên số liệu được cung cấp
2. **Khách quan**: Trình bày cả mặt tích cực và rủi ro
3. **Thực tế**: Crypto là thị trường biến động cao, luôn nhấn mạnh quản lý rủi ro

## ĐỊNH DẠNG OUTPUT:
- Sử dụng emoji phù hợp để tăng tính trực quan
- Sử dụng markdown (bold, bullet points)
- Giữ ngắn gọn, súc tích (tối đa 300 từ)
- Kết thúc bằng 1-2 khuyến nghị hành động cụ thể

## CẢNH BÁO BẮT BUỘC:
Luôn kết thúc bằng: "⚠️ *Đây chỉ là phân tích tham khảo, không phải lời khuyên đầu tư.*"
"""

# ============================================================================
# CHART-SPECIFIC PROMPTS
# ============================================================================

CHART_PROMPTS = {

    # -------------------------------------------------------------------------
    # EDA: Volatility & Risk Analysis
    # -------------------------------------------------------------------------
    
    "rolling_volatility": """## PHÂN TÍCH BIỂU ĐỒ BIẾN ĐỘNG

**Coin:** {coin}
**Tiêu đề:** {chart_title}

### DỮ LIỆU TỪ BIỂU ĐỒ:
- Biến động 14 ngày hiện tại: {vol_14d_latest:.2f}%
- Biến động 30 ngày hiện tại: {vol_30d_latest:.2f}%
- Biến động 14 ngày trung bình: {vol_14d_avg:.2f}%
- Biến động 30 ngày trung bình: {vol_30d_avg:.2f}%
- Xu hướng biến động (so với 30 ngày trước): {volatility_trend}

### YÊU CẦU PHÂN TÍCH:
1. **Mức độ biến động hiện tại**: So sánh với trung bình, cao hay thấp?
2. **Xu hướng**: Biến động đang tăng hay giảm? Điều này có nghĩa gì?
3. **Đánh giá rủi ro**: Mức độ rủi ro cho nhà đầu tư?
4. **Khuyến nghị**: 
   - Nếu biến động cao: Nên làm gì để bảo vệ vốn?
   - Nếu biến động thấp: Cơ hội hay cần cẩn trọng?
""",

    "drawdown": """## PHÂN TÍCH BIỂU ĐỒ DRAWDOWN (SỤT GIẢM TỪ ĐỈNH)

**Coin:** {coin}
**Tiêu đề:** {chart_title}

### DỮ LIỆU TỪ BIỂU ĐỒ:
- Max Drawdown (Mức giảm sâu nhất): {max_drawdown:.2f}%
- Thời gian phục hồi dài nhất: {max_dd_duration} ngày
- Drawdown hiện tại: {current_drawdown:.2f}%
- Số lần drawdown > 20%: {dd_count_20}

### YÊU CẦU PHÂN TÍCH:
1. **Mức độ nghiêm trọng**: Max Drawdown có đáng lo ngại không?
2. **Khả năng phục hồi**: Thời gian phục hồi cho thấy điều gì về sức khỏe coin?
3. **Tình trạng hiện tại**: Coin đang trong trạng thái drawdown hay đã recover?
4. **Quản lý vốn**: 
   - Tỷ lệ vốn tối đa nên đầu tư vào coin này?
   - Mức stop-loss đề xuất?
""",

    "returns_distribution": """## PHÂN TÍCH PHÂN PHỐI LỢI NHUẬN & VaR

**Coin:** {coin}
**Tiêu đề:** {chart_title}

### DỮ LIỆU TỪ BIỂU ĐỒ:
- Lợi nhuận trung bình hàng ngày: {mean_return:.2f}%
- Độ lệch chuẩn: {std_return:.2f}%
- VaR 95%: {var_95:.2f}%
- CVaR 95%: {cvar_95:.2f}%
- Biến động theo năm: {annualized_vol:.2f}%
- Tỷ lệ ngày tăng giá: {positive_days_pct:.1f}%

### YÊU CẦU PHÂN TÍCH:
1. **Hình dạng phân phối**: Có lệch (skewed) không? Có đuôi dày (fat tails)?
2. **Đánh giá VaR/CVaR**: 
   - Với $10,000 đầu tư, mức lỗ tối đa 1 ngày ở 95% confidence?
   - CVaR cho thấy gì về "worst case scenario"?
3. **Risk-Reward**: Lợi nhuận trung bình có đủ bù đắp rủi ro không?
4. **Khuyến nghị position sizing**: Với mức chấp nhận rủi ro 2%/giao dịch?
""",

    # -------------------------------------------------------------------------
    # EDA: Correlation Analysis
    # -------------------------------------------------------------------------
    
    "correlation_matrix": """## PHÂN TÍCH MA TRẬN TƯƠNG QUAN

**Tiêu đề:** {chart_title}

### DỮ LIỆU TỪ BIỂU ĐỒ:
- Tương quan trung bình: {avg_correlation:.2f}
- Cặp tương quan cao nhất: {highest_pair} ({highest_corr:.2f})
- Cặp tương quan thấp nhất: {lowest_pair} ({lowest_corr:.2f})
- Số cặp có tương quan > 0.7: {high_corr_count}
- Số cặp có tương quan < 0.3: {low_corr_count}

### YÊU CẦU PHÂN TÍCH:
1. **Mức độ liên kết thị trường**: Các coin có di chuyển cùng nhau không?
2. **Tiềm năng đa dạng hóa**: 
   - Có thể giảm rủi ro bằng cách đa dạng hóa trong các coin này không?
   - Cặp coin nào tốt nhất cho đa dạng hóa?
3. **Cảnh báo**: Những cặp nào tương quan quá cao (nguy hiểm cho portfolio)?
4. **Khuyến nghị xây dựng danh mục**: 
   - Nên kết hợp những coin nào?
   - Nên tránh kết hợp những coin nào?
""",

    "rolling_correlation": """## PHÂN TÍCH TƯƠNG QUAN LĂN VỚI BITCOIN

**Tiêu đề:** {chart_title}

### DỮ LIỆU TỪ BIỂU ĐỒ:
- Cửa sổ rolling: {window} ngày
- Tương quan trung bình với BTC:
{correlation_summary}
- Coin có tương quan ổn định nhất: {most_stable_coin}
- Coin có tương quan biến động nhất: {most_volatile_coin}

### YÊU CẦU PHÂN TÍCH:
1. **Sự phụ thuộc vào Bitcoin**: Các altcoin có theo sát BTC không?
2. **Xu hướng thay đổi**: Tương quan đang tăng hay giảm theo thời gian?
3. **Cơ hội decoupling**: 
   - Coin nào có khả năng hoạt động độc lập với BTC?
   - Điều này có ý nghĩa gì trong bear/bull market?
4. **Chiến lược giao dịch**: Nên sử dụng BTC làm chỉ báo dẫn dắt như thế nào?
""",

    # -------------------------------------------------------------------------
    # EDA: Price & Volume Analysis
    # -------------------------------------------------------------------------
    
    "price_ma": """## PHÂN TÍCH GIÁ VÀ ĐƯỜNG TRUNG BÌNH ĐỘNG (MA)

**Coin:** {coin}
**Tiêu đề:** {chart_title}

### DỮ LIỆU TỪ BIỂU ĐỒ:
- Giá hiện tại: ${current_price:.2f}
- MA20 (ngắn hạn): ${ma20:.2f}
- MA50 (trung hạn): ${ma50:.2f}
- MA200 (dài hạn): ${ma200:.2f}
- Vị trí giá so với MA20: {price_vs_ma20}
- Vị trí giá so với MA50: {price_vs_ma50}
- Vị trí giá so với MA200: {price_vs_ma200}
- Golden Cross/Death Cross gần nhất: {cross_signal}

### YÊU CẦU PHÂN TÍCH:
1. **Xu hướng tổng thể**: 
   - Ngắn hạn (MA20)?
   - Trung hạn (MA50)?
   - Dài hạn (MA200)?
2. **Tín hiệu kỹ thuật**: 
   - Có Golden Cross (MA ngắn cắt lên MA dài) hay Death Cross?
   - Giá đang test vùng hỗ trợ/kháng cự nào?
3. **Điểm vào lệnh tiềm năng**: 
   - Mức giá nào là vùng mua tiềm năng?
   - Mức giá nào nên cẩn trọng?
4. **Khuyến nghị**: MUA / BÁN / GIỮ và lý do?
""",

    "volume_analysis": """## PHÂN TÍCH KHỐI LƯỢNG GIAO DỊCH

**Coin:** {coin}
**Tiêu đề:** {chart_title}

### DỮ LIỆU TỪ BIỂU ĐỒ:
- Khối lượng trung bình 20 ngày: {avg_volume_20d:,.0f}
- Khối lượng hiện tại so với TB: {volume_vs_avg:.1f}x
- Số đợt đột biến volume (Z > 2): {spike_count}
- Đột biến gần nhất: {latest_spike_date} ({latest_spike_zscore:.1f} Z-score)
- Xu hướng volume: {volume_trend}

### YÊU CẦU PHÂN TÍCH:
1. **Sức khỏe thị trường**: Volume hiện tại cho thấy mức quan tâm như thế nào?
2. **Xác nhận xu hướng**: 
   - Volume có xác nhận xu hướng giá không?
   - Giá tăng + Volume tăng = tốt? Giá tăng + Volume giảm = cẩn trọng?
3. **Đột biến volume**: Các đợt spike có ý nghĩa gì?
4. **Dự báo**: Dựa trên volume pattern, thị trường có thể diễn biến thế nào?
""",

    "returns_histogram": """## PHÂN TÍCH HISTOGRAM LỢI NHUẬN

**Coin:** {coin}
**Tiêu đề:** {chart_title}

### DỮ LIỆU TỪ BIỂU ĐỒ:
- Lợi nhuận TB hàng ngày: {mean_return:.2f}%
- Độ lệch chuẩn: {std_return:.2f}%
- Ngày tăng giá: {positive_days} ({positive_pct:.1f}%)
- Ngày giảm giá: {negative_days} ({negative_pct:.1f}%)
- Lợi nhuận cao nhất 1 ngày: {max_return:.2f}%
- Lỗ lớn nhất 1 ngày: {min_return:.2f}%

### YÊU CẦU PHÂN TÍCH:
1. **Đặc điểm phân phối**: Có cân đối không? Lệch về phía nào?
2. **Tỷ lệ thắng/thua**: Xác suất có lợi nhuận dương trong 1 ngày?
3. **Rủi ro đuôi**: Mức lỗ cực đoan có thể xảy ra?
4. **Kỳ vọng toán học**: Với phân phối này, chiến lược nào phù hợp?
""",

    # -------------------------------------------------------------------------
    # Portfolio Analysis
    # -------------------------------------------------------------------------
    
    "portfolio_returns": """## PHÂN TÍCH LỢI NHUẬN DANH MỤC

**Tiêu đề:** {chart_title}

### DỮ LIỆU TỪ BIỂU ĐỒ:
- Chiến lược so sánh: {strategies}
- Lợi nhuận tích lũy:
{returns_summary}
- Chiến lược tốt nhất: {best_strategy} ({best_return:.2f}%)
- Chiến lược kém nhất: {worst_strategy} ({worst_return:.2f}%)
- Drawdown lớn nhất của mỗi chiến lược:
{drawdown_summary}

### YÊU CẦU PHÂN TÍCH:
1. **So sánh hiệu suất**: Chiến lược nào vượt trội?
2. **Risk-adjusted returns**: Khi xét đến rủi ro, chiến lược nào thực sự tốt hơn?
3. **Phù hợp với ai**: 
   - Nhà đầu tư chấp nhận rủi ro cao?
   - Nhà đầu tư bảo thủ?
4. **Khuyến nghị**: Nên chọn chiến lược nào và tại sao?
""",

    # -------------------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------------------
    
    "prediction_chart": """## PHÂN TÍCH DỰ ĐOÁN GIÁ TỪ MÔ HÌNH AI

**Coin:** {coin}
**Tiêu đề:** {chart_title}

### DỮ LIỆU TỪ BIỂU ĐỒ:
- Mô hình sử dụng: {model_name}
- Giá hiện tại: ${current_price:.2f}
- Dự đoán {forecast_days} ngày tới:
{predictions_summary}
- Giá dự đoán cuối kỳ: ${final_predicted_price:.2f}
- Biến động dự kiến: {expected_change:+.2f}% ({expected_change_usd:+.2f} USD)
- Xu hướng dự đoán: {trend_direction}

### YÊU CẦU PHÂN TÍCH:
1. **Đánh giá dự đoán**: Dựa trên xu hướng gần đây, dự đoán có hợp lý không?
2. **Độ tin cậy**: 
   - Những yếu tố nào có thể làm dự đoán sai?
   - Mô hình AI có giới hạn gì?
3. **Kịch bản**: 
   - Nếu dự đoán đúng: Chiến lược giao dịch?
   - Nếu dự đoán sai: Cách phòng ngừa?
4. **Khuyến nghị hành động**: 
   - Entry point?
   - Stop-loss?
   - Take-profit?
""",

    # -------------------------------------------------------------------------
    # Quant Metrics
    # -------------------------------------------------------------------------
    
    "quant_metrics": """## PHÂN TÍCH CHỈ SỐ ĐỊNH LƯỢNG

**Tiêu đề:** {chart_title}

### DỮ LIỆU TỪ BIỂU ĐỒ:
Bảng xếp hạng các coin theo chỉ số risk-adjusted:

| Coin | Sharpe | Sortino | Calmar | Max DD |
{metrics_table}

- Coin có Sharpe cao nhất: {best_sharpe_coin} ({best_sharpe:.2f})
- Coin có Sortino cao nhất: {best_sortino_coin} ({best_sortino:.2f})
- Coin có Max Drawdown thấp nhất: {lowest_dd_coin} ({lowest_dd:.2f}%)

### GIẢI THÍCH CHỈ SỐ:
- **Sharpe Ratio**: Lợi nhuận trên mỗi đơn vị rủi ro tổng thể (>1 = tốt, >2 = xuất sắc)
- **Sortino Ratio**: Như Sharpe nhưng chỉ tính rủi ro giảm giá (tốt hơn cho crypto)
- **Calmar Ratio**: Lợi nhuận / Max Drawdown (đánh giá recovery)

### YÊU CẦU PHÂN TÍCH:
1. **Xếp hạng tổng hợp**: Coin nào có chỉ số tốt nhất?
2. **Trade-off**: So sánh risk vs return cho từng coin?
3. **Phù hợp với profile đầu tư**: 
   - Conservative: Coin nào?
   - Aggressive: Coin nào?
4. **Khuyến nghị cấu trúc portfolio**: Tỷ trọng phân bổ?
""",

    # -------------------------------------------------------------------------
    # Compare Models
    # -------------------------------------------------------------------------
    
    "model_comparison": """## PHÂN TÍCH SO SÁNH CÁC MÔ HÌNH DỰ ĐOÁN

**Coin:** {coin}
**Tiêu đề:** {chart_title}

### DỮ LIỆU TỪ BIỂU ĐỒ:
Hiệu suất các mô hình:

| Model | RMSE | MAE | MAPE | Direction Acc |
{models_table}

- Mô hình có RMSE thấp nhất: {best_rmse_model}
- Mô hình có Direction Accuracy cao nhất: {best_direction_model}
- Mô hình baseline (Naive): RMSE = {naive_rmse:.4f}

### YÊU CẦU PHÂN TÍCH:
1. **Đánh giá tổng quan**: Mô hình nào hoạt động tốt nhất?
2. **So với baseline**: Các mô hình ML có vượt trội Naive không? Vượt bao nhiêu %?
3. **Direction Accuracy**: Quan trọng cho trading, mô hình nào đoán đúng xu hướng nhất?
4. **Khuyến nghị sử dụng**: 
   - Nên tin tưởng mô hình nào?
   - Khi nào nên/không nên dựa vào dự đoán AI?
""",

    # -------------------------------------------------------------------------
    # Sentiment Analysis
    # -------------------------------------------------------------------------
    
    "sentiment_fng": """## PHÂN TÍCH FEAR & GREED INDEX

**Tiêu đề:** {chart_title}

### DỮ LIỆU TỪ BIỂU ĐỒ:
- Fear & Greed Index hiện tại: {current_fng}
- Phân loại: {fng_classification}
- F&G trung bình 7 ngày: {fng_7d_avg:.1f}
- F&G trung bình 30 ngày: {fng_30d_avg:.1f}
- Xu hướng sentiment: {sentiment_trend}
- Tương quan F&G với lợi nhuận: {fng_return_correlation:.2f}

### THANG ĐO FEAR & GREED:
- 0-24: Extreme Fear (Sợ hãi cực độ) - Có thể là cơ hội mua
- 25-49: Fear (Sợ hãi)
- 50: Neutral (Trung lập)
- 51-74: Greed (Tham lam)
- 75-100: Extreme Greed (Tham lam cực độ) - Cẩn trọng bong bóng

### YÊU CẦU PHÂN TÍCH:
1. **Tâm lý thị trường hiện tại**: Thị trường đang ở trạng thái nào?
2. **Contrarian signal**: 
   - "Buy when others are fearful, sell when others are greedy" - Áp dụng không?
3. **Xu hướng sentiment**: Đang chuyển từ fear sang greed hay ngược lại?
4. **Khuyến nghị**: Nên hành động thế nào dựa trên sentiment?
""",

    "news_sentiment": """## PHÂN TÍCH SENTIMENT TIN TỨC

**Tiêu đề:** {chart_title}

### DỮ LIỆU TỪ BIỂU ĐỒ:
- Tổng số tin tức phân tích: {total_articles}
- Sentiment trung bình: {avg_sentiment:.2f} (thang -1 đến +1)
- Tin tích cực: {positive_count} ({positive_pct:.1f}%)
- Tin tiêu cực: {negative_count} ({negative_pct:.1f}%)
- Tin trung lập: {neutral_count} ({neutral_pct:.1f}%)
- Xu hướng sentiment 7 ngày: {sentiment_7d_trend}

### TOP TIN TỨC:
{top_headlines}

### YÊU CẦU PHÂN TÍCH:
1. **Tổng quan tin tức**: Media đang nói gì về crypto/coin này?
2. **Sentiment tổng thể**: Tích cực, tiêu cực hay trung lập?
3. **Tác động giá**: Tin tức có thể ảnh hưởng giá như thế nào ngắn hạn?
4. **Lời khuyên**: Nên phản ứng thế nào với tin tức hiện tại?
""",

    # -------------------------------------------------------------------------
    # Factor Analysis
    # -------------------------------------------------------------------------
    
    "factor_scatter": """## PHÂN TÍCH SCATTER PLOT CÁC NHÂN TỐ

**Tiêu đề:** {chart_title}

### DỮ LIỆU TỪ BIỂU ĐỒ:
- Trục X: {x_factor}
- Trục Y: {y_factor}
- Số coin trong phân tích: {coin_count}

Vị trí các coin:
{scatter_data}

### YÊU CẦU PHÂN TÍCH:
1. **Phân bố các coin**: Có nhóm (cluster) nào rõ ràng không?
2. **Outliers**: Coin nào nổi bật (tốt hoặc xấu)?
3. **Trade-off giữa 2 nhân tố**: Có thể tối ưu cả hai không?
4. **Khuyến nghị đầu tư**: Nên chọn coin ở vùng nào của scatter plot?
""",

    "factor_cluster": """## PHÂN TÍCH CLUSTER CÁC COIN

**Tiêu đề:** {chart_title}

### DỮ LIỆU TỪ BIỂU ĐỒ:
- Số cluster: {n_clusters}
- Các nhân tố sử dụng: {factors_used}

Chi tiết từng cluster:
{cluster_details}

### YÊU CẦU PHÂN TÍCH:
1. **Đặc điểm từng cluster**: Mỗi nhóm có điểm chung gì?
2. **Cluster hấp dẫn nhất**: Nhóm nào có tiềm năng đầu tư tốt nhất?
3. **Cluster rủi ro nhất**: Nhóm nào nên tránh hoặc cẩn trọng?
4. **Đa dạng hóa**: Nên chọn coin từ các cluster khác nhau như thế nào?
"""
}


def get_prompt(chart_type: str) -> str:
    """Lấy prompt template cho loại biểu đồ cụ thể.
    
    Args:
        chart_type: Loại biểu đồ (key trong CHART_PROMPTS)
        
    Returns:
        Prompt template string
    """
    return CHART_PROMPTS.get(chart_type, "")


def get_system_prompt() -> str:
    """Lấy system prompt cho ChatGPT."""
    return SYSTEM_PROMPT


def list_available_prompts() -> list:
    """Liệt kê tất cả các loại biểu đồ có prompt."""
    return list(CHART_PROMPTS.keys())
