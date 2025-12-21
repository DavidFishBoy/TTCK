# 📚 Deep-Learning-Crypto - Tài Liệu Cấu Trúc Project Chi Tiết

## 📖 Mục Lục
- [1. Tổng Quan Project](#1-tổng-quan-project)
- [2. Cấu Trúc Thư Mục](#2-cấu-trúc-thư-mục)
- [3. Module configs](#3-module-configs)
- [4. Module src](#4-module-src)
  - [4.1 data_collection](#41-data_collection)
  - [4.2 preprocessing](#42-preprocessing)
  - [4.3 training](#43-training)
  - [4.4 analysis](#44-analysis)
  - [4.5 monitoring (Dashboard)](#45-monitoring-dashboard)
  - [4.6 assistant](#46-assistant)
  - [4.7 visualization](#47-visualization)
  - [4.8 utils](#48-utils)
- [5. File Main.py](#5-file-mainpy)
- [6. Thư Mục Data](#6-thư-mục-data)
- [7. Hướng Dẫn Sử Dụng](#7-hướng-dẫn-sử-dụng)

---

## 1. Tổng Quan Project

**Deep-Learning-Crypto** là một dự án phân tích và dự đoán giá tiền điện tử sử dụng kỹ thuật Deep Learning và các mô hình thống kê. Project bao gồm:

- **Thu thập dữ liệu** từ nhiều nguồn API (Binance, CryptoCompare, NewsAPI, Alternative.me)
- **Xử lý và trích xuất đặc trưng** (Technical Indicators)
- **Huấn luyện mô hình** LSTM, ARIMA và các mô hình Baseline
- **Dashboard trực quan** với Streamlit hiển thị 12+ trang phân tích
- **AI Assistant** sử dụng RAG để tư vấn đầu tư

---

## 2. Cấu Trúc Thư Mục

```
Deep-Learning-Crypto/
├── configs/                  # Cấu hình hệ thống
│   └── config.yaml
├── data/                     # Dữ liệu
│   ├── raw/                  # Dữ liệu thô từ API
│   ├── processed/            # Dữ liệu đã xử lý
│   ├── cache/                # Cache
│   └── sentiment/            # Dữ liệu sentiment
├── logs/                     # Log files
├── models/                   # Các model đã train
├── results/                  # Kết quả dự đoán
├── src/                      # Source code chính
│   ├── analysis/             # Module phân tích tài chính
│   ├── assistant/            # AI Assistant (RAG)
│   ├── data_collection/      # Thu thập dữ liệu
│   ├── monitoring/           # Dashboard Streamlit
│   ├── preprocessing/        # Tiền xử lý dữ liệu
│   ├── training/             # Huấn luyện mô hình
│   ├── utils/                # Các tiện ích
│   └── visualization/        # Trực quan hóa
├── visualizations/           # Lưu biểu đồ xuất ra
├── main.py                   # Entry point chính
└── requirements.txt          # Dependencies
```

---

## 3. Module configs

### 📁 `configs/config.yaml`

File cấu hình chính của hệ thống, bao gồm:

| Section | Mô tả |
|---------|-------|
| `data.coins` | Danh sách coins hỗ trợ: ethereum, bitcoin, litecoin, binancecoin, cardano, solana, pancakeswap, axieinfinity, thesandbox |
| `data.days` | Số ngày dữ liệu lịch sử (mặc định: 1000) |
| `data.symbol_mapping` | Mapping symbol Binance (VD: BTCUSDT, ETHUSDT) |
| `data.coin_map` | Mapping tên coin → symbol (VD: bitcoin → BTC) |
| `model` | Cấu hình mô hình LSTM |
| `training` | Cấu hình huấn luyện |
| `preprocessing` | Cấu hình tiền xử lý |
| `paths` | Đường dẫn các thư mục |

**Cấu hình Model:**
- `sequence_length`: 60 ngày (lookback window)
- `prediction_length`: 5 ngày (multi-horizon forecast)
- `target_type`: "log_return" (dự đoán log returns thay vì giá)
- `lstm_units`: [128, 64]
- `dropout_rate`: 0.3
- `learning_rate`: 0.0005

---

## 4. Module src

### 4.1 data_collection

Module thu thập dữ liệu từ nhiều nguồn API.

---

#### 📄 `data_collector.py`

**Chức năng:** Thu thập dữ liệu giá lịch sử từ Binance và dữ liệu market cap từ CryptoCompare.

**APIs sử dụng:**

| API | Endpoint | Mô tả |
|-----|----------|-------|
| **Binance API** | `https://api.binance.com/api/v3/klines` | Dữ liệu OHLCV (Open, High, Low, Close, Volume) |
| **CryptoCompare API** | `https://min-api.cryptocompare.com/data/v2/histoday` | Dữ liệu Market Cap lịch sử |

**Class `DataCollector`:**

| Method | Mô tả |
|--------|-------|
| `fetch_binance_data()` | Lấy dữ liệu OHLCV từ Binance |
| `fetch_cryptocompare_market_cap()` | Lấy dữ liệu market cap |
| `process_raw_data()` | Xử lý dữ liệu thô (fill NaN, cleaning) |
| `handle_outliers()` | Phát hiện và xử lý outliers bằng IQR |
| `collect_all_data()` | Thu thập tất cả dữ liệu cho danh sách coins |

**Dữ liệu thu thập:**
- `open`, `high`, `low`, `close`: Giá OHLC
- `volume`: Khối lượng giao dịch
- `market_cap`: Vốn hóa thị trường
- `quote_volume`: Khối lượng giao dịch tính theo quote asset
- `number_of_trades`: Số lượng giao dịch

---

#### 📄 `news_collector.py`

**Chức năng:** Thu thập tin tức crypto và phân tích sentiment bằng VADER.

**API sử dụng:**

| API | Endpoint | Mô tả |
|-----|----------|-------|
| **NewsAPI** | `https://newsapi.org/v2/everything` | Tin tức crypto (Free tier: 100 requests/ngày) |

**Class `NewsCollector`:**

| Method | Mô tả |
|--------|-------|
| `fetch_news()` | Lấy tin tức từ NewsAPI theo keywords |
| `score_sentiment()` | Chấm điểm sentiment bằng VADER |
| `get_sentiment_label()` | Phân loại: positive/negative/neutral |
| `process_articles()` | Xử lý và chấm điểm sentiment cho articles |
| `aggregate_daily()` | Tổng hợp sentiment theo ngày |

**Sentiment Scoring (VADER):**
- `compound > 0.05`: Positive
- `compound < -0.05`: Negative
- `-0.05 ≤ compound ≤ 0.05`: Neutral

---

#### 📄 `sentiment_collector.py`

**Chức năng:** Thu thập Fear & Greed Index từ Alternative.me.

**API sử dụng:**

| API | Endpoint | Mô tả |
|-----|----------|-------|
| **Alternative.me** | `https://api.alternative.me/fng/?limit=0` | Fear & Greed Index (Miễn phí, không cần API key) |

**Class `SentimentCollector`:**

| Method | Mô tả |
|--------|-------|
| `fetch_fear_greed_index()` | Lấy dữ liệu FnG Index lịch sử |
| `add_lag_features()` | Thêm lag features (0, 1, 3, 7, 14 ngày) |
| `get_extreme_events()` | Phát hiện sự kiện extreme fear/greed |

**Fear & Greed Index Classification:**

| Giá trị | Phân loại |
|---------|-----------|
| 0-25 | Extreme Fear 🔴 |
| 26-49 | Fear 🟠 |
| 50 | Neutral 🟡 |
| 51-74 | Greed 🟢 |
| 75-100 | Extreme Greed 🔵 |

**Cách phân tích:**
- **Fear (< 50):** Thị trường lo sợ → Có thể là cơ hội mua (Buy the fear)
- **Greed (> 50):** Thị trường tham lam → Cẩn thận, có thể điều chỉnh
- **Extreme Fear (< 25):** Panic selling → Cơ hội mua tốt nếu fundamentals tốt
- **Extreme Greed (> 75):** FOMO → Cân nhắc chốt lời

---

### 4.2 preprocessing

Module tiền xử lý và trích xuất đặc trưng.

---

#### 📄 `feature_engineering.py`

**Chức năng:** Tính toán các Technical Indicators.

**Class `FeatureEngineer`:**

**Technical Indicators được tính:**

| Indicator | Công thức | Ý nghĩa |
|-----------|-----------|---------|
| **RSI (14)** | RSI = 100 - (100 / (1 + RS)) | Đo lường momentum; RSI > 70 = overbought, RSI < 30 = oversold |
| **MACD** | MACD = EMA(12) - EMA(26) | Đo lường xu hướng và momentum |
| **MACD Signal** | Signal = EMA(9) của MACD | Tín hiệu mua/bán khi MACD cắt Signal |
| **MACD Histogram** | = MACD - Signal | Độ mạnh của tín hiệu |
| **Bollinger Bands** | Upper/Lower = SMA(20) ± 2*STD | Đo lường biến động |
| **SMA (20, 50)** | Simple Moving Average | Xu hướng ngắn/trung hạn |
| **ROC** | Rate of Change | Tốc độ thay đổi giá |
| **Volume MA** | Moving Average của Volume | Xu hướng khối lượng |
| **Volume ROC** | Rate of Change của Volume | Đột biến khối lượng |

**Cách đọc các chỉ báo:**

**RSI (Relative Strength Index):**
- `RSI > 70`: Overbought → Có thể giảm
- `RSI < 30`: Oversold → Có thể tăng
- `RSI = 50`: Trung tính

**MACD:**
- MACD > Signal + Histogram > 0: Bullish
- MACD < Signal + Histogram < 0: Bearish
- MACD cắt lên Signal: Tín hiệu mua
- MACD cắt xuống Signal: Tín hiệu bán

**Bollinger Bands:**
- Giá chạm Upper Band: Có thể overbought
- Giá chạm Lower Band: Có thể oversold
- Bands thu hẹp: Biến động thấp, chuẩn bị breakout
- Bands mở rộng: Biến động cao

---

#### 📄 `pipeline.py`

**Chức năng:** Pipeline xử lý dữ liệu end-to-end.

**Class `Pipeline`:**

| Method | Mô tả |
|--------|-------|
| `validate_data()` | Kiểm tra dữ liệu đầu vào |
| `create_features()` | Gọi FeatureEngineer để tạo indicators |
| `fit_normalize_features()` | Fit và normalize features |
| `normalize_features()` | Transform features |
| `prepare_sequences()` | Tạo sequences cho LSTM (shape: samples × 60 × features) |
| `split_data()` | Chia train/val/test |
| `inverse_transform_predictions()` | Chuyển log returns → giá thực |

**Scaling:**
- Features: StandardScaler hoặc MinMaxScaler
- Target: RobustScaler (chống outliers)

**Output format:**
- `X_train/X_val/X_test`: shape (samples, 60, num_features)
- `y_train/y_val/y_test`: shape (samples, 5) - 5-day log returns

---

### 4.3 training

Module huấn luyện các mô hình dự đoán.

---

#### 📄 `lstm_model.py`

**Chức năng:** Mô hình LSTM cho dự đoán giá.

**Class `CryptoPredictor`:**

**Kiến trúc mô hình:**
```
Input (60 timesteps × features)
    ↓
LSTM Layer 1 (128 units, return_sequences=True)
    ↓
Dropout (0.3)
    ↓
LSTM Layer 2 (64 units)
    ↓
Dropout (0.3)
    ↓
Dense (64 units, ReLU)
    ↓
Output Dense (5 units) → 5-day log returns
```

**Loss Function:** Direction-Aware Huber Loss
- Kết hợp Huber Loss với penalty cho sai hướng
- Giảm sensitivity với outliers
- Khuyến khích mô hình dự đoán đúng hướng

**Metrics đánh giá:**

| Metric | Mô tả | Giá trị tốt |
|--------|-------|-------------|
| **MAE** | Mean Absolute Error | Càng thấp càng tốt |
| **RMSE** | Root Mean Square Error | Càng thấp càng tốt |
| **Directional Accuracy** | % dự đoán đúng hướng | > 50% (random = 50%) |

---

#### 📄 `baseline_models.py`

**Chức năng:** Các mô hình Baseline để so sánh.

**Các mô hình:**

| Model | Mô tả | Công thức |
|-------|-------|-----------|
| **NaiveModel** | Dự đoán = Giá hôm nay | `P(t+1) = P(t)` |
| **MovingAverageModel** | Dự đoán = MA của N ngày gần nhất | `P(t+1) = mean(P(t-N+1)...P(t))` |
| **ExponentialMovingAverageModel** | Dự đoán = EMA | Weighted average, recent = higher weight |

**Mục đích:** So sánh với LSTM để đánh giá mô hình có tốt hơn baseline không.

---

#### 📄 `arima_predictor.py`

**Chức năng:** Mô hình ARIMA cho time series forecasting.

**Class `ARIMAPredictor`:**

**ARIMA (AutoRegressive Integrated Moving Average):**
- **AR (p):** AutoRegressive - sử dụng p giá trị trước
- **I (d):** Integrated - số lần lấy sai phân
- **MA (q):** Moving Average - sử dụng q residuals trước

**Auto-ARIMA:** Tự động tìm tham số (p, d, q) tối ưu.

**Cách đánh giá:**
- So sánh AIC/BIC để chọn model tốt nhất
- Cross-validation với rolling window
- So sánh với Naive baseline

---

#### 📄 `trainer.py`

**Chức năng:** Huấn luyện mô hình với callbacks.

**Class `ModelTrainer`:**

**Callbacks:**
- **EarlyStopping:** Dừng training khi val_loss không cải thiện sau 30 epochs
- **ModelCheckpoint:** Lưu model tốt nhất
- **ReduceLROnPlateau:** Giảm learning rate khi plateau

---

### 4.4 analysis

Module phân tích tài chính chuyên sâu.

---

#### 📄 `market_analyzer.py`

**Chức năng:** Phân tích thị trường đa coin.

| Function | Mô tả |
|----------|-------|
| `load_all_coins_data()` | Load dữ liệu tất cả coins |
| `calculate_market_breadth()` | Tính % coins tăng/giảm |
| `create_returns_heatmap()` | Tạo heatmap returns |
| `calculate_correlation_matrix()` | Ma trận tương quan |
| `detect_volume_spike()` | Phát hiện đột biến volume |
| `identify_market_regime()` | Xác định Bull/Bear/Sideway |

**Market Regime:**
- **Bull:** Giá > MA200, breadth > 60%
- **Bear:** Giá < MA200, breadth < 40%
- **Sideway:** Các trường hợp còn lại

---

#### 📄 `financial_metrics.py`

**Chức năng:** Tính toán các chỉ số tài chính.

**Risk Metrics:**

| Metric | Công thức | Ý nghĩa | Giá trị tốt |
|--------|-----------|---------|-------------|
| **Volatility** | σ = std(returns) × √365 | Độ biến động hàng năm | Tùy risk appetite |
| **Max Drawdown** | (Peak - Trough) / Peak | Mức giảm tối đa từ đỉnh | < 30% |
| **VaR (95%)** | 5th percentile of returns | Mức lỗ tối đa với 95% confidence | Tùy portfolio |
| **CVaR** | Mean of returns < VaR | Expected shortfall | Tùy portfolio |

**Performance Metrics:**

| Metric | Công thức | Ý nghĩa | Giá trị tốt |
|--------|-----------|---------|-------------|
| **CAGR** | (End/Start)^(1/years) - 1 | Tốc độ tăng trưởng kép | > 0 |
| **Sharpe Ratio** | (Return - Rf) / σ | Return điều chỉnh rủi ro | > 1 = Good, > 2 = Excellent |
| **Sortino Ratio** | (Return - Rf) / σ_downside | Chỉ xét downside risk | > 2 = Good |
| **Calmar Ratio** | CAGR / Max Drawdown | Return/Risk tradeoff | > 1 = Good |

---

#### 📄 `factor_analyzer.py`

**Chức năng:** Phân tích nhân tố đầu tư.

**Factors:**

| Factor | Mô tả | Cách tính |
|--------|-------|-----------|
| **Momentum** | Xu hướng giá | Returns 30d, 90d |
| **Size** | Quy mô | log(Market Cap) |
| **Liquidity** | Thanh khoản | Volume / Market Cap |
| **Volatility** | Biến động | Annualized std |

**Phân tích:**
- **Clustering:** Nhóm coins theo factor characteristics
- **PCA:** Giảm chiều và tìm principal components
- **Factor Scatter:** Ma trận momentum vs volatility

---

#### 📄 `portfolio_engine.py`

**Chức năng:** Xây dựng và backtest danh mục đầu tư.

**Chiến lược phân bổ:**

| Strategy | Mô tả |
|----------|-------|
| **Equal Weight** | Phân bổ đều 1/N cho mỗi coin |
| **Risk Parity** | Phân bổ nghịch đảo với volatility |
| **Volatility Targeting** | Điều chỉnh để đạt target volatility |

**Backtest Metrics:**
- Portfolio equity curve
- Risk contribution của từng coin
- Drawdown analysis
- Rebalancing simulation

---

### 4.5 monitoring (Dashboard)

Dashboard Streamlit với 12 trang phân tích.

---

#### 📄 `dashboard.py`

**Chức năng:** Entry point cho Streamlit dashboard.

```bash
streamlit run src/monitoring/dashboard.py
```

---

#### 📁 `pages/`

##### 📄 `home.py` - Trang chủ

**Nội dung:** Giới thiệu các mục phân tích của dashboard.

---

##### 📄 `prediction.py` - Dự đoán giá

**Biểu đồ:**

| Biểu đồ | Mô tả | Cách đọc |
|---------|-------|----------|
| **Price Forecast Chart** | Biểu đồ giá lịch sử + dự đoán 5 ngày | Đường xanh = thực tế, đường đỏ đứt = dự đoán |
| **Confidence Interval** | Vùng tin cậy của dự đoán | Vùng tô màu cho thấy uncertainty |
| **Model Metrics** | Bảng MAE, RMSE, Direction Accuracy | So sánh các mô hình |

**Mô hình có sẵn:**
- LSTM (Deep Learning)
- ARIMA (Statistical)
- MA, EMA (Baseline)
- Naive (Baseline)

---

##### 📄 `sentiment_analysis.py` - Phân tích tâm lý

**Biểu đồ:**

| Biểu đồ | Mô tả | Cách đọc |
|---------|-------|----------|
| **Fear & Greed Timeline** | Timeline FnG Index | Màu sắc thể hiện mức độ fear/greed |
| **Sentiment-Return Overlay** | FnG vs Price returns | Tìm correlation giữa sentiment và price |
| **Lag Correlation Chart** | Correlation với lag 0-14 ngày | Lag nào có correlation cao nhất? |
| **News Sentiment Timeline** | Sentiment tin tức theo ngày | Positive/Negative ratio |
| **News Headlines** | Tin tức với sentiment score | Top headlines và phân loại |

**Cách phân tích:**
- Correlation âm: Extreme Fear → Có thể là điểm mua
- Lag correlation: Sentiment dẫn trước giá bao nhiêu ngày?
- Event study: Sau extreme fear/greed thì giá thay đổi thế nào?

---

##### 📄 `eda_price_volume.py` - Phân tích Giá & Volume

**Biểu đồ:**

| Biểu đồ | Mô tả | Cách đọc |
|---------|-------|----------|
| **Candlestick Chart** | Biểu đồ nến | Green = tăng, Red = giảm |
| **Volume Bars** | Khối lượng giao dịch | Spike = có sự kiện quan trọng |
| **Volume Z-Score** | Phát hiện volume bất thường | Z > 2 = spike đáng chú ý |
| **Price Statistics** | Min, Max, Mean, Std | Tổng quan phân bố giá |

**Cách phân tích Volume Spike:**
- Volume spike + Price tăng mạnh: Bullish confirmation
- Volume spike + Price giảm mạnh: Panic selling
- Volume spike + Price sideway: Accumulation/Distribution

---

##### 📄 `eda_correlation.py` - Phân tích Tương quan

**Biểu đồ:**

| Biểu đồ | Mô tả | Cách đọc |
|---------|-------|----------|
| **Correlation Heatmap** | Ma trận tương quan | Đỏ = tương quan dương, Xanh = âm |
| **Rolling Correlation with BTC** | Tương quan lăn với Bitcoin | Coin nào follow BTC, coin nào độc lập? |

**Cách phân tích:**
- **Correlation > 0.7:** Coins di chuyển cùng chiều mạnh → Ít đa dạng hóa
- **Correlation < 0.3:** Coins độc lập → Tốt cho đa dạng hóa
- **Correlation âm:** Hedge potential

**Gợi ý đa dạng hóa:**
- Chọn coins có correlation thấp với nhau
- Tránh hold nhiều coins có correlation > 0.8

---

##### 📄 `eda_volatility_risk.py` - Phân tích Biến động & Rủi ro

**Biểu đồ:**

| Biểu đồ | Mô tả | Cách đọc |
|---------|-------|----------|
| **Volatility Comparison** | So sánh volatility các coins | Coin nào biến động mạnh nhất? |
| **Drawdown Chart** | Mức giảm từ đỉnh | Drawdown sâu = rủi ro cao |
| **Rolling Volatility** | Volatility theo thời gian | Giai đoạn nào biến động cao? |
| **VaR/CVaR Table** | Risk metrics | VaR cho biết mức lỗ tối đa |

**Cách phân tích:**
- **High Volatility (> 100%):** Rủi ro cao, cần position size nhỏ
- **Max Drawdown > 50%:** Coin rất rủi ro
- **VaR 95% = -5%:** Có 5% khả năng mất > 5% trong 1 ngày

---

##### 📄 `factor_analysis.py` - Phân tích Nhân tố

**Biểu đồ:**

| Biểu đồ | Mô tả | Cách đọc |
|---------|-------|----------|
| **Factor Scatter Plot** | Momentum vs Volatility | 4 quadrants phân loại coins |
| **Factor Table** | Bảng factor scores | So sánh coins theo nhiều tiêu chí |
| **Cluster Visualization** | Phân nhóm coins | Coins trong cùng cluster có tính chất giống nhau |

**4 Quadrants:**
- **High Momentum + Low Vol:** Stars (Best picks)
- **High Momentum + High Vol:** Risky Winners
- **Low Momentum + Low Vol:** Stable Losers
- **Low Momentum + High Vol:** Avoid

---

##### 📄 `market_overview.py` - Tổng quan Thị trường

**Biểu đồ:**

| Biểu đồ | Mô tả | Cách đọc |
|---------|-------|----------|
| **Returns Heatmap** | Returns theo periods (1d, 7d, 30d) | Green = positive, Red = negative |
| **Market Breadth** | % coins tăng/giảm | > 60% tăng = bullish |
| **Market Regime Indicator** | Bull/Bear/Sideway | Xác định phase thị trường |
| **Ranking Table** | Xếp hạng theo metric | Top performers |

---

##### 📄 `portfolio_analysis.py` - Phân tích Danh mục

**Biểu đồ:**

| Biểu đồ | Mô tả | Cách đọc |
|---------|-------|----------|
| **Portfolio Equity Curve** | Giá trị danh mục theo thời gian | So sánh strategies |
| **Weight Allocation Pie** | Phân bổ trọng số | Risk Parity vs Equal Weight |
| **Risk Contribution** | Đóng góp rủi ro từng coin | Coin nào gây rủi ro nhiều nhất? |
| **Drawdown Comparison** | So sánh drawdown các strategies | Strategy nào ít drawdown hơn? |

---

##### 📄 `quant_metrics.py` - Chỉ số Định lượng

**Biểu đồ:**

| Biểu đồ | Mô tả | Cách đọc |
|---------|-------|----------|
| **Metrics Table** | Bảng tổng hợp tất cả metrics | So sánh toàn diện các coins |
| **Sharpe Ratio Comparison** | Bar chart Sharpe Ratio | Coin nào có risk-adjusted return tốt nhất? |
| **Risk-Return Scatter** | Return vs Volatility | Tìm coins ở efficient frontier |

---

##### 📄 `investment_insights.py` - Khuyến nghị Đầu tư

**Nội dung:**
- Tổng hợp phân tích từ các module
- Khuyến nghị dựa trên market regime
- Risk warnings
- Position sizing suggestions

---

##### 📄 `compare_models.py` - So sánh Mô hình

**Biểu đồ:**

| Biểu đồ | Mô tả | Cách đọc |
|---------|-------|----------|
| **Metrics Comparison Table** | MAE, RMSE, Dir Acc cho tất cả models | Model nào có metrics tốt nhất? |
| **Prediction Comparison Chart** | Actual vs Predicted cho từng model | "Fit" của model |
| **Error Distribution** | Phân bố error | Normal distribution = model ổn định |

**Model Evaluation:**
- **MAE thấp:** Dự đoán gần với thực tế
- **RMSE thấp:** Ít có lỗi lớn
- **Directional Accuracy > 55%:** Có khả năng dự đoán hướng

---

### 4.6 assistant

#### 📄 `rag_assistant.py`

**Chức năng:** AI Assistant sử dụng RAG (Retrieval-Augmented Generation).

**Class `RAGCryptoAssistant`:**

**Tính năng:**
- Chat với AI về cryptocurrencies
- Lấy lời khuyên đầu tư dựa trên dữ liệu
- So sánh nhiều coins
- Phân tích kỹ thuật tự động

**LLM sử dụng:** Google Gemini API

**Methods chính:**
- `chat()`: Chat tự do với AI
- `get_investment_advice()`: Lấy khuyến nghị đầu tư
- `compare_coins()`: So sánh coins
- `get_coin_analysis()`: Phân tích tổng quan 1 coin

---

### 4.7 visualization

#### 📄 `visualizer.py`

**Chức năng:** Tạo các biểu đồ với Plotly.

**Class `CryptoVisualizer`:**

| Method | Output |
|--------|--------|
| `plot_price_history()` | Candlestick + Volume chart |
| `plot_predictions()` | Actual vs Predicted line chart |
| `plot_training_history()` | Loss và metrics qua epochs |
| `plot_correlation_matrix()` | Heatmap correlation |
| `plot_performance_metrics()` | Bar chart metrics |

---

### 4.8 utils

#### 📄 `config.py`
Đọc và parse file config.yaml.

#### 📄 `logger.py`
Setup logging với rotation.

#### 📄 `custom_losses.py`
Custom loss functions cho LSTM:
- `direction_aware_huber_loss`: Huber loss + direction penalty
- `directional_accuracy`: Metric đánh giá % đúng hướng

#### 📄 `callbacks.py`
Custom Keras callbacks.

---

## 5. File Main.py

Entry point chính với các chế độ chạy:

```bash
# Thu thập dữ liệu
python main.py --mode collect-data

# Huấn luyện model
python main.py --mode train

# Dự đoán
python main.py --mode predict

# Full pipeline
python main.py --mode full-pipeline

# So sánh models
python main.py --mode compare-models

# Chọn coins cụ thể
python main.py --mode train --coins bitcoin ethereum
```

---

## 6. Thư Mục Data

```
data/
├── raw/
│   ├── train/          # Dữ liệu training (1000 ngày)
│   └── predict/        # Dữ liệu prediction (100 ngày gần nhất)
├── processed/
│   └── {coin}/
│       ├── X_train.npy, X_val.npy, X_test.npy
│       ├── y_train.npy, y_val.npy, y_test.npy
│       ├── scalers/
│       │   ├── feature_scaler.joblib
│       │   └── target_scaler.joblib
│       └── numeric_features.json
├── cache/              # Cache cho dashboard
└── sentiment/
    ├── fear_greed_daily.csv
    ├── news_articles.csv
    └── news_daily.csv
```

---

## 7. Hướng Dẫn Sử Dụng

### Cài đặt

```bash
pip install -r requirements.txt
```

### Chạy Dashboard

```bash
streamlit run src/monitoring/dashboard.py
```

### Training Model

```bash
python main.py --mode train --coins bitcoin ethereum
```

### Dự đoán

```bash
python main.py --mode predict --coins bitcoin
```

### Environment Variables

```
OPENAI_API_KEY=your_key          # Cho AI Assistant (đã chuyển sang Gemini)
GOOGLE_API_KEY=your_gemini_key   # Cho Gemini AI
NEWSAPI_API_KEY=your_key         # Cho NewsAPI (optional)
```

---

## 📊 Tổng Kết Các Biểu Đồ

| Trang | Số biểu đồ | Loại chính |
|-------|------------|------------|
| Prediction | 3 | Line, Table |
| Sentiment Analysis | 5 | Timeline, Bar, Scatter |
| EDA Price Volume | 4 | Candlestick, Bar |
| EDA Correlation | 2 | Heatmap, Line |
| EDA Volatility Risk | 4 | Bar, Line, Table |
| Factor Analysis | 3 | Scatter, Table, Cluster |
| Market Overview | 4 | Heatmap, Bar, Table |
| Portfolio Analysis | 4 | Line, Pie, Bar |
| Quant Metrics | 3 | Table, Bar, Scatter |
| Compare Models | 3 | Line, Bar, Table |
| **Tổng** | **~35** | Đa dạng |

---

*Tài liệu được tạo tự động bởi AI - Cập nhật: 2025-12-21*
